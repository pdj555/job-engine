"""Vector memory - remembers opportunities and learns your preferences."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from src.models import Opportunity, UserProfile
from config.settings import settings


class OpportunityMemory:
    """
    Semantic memory for opportunities.

    - Stores opportunities with embeddings for similarity search
    - Tracks what you've seen, applied to, liked, passed on
    - Learns from your feedback to improve recommendations
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Collections
        self.opportunities = self.client.get_or_create_collection(
            name="opportunities",
            metadata={"description": "All discovered opportunities"}
        )
        self.preferences = self.client.get_or_create_collection(
            name="preferences",
            metadata={"description": "Your learned preferences from feedback"}
        )
        self.interactions = self.client.get_or_create_collection(
            name="interactions",
            metadata={"description": "Your interactions with opportunities"}
        )

        # OpenAI for embeddings
        self.openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if not self.openai:
            # Return zero vector if no API key (for testing)
            return [0.0] * 1536

        response = self.openai.embeddings.create(
            model=settings.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def _opportunity_to_text(self, opp: Opportunity) -> str:
        """Convert opportunity to searchable text."""
        parts = [
            f"Title: {opp.title}",
            f"Type: {opp.opportunity_type.value}",
            f"Description: {opp.description}",
        ]
        if opp.company:
            parts.append(f"Company: {opp.company}")
        if opp.skills_required:
            parts.append(f"Skills: {', '.join(opp.skills_required)}")
        if opp.income_high:
            parts.append(f"Compensation: up to ${opp.income_high:,}")
        if opp.effort_level:
            parts.append(f"Effort: {opp.effort_level.value}")
        return "\n".join(parts)

    def _opportunity_id(self, opp: Opportunity) -> str:
        """Generate stable ID for opportunity."""
        content = f"{opp.url}:{opp.title}".encode()
        return hashlib.sha256(content).hexdigest()[:16]

    async def store_opportunity(self, opp: Opportunity) -> str:
        """Store an opportunity in memory."""
        opp_id = self._opportunity_id(opp)
        text = self._opportunity_to_text(opp)
        embedding = self._get_embedding(text)

        # Store with all metadata
        self.opportunities.upsert(
            ids=[opp_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "title": opp.title,
                "company": opp.company or "",
                "type": opp.opportunity_type.value,
                "url": opp.url,
                "income_low": opp.income_low or 0,
                "income_high": opp.income_high or 0,
                "remote": opp.remote,
                "source": opp.source,
                "relevance_score": opp.relevance_score,
                "overall_score": opp.overall_score,
                "stored_at": datetime.utcnow().isoformat(),
            }]
        )
        return opp_id

    async def store_opportunities(self, opps: list[Opportunity]) -> list[str]:
        """Store multiple opportunities."""
        return [await self.store_opportunity(opp) for opp in opps]

    async def find_similar(
        self,
        query: str,
        n_results: int = 10,
        min_score: float = 0.0
    ) -> list[dict]:
        """Find opportunities similar to a query."""
        embedding = self._get_embedding(query)

        results = self.opportunities.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        opportunities = []
        if results["ids"] and results["ids"][0]:
            for i, opp_id in enumerate(results["ids"][0]):
                # ChromaDB returns L2 distance, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 / (1 + distance)

                if similarity >= min_score:
                    opportunities.append({
                        "id": opp_id,
                        "document": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity": similarity
                    })

        return opportunities

    async def mark_seen(self, opp_id: str) -> None:
        """Mark opportunity as seen."""
        self.interactions.upsert(
            ids=[f"{opp_id}:seen"],
            documents=["seen"],
            metadatas=[{
                "opportunity_id": opp_id,
                "action": "seen",
                "timestamp": datetime.utcnow().isoformat()
            }]
        )

    async def mark_applied(self, opp_id: str) -> None:
        """Mark opportunity as applied to."""
        self.interactions.upsert(
            ids=[f"{opp_id}:applied"],
            documents=["applied"],
            metadatas=[{
                "opportunity_id": opp_id,
                "action": "applied",
                "timestamp": datetime.utcnow().isoformat()
            }]
        )

    async def mark_feedback(self, opp_id: str, liked: bool, reason: str = "") -> None:
        """Record feedback on an opportunity to improve future recommendations."""
        action = "liked" if liked else "passed"

        # Store interaction
        self.interactions.upsert(
            ids=[f"{opp_id}:{action}"],
            documents=[reason or action],
            metadatas=[{
                "opportunity_id": opp_id,
                "action": action,
                "liked": liked,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }]
        )

        # If there's a reason, store it as a preference signal
        if reason:
            embedding = self._get_embedding(reason)
            pref_type = "positive" if liked else "negative"
            self.preferences.upsert(
                ids=[f"pref:{opp_id}:{pref_type}"],
                embeddings=[embedding],
                documents=[reason],
                metadatas=[{
                    "type": pref_type,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            )

    async def get_seen_ids(self) -> set[str]:
        """Get IDs of all seen opportunities."""
        results = self.interactions.get(
            where={"action": "seen"},
            include=["metadatas"]
        )
        return {m["opportunity_id"] for m in (results.get("metadatas") or [])}

    async def learn_preferences(self, profile: UserProfile) -> str:
        """Store user profile as preference embeddings."""
        # Create searchable text from profile
        profile_text = f"""
        Skills: {', '.join(profile.skills)}
        Industries: {', '.join(profile.industries)}
        Minimum income: ${profile.min_income:,}
        Maximum hours: {profile.max_hours_weekly} per week
        Remote only: {profile.remote_only}
        Experience: {profile.experience_years} years
        Open to equity: {profile.interested_in_equity}
        Open to founding: {profile.interested_in_founding}
        Interested in grants: {profile.interested_in_grants}
        """

        embedding = self._get_embedding(profile_text)
        self.preferences.upsert(
            ids=["user_profile"],
            embeddings=[embedding],
            documents=[profile_text],
            metadatas=[{
                "type": "profile",
                "updated_at": datetime.utcnow().isoformat()
            }]
        )
        return "Profile stored successfully"

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "opportunities_stored": self.opportunities.count(),
            "interactions_recorded": self.interactions.count(),
            "preferences_learned": self.preferences.count(),
        }
