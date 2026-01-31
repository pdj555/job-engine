"""The engine. One class. Does everything."""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Optional

import httpx
from openai import AsyncOpenAI

from src.models import Opportunity
from config.settings import settings


class Engine:
    """
    The opportunity engine.

    find("AI engineer") -> ranked opportunities by $/hour

    That's it.
    """

    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.brave_key = settings.brave_api_key
        self.perplexity_key = settings.perplexity_api_key

    async def find(self, query: str, limit: int = 20) -> list[Opportunity]:
        """
        Find opportunities. Returns ranked by $/hour.

        That's all you need to know.
        """
        # Search everything in parallel
        raw_results = await self._search_all(query)

        # Extract structured data
        opportunities = await self._extract_opportunities(raw_results, query)

        # Rank by efficiency ($/hour)
        ranked = sorted(opportunities, key=lambda x: x.score(), reverse=True)

        return ranked[:limit]

    async def _search_all(self, query: str) -> list[dict]:
        """Search all sources in parallel."""
        searches = [
            self._search_brave(f"{query} remote job hiring"),
            self._search_brave(f"{query} freelance contract"),
            self._search_brave(f"{query} grant funding opportunity"),
            self._search_brave(f"{query} startup equity cofounder"),
        ]

        if self.perplexity_key:
            searches.append(self._search_perplexity(query))

        results = await asyncio.gather(*searches, return_exceptions=True)

        all_results = []
        for r in results:
            if isinstance(r, list):
                all_results.extend(r)

        # Dedupe by URL
        seen = set()
        unique = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)

        return unique

    async def _search_brave(self, query: str) -> list[dict]:
        """Search Brave."""
        if not self.brave_key:
            return []

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": 20, "freshness": "pm"},
                    headers={"X-Subscription-Token": self.brave_key},
                    timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()

                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "source": "brave"
                    }
                    for r in data.get("web", {}).get("results", [])
                ]
            except Exception as e:
                print(f"Brave error: {e}")
                return []

    async def _search_perplexity(self, query: str) -> list[dict]:
        """Deep search with Perplexity."""
        if not self.perplexity_key:
            return []

        prompt = f"""Find the best opportunities for: {query}

Focus on:
- High pay, low hours
- Remote/flexible
- Currently open

Return as JSON array with objects containing:
- title
- company (if known)
- url
- description
- estimated_pay (annual USD, just a number)
- estimated_hours_per_week (just a number)
- remote (boolean)

Only return the JSON array, nothing else."""

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-large-128k-online",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    },
                    timeout=60.0
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                # Parse JSON from response
                try:
                    # Find JSON array in response
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start >= 0 and end > start:
                        data = json.loads(content[start:end])
                        return [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "description": r.get("description", ""),
                                "pay": r.get("estimated_pay"),
                                "hours": r.get("estimated_hours_per_week"),
                                "remote": r.get("remote", True),
                                "source": "perplexity"
                            }
                            for r in data if r.get("url")
                        ]
                except json.JSONDecodeError:
                    pass
                return []
            except Exception as e:
                print(f"Perplexity error: {e}")
                return []

    async def _extract_opportunities(
        self,
        raw_results: list[dict],
        query: str
    ) -> list[Opportunity]:
        """Extract structured opportunities from raw results."""
        if not raw_results:
            return []

        # If we have OpenAI, use it to extract structured data
        if self.openai:
            return await self._extract_with_llm(raw_results, query)

        # Otherwise, create basic opportunities
        return [
            Opportunity(
                title=r.get("title", "Unknown"),
                url=r.get("url", ""),
                description=r.get("description", ""),
                company=r.get("company"),
                pay_high=r.get("pay"),
                hours_per_week=r.get("hours"),
                remote=r.get("remote", True),
                source=r.get("source", "")
            )
            for r in raw_results if r.get("url")
        ]

    async def _extract_with_llm(
        self,
        raw_results: list[dict],
        query: str
    ) -> list[Opportunity]:
        """Use LLM to extract structured opportunity data."""
        # Process in batches
        batch_size = 10
        all_opportunities = []

        for i in range(0, len(raw_results), batch_size):
            batch = raw_results[i:i + batch_size]
            opportunities = await self._extract_batch(batch, query)
            all_opportunities.extend(opportunities)

        return all_opportunities

    async def _extract_batch(
        self,
        batch: list[dict],
        query: str
    ) -> list[Opportunity]:
        """Extract opportunities from a batch of results."""
        batch_text = "\n\n".join([
            f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nDescription: {r.get('description', '')}"
            for r in batch
        ])

        prompt = f"""Extract opportunity data from these search results.
User is looking for: {query}

Results:
{batch_text}

For each result, extract:
- title
- company (if mentioned)
- url
- pay_low (annual USD estimate, null if unknown)
- pay_high (annual USD estimate, null if unknown)
- hours_per_week (estimate, null if unknown)
- remote (true/false, assume true if not specified)

Return JSON array. Be aggressive estimating pay/hours from context clues.
If it looks like full-time, assume 40hrs. If senior role, estimate $150k+.
Only return valid JSON array."""

        try:
            response = await self.openai.chat.completions.create(
                model=settings.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            # Parse response
            data = json.loads(content)
            items = data if isinstance(data, list) else data.get("opportunities", data.get("results", []))

            opportunities = []
            for item in items:
                if not item.get("url"):
                    continue

                opp = Opportunity(
                    title=item.get("title", "Unknown"),
                    company=item.get("company"),
                    url=item.get("url"),
                    description=item.get("description", ""),
                    pay_low=item.get("pay_low"),
                    pay_high=item.get("pay_high"),
                    hours_per_week=item.get("hours_per_week"),
                    remote=item.get("remote", True),
                    source="extracted"
                )
                opp.efficiency = opp.dollars_per_hour
                opportunities.append(opp)

            return opportunities

        except Exception as e:
            print(f"LLM extraction error: {e}")
            # Fallback to basic extraction
            return [
                Opportunity(
                    title=r.get("title", "Unknown"),
                    url=r.get("url", ""),
                    description=r.get("description", ""),
                    source=r.get("source", "")
                )
                for r in batch if r.get("url")
            ]

    async def research(self, opportunity: Opportunity) -> str:
        """Deep dive on a specific opportunity."""
        if not self.perplexity_key:
            return "Perplexity API key required for deep research."

        prompt = f"""Research this opportunity:
{opportunity.title} at {opportunity.company or 'Unknown Company'}
URL: {opportunity.url}

Tell me:
1. Is this legit?
2. What's realistic pay?
3. What's realistic hours?
4. Red flags?
5. Should I apply? Yes/No and why.

Be direct. No fluff."""

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-large-128k-online",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    },
                    timeout=60.0
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Research failed: {e}"


# Singleton
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Get the engine instance."""
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


async def find(query: str, limit: int = 20) -> list[Opportunity]:
    """Find opportunities. The only function you need."""
    return await get_engine().find(query, limit)
