"""Search aggregator - combines all search sources."""

import asyncio
import hashlib
from datetime import datetime
from typing import Optional

from openai import OpenAI

from src.models import (
    Opportunity,
    OpportunityType,
    EffortLevel,
    IncomeType,
    UserProfile,
    SearchQuery,
    SearchResult,
)
from src.search.brave import BraveSearch
from src.search.perplexity import PerplexitySearch
from config.settings import settings


class SearchAggregator:
    """
    Aggregates and normalizes results from all search sources.

    Transforms raw search results into structured Opportunity objects.
    """

    def __init__(self):
        self.brave = BraveSearch()
        self.perplexity = PerplexitySearch()
        self.openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    async def search(
        self,
        query: SearchQuery,
        profile: UserProfile
    ) -> SearchResult:
        """
        Execute search across all sources.

        Args:
            query: What to search for
            profile: User profile for context

        Returns:
            SearchResult with normalized opportunities
        """
        sources_searched = []
        raw_results = []

        # Determine what types to search
        types_to_search = query.opportunity_types or profile.opportunity_types

        # Execute searches in parallel
        tasks = []

        if OpportunityType.JOB in types_to_search or OpportunityType.CONTRACT in types_to_search:
            tasks.append(self._search_jobs(query.query, profile))
            sources_searched.append("jobs")

        if OpportunityType.FREELANCE in types_to_search:
            tasks.append(self._search_freelance(profile))
            sources_searched.append("freelance")

        if OpportunityType.GRANT in types_to_search:
            tasks.append(self._search_grants(query.query, profile))
            sources_searched.append("grants")

        if OpportunityType.VC_FUNDING in types_to_search or OpportunityType.ANGEL in types_to_search:
            tasks.append(self._search_funding(profile))
            sources_searched.append("funding")

        # Also do a general search
        tasks.append(self._search_general(query.query, profile))
        sources_searched.append("web")

        # Gather all results
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in all_results:
            if isinstance(result, list):
                raw_results.extend(result)
            elif isinstance(result, Exception):
                print(f"Search error: {result}")

        # Parse and normalize results
        opportunities = await self._parse_results(raw_results, profile)

        # Deduplicate
        seen_urls = set()
        unique_opps = []
        for opp in opportunities:
            if opp.url not in seen_urls:
                seen_urls.add(opp.url)
                unique_opps.append(opp)

        # Limit results
        unique_opps = unique_opps[:query.max_results]

        return SearchResult(
            opportunities=unique_opps,
            query=query,
            sources_searched=sources_searched,
            timestamp=datetime.utcnow()
        )

    async def _search_jobs(self, query: str, profile: UserProfile) -> list[dict]:
        """Search for job opportunities."""
        results = await self.brave.search_jobs(
            role=query,
            skills=profile.skills,
            remote=profile.remote_only
        )
        for r in results:
            r["opp_type"] = OpportunityType.JOB
        return results

    async def _search_freelance(self, profile: UserProfile) -> list[dict]:
        """Search for freelance opportunities."""
        results = await self.brave.search_freelance(skills=profile.skills)
        for r in results:
            r["opp_type"] = OpportunityType.FREELANCE
        return results

    async def _search_grants(self, query: str, profile: UserProfile) -> list[dict]:
        """Search for grant opportunities."""
        field = profile.industries[0] if profile.industries else "technology"
        results = await self.brave.search_grants(
            field=f"{query} {field}",
            keywords=profile.skills[:3]
        )
        for r in results:
            r["opp_type"] = OpportunityType.GRANT
        return results

    async def _search_funding(self, profile: UserProfile) -> list[dict]:
        """Search for VC/angel funding."""
        industry = profile.industries[0] if profile.industries else "technology"
        results = await self.brave.search_vc(industry=industry)
        for r in results:
            r["opp_type"] = OpportunityType.VC_FUNDING
        return results

    async def _search_general(self, query: str, profile: UserProfile) -> list[dict]:
        """General web search."""
        remote_str = "remote" if profile.remote_only else ""
        full_query = f"{query} opportunity {remote_str} hiring apply"
        results = await self.brave.search(full_query)
        for r in results:
            r["opp_type"] = OpportunityType.JOB  # Default, will be refined
        return results

    async def _parse_results(
        self,
        raw_results: list[dict],
        profile: UserProfile
    ) -> list[Opportunity]:
        """Parse raw results into Opportunity objects."""
        opportunities = []

        for raw in raw_results:
            if not raw.get("url"):
                continue

            # Create basic opportunity
            opp_id = hashlib.sha256(raw["url"].encode()).hexdigest()[:16]

            opp = Opportunity(
                id=opp_id,
                title=raw.get("title", "Unknown"),
                description=raw.get("description", ""),
                opportunity_type=raw.get("opp_type", OpportunityType.JOB),
                url=raw.get("url", ""),
                remote=profile.remote_only,  # Assume based on search
                source=raw.get("source", "unknown"),
            )

            # Use LLM to extract structured data if available
            if self.openai and raw.get("description"):
                enriched = await self._enrich_with_llm(opp, raw)
                if enriched:
                    opp = enriched

            opportunities.append(opp)

        return opportunities

    async def _enrich_with_llm(
        self,
        opp: Opportunity,
        raw: dict
    ) -> Optional[Opportunity]:
        """Use LLM to extract structured data from description."""
        try:
            prompt = f"""
            Extract structured opportunity data from this search result.

            Title: {raw.get('title', '')}
            Description: {raw.get('description', '')}
            URL: {raw.get('url', '')}

            Return JSON with these fields (use null if unknown):
            - company: string or null
            - income_low: number or null (annual USD)
            - income_high: number or null (annual USD)
            - effort_level: "minimal" | "light" | "moderate" | "full" | null
            - hours_per_week: number or null
            - remote: boolean
            - skills_required: string[] (max 5)
            - opportunity_type: "job" | "freelance" | "grant" | "vc_funding" | "contract"

            Only return valid JSON, no explanation.
            """

            response = self.openai.chat.completions.create(
                model=settings.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            import json
            data = json.loads(response.choices[0].message.content)

            # Update opportunity with extracted data
            if data.get("company"):
                opp.company = data["company"]
            if data.get("income_low"):
                opp.income_low = data["income_low"]
            if data.get("income_high"):
                opp.income_high = data["income_high"]
            if data.get("effort_level"):
                try:
                    opp.effort_level = EffortLevel(data["effort_level"])
                except ValueError:
                    pass
            if data.get("hours_per_week"):
                opp.hours_per_week = data["hours_per_week"]
            if data.get("remote") is not None:
                opp.remote = data["remote"]
            if data.get("skills_required"):
                opp.skills_required = data["skills_required"][:5]
            if data.get("opportunity_type"):
                try:
                    opp.opportunity_type = OpportunityType(data["opportunity_type"])
                except ValueError:
                    pass

            return opp

        except Exception as e:
            print(f"LLM enrichment error: {e}")
            return None

    async def deep_research(
        self,
        opportunity: Opportunity
    ) -> dict:
        """Do deep research on a specific opportunity."""
        return await self.perplexity.evaluate_opportunity(
            opportunity_title=opportunity.title,
            company=opportunity.company or "Unknown",
            url=opportunity.url
        )

    async def close(self):
        """Close all clients."""
        await self.brave.close()
        await self.perplexity.close()
