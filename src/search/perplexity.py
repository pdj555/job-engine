"""Perplexity API integration - research-grade AI search."""

from typing import Optional
import httpx

from config.settings import settings


class PerplexitySearch:
    """
    Perplexity API for deep research on opportunities.

    Use when you need synthesized, intelligent answers about:
    - Company research
    - Market analysis
    - Opportunity evaluation
    """

    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.perplexity_api_key
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )

    async def research(
        self,
        query: str,
        model: str = "llama-3.1-sonar-large-128k-online",
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Deep research query with Perplexity.

        Returns synthesized answer with citations.
        """
        if not self.api_key:
            return {"answer": "", "citations": []}

        system = system_prompt or (
            "You are a research assistant helping find high-value opportunities "
            "(jobs, grants, VC funding, contracts) that offer maximum income with "
            "minimum time investment. Be specific and cite sources."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "return_citations": True,
            "return_related_questions": True,
        }

        try:
            response = await self.client.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            return {
                "answer": data["choices"][0]["message"]["content"],
                "citations": data.get("citations", []),
                "related_questions": data.get("related_questions", []),
                "source": "perplexity"
            }

        except httpx.HTTPError as e:
            print(f"Perplexity error: {e}")
            return {"answer": "", "citations": [], "error": str(e)}

    async def find_opportunities(
        self,
        profile_summary: str,
        opportunity_type: str
    ) -> dict:
        """Find opportunities matching a profile."""
        query = f"""
        Find the best {opportunity_type} opportunities for someone with this profile:
        {profile_summary}

        Focus on:
        1. High compensation relative to time investment
        2. Remote/flexible arrangements
        3. Currently accepting applications
        4. Concrete opportunities with URLs where possible
        """
        return await self.research(query)

    async def evaluate_opportunity(
        self,
        opportunity_title: str,
        company: str,
        url: str
    ) -> dict:
        """Deep dive research on a specific opportunity."""
        query = f"""
        Research this opportunity:
        - Title: {opportunity_title}
        - Company: {company}
        - URL: {url}

        Provide:
        1. Company reputation and stability
        2. Typical compensation for this role
        3. Work-life balance expectations
        4. Red flags or concerns
        5. Application tips
        """
        return await self.research(query)

    async def market_research(self, industry: str, skill: str) -> dict:
        """Research market rates and demand for a skill/industry."""
        query = f"""
        What are the current market rates and demand for {skill} professionals
        in the {industry} industry?

        Include:
        1. Salary/rate ranges (hourly, annual)
        2. Remote work availability
        3. Demand trends
        4. Best platforms/sources to find opportunities
        5. Skills that command premium rates
        """
        return await self.research(query)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
