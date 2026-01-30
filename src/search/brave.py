"""Brave Search API integration - real-time web search."""

from typing import Optional
import httpx

from config.settings import settings


class BraveSearch:
    """
    Brave Search API for real-time opportunity discovery.

    Searches job boards, VC listings, grant databases, etc.
    """

    BASE_URL = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.brave_api_key
        self.client = httpx.AsyncClient(
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json"
            },
            timeout=30.0
        )

    async def search(
        self,
        query: str,
        count: int = 20,
        freshness: str = "pw",  # past week
        search_lang: str = "en",
    ) -> list[dict]:
        """
        Execute a web search.

        Args:
            query: Search query
            count: Number of results (max 20)
            freshness: Time filter (pd=day, pw=week, pm=month, py=year)
            search_lang: Language code

        Returns:
            List of search results with title, url, description
        """
        if not self.api_key:
            return []

        params = {
            "q": query,
            "count": min(count, 20),
            "freshness": freshness,
            "search_lang": search_lang,
            "text_decorations": False,
        }

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/web/search",
                params=params
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "age": item.get("age", ""),
                    "source": "brave"
                })

            return results

        except httpx.HTTPError as e:
            print(f"Brave search error: {e}")
            return []

    async def search_jobs(self, role: str, skills: list[str], remote: bool = True) -> list[dict]:
        """Search for job opportunities."""
        skill_str = " OR ".join(skills[:5]) if skills else ""
        remote_str = "remote" if remote else ""
        query = f"{role} {skill_str} job hiring {remote_str}".strip()
        return await self.search(query, freshness="pw")

    async def search_grants(self, field: str, keywords: list[str]) -> list[dict]:
        """Search for grants and funding opportunities."""
        kw_str = " ".join(keywords[:3]) if keywords else ""
        query = f"{field} grant funding opportunity {kw_str} 2024 2025"
        return await self.search(query, freshness="pm")

    async def search_vc(self, industry: str, stage: str = "seed") -> list[dict]:
        """Search for VC and angel investment opportunities."""
        query = f"{industry} {stage} funding startup investment opportunity open"
        return await self.search(query, freshness="pm")

    async def search_freelance(self, skills: list[str], platforms: bool = True) -> list[dict]:
        """Search for freelance/contract opportunities."""
        skill_str = " OR ".join(skills[:5]) if skills else ""
        platform_str = "site:upwork.com OR site:toptal.com OR site:flexjobs.com" if platforms else ""
        query = f"{skill_str} freelance contract remote {platform_str}".strip()
        return await self.search(query, freshness="pw")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
