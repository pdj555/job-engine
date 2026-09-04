"""The engine. One class. Does everything."""

import asyncio
import json
import re
from html import unescape
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

        # Dedupe by URL (trailing slash / case are the same listing)
        seen = set()
        unique = []
        for r in all_results:
            key = _normalize_url(r.get("url") or "")
            if key and key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    async def _search_brave(self, query: str) -> list[dict]:
        """Search Brave, or DuckDuckGo when no Brave key."""
        if not self.brave_key:
            return await self._search_ddg(query)

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

                results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": r.get("description", ""),
                        "source": "brave"
                    }
                    for r in data.get("web", {}).get("results", [])
                ]
                if results:
                    return results
                return await self._search_ddg(query)
            except Exception as e:
                print(f"Brave error: {e}")
                return await self._search_ddg(query)

    async def _search_ddg(self, query: str) -> list[dict]:
        """Free web search fallback."""
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query, "b": ""},
                    headers={"User-Agent": "Mozilla/5.0 (compatible; JobEngine/1.0)"},
                    timeout=30.0,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                return _parse_ddg_html(resp.text)
            except Exception as e:
                print(f"DDG error: {e}")
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

        if self.openai:
            return await self._extract_with_llm(raw_results, query)
        return [o for r in raw_results if (o := _heuristic_opportunity(r))]

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
- url (must be copied exactly from the result above)
- remote (true/false, assume true if not specified)

Do not estimate pay or hours — those are parsed from the listing text in code.
Return a JSON object {{"opportunities": [...]}}.
Only include urls that appear in the results."""

        try:
            response = await self.openai.chat.completions.create(
                model=settings.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            items = _items_from_llm(response.choices[0].message.content)
            by_url = {_normalize_url(r["url"]): r for r in batch if r.get("url")}
            seen: set[str] = set()
            opportunities = []
            for item in items:
                key = _normalize_url(item.get("url") or "")
                raw = by_url.get(key)
                if raw and key not in seen:
                    seen.add(key)
                    opportunities.append(_merge_extracted(raw, item))
            for key, raw in by_url.items():
                if key not in seen and (o := _heuristic_opportunity(raw)):
                    opportunities.append(o)
            if opportunities:
                return opportunities
        except Exception as e:
            print(f"LLM extraction error: {e}")

        return [o for r in batch if (o := _heuristic_opportunity(r))]

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


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/").casefold()


def _items_from_llm(content: Optional[str]) -> list:
    data = json.loads(content or "")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("opportunities", data.get("results", []))
        return items if isinstance(items, list) else []
    return []


def _heuristic_opportunity(raw: dict) -> Optional[Opportunity]:
    url = raw.get("url")
    if not url:
        return None
    title = raw.get("title") or "Unknown"
    desc = raw.get("description") or ""
    remote = raw.get("remote")
    if remote is None:
        remote = _guess_remote(title, desc)
    hours = raw.get("hours")
    if hours is None:
        hours = _guess_hours(title, desc)
    pay_low, pay_high = _compensation_from_raw(raw, title, desc, hours)
    opp = Opportunity(
        title=title,
        url=url,
        description=desc,
        company=raw.get("company"),
        pay_low=pay_low,
        pay_high=pay_high,
        hours_per_week=hours,
        remote=remote,
        source=raw.get("source") or "",
    )
    opp.efficiency = opp.refined_rate
    return opp


def _merge_extracted(raw: dict, item: dict) -> Opportunity:
    title = item.get("title") or raw.get("title") or "Unknown"
    desc = item.get("description") or raw.get("description") or ""
    company = item.get("company") if item.get("company") is not None else raw.get("company")
    guess_title = raw.get("title") or title
    guess_desc = raw.get("description") or desc
    hours = raw.get("hours")
    if hours is None:
        hours = _guess_hours(guess_title, guess_desc)
    pay_low, pay_high = _compensation_from_raw(raw, guess_title, guess_desc, hours)
    if item.get("remote") is not None:
        remote = bool(item["remote"])
    elif "remote" in raw:
        remote = bool(raw["remote"])
    else:
        remote = _guess_remote(guess_title, guess_desc)
    opp = Opportunity(
        title=title,
        company=company,
        url=raw["url"],
        description=desc,
        pay_low=pay_low,
        pay_high=pay_high,
        hours_per_week=hours,
        remote=remote,
        source=raw.get("source") or "extracted",
    )
    opp.efficiency = opp.refined_rate
    return opp


def _compensation_from_raw(
    raw: dict, title: str, description: str, hours: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """Structured source pay wins; otherwise parse listing text. Never invent."""
    if raw.get("pay") is not None:
        return None, raw["pay"]
    return _parse_pay(f"{title} {description}", hours)


def _visible_text(html: str) -> str:
    return unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))).strip()


def _parse_ddg_html(html: str) -> list[dict]:
    """Parse DuckDuckGo HTML results."""
    snippets: dict[str, str] = {}
    for match in re.finditer(
        r'class="result__snippet"\s+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    ):
        key = _normalize_url(unescape(match.group(1)))
        if key and key not in snippets:
            snippets[key] = _visible_text(match.group(2))

    results: list[dict] = []
    seen: set[str] = set()
    for match in re.finditer(
        r'class="result__a"\s+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    ):
        url = unescape(match.group(1))
        title = _visible_text(match.group(2))
        if not url or not title:
            continue
        if "duckduckgo.com/y.js" in url or "bing.com/aclick" in url:
            continue
        if url.startswith("//"):
            url = f"https:{url}"
        key = _normalize_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "title": title,
                "url": url,
                "description": snippets.get(key, ""),
                "source": "duckduckgo",
            }
        )

    return results[:20]


_HOURLY_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:/|\s+per\s+)\s*h(?:r|our)s?\b",
    re.I,
)
_RANGE_K_RE = re.compile(
    r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k?\s*[-–—]\s*\$?\s*(\d{2,3}(?:\.\d+)?)\s*k\b",
    re.I,
)
_ANNUAL_K_RE = re.compile(r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k\b", re.I)
_ANNUAL_FULL_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\b")
_HOURS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:hrs?|hours?)\s*(?:/|\s*per\s*)?\s*(?:wk|week)\b",
    re.I,
)


def _money(raw: str) -> float:
    return float(raw.replace(",", ""))


def _parse_pay(
    text: str, hours: Optional[int] = None
) -> tuple[Optional[int], Optional[int]]:
    """(pay_low, pay_high) annual USD from listing text. (None, None) if unknown."""
    hourly = _HOURLY_RE.search(text)
    if hourly:
        rate = _money(hourly.group(1))
        if 10 <= rate <= 1000:
            annual = int(rate * (hours or 40) * 50)
            return None, annual
    ranged = _RANGE_K_RE.search(text)
    if ranged:
        low, high = int(_money(ranged.group(1)) * 1000), int(_money(ranged.group(2)) * 1000)
        if 10_000 <= low <= high <= 2_000_000:
            return low, high
    k = _ANNUAL_K_RE.search(text)
    if k:
        annual = int(_money(k.group(1)) * 1000)
        if 10_000 <= annual <= 2_000_000:
            return None, annual
    full = _ANNUAL_FULL_RE.search(text)
    if full:
        annual = int(_money(full.group(1)))
        if 10_000 <= annual <= 2_000_000:
            return None, annual
    return None, None


def _guess_pay(title: str, description: str, hours: Optional[int] = None) -> Optional[int]:
    """Best annual pay in the listing text, or None. Does not invent a number."""
    low, high = _parse_pay(f"{title} {description}", hours)
    return high or low


def _guess_hours(title: str, description: str) -> Optional[int]:
    """Hours from the listing text, or None. Does not assume full-time."""
    text = f"{title} {description}"
    match = _HOURS_RE.search(text)
    if match:
        n = int(match.group(1))
        if 1 <= n <= 80:
            return n
    lower = text.lower()
    if "part-time" in lower or "part time" in lower:
        return 20
    if "full-time" in lower or "full time" in lower:
        return 40
    return None


def _guess_remote(title: str, description: str) -> bool:
    text = f"{title} {description}".lower()
    if any(w in text for w in ("onsite", "on-site", "in-office", "in office", "hybrid")):
        return False
    return True


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
