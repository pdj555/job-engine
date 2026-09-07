"""The engine. One class. Does everything."""

import asyncio
import json
import re
from html import unescape
from typing import Optional
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI

from config.settings import settings
from src.compensation import (
    Compensation,
    ats_json_url,
    canonicalize_url,
    is_search_serp,
    parse_ats_json,
    parse_compensation,
    parse_job_posting,
)
from src.models import Opportunity

_LISTING_CAP = 600_000
_FETCH_CONCURRENCY = 8


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
        raw_results = await self._search_all(query)
        opportunities = await self._extract_opportunities(raw_results, query)
        await self.enrich(opportunities)
        return sorted(opportunities, key=lambda x: x.score(), reverse=True)[:limit]

    async def search_web(self, query: str) -> list[dict]:
        """One web search. Brave, or DuckDuckGo when no Brave key."""
        return await self._search_brave(query)

    async def enrich(self, opportunities: list[Opportunity]) -> None:
        """Fill missing pay from ATS JSON, then schema.org JobPosting HTML."""
        need = [o for o in opportunities if not o.pay]
        if not need:
            return
        sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=8.0,
            headers={"User-Agent": "JobEngine/1.0 (listing-schema)"},
            max_redirects=3,
        ) as client:

            async def one(opp: Opportunity) -> None:
                async with sem:
                    ats = await self._fetch_ats(opp.url, client)
                    if ats:
                        _apply_comp(opp, ats, "ats")
                        if ats.posted:
                            return
                    html = await self._fetch_listing(opp.url, client)
                if not html:
                    return
                _apply_comp(opp, parse_job_posting(html), "schema")

            await asyncio.gather(*(one(o) for o in need), return_exceptions=True)

    async def _fetch_ats(self, url: str, client: httpx.AsyncClient) -> Compensation | None:
        endpoint = ats_json_url(url)
        if not endpoint:
            return None
        try:
            resp = await client.get(endpoint, headers={"Accept": "application/json"})
            if resp.status_code >= 400:
                return None
            return parse_ats_json(url, resp.json())
        except Exception:
            return None

    async def _fetch_listing(self, url: str, client: httpx.AsyncClient) -> str | None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or "." not in parts.hostname:
            return None
        try:
            resp = await client.get(url)
            if resp.status_code >= 400:
                return None
            ctype = (resp.headers.get("content-type") or "").lower()
            if ctype and not any(kind in ctype for kind in ("html", "json", "xml", "text")):
                return None
            text = resp.text
            return text[:_LISTING_CAP] if len(text) > _LISTING_CAP else text
        except Exception:
            return None

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

        seen = set()
        unique = []
        for r in all_results:
            url = canonicalize_url(r.get("url", ""))
            if url and url not in seen:
                seen.add(url)
                unique.append({**r, "url": url})

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
        """Deep search with Perplexity. URLs and text only — never estimated pay."""
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
- remote (boolean)

Only return the JSON array, nothing else. Do not invent compensation or copy pay into titles."""

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

                try:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start >= 0 and end > start:
                        data = json.loads(content[start:end])
                        return [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "description": "",
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
        return [o for r in raw_results if (o := opportunity_from_raw(r))]

    async def _extract_with_llm(
        self,
        raw_results: list[dict],
        query: str
    ) -> list[Opportunity]:
        """Use LLM to extract structured opportunity data."""
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

Return JSON array. Do not invent compensation.
Only include urls that appear in the results."""

        try:
            response = await self.openai.chat.completions.create(
                model=settings.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            data = json.loads(content)
            items = data if isinstance(data, list) else data.get("opportunities", data.get("results", []))

            by_url = {canonicalize_url(r["url"]): r for r in batch if r.get("url")}
            opportunities = []
            for item in items:
                raw = by_url.get(canonicalize_url(item.get("url") or ""))
                if not raw:
                    continue
                parsed = opportunity_from_raw({**raw, "source": "extracted"})
                if not parsed:
                    continue
                title = item.get("title")
                if title and (parsed.pay or not parse_compensation(title).posted):
                    parsed.title = title
                if item.get("company"):
                    parsed.company = item["company"]
                if item.get("remote") is not None:
                    parsed.remote = bool(item["remote"])
                opportunities.append(parsed)
            return opportunities

        except Exception as e:
            print(f"LLM extraction error: {e}")
            return [o for r in batch if (o := opportunity_from_raw(r))]

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


def _parse_ddg_html(html: str) -> list[dict]:
    """Parse DuckDuckGo HTML results."""
    results: list[dict] = []
    for match in re.finditer(
        r'class="result__a"\s+href="([^"]+)"[^>]*>([^<]+)</a>',
        html,
    ):
        url = unescape(match.group(1))
        title = unescape(re.sub(r"\s+", " ", match.group(2)).strip())
        if not url or not title:
            continue
        if "duckduckgo.com/y.js" in url or "bing.com/aclick" in url:
            continue
        if url.startswith("//"):
            url = f"https:{url}"
        results.append(
            {
                "title": title,
                "url": url,
                "description": "",
                "source": "duckduckgo",
            }
        )

    for item in results:
        idx = html.find(item["url"])
        if idx < 0:
            continue
        snippet = re.search(
            r'class="result__snippet"[^>]*>([^<]+)',
            html[idx : idx + 1200],
        )
        if snippet:
            item["description"] = unescape(re.sub(r"\s+", " ", snippet.group(1)).strip())

    return results[:20]


def opportunity_from_raw(raw: dict, listing_text: str | None = None) -> Opportunity | None:
    """Build an opportunity from a search hit. Pay/hours only if the text states them."""
    url = canonicalize_url(raw.get("url", ""))
    if not url or is_search_serp(url):
        return None
    title = raw.get("title") or "Unknown"
    description = raw.get("description") or ""
    source = raw.get("source") or ""
    if listing_text is None and source == "perplexity":
        listing_text = ""
    blob = f"{title} {description}" if listing_text is None else listing_text
    parsed = parse_compensation(blob)
    remote = raw.get("remote")
    if remote is None:
        remote = _guess_remote(title, description)
    opp = Opportunity(
        title=title,
        url=url,
        description=description,
        company=raw.get("company"),
        pay_low=parsed.pay_low,
        pay_high=parsed.pay_high,
        hours_per_week=parsed.hours,
        remote=bool(remote),
        source=source,
        pay_source="posted" if parsed.posted else None,
        hours_source="posted" if parsed.hours else None,
    )
    opp.efficiency = opp.dollars_per_hour
    return opp


def _apply_comp(opp: Opportunity, parsed: Compensation, source: str) -> None:
    if parsed.posted:
        opp.pay_low = parsed.pay_low
        opp.pay_high = parsed.pay_high
        opp.pay_source = source
    if parsed.hours and not opp.hours_per_week:
        opp.hours_per_week = parsed.hours
        opp.hours_source = source
    if parsed.remote is not None:
        opp.remote = parsed.remote
    if parsed.company and not opp.company:
        opp.company = parsed.company
    if parsed.title and opp.title in {"Unknown", ""}:
        opp.title = parsed.title
    opp.efficiency = opp.dollars_per_hour


def _guess_remote(title: str, description: str) -> bool:
    text = f"{title} {description}".lower()
    if any(w in text for w in ("onsite", "on-site", "in-office", "in office", "hybrid")):
        return False
    return True


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
