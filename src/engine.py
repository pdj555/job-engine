"""The engine. One class. Does everything."""

import asyncio
import calendar
import json
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from html import unescape
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from openai import AsyncOpenAI

from src.models import Opportunity
from config.settings import settings

_LISTING_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; JobEngine/1.0)"}


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
        self._ddg_sem = asyncio.Semaphore(3)

    async def find(self, query: str, limit: int = 20) -> list[Opportunity]:
        """
        Find opportunities. Returns ranked by $/hour.

        That's all you need to know.
        """
        # Search everything in parallel
        raw_results = await self._search_all(query)

        # Extract structured data
        opportunities = await self._extract_opportunities(raw_results, query)
        await self._enrich_pay(opportunities)
        ranked = sorted(opportunities, key=lambda x: x.score(), reverse=True)
        return _dedupe_opportunities(ranked)[:limit]

    async def _enrich_pay(self, opps: list[Opportunity]) -> None:
        """Fill pay/hours/company from the listing page. Never invent."""
        if opps:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=8.0,
                headers=_LISTING_HEADERS,
            ) as client:
                self._http_client = client
                try:
                    texts = await asyncio.gather(
                        *(self._listing_text(o.url) for o in opps),
                        return_exceptions=True,
                    )
                finally:
                    self._http_client = None
            gone = []
            for o, text in zip(opps, texts):
                if text is None:
                    gone.append(o)
                    continue
                if not isinstance(text, str) or not text:
                    continue
                if _html_is_index(text, o.url) or _html_is_gone(text):
                    gone.append(o)
                    continue
                listed = _apply_listing(o, text)
                if _foreign_salary(text) and not listed:
                    gone.append(o)
            if gone:
                opps[:] = [o for o in opps if o not in gone]
        _unify_board_companies(opps)

    async def _listing_text(self, url: str) -> Optional[str]:
        if not _public_http_url(url):
            return ""
        client = getattr(self, "_http_client", None)

        async def fetch(target: str) -> Optional[str]:
            if client is not None:
                return await _http_get_text(client, target)
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=8.0,
                    headers=_LISTING_HEADERS,
                ) as owned:
                    return await _http_get_text(owned, target)
            except Exception:
                return ""

        api = _greenhouse_api_url(url)
        if api:
            raw = await fetch(api)
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and not data.get("error"):
                    return _greenhouse_to_html(data)
        if _greenhouse_is_board(url):
            return None
        hosted = _greenhouse_hosted_ids(url)
        if hosted:
            raw = await fetch(_greenhouse_boards_api_url(hosted))
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and not data.get("error"):
                    return _greenhouse_to_html(data)
        md = _workable_md_url(url)
        if md:
            raw = await fetch(md)
            if raw and raw.lstrip().startswith("#"):
                return _workable_to_html(raw)
        jobs_api = _workable_jobs_api_url(url)
        if jobs_api:
            raw = await fetch(jobs_api)
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and data.get("title"):
                    return _workable_jobs_to_html(data)
        lever_api = _lever_api_url(url)
        if lever_api:
            raw = await fetch(lever_api)
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and (data.get("text") or data.get("id")):
                    return _lever_to_html(data, _company_from_url(url))
        if _lever_is_board(url):
            return None
        sr_api = _smartrecruiters_api_url(url)
        if sr_api:
            raw = await fetch(sr_api)
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and (data.get("name") or data.get("id")):
                    return _smartrecruiters_to_html(data)
        wd_api = _workday_api_url(url)
        if wd_api:
            raw = await fetch(wd_api)
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and isinstance(data.get("jobPostingInfo"), dict):
                    return _workday_to_html(data)
            html = await fetch(_lever_job_url(url))
            if html is None:
                return None
            if not html:
                return ""
            if _html_title(html) or _job_posting(html):
                return html
            return None
        iframe = _icims_iframe_url(url)
        if iframe:
            raw = await fetch(iframe)
            if raw is None:
                return None
            return raw or ""
        jv = _jobvite_job_url(url)
        if jv:
            raw = await fetch(jv)
            if raw is None:
                return None
            if not raw or _jobvite_html_is_gone(raw):
                return None
            return raw
        rt_api = _recruitee_api_url(url)
        if rt_api:
            raw = await fetch(rt_api)
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                offer = _recruitee_offer(data) if isinstance(data, dict) else None
                if offer:
                    return _recruitee_to_html(offer)
                if isinstance(data, dict) and data.get("error"):
                    return None
            return ""
        if _rippling_ids(url):
            raw = await fetch(_rippling_job_url(url))
            if raw is None:
                return None
            if raw:
                parsed = _rippling_from_next(raw)
                if parsed is None:
                    return None
                if parsed:
                    return parsed
            return ""
        bz = _breezy_ids(url)
        if bz:
            raw = await fetch(_breezy_json_url(url))
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, list):
                    job = _breezy_job(data, bz[1])
                    if job:
                        return _breezy_to_html(job)
                    return None
            return ""
        pp = _pinpoint_ids(url)
        if pp:
            raw = await fetch(_pinpoint_json_url(url))
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                rows = data.get("data") if isinstance(data, dict) else data
                job = _pinpoint_job(rows, pp[1])
                if job:
                    return _pinpoint_to_html(job, pp[0])
                if isinstance(rows, list):
                    return None
            return ""
        cm = _comeet_ids(url)
        if cm:
            page = await fetch(_comeet_job_url(url))
            if page is None:
                return None
            token = _comeet_token(page)
            if token:
                raw = await fetch(_comeet_api_url(cm, token))
                if raw is None:
                    return None
                if raw:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = None
                    if isinstance(data, dict) and (data.get("name") or data.get("uid")):
                        return _comeet_to_html(data)
            return ""
        bb = _bamboohr_ids(url)
        if bb:
            raw = await fetch(_bamboohr_detail_url(url))
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return None
                job = _bamboohr_opening(data)
                if job:
                    return _bamboohr_to_html(job, bb[0])
                return None
            return ""
        jz = _jazzhr_ids(url)
        if jz:
            raw = await fetch(_jazzhr_job_url(url))
            if raw is None:
                return None
            if raw:
                if _job_posting(raw):
                    return raw
                return None
            return ""
        dv = _dover_ids(url)
        if dv:
            raw = await fetch(_dover_api_url(url))
            if raw is None:
                return None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return None
                job = _dover_job(data)
                if job:
                    return _dover_to_html(job)
                return None
            return ""
        gem = _gem_ids(url)
        if gem:
            if client is not None:
                posting = await _gem_posting(client, *gem)
            else:
                try:
                    async with httpx.AsyncClient(
                        follow_redirects=True,
                        timeout=8.0,
                        headers=_LISTING_HEADERS,
                    ) as owned:
                        posting = await _gem_posting(owned, *gem)
                except Exception:
                    posting = {}
            if posting is None:
                return None
            if posting:
                return _gem_to_html(posting, gem[0])
            return ""
        wm = _walmart_ids(url)
        if wm:
            raw = await fetch(_walmart_job_url(url))
            if raw is None:
                return None
            if raw:
                job = _walmart_details(raw, wm)
                if job is None:
                    return None
                if job:
                    return _walmart_to_html(job)
                return None
            return ""
        ap = _apple_ids(url)
        if ap:
            raw = await fetch(_apple_job_url(url))
            if raw is None:
                return None
            if raw:
                job = _apple_job(raw)
                if job is None:
                    return None
                if job:
                    return _apple_to_html(job)
                return None
            return ""
        ashby = _ashby_ids(url)
        if ashby:
            if client is not None:
                posting = await _ashby_posting(client, *ashby)
            else:
                try:
                    async with httpx.AsyncClient(
                        follow_redirects=True,
                        timeout=8.0,
                        headers=_LISTING_HEADERS,
                    ) as owned:
                        posting = await _ashby_posting(owned, *ashby)
                except Exception:
                    posting = {}
            if posting is None:
                return None
            if posting:
                return _ashby_to_html(posting)
        if _ashby_is_board(url):
            return None
        pn = _personio_ids(url)
        if pn:
            raw = await fetch(_personio_xml_url(url))
            if raw:
                pos = _personio_position(raw, pn)
                if pos is None:
                    return None
                if pos:
                    return _personio_to_html(pos)
        html = await fetch(_lever_job_url(url))
        if html is None:
            return None
        hosted = _greenhouse_hosted_ids(url, html)
        if hosted:
            raw = await fetch(_greenhouse_boards_api_url(hosted))
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict) and not data.get("error"):
                    return _greenhouse_to_html(data)
        if html and _html_is_gone(html):
            return None
        return html

    async def _search_all(self, query: str) -> list[dict]:
        """Search ATS site: queries first, then generic angles; retry empty site: queries."""
        angles = _search_angles(query)
        generic = [q for q in angles if "site:" not in q.casefold()]
        sites = [q for q in angles if "site:" in q.casefold()]
        results = []
        empty_sites = []
        for q in sites:
            try:
                rows = await self._search_brave(q)
            except Exception as e:
                results.append(e)
                empty_sites.append(q)
                continue
            results.append(rows)
            if not rows:
                empty_sites.append(q)
        searches = [self._search_brave(q) for q in generic]
        if self.perplexity_key:
            searches.append(self._search_perplexity(query))
        if searches:
            results.extend(
                await asyncio.gather(*searches, return_exceptions=True)
            )
        for q in empty_sites:
            try:
                rows = await self._search_brave(q)
            except Exception:
                continue
            if rows:
                results.append(rows)

        all_results = []
        for r in results:
            if isinstance(r, list):
                all_results.extend(r)

        seen = set()
        unique = []
        for r in all_results:
            key = _normalize_url(r.get("url") or "")
            if key and key not in seen and not _is_index_page(r):
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
        """Free web search fallback. Retry DDG 202s; lite HTML when html.duckduckgo is empty."""
        try:
            async with httpx.AsyncClient() as client:
                for attempt in range(4):
                    async with self._ddg_sem:
                        resp = await client.post(
                            "https://html.duckduckgo.com/html/",
                            data={"q": query, "b": ""},
                            headers=_LISTING_HEADERS,
                            timeout=30.0,
                            follow_redirects=True,
                        )
                    rows = _parse_ddg_html(resp.text)
                    if rows:
                        return rows
                    if resp.status_code >= 400 and resp.status_code != 202:
                        return []
                    async with self._ddg_sem:
                        lite = await client.get(
                            "https://lite.duckduckgo.com/lite/",
                            params={"q": query},
                            headers=_LISTING_HEADERS,
                            timeout=30.0,
                            follow_redirects=True,
                        )
                    rows = _parse_ddg_html(lite.text)
                    if rows:
                        return rows
                    if attempt == 3:
                        break
                    await asyncio.sleep(0.4 * (attempt + 1))
        except Exception as e:
            print(f"DDG error: {e}")
            return []
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
                    if not _is_index_page(raw):
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
    return _lever_job_url(url).strip().rstrip("/").casefold()


def _lever_job_url(url: str) -> str:
    """Job page, not the apply form (Lever /apply, Ashby /application, Workable .md)."""
    gh = _greenhouse_ids(url)
    if gh:
        return f"https://job-boards.greenhouse.io/{gh[0]}/jobs/{gh[1]}"
    sr = _smartrecruiters_ids(url)
    if sr:
        return f"https://jobs.smartrecruiters.com/{sr[0]}/{sr[1]}"
    icims = _icims_ids(url)
    if icims:
        return f"https://{icims[0]}/jobs/{icims[1]}/job"
    jv = _jobvite_ids(url)
    if jv:
        return f"https://jobs.jobvite.com/{jv[0]}/job/{jv[1]}"
    rt = _recruitee_job_url(url)
    if rt:
        return rt
    rp = _rippling_job_url(url)
    if rp:
        return rp
    bz = _breezy_job_url(url)
    if bz:
        return bz
    pp = _pinpoint_job_url(url)
    if pp:
        return pp
    cm = _comeet_job_url(url)
    if cm:
        return cm
    bb = _bamboohr_job_url(url)
    if bb:
        return bb
    jz = _jazzhr_job_url(url)
    if jz:
        return jz
    dv = _dover_job_url(url)
    if dv:
        return dv
    gm = _gem_job_url(url)
    if gm:
        return gm
    wm = _walmart_job_url(url)
    if wm:
        return wm
    ap = _apple_job_url(url)
    if ap:
        return ap
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/")
    cut = None
    if host.endswith("lever.co") and path.casefold().endswith("/apply"):
        cut = "/apply"
    elif host.endswith("ashbyhq.com") and path.casefold().endswith("/application"):
        cut = "/application"
    elif host.endswith("apply.workable.com"):
        m = re.match(
            r"(?i)^/([^/]+)/jobs/view/([A-Za-z0-9]+)(?:\.md)?$",
            path,
        )
        if m:
            path = f"/{m.group(1)}/j/{m.group(2)}"
            return parsed._replace(path=path, query="", fragment="").geturl()
    if cut:
        path = path[: -len(cut)] or "/"
        return parsed._replace(path=path, query="", fragment="").geturl()
    return url


_LEVER_JOB_RE = re.compile(
    r"(?i)https?://(jobs(?:\.[a-z]+)?)\.lever\.co/([^/]+)/"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_PERIOD_NEEDLES = (
    ("every two week", "BIWEEKLY"),
    ("every 2 week", "BIWEEKLY"),
    ("every other week", "BIWEEKLY"),
    ("bi week", "BIWEEKLY"),
    ("biweek", "BIWEEKLY"),
    ("fortnight", "BIWEEKLY"),
    ("twice a month", "SEMIMONTHLY"),
    ("twice per month", "SEMIMONTHLY"),
    ("twice monthly", "SEMIMONTHLY"),
    ("semi month", "SEMIMONTHLY"),
    ("semimonth", "SEMIMONTHLY"),
    ("hour", "HOUR"),
    ("day", "DAY"),
    ("month", "MONTH"),
    ("week", "WEEK"),
    ("year", "YEAR"),
    ("annual", "YEAR"),
)


_PERIOD_FROM_PAY = {
    "hour": "HOUR",
    "day": "DAY",
    "week": "WEEK",
    "biweek": "BIWEEKLY",
    "semimonth": "SEMIMONTHLY",
    "month": "MONTH",
    "year": "YEAR",
}


def _period_unit(period: str) -> Optional[str]:
    """Map an ATS period string to JSON-LD unitText. Longer units win."""
    raw = str(period or "").lower()
    if not raw.strip():
        return None
    blob = raw.replace("_", " ").replace("-", " ")
    for needle, name in _PERIOD_NEEDLES:
        if needle in raw or needle in blob:
            return name
    token = str(period or "").rsplit("/", 1)[-1].upper().replace("-", "_").strip()
    token = re.sub(r"^(?:USD|US\$|US|\$)\s*", "", token).strip()
    mapped = _PAY_UNITS.get(token) or _PAY_UNITS.get(token.replace("_", " "))
    return _PERIOD_FROM_PAY.get(mapped) if mapped else None


def _ats_period(raw: dict) -> str:
    """Occupied period / interval / frequency / unitText / unitCode / salaryUnit / duration."""
    text = (
        _ld_text(raw.get("period"))
        or _ld_text(raw.get("interval"))
        or _ld_text(raw.get("frequency"))
        or _ld_text(raw.get("unitText"))
        or _ld_text(raw.get("unit_text"))
        or _ld_text(raw.get("unit"))
        or _ld_text(raw.get("unitCode"))
        or _ld_text(raw.get("unit_code"))
        or _ld_text(raw.get("salaryUnit"))
        or _ld_text(raw.get("salary_unit"))
        or ""
    )
    if text:
        return text
    duration = _ld_text(raw.get("duration"))
    if duration:
        return _duration_unit(duration) or ""
    return ""


def _ats_currency(*values, default: str = "USD") -> str:
    """Occupied currency scalar or {name,@value,@id}. Empty is USD."""
    for value in values:
        text = _ld_text(value)
        if text:
            return text.upper()
    return default


def _lever_api_url(url: str) -> Optional[str]:
    m = _LEVER_JOB_RE.search(_lever_job_url(url) or "")
    if not m:
        return None
    host, board, jid = m.group(1).casefold(), m.group(2), m.group(3)
    if host == "jobs":
        api = "api.lever.co"
    else:
        api = f"api.{host.split('.', 1)[1]}.lever.co"
    return f"https://{api}/v0/postings/{board}/{jid}"


def _lever_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.startswith("jobs") or not host.endswith("lever.co"):
        return False
    return _lever_api_url(url) is None


def _lever_to_html(data: dict, company: Optional[str] = None) -> str:
    """Turn Lever posting JSON into listing HTML. Never invent pay."""
    title = str(data.get("text") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    cats = data.get("categories")
    if isinstance(cats, dict):
        commit = str(cats.get("commitment") or "").lower()
        if "part" in commit:
            posting["employmentType"] = "PART_TIME"
        elif "full" in commit:
            posting["employmentType"] = "FULL_TIME"
    rng = data.get("salaryRange")
    if isinstance(rng, str):
        pay = _span_pay_ld(rng)
        if pay:
            posting["baseSalary"] = pay
    elif isinstance(rng, dict):
        low, high = _bound_nums(rng)
        if low or high:
            unit = _period_unit(_ats_period(rng)) or "YEAR"
            value: dict = {"unitText": unit}
            if low is not None and high is not None:
                value["minValue"] = low
                value["maxValue"] = high
            else:
                value["value"] = high or low
            posting["baseSalary"] = {
                "currency": _ats_currency(rng.get("currency")),
                "value": value,
            }
    if "baseSalary" not in posting:
        pay = _span_pay_ld(str(data.get("salaryDescription") or ""))
        if pay:
            posting["baseSalary"] = pay
    if "baseSalary" not in posting:
        for item in data.get("lists") or []:
            if not isinstance(item, dict):
                continue
            head = str(item.get("text") or "").strip()
            if not _GH_PAY_META_RE.fullmatch(head):
                continue
            pay = _named_pay_ld(head, str(item.get("content") or ""))
            if pay:
                posting["baseSalary"] = pay
                break
    parts = []
    loc = str(cats.get("location") or "").strip() if isinstance(cats, dict) else ""
    place = str(data.get("workplaceType") or "").strip()
    _apply_workplace(posting, place, loc)
    _copy_hours(posting, data)
    for label in (place, loc):
        if label:
            parts.append(f"<p>{label}</p>")
    for key in ("description", "additional", "salaryDescription"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    for item in data.get("lists") or []:
        if not isinstance(item, dict):
            continue
        head = str(item.get("text") or "").strip()
        body = str(item.get("content") or "").strip()
        if head:
            parts.append(f"<h3>{head}</h3>")
        if body:
            parts.append(body)
    return (
        f"<title>{title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


def _public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme not in ("http", "https") or not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
        return False
    if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", host):
        return False
    return True


_CF_CHALLENGE_RE = re.compile(
    r"(?is)<title[^>]*>\s*just a moment\.\.\.\s*</title>"
    r"|cdn-cgi/challenge"
    r"|cf-browser-verification"
    r"|cf-challenge-running"
)


def _cloudflare_challenge(html: str) -> bool:
    """True for a Cloudflare interstitial, not a job listing."""
    return bool(html and _CF_CHALLENGE_RE.search(html[:8000]))


async def _http_get_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        resp = await client.get(url)
        if resp.status_code in (404, 410):
            return None
        text = resp.text
        if _cloudflare_challenge(text):
            return None
        if resp.status_code >= 400:
            return ""
        return text
    except Exception:
        return ""


_ASHBY_JOB_RE = re.compile(
    r"(?i)https?://jobs\.ashbyhq.com/([^/]+)/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_ASHBY_JOB_QUERY = """
query ApiJobPosting($organizationHostedJobsPageName: String!, $jobPostingId: String!) {
  jobPosting(organizationHostedJobsPageName: $organizationHostedJobsPageName, jobPostingId: $jobPostingId) {
    id
    title
    employmentType
    workplaceType
    locationName
    descriptionHtml
    compensationTierSummary
    scrapeableCompensationSalarySummary
  }
}
"""


def _ashby_ids(url: str) -> Optional[tuple[str, str]]:
    m = _ASHBY_JOB_RE.search(_lever_job_url(url) or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def _ashby_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.endswith("ashbyhq.com"):
        return False
    return _ashby_ids(url) is None


async def _ashby_posting(client: httpx.AsyncClient, board: str, jid: str) -> Optional[dict]:
    """None if the posting is gone. Empty dict if the API failed."""
    try:
        resp = await client.post(
            "https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobPosting",
            json={
                "operationName": "ApiJobPosting",
                "variables": {
                    "organizationHostedJobsPageName": board,
                    "jobPostingId": jid,
                },
                "query": _ASHBY_JOB_QUERY,
            },
            headers=_LISTING_HEADERS,
        )
        if resp.status_code in (404, 410):
            return None
        if resp.status_code >= 400:
            return {}
        data = json.loads(resp.text)
        if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
            return {}
        posting = data["data"].get("jobPosting")
        if posting is None and "jobPosting" in data["data"]:
            return None
        if isinstance(posting, dict) and posting:
            return posting
        return {}
    except Exception:
        return {}


def _ashby_to_html(data: dict) -> str:
    """Turn Ashby GraphQL posting into listing HTML. Never invent pay."""
    title = str(data.get("title") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    emp = str(data.get("employmentType") or "")
    if "part" in emp.lower():
        posting["employmentType"] = "PART_TIME"
    elif "full" in emp.lower():
        posting["employmentType"] = "FULL_TIME"
    summary = str(
        data.get("scrapeableCompensationSalarySummary")
        or data.get("compensationTierSummary")
        or ""
    )
    if summary and not _foreign_pay_text(summary):
        low, high = _parse_pay(summary)
        if high or low:
            value: dict = {"unitText": "YEAR"}
            if low and high:
                value["minValue"] = low
                value["maxValue"] = high
            else:
                value["value"] = high or low
            posting["baseSalary"] = {"currency": "USD", "value": value}
    desc = str(data.get("descriptionHtml") or "")
    place = str(data.get("workplaceType") or "").strip()
    loc_name = str(data.get("locationName") or "").strip()
    _apply_workplace(posting, place, loc_name)
    loc = "".join(f"<p>{p}</p>" for p in (place, loc_name) if p)
    return (
        f"<title>{title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{loc}<p>{summary}</p>{desc}"
    )


_ATS_TITLE_TAIL_RE = re.compile(
    r"(?i)\s*[-–—|]\s*(?:jobs\.(?:lever\.co|ashbyhq\.com|workable\.com)|"
    r"jobs by workable|built\s*in(?:\s+[A-Za-z]{2,})?|wellfound)\s*$"
)


def _strip_ats_title(title: str) -> str:
    t = _ATS_TITLE_TAIL_RE.sub("", title or "")
    return re.sub(r"(?i)^job application for\s+", "", t)


def _role_title(title: str) -> str:
    t = _strip_ats_title(title).strip()
    return t or (title or "Unknown").strip() or "Unknown"


def _title_parts(title: str, company: Optional[str] = None) -> tuple[str, str]:
    """Employer + role tokens after stripping board wrappers and company suffixes."""
    t = _strip_ats_title(title)
    org = re.sub(r"\s+", " ", (company or "").strip())
    if org:
        c = re.escape(org)
        t = re.sub(rf"(?i)^{c}\s*[-:|]\s*", "", t)
        t = re.sub(rf"(?i)\s+at\s+{c}\b.*$", "", t)
        t = re.sub(rf"(?i)\s+@\s+{c}\s*$", "", t)
        t = re.sub(rf"(?i)\s*[|\-–—]\s*{c}\s*$", "", t)
    t = re.sub(r"(?i)\s+in remote\b.*$", "", t)
    role = re.sub(r"\W+", " ", t).casefold().strip()
    return org.casefold(), role


def _title_key(title: str, company: Optional[str] = None) -> str:
    """Role identity across boards: same employer + role, after stripping wrappers."""
    org, role = _title_parts(title, company)
    return f"{org}\t{role}" if org else role


_ROLE_CHANGE_RE = re.compile(
    r"(?i)\b(?:senior|staff|principal|lead|jr|junior|intern|manager|director|"
    r"head|scientist|engineer|analyst|architect|specialist|i{1,3}|iv|v)\b"
)


def _same_role(a: str, b: str) -> bool:
    """True when titles are the same job; team suffixes match, seniority changes do not."""
    if a == b:
        return True
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not long.startswith(short + " "):
        return False
    extra = long[len(short) :].strip()
    return bool(extra) and not _ROLE_CHANGE_RE.search(extra)


def _dedupe_opportunities(opps: list) -> list:
    """Keep the first of each employer+role. Call after sorting so the best score wins."""
    seen: list[tuple[str, str]] = []
    unique: list = []
    for o in opps:
        org, role = _title_parts(o.title, o.company)
        if not org and not role:
            continue
        dup = False
        for i, (s_org, s_role) in enumerate(seen):
            if org != s_org:
                continue
            match = _same_role(s_role, role) if org else s_role == role
            if not match:
                continue
            if o.score() == unique[i].score() and len(role) > len(s_role):
                unique[i] = o
                seen[i] = (org, role)
            dup = True
            break
        if dup:
            continue
        seen.append((org, role))
        unique.append(o)
    return unique


def _with_terms(query: str, *terms: str) -> str:
    have = query.casefold()
    extra = [t for t in terms if t.casefold() not in have]
    return f"{query} {' '.join(extra)}".strip() if extra else query


def _search_angles(query: str) -> list[str]:
    """Web queries for this goal. Grants and equity only when the user asked."""
    text = query.casefold()
    angles = [query]
    for extra in (("remote", "job", "hiring"), ("freelance", "contract")):
        q = _with_terms(query, *extra)
        if q not in angles:
            angles.append(q)
    if "site:" not in text:
        for site in (
            "greenhouse.io",
            "jobs.lever.co",
            "jobs.eu.lever.co",
            "jobs.ashbyhq.com",
            "jobs.workable.com",
            "apply.workable.com",
            "jobs.smartrecruiters.com",
            "myworkdayjobs.com",
            "icims.com",
            "jobvite.com",
            "teamtailor.com",
            "personio.com",
            "personio.de",
            "recruitee.com",
            "ats.rippling.com",
            "breezy.hr",
            "pinpointhq.com",
            "comeet.com",
            "bamboohr.com",
            "applytojob.com",
            "app.dover.com",
            "jobs.gem.com",
            "careers.walmart.com",
            "jobs.apple.com",
            "wellfound.com",
            "builtin.com",
        ):
            q = f"{query} site:{site}"
            if q not in angles:
                angles.append(q)
    if any(w in text for w in ("grant", "funding", "fellowship", "scholarship")):
        q = _with_terms(query, "grant", "funding", "opportunity")
        if q not in angles:
            angles.append(q)
    if any(w in text for w in ("equity", "cofounder", "co-founder", "startup")):
        q = _with_terms(query, "startup", "equity", "cofounder")
        if q not in angles:
            angles.append(q)
    return angles


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
    if not url or _is_index_page(raw):
        return None
    url = _lever_job_url(url)
    title = _role_title(raw.get("title") or "")
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
        company=raw.get("company") or _guess_company(title, url),
        pay_low=pay_low,
        pay_high=pay_high,
        hours_per_week=hours,
        remote=remote,
        source=raw.get("source") or "",
    )
    opp.efficiency = opp.refined_rate
    return opp


def _merge_extracted(raw: dict, item: dict) -> Opportunity:
    title = _role_title(item.get("title") or raw.get("title") or "")
    desc = item.get("description") or raw.get("description") or ""
    company = item.get("company") if item.get("company") is not None else raw.get("company")
    if not company:
        url = raw.get("url") or ""
        company = (
            _company_from_title(title, url)
            or _company_from_title(raw.get("title") or "", url)
            or _company_from_url(url)
        )
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
        url=_lever_job_url(raw["url"]),
        description=desc,
        pay_low=pay_low,
        pay_high=pay_high,
        hours_per_week=hours,
        remote=remote,
        source=raw.get("source") or "extracted",
    )
    opp.efficiency = opp.refined_rate
    return opp


_PLACE_RE = re.compile(r"(?i)^(remote|home|office|onsite|hybrid)\b")


_ROLE_START_RE = re.compile(
    r"(?i)^(senior|staff|principal|lead|jr|junior|intern|contract|freelance)\b"
)


def _clean_company_name(name: str) -> str | None:
    name = name.strip(" .,-")
    if not name or re.fullmatch(r"\d+", name) or _PLACE_RE.search(name) or _ROLE_START_RE.search(name):
        return None
    return name


def _company_from_title(title: str, url: str = "") -> str | None:
    """Employer from ` at X`, ` @ X`, Lever `Company - Role`, or Workable suffixes."""
    t = _strip_ats_title(title)
    m = re.search(r"(?i)\bat\s+(.+)$", t)
    if m:
        name = re.split(r"\s*[•|]\s*", m.group(1).strip(" .,-"), maxsplit=1)[0].strip()
        name = _clean_company_name(name)
        if name:
            return name
    m = re.search(r"(?i)\s+@\s+(.+)$", t)
    if m:
        name = re.split(r"\s*[•|]\s*", m.group(1).strip(" .,-"), maxsplit=1)[0].strip()
        name = _clean_company_name(name)
        if name:
            return name
    host = (urlparse(url).hostname or "").casefold()
    if host.endswith("lever.co"):
        m = re.match(r"^(.+?)\s+[-–—]\s+(\S.*)$", t)
        if m and not re.fullmatch(r"\d+", m.group(2).strip()):
            name = _clean_company_name(m.group(1))
            if name:
                return name
    if host.endswith("workable.com"):
        parts = [p.strip(" .,-") for p in re.split(r"\s*\|\s*", t) if p.strip(" .,-")]
        if len(parts) >= 2:
            name = _clean_company_name(parts[-1])
            if name:
                return name
        if host.endswith("apply.workable.com"):
            m = re.search(r"\s+[-–—]\s+(.+)$", t)
            if m:
                name = _clean_company_name(m.group(1))
                if name:
                    return name
    if "builtin" in host or re.search(r"(?i)\|\s*built\s*in\b", title or ""):
        parts = [p.strip() for p in re.split(r"\s+[-–—]\s+", t) if p.strip()]
        if len(parts) >= 2:
            name = _clean_company_name(parts[-1])
            if name:
                return name
    return None


def _company_from_url(url: str) -> str | None:
    """Board slug when the title has no employer: jobs.lever.co/swordhealth/…"""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return None
    if host.endswith("greenhouse.io"):
        slug = parts[0]
        if slug in {"jobs", "embed"}:
            return None
    elif host.endswith("lever.co") or host.endswith("ashbyhq.com"):
        slug = parts[0]
        if host.endswith("ashbyhq.com") and slug in {"jobs", "application"}:
            return None
    elif host.endswith("apply.workable.com"):
        slug = parts[0]
        if slug in {"j", "jobs", "view"}:
            return None
    elif host.endswith("smartrecruiters.com"):
        slug = parts[0]
        if slug.isdigit() or slug in {"jobs", "app"}:
            return None
    elif host.endswith("myworkdayjobs.com"):
        slug = host.split(".")[0]
    elif host.endswith("jobvite.com"):
        slug = parts[0]
        if slug in {"jobs", "careers", "job"}:
            return None
    elif host.endswith(".recruitee.com"):
        slug = host.split(".")[0]
        if slug in {"www", "app", "careers"}:
            return None
    elif host == "ats.rippling.com":
        slug = parts[0]
        if slug in {"jobs", "apply"}:
            return None
    elif host.endswith(".breezy.hr"):
        slug = host.split(".")[0]
        if slug in {"www", "app"}:
            return None
    elif host.endswith(".pinpointhq.com"):
        slug = host.split(".")[0]
        if slug in {"www", "app"}:
            return None
    elif host.endswith("comeet.com"):
        if len(parts) >= 2 and parts[0].casefold() == "jobs":
            slug = parts[1]
        else:
            return None
    elif host.endswith(".bamboohr.com"):
        labels = host.split(".")
        slug = labels[0]
        if slug == "www" and len(labels) > 3:
            slug = labels[1]
        if slug in {"www", "app", "careers"}:
            return None
    elif host.endswith(".applytojob.com"):
        slug = host.split(".")[0]
        if slug in {"www", "app", "careers"}:
            return None
    elif host in {"app.dover.com", "www.app.dover.com"}:
        if len(parts) >= 2 and parts[0].casefold() == "apply":
            slug = parts[1]
            if slug.casefold() in {"jobs", "apply", "careers"}:
                return None
        else:
            return None
    elif host in {"jobs.gem.com", "www.jobs.gem.com"}:
        slug = parts[0]
        if slug.casefold() in {"jobs", "apply", "application"}:
            return None
    elif host in {"careers.walmart.com", "www.careers.walmart.com"}:
        slug = "walmart"
    elif host in {"jobs.apple.com", "www.jobs.apple.com"}:
        slug = "apple"
    else:
        return None
    name = slug.replace("-", " ").replace("_", " ").strip()
    if not name or _PLACE_RE.search(name):
        return None
    return name.title()


def _guess_company(title: str, url: str = "") -> str | None:
    return _company_from_title(title, url) or _company_from_url(url)


def _ats_board_key(url: str) -> Optional[str]:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return None
    if host.endswith("greenhouse.io"):
        return f"gh:{parts[0].casefold()}"
    if host.endswith("lever.co"):
        return f"lever:{parts[0].casefold()}"
    if host.endswith("ashbyhq.com"):
        return f"ashby:{parts[0].casefold()}"
    if host.endswith("apply.workable.com"):
        return f"workable:{parts[0].casefold()}"
    if host.endswith("smartrecruiters.com"):
        return f"sr:{parts[0].casefold()}"
    if host.endswith("myworkdayjobs.com"):
        return f"wd:{(parsed.hostname or '').split('.')[0].casefold()}"
    if host.endswith("jobvite.com"):
        slug = parts[0].casefold()
        if slug in {"jobs", "careers", "job"}:
            return None
        return f"jobvite:{slug}"
    if host.endswith(".recruitee.com"):
        slug = host.split(".")[0].casefold()
        if slug in {"www", "app", "careers"}:
            return None
        return f"recruitee:{slug}"
    if host == "ats.rippling.com":
        slug = parts[0].casefold()
        if slug in {"jobs", "apply"}:
            return None
        return f"rippling:{slug}"
    if host.endswith(".breezy.hr"):
        slug = host.split(".")[0].casefold()
        if slug in {"www", "app"}:
            return None
        return f"breezy:{slug}"
    if host.endswith(".pinpointhq.com"):
        slug = host.split(".")[0].casefold()
        if slug in {"www", "app"}:
            return None
        return f"pinpoint:{slug}"
    if host.endswith("comeet.com"):
        if len(parts) >= 2 and parts[0].casefold() == "jobs":
            return f"comeet:{parts[1].casefold()}"
        return None
    if host.endswith(".bamboohr.com"):
        labels = host.split(".")
        slug = labels[0].casefold()
        if slug == "www" and len(labels) > 3:
            slug = labels[1].casefold()
        if slug in {"www", "app", "careers"}:
            return None
        return f"bamboohr:{slug}"
    if host.endswith(".applytojob.com"):
        slug = host.split(".")[0].casefold()
        if slug in {"www", "app", "careers"}:
            return None
        return f"jazzhr:{slug}"
    if host in {"app.dover.com", "www.app.dover.com"}:
        if len(parts) >= 2 and parts[0].casefold() == "apply":
            slug = parts[1].casefold()
            if slug in {"jobs", "apply", "careers"}:
                return None
            return f"dover:{slug}"
        return None
    if host in {"jobs.gem.com", "www.jobs.gem.com"}:
        slug = parts[0].casefold()
        if slug in {"jobs", "apply", "application"}:
            return None
        return f"gem:{slug}"
    return None


def _unify_board_companies(opps: list) -> None:
    """Prefer JSON-LD / title employer over a title-cased URL slug on the same board."""
    best: dict[str, str] = {}
    for o in opps:
        key = _ats_board_key(o.url)
        if not key or not o.company:
            continue
        slug = _company_from_url(o.url)
        if slug and o.company.casefold() == slug.casefold():
            continue
        best[key] = o.company
    for o in opps:
        key = _ats_board_key(o.url)
        if key not in best:
            continue
        slug = _company_from_url(o.url)
        if not o.company or (slug and o.company.casefold() == slug.casefold()):
            o.company = best[key]


_INDEX_URL_RE = re.compile(
    r"(?:indeed\.com/|linkedin\.com/(?:jobs|in)/"
    r"|glassdoor\.com/"
    r"|simplyhired\.com/"
    r"|/search\?q="
    r"|monster\.com/"
    r"|dice\.com/"
    r"|jooble\."
    r"|adzuna\."
    r"|talent\.com/"
    r"|careerbuilder\.com/"
    r"|upwork\.com/freelance-jobs/apply/"
    r"|grants\.gov/search"
    r"|lemon\.io/for-developers/"
    r"|magic\.lemon\.io/share/"
    r"|docs\.lemon\.io/"
    r"|corptocorp\.org/"
    r"|karkidi\.com/"
    r"|jobleads\.com/"
    r"|remoterocketship\.com/"
    r"|migratemate\.co/"
    r"|builtin[a-z]*\.com/(?!job/)"
    r"|ziprecruiter\.com/"
    r"|lever\.co/jobgether/"
    r"|jobgether\.com/"
    r"|remotesource\.com/"
    r"|arc\.dev/remote-jobs"
    r"|jobdescription\.org/"
    r"|ai\.engineer/jobs"
    r"|remotely\.works/blog/"
    r"|peopleinai\.com/"
    r"|7seventy\.net/"
    r"|globalcareer\.io/"
    r"|visa-hunt\.com/"
    r"|dailyremote\.com/"
    r"|optiveum\.com/"
    r"|salaryexpert\.com/"
    r"|levels\.fyi/"
    r"|payscale\.com/"
    r"|salary\.com/"
    r"|jobera\.com/"
    r"|jobright\.ai/"
    r"|aijobs\.net/"
    r"|aijobs\.com/"
    r"|aijobs\.mlyearning\.org/"
    r"|wellfound\.com/(?!jobs/\d)"
    r"|angel\.co/(?!jobs/\d)"
    r"|h1bscope\.com/"
    r"|salarybyrole\.com/"
    r"|salarysolver\.com/"
    r"|salarycube\.com/"
    r"|motionrecruitment\.com/it-salary"
    r"|hackerx\.org/[^\"'\s]*salary"
    r"|greenhouse\.com/"
    r"|remoteok\.com/"
    r"|opentoworkremote\.com/"
    r"|bilingualjobs\.io/"
    r"|jobquip\.com/"
    r"|developers\.comeet\.com/"
    r"|personio\.com/blog/"
    r"|personio\.com/careers)",
    re.I,
)
_INDEX_TITLE_RE = re.compile(
    r"(?i)^hire\b|\bcurrent (?:openings|positions|roles|listings|opportunities)\b"
    r"|\bopen (?:positions|roles|listings|opportunities)\b"
    r"|\b(?:job|role|position|listing) openings\b"
    r"|\ball (?:openings|positions|roles|listings|opportunities)\b"
    r"|\bcareer (?:opportunities|listings|roles|positions)\b"
    r"|^careers at\b"
    r"|^join our team(?:\s*[|\-–—].*)?$"
    r"|^work with us(?:\s*[|\-–—].*)?$"
    r"|^we(?:'re| are) hiring(?:\s*[|\-–—].*)?$"
    r"|^opportunities(?:\s*[|\-–—].*)?$"
    r"|^vacancies(?:\s*[|\-–—].*)?$"
    r"|^hiring(?:\s*[|\-–—].*)?$"
    r"|^explore careers\b"
    r"|^browse careers\b"
    r"|^find careers\b"
    r"|^search careers\b"
    r"|^view careers\b"
    r"|^discover careers\b"
    r"|^see careers\b"
    r"|^apply careers\b"
    r"|^open careers\b"
    r"|\b(?:job|role|position|listing) vacancies\b"
    r"|\bavailable (?:positions|roles|listings|opportunities|openings)\b"
    r"|\b(?:featured|latest|popular|hot|new|trending|recommended|matching|similar|suggested|related|other|browse|explore|view|discover|see|find|search|apply) (?:positions|roles|listings|openings|opportunities)\b"
    r"|^life at\b"
    r"|^meet (?:the|our) team(?:\s*[|\-–—].*)?$"
    r"|^our (?:team|people)(?:\s*[|\-–—].*)?$"
    r"|^about (?:the|our) team(?:\s*[|\-–—].*)?$"
    r"|^team(?:\s*[|\-–—].*)?$"
    r"|^why \S+(?:\s*[|\-–—].*)?$"
    r"|\binternships\b"
    r"|^university recruiting(?:\s*[|\-–—].*)?$"
    r"|^campus recruiting(?:\s*[|\-–—].*)?$"
    r"|^early careers?(?:\s*[|\-–—].*)?$"
    r"|^student programs?(?:\s*[|\-–—].*)?$"
    r"|^graduate programs?(?:\s*[|\-–—].*)?$"
    r"|^university programs?(?:\s*[|\-–—].*)?$"
    r"|^job search(?:\s*[|\-–—].*)?$"
    r"|^career search(?:\s*[|\-–—].*)?$"
    r"|^careers(?:\s*[|\-–—].*)?$"
    r"|^benefits(?:\s*[|\-–—].*)?$"
    r"|^our benefits(?:\s*[|\-–—].*)?$"
    r"|^culture(?:\s*[|\-–—].*)?$"
    r"|^our culture(?:\s*[|\-–—].*)?$"
    r"|^leadership(?:\s*[|\-–—].*)?$"
    r"|^our leadership(?:\s*[|\-–—].*)?$"
    r"|^about us(?:\s*[|\-–—].*)?$"
    r"|^about(?:\s*[|\-–—].*)?$"
    r"|^our values(?:\s*[|\-–—].*)?$"
    r"|^values(?:\s*[|\-–—].*)?$"
    r"|^our mission(?:\s*[|\-–—].*)?$"
    r"|^locations(?:\s*[|\-–—].*)?$"
    r"|^our locations(?:\s*[|\-–—].*)?$"
    r"|^diversity(?:\s*[|\-–—].*)?$"
    r"|^inclusion(?:\s*[|\-–—].*)?$"
    r"|^dei(?:\s*[|\-–—].*)?$"
    r"|^our dei(?:\s*[|\-–—].*)?$"
    r"|^diversity equity(?:\s+and)?\s+inclusion(?:\s*[|\-–—].*)?$"
    r"|^our story(?:\s*[|\-–—].*)?$"
    r"|^faqs?(?:\s*[|\-–—].*)?$"
    r"|^news(?:\s*[|\-–—].*)?$"
    r"|^press(?:\s*[|\-–—].*)?$"
    r"|^blog(?:\s*[|\-–—].*)?$"
    r"|^our blog(?:\s*[|\-–—].*)?$"
    r"|^newsroom(?:\s*[|\-–—].*)?$"
    r"|^press releases?(?:\s*[|\-–—].*)?$"
    r"|^our news(?:\s*[|\-–—].*)?$"
    r"|^investors?(?:\s*[|\-–—].*)?$"
    r"|^investor relations(?:\s*[|\-–—].*)?$"
    r"|^sustainability(?:\s*[|\-–—].*)?$"
    r"|^our sustainability(?:\s*[|\-–—].*)?$"
    r"|^esg(?:\s*[|\-–—].*)?$"
    r"|^impact(?:\s*[|\-–—].*)?$"
    r"|^our impact(?:\s*[|\-–—].*)?$"
    r"|^community(?:\s*[|\-–—].*)?$"
    r"|^our community(?:\s*[|\-–—].*)?$"
    r"|^csr(?:\s*[|\-–—].*)?$"
    r"|^social responsibility(?:\s*[|\-–—].*)?$"
    r"|^purpose(?:\s*[|\-–—].*)?$"
    r"|^our purpose(?:\s*[|\-–—].*)?$"
    r"|^mission(?:\s*[|\-–—].*)?$"
    r"|^people(?:\s*[|\-–—].*)?$"
    r"|^ethics(?:\s*[|\-–—].*)?$"
    r"|^governance(?:\s*[|\-–—].*)?$"
    r"|^environment(?:\s*[|\-–—].*)?$"
    r"|^history(?:\s*[|\-–—].*)?$"
    r"|^our history(?:\s*[|\-–—].*)?$"
    r"|^media center(?:\s*[|\-–—].*)?$"
    r"|^press center(?:\s*[|\-–—].*)?$"
    r"|^foundation(?:\s*[|\-–—].*)?$"
    r"|^our foundation(?:\s*[|\-–—].*)?$"
    r"|^giving(?:\s*[|\-–—].*)?$"
    r"|^our giving(?:\s*[|\-–—].*)?$"
    r"|^philanthropy(?:\s*[|\-–—].*)?$"
    r"|^citizenship(?:\s*[|\-–—].*)?$"
    r"|^corporate citizenship(?:\s*[|\-–—].*)?$"
    r"|^volunteering(?:\s*[|\-–—].*)?$"
    r"|^charity(?:\s*[|\-–—].*)?$"
    r"|^responsibility(?:\s*[|\-–—].*)?$"
    r"|\bfreelancers\b"
    r"|\bsalary guide\b"
    r"|\bsalary data\b"
    r"|\bcompensation (?:guide|data|benchmarks?|for)\b"
    r"|\bh-?1b visa salary\b"
    r"|\bmedian pay\b"
    r"|\bhow much (?:do|can)\b"
    r"|salary 20\d{2}"
    r"|20\d{2}\b.{0,60}\bsalary\b"
)
_JOBS_WORD_RE = re.compile(r"(?i)\bjobs\b(?!\.)(?! by workable)")
_ROLE_JOBS_AT_RE = re.compile(r"(?i).+\bjobs (?:at|bei) \S")


def _title_is_index(title: str) -> bool:
    """Board/catalog titles. 'Role Jobs at Employer' is a listing, not a board."""
    t = title or ""
    if _INDEX_TITLE_RE.search(t):
        return True
    if _ROLE_JOBS_AT_RE.search(t) and not re.match(r"(?i)\s*jobs\b", t):
        return False
    return bool(_JOBS_WORD_RE.search(t))


_GH_HOST = r"(?:job-boards(?:\.[a-z]+)?|boards(?:\.[a-z]+)?)\.greenhouse\.io"
_GH_JOB_RE = re.compile(
    rf"(?i)https?://{_GH_HOST}/(?!embed\b)([^/]+)/jobs/(\d+)",
)
_GH_EMBED_FOR_RE = re.compile(
    rf"(?i)https?://{_GH_HOST}/embed/job_app\?[^\"'\s<>]*\bfor=([a-z0-9_-]+)",
)
_GH_HOSTED_JID_RE = re.compile(r"(?i)(?:[?&]gh_jid=|greenhouse-job-)(\d{5,})")
_GH_BOARD_SKIP = frozenset({"www", "careers", "jobs", "job", "app", "cdn", "api"})


def _greenhouse_ids(url: str) -> Optional[tuple[str, str]]:
    """Board token and numeric job id from job-boards, boards.eu, or embed URLs."""
    m = _GH_JOB_RE.search(url or "")
    if m:
        return m.group(1), m.group(2)
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    if not host.endswith("greenhouse.io"):
        return None
    q = parse_qs(parsed.query)
    board = (q.get("for") or [""])[0].strip()
    jid = (q.get("token") or q.get("gh_jid") or [""])[0].strip()
    path = (parsed.path or "").casefold()
    if board and jid.isdigit() and ("/embed/" in path or q.get("token") or q.get("gh_jid")):
        return board, jid
    return None


def _greenhouse_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host.endswith("greenhouse.com"):
        return True
    if not host.endswith("greenhouse.io"):
        return False
    return _greenhouse_ids(url) is None


def _greenhouse_api_url(url: str) -> Optional[str]:
    ids = _greenhouse_ids(url)
    if not ids:
        return None
    return _greenhouse_boards_api_url(ids)


def _greenhouse_boards_api_url(ids: tuple[str, str]) -> str:
    return (
        f"https://boards-api.greenhouse.io/v1/boards/{ids[0]}/jobs/{ids[1]}"
        "?pay_transparency=true"
    )


def _greenhouse_board_from_host(url: str) -> Optional[str]:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host or host.endswith("greenhouse.io") or host.endswith("greenhouse.com"):
        return None
    labels = [p for p in host.split(".") if p and p not in _GH_BOARD_SKIP]
    if len(labels) < 2:
        return None
    board = labels[-2]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", board):
        return None
    return board


def _greenhouse_hosted_ids(url: str, html: str = "") -> Optional[tuple[str, str]]:
    """Board token and job id for a company-hosted Greenhouse page.

    greenhouse.io URLs stay on `_greenhouse_ids`. A wrong host board must not
    mark the posting gone — callers fall through to page HTML on API 404.
    """
    if _greenhouse_ids(url):
        return None
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    if host.endswith("greenhouse.io") or host.endswith("greenhouse.com"):
        return None
    q = parse_qs(parsed.query)
    jid = (q.get("gh_jid") or [""])[0].strip()
    job = _GH_JOB_RE.search(html or "")
    if not (jid.isdigit() and len(jid) >= 5):
        m = _GH_HOSTED_JID_RE.search(html or "")
        jid = m.group(1) if m else (job.group(2) if job else "")
    if not (jid.isdigit() and len(jid) >= 5):
        return None
    board = (q.get("for") or [""])[0].strip()
    if not board:
        embed = job or _GH_EMBED_FOR_RE.search(html or "")
        if embed:
            board = embed.group(1)
    if not board:
        board = _greenhouse_board_from_host(url) or ""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", board):
        return None
    return board, jid


def _cents_to_annual(cents) -> Optional[int]:
    n = _num(cents)
    if n is None:
        return None
    annual = int(n) // 100
    if 10_000 <= annual <= 2_000_000:
        return annual
    return None


_GH_PAY_META_RE = re.compile(
    r"(?i)^(?:(?:base|annual|yearly|hourly|monthly|every\s+(?:two|2|other)\s+weeks?|fortnightly|bi[-\s]?weekly|semi[-\s]?monthly|weekly)\s+)*(?:salary|compensation|pay)(?:\s+(?:range|band|rate))?$"
    r"|^(?:(?:base|annual|yearly|hourly|monthly|every\s+(?:two|2|other)\s+weeks?|fortnightly|bi[-\s]?weekly|semi[-\s]?monthly|weekly)\s+)+rate$"
)


def _greenhouse_pay_ld(data: dict) -> Optional[dict]:
    """Listed baseSalary from pay_input_ranges. Prefer USD; keep stated foreign currency."""
    foreign = None
    for row in data.get("pay_input_ranges") or []:
        if not isinstance(row, dict):
            continue
        low = _cents_to_annual(row.get("min_cents"))
        high = _cents_to_annual(row.get("max_cents"))
        unit = None
        if not high and not low:
            unit = _period_unit(_ats_period(row))
            bound_low, bound_high = _bound_nums(row)
            if bound_low is None and bound_high is None and unit:
                cents_low, cents_high = _num(row.get("min_cents")), _num(row.get("max_cents"))
                bound_low = cents_low / 100 if cents_low is not None else None
                bound_high = cents_high / 100 if cents_high is not None else None
            if unit:
                low = int(bound_low) if bound_low else None
                high = int(bound_high) if bound_high else None
            else:
                low = int(bound_low) if bound_low and 10_000 <= bound_low <= 2_000_000 else None
                high = int(bound_high) if bound_high and 10_000 <= bound_high <= 2_000_000 else None
        if not high and not low:
            spanned = _span_pay_ld(
                str(row.get("title") or row.get("label") or row.get("text") or "")
            )
            if not spanned:
                continue
            cur = _ats_currency(row.get("currency_type"), spanned.get("currency"))
            spanned["currency"] = cur
            if _usd(cur):
                return spanned
            if foreign is None:
                foreign = spanned
            continue
        cur = _ats_currency(row.get("currency_type"))
        value: dict = {"unitText": unit or "YEAR"}
        if low and high:
            value["minValue"] = low
            value["maxValue"] = high
        else:
            value["value"] = high or low
        blob = {"currency": cur, "value": value}
        if _usd(cur):
            return blob
        if foreign is None:
            foreign = blob
    if foreign:
        return foreign
    for item in data.get("metadata") or []:
        if not isinstance(item, dict):
            continue
        if not _GH_PAY_META_RE.fullmatch(str(item.get("name") or "").strip()):
            continue
        spanned = _named_pay_ld(str(item.get("name") or ""), str(item.get("value") or ""))
        if not spanned:
            continue
        if _usd(spanned.get("currency")):
            return spanned
        if foreign is None:
            foreign = spanned
    return foreign


def _greenhouse_to_html(data: dict) -> str:
    """Turn Greenhouse job JSON into listing HTML. Never invent pay."""
    company = str(data.get("company_name") or "").strip()
    title = str(data.get("title") or "").strip()
    loc = ""
    location = data.get("location")
    if isinstance(location, dict):
        loc = str(location.get("name") or "")
    content = unescape(data.get("content") or "")
    posting = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    pay = _greenhouse_pay_ld(data)
    if pay:
        posting["baseSalary"] = pay
    meta = {}
    for item in data.get("metadata") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().casefold()
        val = item.get("value")
        if name and val not in (None, ""):
            meta[name] = val
    n = _num(meta.get("scheduled weekly hours"))
    if n is None or not (1 <= n <= 80):
        n = None
        for name, val in meta.items():
            stated = _stated_hours("", f"{name}: {val}")
            if stated:
                n = stated
                break
    if n is not None and 1 <= n <= 80:
        posting["workHours"] = str(int(n))
    _copy_hours(posting, data)
    time_type = str(meta.get("time type") or meta.get("employment type") or "").lower()
    if "part" in time_type:
        posting["employmentType"] = "PART_TIME"
    elif "full" in time_type:
        posting["employmentType"] = "FULL_TIME"
    _apply_workplace(posting, loc)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"<p>{loc}</p>{content}"
    )


_WORKABLE_JOB_RE = re.compile(
    r"(?i)https?://apply\.workable\.com/([^/]+)/(?:j|jobs/view)/([A-Za-z0-9]+)",
)
_WORKABLE_SALARY_RE = re.compile(
    r"(?i)\*\*Salary:\*\*\s*(?:USD|US\$)\s*([\d,]+)(?:\s*[–—-]\s*(?:USD|US\$)?\s*([\d,]+))?"
)
_WORKABLE_WORKPLACE_RE = re.compile(r"(?im)^\*\*Workplace:\*\*\s*(.+)$")


def _workable_md_url(url: str) -> Optional[str]:
    m = _WORKABLE_JOB_RE.search(url or "")
    if not m:
        return None
    return f"https://apply.workable.com/{m.group(1)}/jobs/view/{m.group(2)}.md"


_WORKABLE_VIEW_RE = re.compile(
    r"(?i)https?://jobs\.workable\.com/view/([A-Za-z0-9]+)",
)


def _workable_jobs_api_url(url: str) -> Optional[str]:
    m = _WORKABLE_VIEW_RE.search(url or "")
    if not m:
        return None
    return f"https://jobs.workable.com/api/v1/jobs/{m.group(1)}"




def _workable_pay_ld(data: dict) -> Optional[dict]:
    """USD or stated foreign salary from jobs.workable.com JSON."""
    for key in ("salary", "salaryRange", "compensation", "payRange"):
        raw = data.get(key)
        if isinstance(raw, str):
            pay = _span_pay_ld(raw)
            if pay:
                return pay
            continue
        if not isinstance(raw, dict):
            continue
        low, high = _bound_nums(raw)
        if low is None and high is None:
            continue
        unit = _period_unit(_ats_period(raw)) or "YEAR"
        value: dict = {"unitText": unit}
        if low is not None and high is not None:
            value["minValue"] = low
            value["maxValue"] = high
        else:
            value["value"] = high or low
        return {
            "currency": _ats_currency(raw.get("currency")),
            "value": value,
        }
    return None


def _workable_jobs_to_html(data: dict) -> str:
    """Turn jobs.workable.com job JSON into listing HTML. Never invent pay."""
    company = ""
    org = data.get("company")
    if isinstance(org, dict):
        company = str(org.get("title") or org.get("name") or "").strip()
        if company and _PLACE_RE.search(company):
            company = ""
    title = str(data.get("title") or "").strip()
    parts = []
    for key in ("description", "requirementsSection", "benefitsSection"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    posting = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = data.get("employmentType")
    if isinstance(emp, str) and emp.strip():
        posting["employmentType"] = emp.strip()
    place = str(data.get("workplace") or "").strip()
    loc = f"<p>{place}</p>" if place else ""
    pay = _workable_pay_ld(data)
    if pay:
        posting["baseSalary"] = pay
    _copy_hours(posting, data)
    page_title = f"{title} at {company}" if company else title
    _apply_workplace(posting, place)
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{loc}{''.join(parts)}"
    )


def _workable_is_board(url: str) -> bool:
    """True for apply.workable.com/{org} career home, not /j/{id} listings."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    parts = [p for p in parsed.path.split("/") if p]
    if host.endswith("apply.workable.com"):
        if not parts:
            return True
        if len(parts) >= 3 and parts[1].casefold() == "j":
            return False
        if len(parts) >= 4 and parts[1].casefold() == "jobs" and parts[2].casefold() == "view":
            return False
        return True
    if host == "jobs.workable.com" or host.endswith(".jobs.workable.com"):
        if not parts:
            return True
        if parts[0].casefold() == "view" and len(parts) >= 2:
            return False
        return True
    return False


_SR_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?jobs\.smartrecruiters\.com/([^/]+)/(\d+)",
)


def _smartrecruiters_ids(url: str) -> Optional[tuple[str, str]]:
    m = _SR_JOB_RE.search(url or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def _smartrecruiters_api_url(url: str) -> Optional[str]:
    ids = _smartrecruiters_ids(url)
    if not ids:
        return None
    return f"https://api.smartrecruiters.com/v1/companies/{ids[0]}/postings/{ids[1]}"


def _smartrecruiters_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host != "jobs.smartrecruiters.com":
        return False
    return _smartrecruiters_ids(url) is None


def _smartrecruiters_pay_ld(data: dict) -> Optional[dict]:
    """USD or stated foreign salary from compensation. Skip rows with no amounts."""
    comp = data.get("compensation")
    if isinstance(comp, str):
        return _span_pay_ld(comp)
    if not isinstance(comp, dict):
        return None
    low, high = _bound_nums(comp)
    if low is None and high is None:
        return None
    unit = _period_unit(_ats_period(comp))
    value: dict = {}
    if unit:
        value["unitText"] = unit
    if low is not None and high is not None:
        value["minValue"] = int(low) if low == int(low) else low
        value["maxValue"] = int(high) if high == int(high) else high
    else:
        amount = high if high is not None else low
        value["value"] = int(amount) if amount == int(amount) else amount
    currency = _ats_currency(comp.get("currency"))
    return {"currency": currency, "value": value}


def _smartrecruiters_to_html(data: dict) -> str:
    """Turn SmartRecruiters posting JSON into listing HTML. Never invent pay."""
    title = str(data.get("name") or "").strip()
    company = ""
    org = data.get("company")
    if isinstance(org, dict):
        company = str(org.get("name") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = data.get("typeOfEmployment")
    label = ""
    if isinstance(emp, dict):
        label = str(emp.get("label") or emp.get("id") or "")
    elif isinstance(emp, str):
        label = emp
    lower = label.lower()
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower or "permanent" in lower:
        posting["employmentType"] = "FULL_TIME"
    loc = data.get("location") if isinstance(data.get("location"), dict) else {}
    if loc.get("remote") is True and loc.get("hybrid") is not True:
        place = "remote"
    elif loc.get("hybrid") is True:
        place = "hybrid"
    elif loc.get("remote") is False:
        place = "onsite"
    else:
        place = str(loc.get("fullLocation") or loc.get("city") or "")
    _apply_workplace(posting, place)
    pay = _smartrecruiters_pay_ld(data)
    if pay:
        posting["baseSalary"] = pay
    _copy_hours(posting, data)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    secs = (data.get("jobAd") or {}).get("sections") if isinstance(data.get("jobAd"), dict) else None
    if isinstance(secs, dict):
        for key in (
            "jobDescription",
            "qualifications",
            "additionalInformation",
            "companyDescription",
        ):
            sec = secs.get(key)
            text = sec.get("text") if isinstance(sec, dict) else None
            if isinstance(text, str) and text.strip():
                parts.append(text)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


def _workable_to_html(md: str) -> str:
    """Turn Workable job markdown into listing HTML. Never invent pay."""
    title = ""
    m = re.search(r"(?m)^#\s+(.+)$", md)
    if m:
        title = m.group(1).strip()
    company = ""
    m = re.search(r"(?m)^>\s*([^·\n]+)", md)
    if m:
        company = m.group(1).strip(" \t-")
        if company and _PLACE_RE.search(company):
            company = ""
    posting = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    sm = _WORKABLE_SALARY_RE.search(md)
    if sm:
        low = int(sm.group(1).replace(",", ""))
        high = int(sm.group(2).replace(",", "")) if sm.group(2) else None
        value: dict = {"unitText": "YEAR"}
        if high:
            value["minValue"] = low
            value["maxValue"] = high
        else:
            value["value"] = low
        posting["baseSalary"] = {"currency": "USD", "value": value}
    if re.search(r"(?i)\bfull-time\b", md):
        posting["employmentType"] = "FULL_TIME"
    elif re.search(r"(?i)\bpart-time\b", md):
        posting["employmentType"] = "PART_TIME"
    place = ""
    wm = _WORKABLE_WORKPLACE_RE.search(md)
    if wm:
        place = wm.group(1).strip()
    loc = ""
    hm = re.search(r"(?m)^>\s*(.+)$", md)
    if hm:
        fields = [p.strip() for p in hm.group(1).split("·")]
        if len(fields) > 1:
            loc = fields[1]
    _apply_workplace(posting, place, loc)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"<pre>{md}</pre>"
    )


_WD_HOST_RE = re.compile(r"(?i)^([a-z0-9-]+)\.wd\d+\.myworkdayjobs\.com$")


def _workday_ids(url: str) -> Optional[tuple[str, str, str, str]]:
    """host, tenant, site, job slug for a myworkdayjobs.com posting."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    m = _WD_HOST_RE.match(host)
    if not m:
        return None
    tenant = m.group(1)
    parts = [p for p in parsed.path.split("/") if p]
    if parts and re.match(r"(?i)^[a-z]{2}-[a-z]{2}$", parts[0]):
        parts = parts[1:]
    try:
        ji = next(i for i, p in enumerate(parts) if p.casefold() == "job")
    except StopIteration:
        return None
    if ji < 1 or ji + 1 >= len(parts):
        return None
    site = parts[0]
    slug = parts[-1]
    if not slug or slug.casefold() == "job":
        return None
    return host, tenant, site, slug


def _workday_api_url(url: str) -> Optional[str]:
    ids = _workday_ids(url)
    if not ids:
        return None
    host, tenant, site, slug = ids
    return f"https://{host}/wday/cxs/{tenant}/{site}/job/{slug}"


def _workday_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not _WD_HOST_RE.match(host):
        return False
    return _workday_ids(url) is None


def _workday_to_html(data: dict) -> str:
    """Turn Workday CXS posting JSON into listing HTML. Never invent pay."""
    info = data.get("jobPostingInfo") if isinstance(data.get("jobPostingInfo"), dict) else {}
    title = str(info.get("title") or "").strip()
    company = ""
    org = data.get("hiringOrganization")
    if isinstance(org, dict):
        company = str(org.get("name") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    time_type = str(info.get("timeType") or "").lower()
    if "part" in time_type:
        posting["employmentType"] = "PART_TIME"
    elif "full" in time_type:
        posting["employmentType"] = "FULL_TIME"
    place = str(info.get("remoteType") or "").strip()
    loc = str(info.get("location") or "").strip()
    extras = []
    raw_extras = info.get("additionalLocations")
    if isinstance(raw_extras, list):
        extras = [str(x).strip() for x in raw_extras if str(x).strip()]
    _apply_workplace(posting, place, *extras, loc)
    desc = str(info.get("jobDescription") or "")
    page_title = f"{title} at {company}" if company else title
    bits = []
    for label in (place, loc, *extras):
        if label:
            bits.append(f"<p>{label}</p>")
    bits.append(desc)
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(bits)}"
    )


_ICIMS_JOB_RE = re.compile(r"(?i)/jobs/(\d+)(?:/|$)")


def _icims_ids(url: str) -> Optional[tuple[str, str]]:
    """Host and numeric job id from careers-*.icims.com/jobs/{id}/..."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").casefold()
    if not host.endswith("icims.com"):
        return None
    m = _ICIMS_JOB_RE.search(parsed.path or "")
    if not m:
        return None
    return host, m.group(1)


def _icims_iframe_url(url: str) -> Optional[str]:
    ids = _icims_ids(url)
    if not ids:
        return None
    host, jid = ids
    return f"https://{host}/jobs/{jid}/job?in_iframe=1"


def _icims_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.endswith("icims.com"):
        return False
    return _icims_ids(url) is None


_JOBVITE_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?(?:jobs|careers)\.jobvite\.com/([^/]+)/job/([A-Za-z0-9]+)"
)
_JOBVITE_GONE_RE = re.compile(r"(?i)the job listing no longer exists")


def _jobvite_ids(url: str) -> Optional[tuple[str, str]]:
    m = _JOBVITE_JOB_RE.search(url or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def _jobvite_job_url(url: str) -> Optional[str]:
    ids = _jobvite_ids(url)
    if not ids:
        return None
    return f"https://jobs.jobvite.com/{ids[0]}/job/{ids[1]}"


def _jobvite_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.endswith("jobvite.com"):
        return False
    return _jobvite_ids(url) is None


def _jobvite_html_is_gone(html: str) -> bool:
    return bool(_JOBVITE_GONE_RE.search(html or ""))


_TEAMTAILOR_JOB_RE = re.compile(r"(?i)https?://[^/]*teamtailor\.com/jobs/(\d+)")
_PERSONIO_JOB_RE = re.compile(
    r"(?i)https?://[^/]*jobs\.personio\.(?:com|de)/job/(\d+)"
)


def _teamtailor_ids(url: str) -> Optional[str]:
    m = _TEAMTAILOR_JOB_RE.search(url or "")
    return m.group(1) if m else None


def _teamtailor_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.endswith("teamtailor.com"):
        return False
    return _teamtailor_ids(url) is None


def _personio_ids(url: str) -> Optional[str]:
    m = _PERSONIO_JOB_RE.search(url or "")
    return m.group(1) if m else None


def _personio_xml_url(url: str) -> Optional[str]:
    if not _personio_ids(url):
        return None
    host = (urlparse(url).hostname or "").casefold()
    return f"https://{host}/xml"


def _personio_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if "jobs.personio." not in host:
        return False
    return _personio_ids(url) is None


def _personio_position(xml: str, jid: str) -> Optional[dict]:
    """Position dict, None if the board XML omits this id, {} if XML is unusable."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}
    needle = str(jid or "").strip()
    found = False
    for pos in root.iter("position"):
        found = True
        if (pos.findtext("id") or "").strip() != needle:
            continue
        offices = []
        office = (pos.findtext("office") or "").strip()
        if office:
            offices.append(office)
        extras = pos.find("additionalOffices")
        if extras is not None:
            for el in extras.findall("office"):
                label = (el.text or "").strip()
                if label:
                    offices.append(label)
        descs = []
        pay_text = None
        pay_name = None
        hours = None
        for block in pos.findall("jobDescriptions/jobDescription"):
            name = (block.findtext("name") or "").strip()
            if re.search(r"reward|referr", name, re.I):
                continue
            if re.fullmatch(r"(?i)(?:desired|expected|target)\s+salary", name):
                continue
            value_el = block.find("value")
            text = ""
            if value_el is not None:
                text = value_el.text or ""
                for child in list(value_el):
                    text += ET.tostring(child, encoding="unicode")
                    text += child.tail or ""
                text = text.strip()
            if not text:
                continue
            descs.append(text)
            if pay_text is None and _GH_PAY_META_RE.fullmatch(name):
                pay_text = text
                pay_name = name
            if hours is None:
                hours = _stated_hours("", f"{name}: {text}")
        return {
            "name": (pos.findtext("name") or "").strip(),
            "subcompany": (pos.findtext("subcompany") or "").strip(),
            "offices": offices,
            "schedule": (pos.findtext("schedule") or "").strip(),
            "descriptions": descs,
            "pay_text": pay_text,
            "pay_name": pay_name,
            "hours": hours,
        }
    if found or root.tag.endswith("workzag-jobs"):
        return None
    return {}


def _personio_to_html(pos: dict) -> str:
    """Turn Personio XML position into listing HTML. Never invent pay.

    Omit referral and desired-salary jobDescription blocks — those are not listed pay.
    """
    title = str(pos.get("name") or "").strip()
    company = str(pos.get("subcompany") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    sched = str(pos.get("schedule") or "").lower().replace("_", " ").replace("-", " ")
    if "part" in sched and "full" not in sched:
        posting["employmentType"] = "PART_TIME"
    elif "full" in sched and "part" not in sched:
        posting["employmentType"] = "FULL_TIME"
    offices = pos.get("offices") if isinstance(pos.get("offices"), list) else []
    labels = [str(o).strip() for o in offices if str(o).strip()]
    _apply_workplace(posting, *labels)
    n = _num(pos.get("hours"))
    if n is not None and 1 <= n <= 80:
        posting["workHours"] = str(int(n))
    pay = _named_pay_ld(str(pos.get("pay_name") or ""), str(pos.get("pay_text") or ""))
    if pay:
        posting["baseSalary"] = pay
    parts = [f"<p>{label}</p>" for label in labels]
    for desc in pos.get("descriptions") or []:
        if isinstance(desc, str) and desc.strip():
            parts.append(desc)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{' '.join(parts)}"
    )


_RECRUITEE_JOB_RE = re.compile(
    r"(?i)https?://([a-z0-9-]+)\.recruitee\.com/o/([A-Za-z0-9_-]+)"
)


def _recruitee_ids(url: str) -> Optional[tuple[str, str]]:
    m = _RECRUITEE_JOB_RE.search(url or "")
    if not m:
        return None
    board = m.group(1).casefold()
    if board in {"www", "app"}:
        return None
    return f"{board}.recruitee.com", m.group(2)


def _recruitee_job_url(url: str) -> Optional[str]:
    ids = _recruitee_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}/o/{ids[1]}"


def _recruitee_api_url(url: str) -> Optional[str]:
    ids = _recruitee_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}/api/offers/{ids[1]}"


def _recruitee_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.endswith(".recruitee.com"):
        return False
    return _recruitee_ids(url) is None


def _recruitee_offer(data: dict) -> Optional[dict]:
    offer = data.get("offer") if isinstance(data.get("offer"), dict) else data
    if not isinstance(offer, dict):
        return None
    if offer.get("title") or offer.get("slug"):
        return offer
    return None


def _recruitee_pay_ld(data: dict) -> Optional[dict]:
    """USD or stated foreign salary. Skip currency-only rows with no amounts."""
    for key in ("salary", "salaryRange", "compensation"):
        sal = data.get(key)
        if isinstance(sal, str) and sal.strip():
            pay = _span_pay_ld(sal)
            if pay:
                return pay
            continue
        if not isinstance(sal, dict):
            continue
        low, high = _bound_nums(sal)
        if low is None and high is None:
            continue
        unit = _period_unit(_ats_period(sal))
        value: dict = {}
        if unit:
            value["unitText"] = unit
        if low is not None and high is not None:
            value["minValue"] = int(low) if low == int(low) else low
            value["maxValue"] = int(high) if high == int(high) else high
        else:
            amount = high if high is not None else low
            value["value"] = int(amount) if amount == int(amount) else amount
        currency = _ats_currency(sal.get("currency"))
        return {"currency": currency, "value": value}
    return None


def _recruitee_to_html(data: dict) -> str:
    """Turn Recruitee offer JSON into listing HTML. Never invent pay.

    Omit open_questions — those are applicant form bands, not listed compensation.
    """
    title = str(data.get("title") or "").strip()
    company = str(data.get("company_name") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    code = str(data.get("employment_type_code") or "").lower()
    if "part" in code:
        posting["employmentType"] = "PART_TIME"
    elif "full" in code:
        posting["employmentType"] = "FULL_TIME"
    n = _num(data.get("min_hours_per_week"))
    if n is not None and 1 <= n <= 80:
        posting["workHours"] = str(int(n))
    else:
        _copy_hours(posting, data)
    if data.get("remote") is True:
        place = "remote"
    elif data.get("hybrid") is True:
        place = "hybrid"
    elif data.get("on_site") is True:
        place = "onsite"
    else:
        place = str(data.get("location") or "").strip()
    _apply_workplace(posting, place)
    pay = _recruitee_pay_ld(data)
    if pay:
        posting["baseSalary"] = pay
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    for key in ("description", "requirements"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_RIPPLING_JOB_RE = re.compile(
    r"(?i)https?://ats\.rippling\.com/([^/]+)/jobs/"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_NEXT_DATA_RE = re.compile(
    r'(?is)<script id="__NEXT_DATA__"[^>]*>(.*?)</script>'
)


def _rippling_ids(url: str) -> Optional[tuple[str, str]]:
    m = _RIPPLING_JOB_RE.search(url or "")
    if not m:
        return None
    return m.group(1), m.group(2)


def _rippling_job_url(url: str) -> Optional[str]:
    ids = _rippling_ids(url)
    if not ids:
        return None
    return f"https://ats.rippling.com/{ids[0]}/jobs/{ids[1]}"


def _rippling_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host in {"rippling.com", "www.rippling.com"}:
        return True
    if host != "ats.rippling.com":
        return False
    return _rippling_ids(url) is None


def _rippling_from_next(html: str) -> Optional[str]:
    """Listing HTML from Rippling NEXT_DATA. None if the job UUID is gone."""
    m = _NEXT_DATA_RE.search(html or "")
    if not m:
        return html
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return html
    api = ((data.get("props") or {}).get("pageProps") or {}).get("apiData")
    if not isinstance(api, dict):
        return html
    post = api.get("jobPost")
    if isinstance(post, dict) and (post.get("name") or post.get("uuid")):
        return _rippling_to_html(post, api)
    return None


def _rippling_place(post: dict) -> str:
    locs = post.get("workLocations") or []
    names = []
    if isinstance(locs, list):
        for loc in locs:
            if isinstance(loc, str) and loc.strip():
                names.append(loc.strip())
            elif isinstance(loc, dict):
                label = str(loc.get("name") or loc.get("workplaceType") or "").strip()
                if label:
                    names.append(label)
    rows = post.get("payRangeDetails")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("isRemote") is True:
                return "remote"
            loc = str(row.get("location") or "").strip()
            if loc:
                names.append(loc)
    if any(_workplace_remote(n) is True for n in names):
        return "remote"
    return ", ".join(names)


def _rippling_pay_ld(post: dict) -> Optional[dict]:
    rows = post.get("payRangeDetails")
    if not isinstance(rows, list):
        return None
    foreign = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        low, high = _num(row.get("rangeStart")), _num(row.get("rangeEnd"))
        if low is None and high is None:
            low, high = _bound_nums(row)
        if low is None and high is None:
            continue
        cur = _ats_currency(row.get("currency"))
        value: dict = {}
        unit = _period_unit(_ats_period(row))
        if unit:
            value["unitText"] = unit
        if low is not None and high is not None:
            value["minValue"] = int(low) if low == int(low) else low
            value["maxValue"] = int(high) if high == int(high) else high
        else:
            amount = high if high is not None else low
            value["value"] = int(amount) if amount == int(amount) else amount
        blob = {"currency": cur, "value": value}
        if _usd(cur):
            return blob
        if foreign is None:
            foreign = blob
    return foreign


def _rippling_to_html(post: dict, api: Optional[dict] = None) -> str:
    """Turn Rippling jobPost JSON into listing HTML. Never invent pay."""
    title = str(post.get("name") or "").strip()
    company = str(post.get("companyName") or "").strip()
    if not company:
        board = post.get("board") if isinstance(post.get("board"), dict) else None
        if board:
            company = str(board.get("companyName") or "").strip()
        elif isinstance(api, dict) and isinstance(api.get("jobBoard"), dict):
            company = str(api["jobBoard"].get("companyName") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = post.get("employmentType")
    label = ""
    if isinstance(emp, dict):
        label = str(emp.get("id") or emp.get("label") or "")
    elif isinstance(emp, str):
        label = emp
    lower = label.lower()
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower or "salaried_ft" in lower:
        posting["employmentType"] = "FULL_TIME"
    place = _rippling_place(post)
    _apply_workplace(posting, place)
    pay = _rippling_pay_ld(post)
    if pay:
        posting["baseSalary"] = pay
    _copy_hours(posting, post)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    desc = post.get("description")
    if isinstance(desc, dict):
        for key in ("company", "role"):
            val = desc.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
    elif isinstance(desc, str) and desc.strip():
        parts.append(desc)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_BREEZY_JOB_RE = re.compile(
    r"(?i)https?://([a-z0-9-]+)\.breezy\.hr/p/([a-f0-9]+)"
)


def _breezy_ids(url: str) -> Optional[tuple[str, str]]:
    m = _BREEZY_JOB_RE.search(url or "")
    if not m:
        return None
    board = m.group(1).casefold()
    if board in {"www", "app"}:
        return None
    return board, m.group(2).casefold()


def _breezy_job_url(url: str) -> Optional[str]:
    ids = _breezy_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}.breezy.hr/p/{ids[1]}"


def _breezy_json_url(url: str) -> Optional[str]:
    ids = _breezy_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}.breezy.hr/json"


def _breezy_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.endswith(".breezy.hr") and host != "breezy.hr":
        return False
    return _breezy_ids(url) is None


def _breezy_job(data, jid: str) -> Optional[dict]:
    if not isinstance(data, list):
        return None
    needle = (jid or "").casefold()
    for row in data:
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or "").casefold() == needle:
            return row
        fid = str(row.get("friendly_id") or "").casefold()
        if fid == needle or fid.startswith(f"{needle}-"):
            return row
    return None


def _breezy_pay_ld(job: dict) -> Optional[dict]:
    """USD or stated foreign salary from a salary object. Strings stay HTML."""
    raw = job.get("salary")
    if not isinstance(raw, dict):
        return None
    low, high = _bound_nums(raw)
    if low is None and high is None:
        return None
    unit = _period_unit(_ats_period(raw))
    value: dict = {}
    if unit:
        value["unitText"] = unit
    if low is not None and high is not None:
        value["minValue"] = int(low) if low == int(low) else low
        value["maxValue"] = int(high) if high == int(high) else high
    else:
        amount = high if high is not None else low
        value["value"] = int(amount) if amount == int(amount) else amount
    currency = _ats_currency(raw.get("currency"))
    return {"currency": currency, "value": value}


def _breezy_to_html(job: dict) -> str:
    """Turn a Breezy board JSON row into listing HTML. Never invent pay."""
    title = str(job.get("name") or "").strip()
    company = ""
    org = job.get("company")
    if isinstance(org, dict):
        company = str(org.get("name") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    kind = job.get("type")
    label = ""
    if isinstance(kind, dict):
        label = str(kind.get("id") or kind.get("name") or "")
    elif isinstance(kind, str):
        label = kind
    lower = label.lower().replace("_", " ").replace("-", " ")
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower:
        posting["employmentType"] = "FULL_TIME"
    loc = job.get("location") if isinstance(job.get("location"), dict) else {}
    remote = loc.get("is_remote")
    if remote is True:
        place = "remote"
    elif remote is False:
        place = "onsite"
    else:
        details = loc.get("remote_details") if isinstance(loc.get("remote_details"), dict) else {}
        place = str(details.get("value") or loc.get("name") or "").strip()
    _apply_workplace(posting, place)
    pay = _breezy_pay_ld(job)
    if pay:
        posting["baseSalary"] = pay
    _copy_hours(posting, job)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    salary = job.get("salary")
    if isinstance(salary, str) and salary.strip():
        parts.append(f"<p>{salary.strip()}</p>")
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_PINPOINT_JOB_RE = re.compile(
    r"(?i)https?://([a-z0-9-]+)\.pinpointhq\.com/(?:en/)?postings/"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def _pinpoint_ids(url: str) -> Optional[tuple[str, str]]:
    m = _PINPOINT_JOB_RE.search(url or "")
    if not m:
        return None
    board = m.group(1).casefold()
    if board in {"www", "app"}:
        return None
    return board, m.group(2).casefold()


def _pinpoint_job_url(url: str) -> Optional[str]:
    ids = _pinpoint_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}.pinpointhq.com/postings/{ids[1]}"


def _pinpoint_json_url(url: str) -> Optional[str]:
    ids = _pinpoint_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}.pinpointhq.com/postings.json"


def _pinpoint_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if not host.endswith(".pinpointhq.com") and host != "pinpointhq.com":
        return False
    return _pinpoint_ids(url) is None


def _pinpoint_job(rows, uuid: str) -> Optional[dict]:
    if not isinstance(rows, list):
        return None
    needle = (uuid or "").casefold()
    for row in rows:
        if not isinstance(row, dict):
            continue
        blob = f"{row.get('url') or ''} {row.get('path') or ''}".casefold()
        if needle and needle in blob:
            return row
    return None


def _pinpoint_pay_ld(job: dict) -> Optional[dict]:
    raw = job.get("compensation")
    nums = (
        _nums(job.get("compensation_minimum"))
        or _nums(job.get("compensation_min"))
    ) + (
        _nums(job.get("compensation_maximum"))
        or _nums(job.get("compensation_max"))
    )
    if nums:
        low, high = min(nums), max(nums)
    else:
        low = high = None
        if isinstance(raw, dict):
            low, high = _bound_nums(raw)
        if low is None and high is None:
            return None
    cur = _ats_currency(job.get("compensation_currency"))
    value: dict = {}
    unit = _period_unit(
        job.get("compensation_frequency")
        or (_ats_period(raw) if isinstance(raw, dict) else "")
    )
    if unit:
        value["unitText"] = unit
    if low is not None and high is not None:
        value["minValue"] = int(low) if low == int(low) else low
        value["maxValue"] = int(high) if high == int(high) else high
    else:
        amount = high if high is not None else low
        value["value"] = int(amount) if amount == int(amount) else amount
    return {"currency": cur, "value": value}


def _pinpoint_to_html(job: dict, board: str = "") -> str:
    """Turn a Pinpoint postings.json row into listing HTML. Never invent pay."""
    title = str(job.get("title") or "").strip()
    company = board.replace("-", " ").replace("_", " ").strip().title() if board else ""
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = str(job.get("employment_type") or job.get("employment_type_text") or "")
    lower = emp.lower().replace("_", " ").replace("-", " ")
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower:
        posting["employmentType"] = "FULL_TIME"
    work = str(job.get("workplace_type") or job.get("workplace_type_text") or "").strip()
    loc = job.get("location") if isinstance(job.get("location"), dict) else {}
    loc_name = str(loc.get("name") or "").strip()
    place = work.replace("_", " ") if work else loc_name
    _apply_workplace(posting, place if work else "", loc_name)
    pay = _pinpoint_pay_ld(job)
    if pay:
        posting["baseSalary"] = pay
    _copy_hours(posting, job)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    comp = str(job.get("compensation") or "").strip()
    if comp:
        parts.append(f"<p>{comp}</p>")
    for key in ("description", "key_responsibilities"):
        val = job.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_COMEET_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?comeet\.com/jobs/([^/]+)/([^/]+)/([^/]+)/([^/?#]+)"
)
_COMEET_TOKEN_RE = re.compile(r'"token"\s*:\s*"([A-Fa-f0-9]{20,})"')


def _comeet_ids(url: str) -> Optional[tuple[str, str, str, str]]:
    m = _COMEET_JOB_RE.search(url or "")
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _comeet_job_url(url: str) -> Optional[str]:
    ids = _comeet_ids(url)
    if not ids:
        return None
    return f"https://www.comeet.com/jobs/{ids[0]}/{ids[1]}/{ids[2]}/{ids[3]}"


def _comeet_api_url(ids: tuple[str, str, str, str], token: str) -> str:
    return (
        f"https://www.comeet.co/careers-api/2.0/company/{ids[1]}"
        f"/positions/{ids[3]}?token={token}&details=true"
    )


def _comeet_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host not in {"comeet.com", "www.comeet.com"}:
        return False
    return _comeet_ids(url) is None


def _comeet_token(html: str) -> Optional[str]:
    m = _COMEET_TOKEN_RE.search(html or "")
    return m.group(1) if m else None


def _comeet_to_html(data: dict) -> str:
    """Turn Comeet position JSON into listing HTML. Never invent pay.

    Omit referral rewards and desired-salary prompts — those are not listed pay.
    """
    title = str(data.get("name") or "").strip()
    company = str(data.get("company_name") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = str(data.get("employment_type") or "")
    lower = emp.lower().replace("_", " ").replace("-", " ")
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower:
        posting["employmentType"] = "FULL_TIME"
    loc = data.get("location") if isinstance(data.get("location"), dict) else {}
    work = str(data.get("workplace_type") or "").strip()
    loc_name = str(loc.get("name") or "").strip()
    if loc.get("is_remote") is True:
        place = "remote"
    else:
        place = work or loc_name
    _apply_workplace(posting, place if loc.get("is_remote") is True else work, loc_name)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    pay = None
    for item in data.get("details") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if re.search(r"reward|referr", name, re.I):
            continue
        if re.fullmatch(r"(?i)(?:desired|expected|target)\s+salary", name):
            continue
        val = item.get("value")
        if isinstance(val, str) and val.strip():
            parts.append(val)
            if pay is None and _GH_PAY_META_RE.fullmatch(name):
                pay = _named_pay_ld(name, val)
            stated = _stated_hours("", f"{name}: {val}")
            if stated and posting.get("workHours") is None:
                posting["workHours"] = str(stated)
    if pay:
        posting["baseSalary"] = pay
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{' '.join(parts)}"
    )


_BAMBOOHR_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?([a-z0-9-]+)\.bamboohr\.com/careers/(\d+)"
)


def _bamboohr_ids(url: str) -> Optional[tuple[str, str]]:
    m = _BAMBOOHR_JOB_RE.search(url or "")
    if not m:
        return None
    board = m.group(1).casefold()
    if board in {"www", "app", "careers"}:
        return None
    return board, m.group(2)


def _bamboohr_job_url(url: str) -> Optional[str]:
    ids = _bamboohr_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}.bamboohr.com/careers/{ids[1]}"


def _bamboohr_detail_url(url: str) -> Optional[str]:
    ids = _bamboohr_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}.bamboohr.com/careers/{ids[1]}/detail"


def _bamboohr_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host in {"bamboohr.com", "www.bamboohr.com"}:
        return True
    if not host.endswith(".bamboohr.com"):
        return False
    return _bamboohr_ids(url) is None


def _bamboohr_opening(data: dict) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if isinstance(result, dict) and isinstance(result.get("jobOpening"), dict):
        job = result["jobOpening"]
    elif isinstance(data.get("jobOpening"), dict):
        job = data["jobOpening"]
    elif data.get("jobOpeningName"):
        job = data
    else:
        return None
    if job.get("jobOpeningName") or job.get("description"):
        return job
    return None


def _bamboohr_place(job: dict) -> str:
    lt = job.get("locationType")
    try:
        n = int(lt)
    except (TypeError, ValueError):
        n = None
    if n == 1:
        return "remote"
    if n == 2:
        return "hybrid"
    if n == 0:
        return "onsite"
    if isinstance(lt, str) and lt.strip():
        return lt.strip()
    loc = job.get("location") if isinstance(job.get("location"), dict) else {}
    ats = job.get("atsLocation") if isinstance(job.get("atsLocation"), dict) else {}
    parts = [
        str(loc.get("city") or ats.get("city") or "").strip(),
        str(loc.get("state") or ats.get("state") or "").strip(),
        str(loc.get("addressCountry") or ats.get("country") or "").strip(),
    ]
    return ", ".join(p for p in parts if p)


def _bamboohr_pay_ld(job: dict) -> Optional[dict]:
    """USD or stated foreign salary from a compensation object. Strings stay HTML."""
    raw = job.get("compensation")
    if not isinstance(raw, dict):
        return None
    low, high = _bound_nums(raw)
    if low is None and high is None:
        return None
    unit = _period_unit(_ats_period(raw))
    value: dict = {}
    if unit:
        value["unitText"] = unit
    if low is not None and high is not None:
        value["minValue"] = int(low) if low == int(low) else low
        value["maxValue"] = int(high) if high == int(high) else high
    else:
        amount = high if high is not None else low
        value["value"] = int(amount) if amount == int(amount) else amount
    currency = _ats_currency(raw.get("currency"))
    return {"currency": currency, "value": value}


def _bamboohr_to_html(job: dict, board: str = "") -> str:
    """Turn BambooHR detail JSON into listing HTML. Never invent pay.

    Omit formFields — desiredPay is an applicant prompt, not listed compensation.
    """
    title = str(job.get("jobOpeningName") or "").strip()
    company = board.replace("-", " ").replace("_", " ").strip().title() if board else ""
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = str(job.get("employmentStatusLabel") or job.get("employmentType") or "")
    lower = emp.lower().replace("_", " ").replace("-", " ")
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower:
        posting["employmentType"] = "FULL_TIME"
    place = _bamboohr_place(job)
    _apply_workplace(posting, place)
    pay = _bamboohr_pay_ld(job)
    if pay:
        posting["baseSalary"] = pay
    _copy_hours(posting, job)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    comp = job.get("compensation")
    if isinstance(comp, str) and comp.strip():
        parts.append(f"<p>{comp.strip()}</p>")
    desc = job.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_JAZZHR_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?([a-z0-9-]+)\.applytojob\.com/apply/([A-Za-z0-9]{6,})"
)


def _jazzhr_ids(url: str) -> Optional[tuple[str, str]]:
    m = _JAZZHR_JOB_RE.search(url or "")
    if not m:
        return None
    board = m.group(1).casefold()
    if board in {"www", "app", "careers"}:
        return None
    return board, m.group(2)


def _jazzhr_job_url(url: str) -> Optional[str]:
    ids = _jazzhr_ids(url)
    if not ids:
        return None
    return f"https://{ids[0]}.applytojob.com/apply/{ids[1]}"


def _jazzhr_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host in {"applytojob.com", "www.applytojob.com"}:
        return True
    if not host.endswith(".applytojob.com"):
        return False
    return _jazzhr_ids(url) is None


_DOVER_UUID = (
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_DOVER_APPLY_RE = re.compile(
    rf"(?i)https?://(?:www\.)?app\.dover\.com/apply/([^/]+)/({_DOVER_UUID})"
)
_DOVER_CAREERS_RE = re.compile(
    rf"(?i)https?://(?:www\.)?app\.dover\.com/dover/careers/({_DOVER_UUID})"
)


def _dover_ids(url: str) -> Optional[str]:
    apply = _DOVER_APPLY_RE.search(url or "")
    if apply:
        return apply.group(2).casefold()
    careers = _DOVER_CAREERS_RE.search(url or "")
    if careers:
        return careers.group(1).casefold()
    return None


def _dover_job_url(url: str) -> Optional[str]:
    apply = _DOVER_APPLY_RE.search(url or "")
    if apply:
        return (
            f"https://app.dover.com/apply/{apply.group(1)}/{apply.group(2).casefold()}"
        )
    careers = _DOVER_CAREERS_RE.search(url or "")
    if careers:
        return f"https://app.dover.com/dover/careers/{careers.group(1).casefold()}"
    return None


def _dover_api_url(url: str) -> Optional[str]:
    jid = _dover_ids(url)
    if not jid:
        return None
    return f"https://app.dover.com/api/v1/inbound/application-portal-job/{jid}"


def _dover_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host not in {"app.dover.com", "www.app.dover.com"}:
        return False
    return _dover_ids(url) is None


def _dover_job(data) -> Optional[dict]:
    if not isinstance(data, dict):
        return None
    if data.get("active") is False or data.get("is_private") is True:
        return None
    if data.get("title") or data.get("id"):
        return data
    return None


def _dover_place(job: dict) -> str:
    work = str(job.get("workplace_type") or "").strip()
    if work:
        return work.replace("_", " ")
    rows = job.get("locations") if isinstance(job.get("locations"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("location_type") or "").replace("_", " ")
        name = str(row.get("name") or "").strip()
        if kind or name:
            return kind or name
    return str(job.get("location") or "").strip()


def _dover_pay_ld(job: dict) -> Optional[dict]:
    raw = job.get("compensation")
    if isinstance(raw, str):
        return _span_pay_ld(raw)
    comp = raw if isinstance(raw, dict) else {}
    low, high = _num(comp.get("lower_bound")), _num(comp.get("upper_bound"))
    if low is None and high is None:
        low, high = _bound_nums(comp)
    if low is None and high is None:
        return None
    cur = _ats_currency(comp.get("currency_code"))
    value: dict = {}
    unit = _period_unit(comp.get("salary_range_type") or _ats_period(comp))
    if unit:
        value["unitText"] = unit
    if low is not None and high is not None:
        value["minValue"] = int(low) if low == int(low) else low
        value["maxValue"] = int(high) if high == int(high) else high
    else:
        amount = high if high is not None else low
        value["value"] = int(amount) if amount == int(amount) else amount
    return {"currency": cur, "value": value}


def _dover_to_html(job: dict) -> str:
    """Turn Dover inbound job JSON into listing HTML. Never invent pay.

    Omit application_questions — those are applicant prompts, not listed pay.
    """
    title = str(job.get("title") or "").strip()
    company = str(job.get("client_name") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = ""
    comp = job.get("compensation") if isinstance(job.get("compensation"), dict) else {}
    emp = str(comp.get("employment_type") or "")
    lower = emp.lower().replace("_", " ").replace("-", " ")
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower:
        posting["employmentType"] = "FULL_TIME"
    place = _dover_place(job)
    _apply_workplace(posting, place)
    pay = _dover_pay_ld(job)
    if pay:
        posting["baseSalary"] = pay
    _copy_hours(posting, job)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    desc = job.get("user_provided_description")
    if isinstance(desc, str) and desc.strip():
        parts.append(desc)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_GEM_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?jobs\.gem\.com/([^/]+)/([^/?#]+)(?:/application)?"
)
_GEM_JOB_QUERY = """
query ExternalJobPosting($boardId: String!, $extId: String!) {
  oatsExternalJobPosting(boardId: $boardId, extId: $extId) {
    id
    title
    descriptionHtml
    compensationHtml
    isUnlistedExternally
    locations { name city isoCountry isRemote }
    job { locationType employmentType teamDisplayName }
  }
}
"""


def _gem_ids(url: str) -> Optional[tuple[str, str]]:
    m = _GEM_JOB_RE.search(url or "")
    if not m:
        return None
    board = m.group(1)
    jid = m.group(2)
    if board.casefold() in {"jobs", "apply", "application"}:
        return None
    if jid.casefold() in {"jobs", "apply", "application"}:
        return None
    return board, jid


def _gem_job_url(url: str) -> Optional[str]:
    ids = _gem_ids(url)
    if not ids:
        return None
    return f"https://jobs.gem.com/{ids[0]}/{ids[1]}"


def _gem_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host not in {"jobs.gem.com", "www.jobs.gem.com"}:
        return False
    return _gem_ids(url) is None


async def _gem_posting(client: httpx.AsyncClient, board: str, jid: str) -> Optional[dict]:
    """None if the posting is gone. Empty dict if the API failed."""
    try:
        resp = await client.post(
            "https://jobs.gem.com/api/public/graphql",
            json={
                "operationName": "ExternalJobPosting",
                "variables": {"boardId": board, "extId": jid},
                "query": _GEM_JOB_QUERY,
            },
            headers=_LISTING_HEADERS,
        )
        if resp.status_code in (404, 410):
            return None
        if resp.status_code >= 400:
            return {}
        data = json.loads(resp.text)
        if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
            return {}
        posting = data["data"].get("oatsExternalJobPosting")
        if posting is None and "oatsExternalJobPosting" in data["data"]:
            return None
        if isinstance(posting, dict) and posting:
            if posting.get("isUnlistedExternally") is True:
                return None
            return posting
        return {}
    except Exception:
        return {}


def _gem_place(post: dict) -> str:
    job = post.get("job") if isinstance(post.get("job"), dict) else {}
    kind = str(job.get("locationType") or "").strip()
    if kind:
        return kind.replace("_", " ")
    rows = post.get("locations") if isinstance(post.get("locations"), list) else []
    if any(isinstance(r, dict) and r.get("isRemote") is True for r in rows):
        return "remote"
    for row in rows:
        if isinstance(row, dict) and str(row.get("name") or "").strip():
            return str(row.get("name")).strip()
    return ""


def _gem_to_html(post: dict, board: str = "") -> str:
    """Turn Gem oatsExternalJobPosting JSON into listing HTML. Never invent pay.

    Omit application questions — those are applicant prompts, not listed pay.
    """
    title = str(post.get("title") or "").strip()
    job = post.get("job") if isinstance(post.get("job"), dict) else {}
    company = str(job.get("teamDisplayName") or "").strip()
    if not company and board:
        company = board.replace("-", " ").replace("_", " ").strip().title()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    emp = str(job.get("employmentType") or "")
    lower = emp.lower().replace("_", " ").replace("-", " ")
    if "part" in lower:
        posting["employmentType"] = "PART_TIME"
    elif "full" in lower:
        posting["employmentType"] = "FULL_TIME"
    place = _gem_place(post)
    _apply_workplace(posting, place)
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    for key in ("compensationHtml", "descriptionHtml"):
        val = post.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_WALMART_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?careers\.walmart\.com/(?:[a-z]{2}/[a-z]{2}/)?jobs/(R-\d+)"
)
_NEXT_DATA_RE = re.compile(
    r'<script[^>]*\bid=["\']?__NEXT_DATA__["\']?[^>]*>(.*?)</script>',
    re.I | re.S,
)


def _walmart_ids(url: str) -> Optional[str]:
    m = _WALMART_JOB_RE.search(url or "")
    return m.group(1) if m else None


def _walmart_job_url(url: str) -> Optional[str]:
    jid = _walmart_ids(url)
    if not jid:
        return None
    return f"https://careers.walmart.com/us/en/jobs/{jid}"


def _walmart_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host not in {"careers.walmart.com", "www.careers.walmart.com"}:
        return False
    return _walmart_ids(url) is None


def _walmart_details(html: str, jid: str) -> Optional[dict]:
    """Open jobDetails. None if gone. Empty dict if the page is unusable."""
    m = _NEXT_DATA_RE.search(html or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    props = data.get("props")
    pp = props.get("pageProps") if isinstance(props, dict) else None
    if not isinstance(pp, dict):
        return {}
    details = pp.get("jobDetails")
    if details is None:
        return None
    if not isinstance(details, dict):
        return {}
    if not (details.get("title") or details.get("jobPostingTitle")):
        return {}
    page_id = str(pp.get("jobId") or details.get("jobId") or "").strip()
    if page_id and page_id.casefold() != jid.casefold():
        return None
    if details.get("active") is False:
        return None
    if details.get("positionAvailable") == 0:
        return None
    return details


def _walmart_place(details: dict) -> str:
    names = []
    primary = details.get("primaryLocation")
    if isinstance(primary, dict):
        city = str(primary.get("city") or "").strip().title()
        state = str(primary.get("stateCode") or "").strip()
        names.append(", ".join(p for p in (city, state) if p))
    for row in details.get("additionalLocations") or []:
        if not isinstance(row, dict):
            continue
        city = str(row.get("city") or "").strip().title()
        state = str(row.get("stateCode") or "").strip()
        names.append(", ".join(p for p in (city, state) if p) or str(row.get("locationName") or ""))
    blob = " ".join(names)
    if re.search(r"(?i)\bremote\b", blob):
        return "Remote"
    if names:
        return f"{names[0]} onsite"
    return "onsite"


def _walmart_currency(details: dict) -> str:
    plan = details.get("payPlanData")
    if not isinstance(plan, dict):
        return ""
    cref = plan.get("currencyReference")
    if not isinstance(cref, dict):
        return ""
    return (_ld_text(cref.get("currencyId")) or "").strip()


def _walmart_to_html(details: dict) -> str:
    """Turn Walmart careers jobDetails into listing HTML. Never invent pay.

    Omit questionnaires — those are applicant prompts, not listed pay.
    """
    company = str(details.get("brand") or "").strip()
    title = str(details.get("jobPostingTitle") or details.get("title") or "").strip()
    posting: dict = {"@type": "JobPosting", "title": title}
    if company:
        posting["hiringOrganization"] = {"@type": "Organization", "name": company}
    place = _walmart_place(details)
    _apply_workplace(posting, place)
    currency = _walmart_currency(details)
    usd = _usd(currency) if currency else True
    parts = []
    if place:
        parts.append(f"<p>{place}</p>")
    first = None
    if usd:
        for row in details.get("payRange") or []:
            if not isinstance(row, dict):
                continue
            low = _num(row.get("min"))
            high = _num(row.get("max"))
            loc = str(row.get("location") or "").strip()
            if not (low or high):
                continue
            lo = int(low) if low and 10_000 <= low <= 2_000_000 else None
            hi = int(high) if high and 10_000 <= high <= 2_000_000 else None
            if not (lo or hi):
                continue
            if first is None:
                first = (lo, hi)
            band = " - ".join(f"${n:,}" for n in (lo, hi) if n)
            parts.append(f"<p>{loc}: {band}</p>" if loc else f"<p>{band}</p>")
        if first:
            value: dict = {"unitText": "YEAR"}
            lo, hi = first
            if lo and hi:
                value["minValue"] = lo
                value["maxValue"] = hi
            else:
                value["value"] = hi or lo
            posting["baseSalary"] = {"currency": "USD", "value": value}
    elif currency:
        posting["baseSalary"] = {"currency": currency, "value": {"unitText": "YEAR"}}
        parts.append(f"<p>{currency}</p>")
    for key in ("jobPostingDescription", "description"):
        val = details.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(unescape(val))
            break
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_APPLE_JOB_RE = re.compile(
    r"(?i)https?://(?:www\.)?jobs\.apple\.com/(?:[a-z]{2}(?:-[a-z]{2})?/)?"
    r"details/(\d+(?:-\d+)?)"
)
_APPLE_HYDRATION_RE = re.compile(
    r'window\.__staticRouterHydrationData\s*=\s*JSON\.parse\("((?:\\.|[^"\\])*)"\)',
)


def _apple_ids(url: str) -> Optional[str]:
    m = _APPLE_JOB_RE.search(url or "")
    return m.group(1) if m else None


def _apple_job_url(url: str) -> Optional[str]:
    jid = _apple_ids(url)
    if not jid:
        return None
    return f"https://jobs.apple.com/en-us/details/{jid}"


def _apple_is_board(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").casefold()
    if host not in {"jobs.apple.com", "www.jobs.apple.com"}:
        return False
    return _apple_ids(url) is None


def _apple_hydration(html: str) -> Optional[dict]:
    m = _APPLE_HYDRATION_RE.search(html or "")
    if not m:
        return None
    try:
        blob = json.loads(f'"{m.group(1)}"')
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _apple_job(html: str) -> Optional[dict]:
    """Open jobsData. None if gone. Empty dict if the page is unusable."""
    data = _apple_hydration(html)
    if not data:
        if re.search(r"(?i)this role does not exist|no longer available", html or ""):
            return None
        return {}
    errors = data.get("errors")
    if isinstance(errors, dict):
        err = errors.get("jobDetails")
        if isinstance(err, dict) and err.get("status") in (404, 410):
            return None
    ld = data.get("loaderData")
    details = ld.get("jobDetails") if isinstance(ld, dict) else None
    job = details.get("jobsData") if isinstance(details, dict) else None
    if job is None and isinstance(details, dict) and "jobsData" in details:
        return None
    if not isinstance(job, dict) or not (job.get("postingTitle") or job.get("id")):
        return {}
    return job


def _apple_place(job: dict) -> str:
    if job.get("homeOffice") is True:
        return "Remote"
    for row in job.get("locations") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("city") or "").strip()
        if name:
            return f"{name} onsite"
    return "onsite"


def _apple_to_html(job: dict) -> str:
    """Turn Apple jobsData into listing HTML. Never invent pay.

    Omit questionnaires — those are applicant prompts, not listed pay.
    """
    title = str(job.get("postingTitle") or "").strip()
    posting: dict = {
        "@type": "JobPosting",
        "title": title,
        "hiringOrganization": {"@type": "Organization", "name": "Apple"},
    }
    place = _apple_place(job)
    _apply_workplace(posting, place)
    n = _num(job.get("standardWeeklyHours"))
    if n is not None and 1 <= n <= 80:
        posting["workHours"] = str(int(n))
    emp = str(job.get("employmentType") or "").lower().replace("_", " ").replace("-", " ")
    if "part" in emp and "full" not in emp:
        posting["employmentType"] = "PART_TIME"
    elif "full" in emp:
        posting["employmentType"] = "FULL_TIME"
    parts = [f"<p>{place}</p>"] if place else []
    for key in ("jobSummary", "description"):
        val = job.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"<p>{unescape(val)}</p>")
            break
    page_title = f"{title} at Apple" if title else "Apple"
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
    )


_INDEX_PATH_RE = re.compile(
    r"^/(?:category|categories|tag|tags|topics?|major)(?:/|$)|/search(?:/|$)"
    r"|^/(?:careers|jobs)/?$"
    r"|^/hire(?:/|$)"
    r"|/(?:open[-_]?(?:positions|roles|jobs|listings|opportunities)|current[-_]?(?:openings|positions|roles|listings|opportunities))/?$"
    r"|/(?:career[-_]?(?:opportunities|listings|roles|positions)|(?:job|role|position|listing)[-_]?openings|all[-_]?(?:openings|positions|roles|listings|opportunities|jobs))/?$"
    r"|/(?:join[-_]?our[-_]?team|work[-_]?with[-_]?us|we(?:re)?[-_]?hiring)/?$"
    r"|/opportunities/?$"
    r"|/(?:(?:job|role|position|listing)[-_]?vacancies|available[-_]?(?:positions|roles|jobs|listings|opportunities|openings)|vacancies|explore[-_]?careers|browse[-_]?careers|find[-_]?careers|search[-_]?careers|view[-_]?careers|discover[-_]?careers|see[-_]?careers|apply[-_]?careers|open[-_]?careers|hiring)/?$"
    r"|/(?:featured|latest|popular|hot|new|trending|recommended|matching|similar|suggested|related|other|browse|explore|view|discover|see|find|search|apply)[-_]?(?:positions|roles|listings|openings|opportunities)/?$"
    r"|/(?:internships|university[-_]?recruiting|campus[-_]?recruiting|early[-_]?careers?|student[-_]?programs?|graduate[-_]?programs?|university[-_]?programs?|job[-_]?search|career[-_]?search|life[-_]?at(?:[-_][^/]+)?|team|meet[-_]?(?:the|our)[-_]?team|our[-_]?(?:team|people)|benefits|our[-_]?benefits|culture|our[-_]?culture|leadership|our[-_]?leadership|about[-_]?us|about|our[-_]?values|values|our[-_]?mission|locations|our[-_]?locations|diversity|inclusion|dei|our[-_]?dei|diversity[-_]?equity(?:[-_]?and)?[-_]?inclusion|our[-_]?story|faqs?|news|press|blog|our[-_]?blog|newsroom|press[-_]?releases?|our[-_]?news|investors?|investor[-_]?relations|sustainability|our[-_]?sustainability|esg|impact|our[-_]?impact|community|our[-_]?community|csr|social[-_]?responsibility|purpose|our[-_]?purpose|mission|people|ethics|governance|environment|history|our[-_]?history|media[-_]?center|press[-_]?center|foundation|our[-_]?foundation|giving|our[-_]?giving|philanthropy|citizenship|corporate[-_]?citizenship|volunteering|charity|responsibility)/?$"
    r"|/(?:salaries|salary)(?:/|$)"
    r"|/apply/?$",
    re.I,
)


def _ats_job_url(url: str) -> bool:
    """True when the URL is a specific ATS posting, not a board or catalog."""
    return bool(
        _greenhouse_ids(url)
        or _greenhouse_hosted_ids(url)
        or _lever_api_url(url)
        or _ashby_ids(url)
        or _workable_md_url(url)
        or _workable_jobs_api_url(url)
        or _smartrecruiters_ids(url)
        or _workday_ids(url)
        or _icims_ids(url)
        or _jobvite_ids(url)
        or _teamtailor_ids(url)
        or _personio_ids(url)
        or _recruitee_ids(url)
        or _rippling_ids(url)
        or _breezy_ids(url)
        or _pinpoint_ids(url)
        or _comeet_ids(url)
        or _bamboohr_ids(url)
        or _jazzhr_ids(url)
        or _dover_ids(url)
        or _gem_ids(url)
        or _walmart_ids(url)
        or _apple_ids(url)
    )


def _is_index_page(raw: dict) -> bool:
    """True for search/board/home/category pages, not a single opportunity."""
    url = raw.get("url") or ""
    title = raw.get("title") or ""
    desc = raw.get("description") or ""
    if _INDEX_URL_RE.search(url):
        return True
    if _ats_job_url(url):
        return False
    if _title_is_index(title):
        return True
    if re.match(r"(?i)\s*browse\s+\d+", desc):
        return True
    if re.match(r"(?i)\s*home\s*[|\-–]", title):
        return True
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    if path == "/":
        return True
    if _INDEX_PATH_RE.search(parsed.path):
        return True
    if _lever_is_board(url):
        return True
    if _greenhouse_is_board(url):
        return True
    if _ashby_is_board(url):
        return True
    if _workable_is_board(url):
        return True
    if _smartrecruiters_is_board(url):
        return True
    if _workday_is_board(url):
        return True
    if _icims_is_board(url):
        return True
    if _jobvite_is_board(url):
        return True
    if _teamtailor_is_board(url):
        return True
    if _personio_is_board(url):
        return True
    if _recruitee_is_board(url):
        return True
    if _rippling_is_board(url):
        return True
    if _breezy_is_board(url):
        return True
    if _pinpoint_is_board(url):
        return True
    if _comeet_is_board(url):
        return True
    if _bamboohr_is_board(url):
        return True
    if _jazzhr_is_board(url):
        return True
    if _dover_is_board(url):
        return True
    if _gem_is_board(url):
        return True
    if _walmart_is_board(url):
        return True
    if _apple_is_board(url):
        return True
    return False


def _compensation_from_raw(
    raw: dict, title: str, description: str, hours: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """Structured source pay wins; otherwise parse listing text. Never invent."""
    if raw.get("pay") is not None:
        return None, raw["pay"]
    return _parse_pay(f"{title} {description}", hours)


_RELATED_JOBS_RE = re.compile(
    r"(?is)(?:similar|related|recommended|other|more|featured|popular|suggested|matching)\s+jobs.*$"
    r"|jobs\s+(?:you\s+(?:may|might)\s+like|near\s+you).*$"
    r"|you\s+(?:may|might)\s+(?:also\s+)?like.*$"
    r"|(?:people|applicants|candidates)\s+also\s+viewed.*$"
    r"|(?:similar|related|recommended|other|more|featured|popular|suggested|matching|latest|current|available|hot|new|trending)\s+(?:roles|openings|positions).*$"
    r"|(?:similar|related|recommended|other|featured|popular|suggested|matching|latest|current|available|hot|new|trending)\s+(?:opportunities|listings).*$"
    r"|(?:similar|related|recommended|featured|suggested|matching)\s*:?\s*\$.*$"
)
_RELATED_HEADING_RE = re.compile(
    r"(?is)(</(?:p|h1|article|section|div|ul|ol|li|main)>)(\s*)"
    r"<(h[1-6])(?:\s[^>]*)?>\s*"
    r"(?:browse(?:\s+all)?(?:\s+open)?\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|new\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|see\s+also"
    r"|see\s+more"
    r"|discover\s+more"
    r"|hot\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|latest\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|explore(?:\s+all)?\s+(?:jobs|roles|openings|positions|listings|opportunities)"
    r"|continue\s+browsing(?:\s+(?:jobs|roles|positions|listings|opportunities))?"
    r"|more\s+opportunities"
    r"|open\s+(?:positions|roles|jobs|listings|opportunities)"
    r"|current\s+(?:openings|roles|jobs|positions|listings|opportunities)"
    r"|view(?:\s+all)?\s+(?:jobs|roles|openings|positions|listings|opportunities)"
    r"|recommended\s+for\s+you"
    r"|jobs\s+recommended\s+for\s+you"
    r"|roles\s+recommended\s+for\s+you"
    r"|(?:positions|listings|opportunities)\s+recommended\s+for\s+you"
    r"|more\s+from\s+this\s+company"
    r"|more\s+from\s+\S+"
    r"|jobs\s+at\s+this\s+company"
    r"|jobs\s+at\s+\S+"
    r"|all\s+(?:jobs|openings|roles|positions|listings|opportunities)"
    r"|see(?:\s+all)?\s+(?:jobs|roles|openings|positions|listings|opportunities)"
    r"|discover\s+(?:jobs|openings|roles|positions|listings|opportunities)"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+you\s+applied\s+to"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+you\s+(?:may|might)\s+be\s+interested\s+in"
    r"|browse(?:\s+all)?\s+openings"
    r"|explore\s+careers"
    r"|browse\s+careers"
    r"|latest\s+openings"
    r"|new\s+openings"
    r"|hot\s+openings"
    r"|available\s+openings"
    r"|matching\s+openings"
    r"|matching\s+positions"
    r"|people\s+also\s+applied(?:\s+for)?"
    r"|applicants\s+also\s+applied(?:\s+for)?"
    r"|available\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|your\s+recent\s+searches"
    r"|recent\s+searches"
    r"|hiring\s+in\s+your\s+area"
    r"|explore\s+more"
    r"|browse\s+more"
    r"|view\s+more"
    r"|jobs\s+for\s+you(?:\s+nearby)?"
    r"|roles\s+for\s+you"
    r"|opportunities\s+for\s+you"
    r"|(?:positions|listings)\s+for\s+you(?:\s+nearby)?"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+in\s+your\s+area"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+nearby"
    r"|nearby\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+near\s+you"
    r"|because\s+you\s+searched(?:\s+for)?(?:\s+this(?:\s+(?:job|role|position|listing|opportunit(?:y|ies)))?)?"
    r"|because\s+you\s+applied(?:\s+to(?:\s+this(?:\s+(?:job|role|position|listing|opportunit(?:y|ies)))?)?)?"
    r"|because\s+you\s+liked(?:\s+this(?:\s+(?:job|role|position|listing|opportunit(?:y|ies)))?)?"
    r"|because\s+you\s+saved(?:\s+this(?:\s+(?:job|role|position|listing|opportunit(?:y|ies)))?)?"
    r"|your\s+(?:recent\s+)?applications"
    r"|your\s+saved\s+searches"
    r"|recently\s+saved(?:\s+jobs)?"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+you\s+saved"
    r"|saved\s+for\s+later"
    r"|keep\s+scrolling"
    r"|continue\s+scrolling"
    r"|keep\s+discovering"
    r"|continue\s+discovering"
    r"|continue\s+looking"
    r"|people\s+also\s+liked"
    r"|explore\s+(?:similar|related)"
    r"|(?:discover|see|browse)\s+similar"
    r"|others\s+also\s+applied(?:\s+for)?"
    r"|based\s+on\s+your\s+search"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+based\s+on\s+your\s+search"
    r"|trending\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|(?:your\s+)?saved\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|applied\s+(?:jobs|roles|positions|listings|opportunities)"
    r"|keep\s+looking"
    r"|hiring\s+nearby"
    r"|hiring\s+near\s+you"
    r"|top\s+picks"
    r"|more\s+like\s+this(?:\s+(?:job|role|position|listing|opportunit(?:y|ies)))?"
    r"|more\s+like\s+these"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+like\s+this"
    r"|you\s+applied"
    r"|you\s+recently\s+viewed"
    r"|recently\s+viewed(?:\s+(?:jobs?|roles|positions|listings|opportunit(?:y|ies)))?"
    r"|others\s+also\s+viewed"
    r"|because\s+you\s+viewed(?:\s+this(?:\s+(?:job|role|position|listing|opportunit(?:y|ies)))?)?"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+you\s+viewed"
    r"|keep\s+browsing(?:\s+(?:jobs|roles|positions|listings|opportunities))?"
    r"|keep\s+exploring(?:\s+(?:jobs|roles|positions|listings|opportunities))?"
    r"|continue\s+exploring(?:\s+(?:jobs|roles|positions|listings|opportunities))?"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+similar\s+to\s+this(?:\s+(?:job|role|position|listing|opportunit(?:y|ies)))?"
    r"|based\s+on\s+your\s+activity"
    r"|(?:jobs|roles|positions|listings|opportunities)\s+based\s+on\s+your\s+activity"
    r"|people\s+also\s+searched"
    r"|recently\s+applied(?:\s+jobs)?"
    r"|similar\s+careers"
    r"|related\s+careers"
    r"|other\s+careers"
    r"|similar\s+listings"
    r"|(?:featured|related|other|recommended|more|popular|open)\s+listings"
    r"|similar\s+job"
    r"|matching\s+roles"
    r"|related"
    r"|similar"
    r"|recommended)\s*"
    r"</\3>.*$"
)


def _visible_text(html: str) -> str:
    return unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))).strip()


def _listing_plain_text(html: str) -> str:
    """Visible listing copy only — scripts, styles, and related-job cards are not pay."""
    html = re.sub(r"(?is)<script\b[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style\b[^>]*>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript\b[^>]*>.*?</noscript>", " ", html)
    html = _RELATED_HEADING_RE.sub(r"\1", html)
    return _RELATED_JOBS_RE.sub("", _visible_text(html))


_LD_SCRIPT_RE = re.compile(
    r'<script\b[^>]*\btype\s*=\s*["\']?application/ld\+json[^>]*>\s*(.*?)\s*</script>',
    re.I | re.S,
)
_LD_COMMENT_RE = re.compile(
    r"(?is)^\s*(?:<!--\s*|(?://\s*)?<!\[CDATA\[\s*|//[^\n]*\n\s*|/\*.*?\*/\s*)"
)
_LD_COMMENT_TAIL_RE = re.compile(r"(?is)\s*(?:-->|(?://\s*)?\]\]>)\s*$")
_PAY_UNITS = {
    "HOUR": "hour",
    "HOURLY": "hour",
    "HR": "hour",
    "HRS": "hour",
    "HOURS": "hour",
    "HUR": "hour",
    "YEAR": "year",
    "ANNUAL": "year",
    "ANNUALLY": "year",
    "ANNUM": "year",
    "YEARLY": "year",
    "YR": "year",
    "ANN": "year",
    "PER_YEAR": "year",
    "PER YEAR": "year",
    "PER_HOUR": "hour",
    "PER HOUR": "hour",
    "AN_HOUR": "hour",
    "AN HOUR": "hour",
    "PER_WEEK": "week",
    "PER WEEK": "week",
    "PER_MONTH": "month",
    "PER MONTH": "month",
    "PER_DAY": "day",
    "PER DAY": "day",
    "WEEK": "week",
    "WEEKLY": "week",
    "WEE": "week",
    "MONTH": "month",
    "MONTHLY": "month",
    "MON": "month",
    "DAY": "day",
    "DAILY": "day",
    "DIEM": "day",
    "PER_DIEM": "day",
    "PERDIEM": "day",
    "BIWEEKLY": "biweek",
    "BI_WEEKLY": "biweek",
    "BI-WEEKLY": "biweek",
    "FORTNIGHT": "biweek",
    "FORTNIGHTLY": "biweek",
    "SEMIMONTHLY": "semimonth",
    "SEMI_MONTHLY": "semimonth",
    "SEMI-MONTHLY": "semimonth",
}


def _ld_types(value) -> set[str]:
    if isinstance(value, str):
        return {value.rsplit("/", 1)[-1]}
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out |= _ld_types(item)
        return out
    if isinstance(value, dict):
        return _ld_types(value.get("@type")) | _ld_types(_ld_text(value))
    return set()


_LD_REF_KEYS = frozenset({"@id", "@type", "@context"})


def _walk_ld(obj, in_itemlist: bool = False):
    """Yield (node, inside_itemlist). Related cards live under ItemList."""
    if isinstance(obj, list):
        for item in obj:
            yield from _walk_ld(item, in_itemlist)
    elif isinstance(obj, dict):
        listed = in_itemlist or "ItemList" in _ld_types(obj.get("@type"))
        yield obj, listed
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from _walk_ld(value, listed)


def _ld_ids(obj) -> dict[str, dict]:
    """Index JSON-LD nodes that actually hold data, including #fragment aliases."""
    out: dict[str, dict] = {}
    for node, _ in _walk_ld(obj):
        ident = node.get("@id")
        if not isinstance(ident, str):
            continue
        ident = ident.strip()
        if not ident or not (set(node) - _LD_REF_KEYS):
            continue
        out[ident] = node
        if "#" in ident:
            frag = "#" + ident.rsplit("#", 1)[-1]
            if frag != ident and frag != "#":
                out.setdefault(frag, node)
    return out


def _ld_resolve(value, index: dict, stack: frozenset = frozenset()):
    """Replace {@id} stubs with the graph node. Salary often lives on another node."""
    if isinstance(value, list):
        return [_ld_resolve(v, index, stack) for v in value]
    if not isinstance(value, dict):
        return value
    ident = value.get("@id")
    if isinstance(ident, str) and ident not in index and "#" in ident:
        frag = "#" + ident.rsplit("#", 1)[-1]
        if frag in index:
            ident = frag
    if (
        isinstance(ident, str)
        and ident in index
        and not (set(value) - _LD_REF_KEYS)
        and ident not in stack
    ):
        return _ld_resolve(index[ident], index, stack | {ident})
    return {k: _ld_resolve(v, index, stack) for k, v in value.items()}


def _ld_title_hit(posting: dict, blob: str) -> int:
    """Length of posting title if blob names that role, else -1."""
    pt = (_ld_text(posting.get("title")) or "").strip().casefold()
    if not pt or not (blob or "").strip():
        return -1
    title = blob.casefold().strip()
    head = re.split(r"\s*[•|]\s*", title, maxsplit=1)[0].strip()
    role_head = re.split(r"\s+at\s+", head, maxsplit=1)[0].strip()
    if role_head == pt or role_head.startswith(pt + " ") or head.startswith(pt):
        return len(pt)
    return -1


def _ld_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        if "'" in cleaned and '"' not in cleaned:
            cleaned = cleaned.replace("'", '"')
        if cleaned == text:
            return None
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _ld_payload(raw: str):
    """Parse JSON-LD script text. CMS wrappers and HTML entities are not pay."""
    text = _LD_COMMENT_TAIL_RE.sub("", _LD_COMMENT_RE.sub("", raw or ""))
    text = text.lstrip("\ufeff").strip()
    if not text:
        return None
    for candidate in (text, unescape(text).lstrip("\ufeff").strip()):
        data = _ld_json(candidate)
        if data is not None:
            return data
    return None


def _job_posting(html: str, role: str = "") -> Optional[dict]:
    """The listing's JobPosting. Related-job JSON-LD in an ItemList is not pay."""
    posts: list[tuple[dict, bool]] = []
    seen: set[int] = set()
    index: dict[str, dict] = {}
    for raw in _LD_SCRIPT_RE.findall(html or ""):
        data = _ld_payload(raw)
        if data is None:
            continue
        index.update(_ld_ids(data))
        for obj, listed in _walk_ld(data):
            if "JobPosting" not in _ld_types(obj.get("@type")):
                continue
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            posts.append((obj, listed))
    if not posts:
        return None
    standalone = [p for p, listed in posts if not listed]
    pool = standalone or [p for p, _ in posts]
    posting = None
    if len(pool) == 1:
        posting = pool[0]
    else:
        page = _html_title(html)
        best = None
        best_n = -1
        for item in pool:
            n = max(_ld_title_hit(item, page), _ld_title_hit(item, role))
            if n > best_n:
                best, best_n = item, n
        if best_n >= 0:
            posting = best
        elif standalone:
            posting = standalone[0]
    if posting is None:
        return None
    return _ld_resolve(posting, index)


def _num(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.replace(",", "").replace("$", "").strip()
        s = re.sub(r"^(?:USD|US)\s*", "", s, flags=re.I)
        s = re.sub(r"\s*(?:USD|US)$", "", s, flags=re.I)
        s = re.sub(
            r"(?:\s*(?:USD|US))?(?:"
            r"\s*/\s*(?:yearly|annual(?:ly)?|year(?!s)|yr|annum|hourly|hours?|hrs?|hr|daily|days?|day|diem|weekly|weeks?|week|monthly|months?|month)"
            r"|\s+per\s+(?:yearly|annual(?:ly)?|year(?!s)|yr|annum|hour|hr|day|diem|week|month)"
            r"|\s+an?\s+(?:year(?!s)|hour|day|week|month)"
            r"|\s+yearly"
            r"|\s+annual(?:ly)?"
            r"|\s+hourly"
            r")\s*$",
            "",
            s,
            flags=re.I,
        )
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*k", s, flags=re.I)
        if m:
            return float(m.group(1)) * 1000
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("@value", "value", "name"):
            nested = value.get(key)
            if nested is not None and nested is not value:
                amount = _num(nested)
                if amount is not None:
                    return amount
        return _num(_ld_text(value))
    return None


def _bound_nums(raw: dict) -> tuple[Optional[float], Optional[float]]:
    """min/max, from/to, minValue/maxValue, minimum/maximum, low/high, minSalary, or salaryFrom."""
    low = high = None
    for a, b in (
        ("min", "max"),
        ("from", "to"),
        ("minValue", "maxValue"),
        ("min_value", "max_value"),
        ("minimum", "maximum"),
        ("low", "high"),
        ("minSalary", "maxSalary"),
        ("salaryMin", "salaryMax"),
        ("salaryMinimum", "salaryMaximum"),
        ("minimumSalary", "maximumSalary"),
        ("salaryFrom", "salaryTo"),
        ("minCompensation", "maxCompensation"),
        ("salaryRangeMin", "salaryRangeMax"),
        ("min_salary", "max_salary"),
        ("minPay", "maxPay"),
        ("payMin", "payMax"),
        ("min_pay", "max_pay"),
        ("pay_min", "pay_max"),
        ("salary_from", "salary_to"),
        ("compensationMin", "compensationMax"),
        ("salary_min", "salary_max"),
        ("compensation_min", "compensation_max"),
        ("min_compensation", "max_compensation"),
        ("minAmount", "maxAmount"),
        ("min_amount", "max_amount"),
        ("rangeStart", "rangeEnd"),
        ("range_start", "range_end"),
        ("lower_bound", "upper_bound"),
        ("lowerBound", "upperBound"),
    ):
        a_nums = _nums(raw.get(a))
        b_nums = _nums(raw.get(b))
        if low is None and a_nums:
            low = min(a_nums)
            if high is None and len(a_nums) >= 2:
                high = max(a_nums)
        if high is None and b_nums:
            high = max(b_nums)
            if low is None and len(b_nums) >= 2:
                low = min(b_nums)
    if low is None and high is None:
        for key in (
            "compensation",
            "salary",
            "salaryRange",
            "payRange",
            "estimatedSalary",
            "estimated_salary",
            "baseCompensation",
            "base_compensation",
            "jobCompensation",
            "job_compensation",
            "offeredSalary",
            "offered_salary",
            "salaryOffered",
            "salary_offered",
            "annualSalary",
            "annual_salary",
            "yearlySalary",
            "yearly_salary",
            "annualPay",
            "annual_pay",
            "yearlyPay",
            "yearly_pay",
            "jobSalary",
            "job_salary",
            "basePay",
            "base_pay",
            "salary_range",
            "pay_range",
        ):
            nested = raw.get(key)
            if isinstance(nested, dict):
                return _bound_nums(nested)
        nums = _nums(raw.get("amount"))
        if nums:
            return min(nums), max(nums)
    return low, high


def _pay_unit(raw) -> Optional[str]:
    token = str(raw or "").rsplit("/", 1)[-1].upper().replace("-", "_").strip()
    token = re.sub(r"^(?:USD|US\$|US|\$)\s*", "", token).strip()
    return (
        _PAY_UNITS.get(token)
        or _PAY_UNITS.get(token.replace("_", " "))
        or _PAY_UNITS.get(_period_unit(raw) or "")
    )


def _usd(currency) -> bool:
    if not currency:
        return True
    token = str(currency).upper().replace("$", "").strip()
    token = token.rsplit("/", 1)[-1].rsplit("#", 1)[-1].strip()
    if token in {"USD", "US", "USA"}:
        return True
    compact = re.sub(r"[^A-Z]", "", token)
    return compact in {
        "USDOLLAR",
        "USDOLLARS",
        "USDDOLLAR",
        "USDDOLLARS",
        "UNITEDSTATESDOLLAR",
        "UNITEDSTATESDOLLARS",
    }


_FOREIGN_PAY_RE = re.compile(
    r"(?:€|£)\s*\d{1,3}(?:,\d{3}){1,2}"
    r"|(?:€|£)\s*\d{5,7}\b"
    r"|(?:€|£)\s*\d{2,3}(?:\.\d+)?\s*k\b"
    r"|\b(?:EUR|GBP|euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?)\s*\d{1,3}(?:[,'’.\s]\d{3}){1,2}"
    r"|\b(?:EUR|GBP|euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?)\s*\d{5,7}\b"
    r"|\b(?:EUR|GBP|euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?)\s*\d{2,3}(?:\.\d+)?\s*k\b"
    r"|(?:₹|¥|₩|₪|฿|₫|₦|₴|₸|₺|₽|₱|৳|₾|₼|₡|₭|៛)\s*\d"
    r"|\b(?:CHF|INR|JPY|AED|CNY|KRW|HUF|SEK|NOK|DKK|PLN|BRL|ZAR|ILS|CZK|TRY|RON|BGN|THB|TWD|MYR|IDR|RUB|UAH|ISK|HRK|VND|NGN|PKR|BDT|LKR|SAR|QAR|EGP|KES|GHS|KWD|BHD|OMR|JOD|RSD|GEL|KZT|TND|DZD|BAM|AZN|MDL|MKD|BYN|NPR|UZS|KGS|MMK|KHR|LAK|MNT|UYU|PYG|BOB|GTQ|IQD|LBP|TZS|UGX|ETB|RWF|BND|MOP|MUR|XOF|XAF|NAD|BWP|MWK|ZMW|AOA|GMD|SLL|LRD|SLE|CVE|SZL|LSL|GNF|STN|STD|GIP|FJD|PGK|WST|SBD|VUV|XPF|XCD|BBD|JMD|TTD|KYD|BZD|HTG|DOP|CUC|HNL|PAB|AWG|SRD|GYD|VES|VEF|MZN|MGA|DJF|ERN|SSP|LYD|MRU|KMF|BIF|ZWL|SCR|AFN|ANG|BMD|BTN|FKP|SYP|TJS|YER|VED|MVR|KPW)\s*['’]?\d"
    r"|\b(?:PHP|CRC|BSD|SDG|CDF|NIO|MRO|IRR|SVC|TMT|SHP)\s*['’]?(?:\d{1,3}(?:[,'’.\s]\d{3}){1,2}|\d{5,7}|\d{2,3}(?:\.\d+)?\s*k\b)"
    r"|\b(?:RM|Rp)\s*['’]?(?:\d{1,3}(?:[,'’.\s]\d{3}){1,2}|\d{5,7}|\d{2,3}(?:\.\d+)?\s*k\b)"
    r"|\d{1,3}(?:[,'’.\s]\d{3}){1,2}\s*(?:EUR|GBP|euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?)\b"
    r"|\d{5,7}\s*(?:EUR|GBP|euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?)\b"
    r"|\d{2,3}(?:\.\d+)?\s*k\s*(?:EUR|GBP|euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?)\b"
    r"|\d{1,3}(?:[,'’.\s]\d{3}){1,2}\s*(?:CHF|INR|JPY|AED|CNY|KRW|HUF|SEK|NOK|DKK|PLN|BRL|ZAR|ILS|CZK|TRY|RON|BGN|THB|TWD|PHP|MYR|IDR|RUB|UAH|ISK|HRK|VND|NGN|PKR|BDT|LKR|SAR|QAR|EGP|KES|GHS|KWD|BHD|OMR|JOD|RSD|GEL|KZT|TND|DZD|BAM|AZN|MDL|MKD|BYN|NPR|UZS|KGS|MMK|KHR|LAK|MNT|UYU|PYG|BOB|GTQ|IQD|LBP|TZS|UGX|ETB|RWF|BND|MOP|MUR|CRC|XOF|XAF|NAD|BWP|MWK|ZMW|AOA|GMD|SLL|LRD|SLE|CVE|SZL|LSL|GNF|STN|STD|GIP|FJD|PGK|WST|SBD|VUV|XPF|XCD|BBD|JMD|TTD|BSD|KYD|BZD|HTG|DOP|CUC|HNL|PAB|AWG|SRD|GYD|VES|VEF|MZN|MGA|SDG|DJF|ERN|SSP|LYD|MRU|KMF|BIF|CDF|ZWL|SCR|NIO|MRO|AFN|ANG|BMD|BTN|FKP|SYP|TJS|YER|VED|MVR|KPW|IRR|SVC|TMT|SHP)\b"
    r"|\d{5,7}\s*(?:CHF|INR|JPY|AED|CNY|KRW|HUF|SEK|NOK|DKK|PLN|BRL|ZAR|ILS|CZK|TRY|RON|BGN|THB|TWD|PHP|MYR|IDR|RUB|UAH|ISK|HRK|VND|NGN|PKR|BDT|LKR|SAR|QAR|EGP|KES|GHS|KWD|BHD|OMR|JOD|RSD|GEL|KZT|TND|DZD|BAM|AZN|MDL|MKD|BYN|NPR|UZS|KGS|MMK|KHR|LAK|MNT|UYU|PYG|BOB|GTQ|IQD|LBP|TZS|UGX|ETB|RWF|BND|MOP|MUR|CRC|XOF|XAF|NAD|BWP|MWK|ZMW|AOA|GMD|SLL|LRD|SLE|CVE|SZL|LSL|GNF|STN|STD|GIP|FJD|PGK|WST|SBD|VUV|XPF|XCD|BBD|JMD|TTD|BSD|KYD|BZD|HTG|DOP|CUC|HNL|PAB|AWG|SRD|GYD|VES|VEF|MZN|MGA|SDG|DJF|ERN|SSP|LYD|MRU|KMF|BIF|CDF|ZWL|SCR|NIO|MRO|AFN|ANG|BMD|BTN|FKP|SYP|TJS|YER|VED|MVR|KPW|IRR|SVC|TMT|SHP)\b"
    r"|\d{2,3}(?:\.\d+)?\s*k\s*(?:CHF|INR|JPY|AED|CNY|KRW|HUF|SEK|NOK|DKK|PLN|BRL|ZAR|ILS|CZK|TRY|RON|BGN|THB|TWD|PHP|MYR|IDR|RUB|UAH|ISK|HRK|VND|NGN|PKR|BDT|LKR|SAR|QAR|EGP|KES|GHS|KWD|BHD|OMR|JOD|RSD|GEL|KZT|TND|DZD|BAM|AZN|MDL|MKD|BYN|NPR|UZS|KGS|MMK|KHR|LAK|MNT|UYU|PYG|BOB|GTQ|IQD|LBP|TZS|UGX|ETB|RWF|BND|MOP|MUR|CRC|XOF|XAF|NAD|BWP|MWK|ZMW|AOA|GMD|SLL|LRD|SLE|CVE|SZL|LSL|GNF|STN|STD|GIP|FJD|PGK|WST|SBD|VUV|XPF|XCD|BBD|JMD|TTD|BSD|KYD|BZD|HTG|DOP|CUC|HNL|PAB|AWG|SRD|GYD|VES|VEF|MZN|MGA|SDG|DJF|ERN|SSP|LYD|MRU|KMF|BIF|CDF|ZWL|SCR|NIO|MRO|AFN|ANG|BMD|BTN|FKP|SYP|TJS|YER|VED|MVR|KPW|IRR|SVC|TMT|SHP)\b"
    r"|\d{1,3}(?:[,'’.\s]\d{3}){1,2}\s*(?:kr|zł|RM|Rp)\b"
    r"|\d{5,7}\s*(?:kr|zł|RM|Rp)\b"
    r"|\b(?:kr|zł)\s*['’]?\d"
    r"|\d{2,3}(?:\.\d+)?\s*k\s*(?:kr|zł|RM|Rp)\b"
    r"|\d{1,3}(?:[,'’.\s]\d{3}){1,2}\s*:-"
    r"|\bRs\.?\s*['’]?\d"
    r"|\d{1,3}(?:[,'’.\s]\d{3}){1,2}\s*Rs\.?\b"
    r"|\d{5,7}\s*Rs\.?\b"
    r"|\d{2,3}(?:\.\d+)?\s*k\s*Rs\.?\b"
    r"|\d{1,2}(?:\.\d+)?\s*(?:[-–—]\s*\d{1,2}(?:\.\d+)?)?\s*(?:lpa|lacs?|lakhs?)\b",
    re.I,
)
_FOREIGN_DOLLAR_RE = re.compile(
    r"\b(?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|CLP|COP|PEN|TTD|JMD|BBD|BZD|KYD|XCD|FJD|SBD|GYD|SRD|LRD|NAD|ZWL|UYU|BRL|DOP|CUC|AWG|BMD)\s*\$?\s*\d"
    r"|\b(?:CHF|INR|JPY|AED|CNY|KRW|HUF|SEK|NOK|DKK|PLN|BRL|ZAR|ILS|CZK|TRY|RON|BGN|THB|TWD|MYR|IDR|RUB|UAH|ISK|HRK|VND|NGN|PKR|BDT|LKR|SAR|QAR|EGP|KES|GHS|KWD|BHD|OMR|JOD|RSD|GEL|KZT|TND|DZD|BAM|AZN|MDL|MKD|BYN|NPR|UZS|KGS|MMK|KHR|LAK|MNT|UYU|PYG|BOB|GTQ|IQD|LBP|TZS|UGX|ETB|RWF|BND|MOP|MUR|XOF|XAF|NAD|BWP|MWK|ZMW|AOA|GMD|SLL|LRD|SLE|CVE|SZL|LSL|GNF|STN|STD|GIP|FJD|PGK|WST|SBD|VUV|XPF|XCD|BBD|JMD|TTD|KYD|BZD|HTG|DOP|CUC|HNL|PAB|AWG|SRD|GYD|VES|VEF|MZN|MGA|DJF|ERN|SSP|LYD|MRU|KMF|BIF|ZWL|SCR|AFN|ANG|BMD|BTN|FKP|SYP|TJS|YER|VED|MVR|KPW|EUR|GBP|PHP|CRC|BSD|SDG|CDF|NIO|MRO|IRR|SVC|TMT|SHP|RM|Rp|Rs\.?|kr|zł)\s*\$\s*\d"
    r"|\b(?:euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?|lacs?|lakhs?|lpa|crores?)\s*\$\s*\d"
    r"|(?<![A-Za-z])(?:AU|CA|SG|MX|AR|CL|HK|NZ|NT|COL|CO|PE|C|A|R|S)\$\s*\d"
    r"|\b(?:salario|mensual|pesos?)\b.{0,80}\$\s*\d"
    r"|\$\s*\d[\d,]*.{0,80}salary\s+monthly"
    r"|\$\s*\d[\d,]*(?:\.\d+)?\s*(?:k\b)?"
    r"(?:\s*[—–-]\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*(?:k\b)?)?"
    r"\s*(?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|CLP|COP|PEN|TTD|JMD|BBD|BZD|KYD|XCD|FJD|SBD|GYD|SRD|LRD|NAD|ZWL|UYU|BRL|DOP|CUC|AWG|BMD|pesos?)\b"
    r"|\$\s*\d[\d,]*.{0,40}\((?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|CLP|COP|PEN|CHF|INR|JPY|EUR|GBP|AED|CNY|KRW|HUF|SEK|NOK|DKK|PLN|BRL|ZAR|ILS|CZK|TRY|RON|BGN|THB|TWD|MYR|IDR|RUB|UAH|ISK|HRK|VND|NGN|PKR|BDT|LKR|SAR|QAR|EGP|KES|GHS|KWD|BHD|OMR|JOD|RSD|GEL|KZT|TND|DZD|BAM|AZN|MDL|MKD|BYN|NPR|UZS|KGS|MMK|KHR|LAK|MNT|UYU|PYG|BOB|GTQ|IQD|LBP|TZS|UGX|ETB|RWF|BND|MOP|MUR|XOF|XAF|NAD|BWP|MWK|ZMW|AOA|GMD|SLL|LRD|SLE|CVE|SZL|LSL|GNF|STN|STD|GIP|FJD|PGK|WST|SBD|VUV|XPF|XCD|BBD|JMD|TTD|BSD|KYD|BZD|HTG|DOP|CUC|HNL|PAB|AWG|SRD|GYD|VES|VEF|MZN|MGA|SDG|DJF|ERN|SSP|LYD|MRU|KMF|BIF|CDF|ZWL|SCR|NIO|MRO|AFN|ANG|BMD|BTN|FKP|SYP|TJS|YER|VED|MVR|KPW|IRR|SVC|TMT|SHP|PHP|CRC|RM|Rp|Rs|kr|zł|euros?|pounds?|yen|sterling|quid|zloty|kroner|kronor|francs?|dirhams?|rand|baht|shekels?|forint|koruna|rupees?|lacs?|lakhs?|lpa|crores?)\)"
    r"|\d{1,3}(?:[,'’.\s]\d{3}){1,2}\s*(?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|CLP|COP|PEN|TTD|JMD|BBD|BZD|KYD|XCD|FJD|SBD|GYD|SRD|LRD|NAD|ZWL|UYU|BRL|DOP|CUC|AWG|BMD)\b"
    r"|\d{5,7}\s*(?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|CLP|COP|PEN|TTD|JMD|BBD|BZD|KYD|XCD|FJD|SBD|GYD|SRD|LRD|NAD|ZWL|UYU|BRL|DOP|CUC|AWG|BMD)\b"
    r"|\d{2,3}(?:\.\d+)?\s*k\s*(?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|CLP|COP|PEN|TTD|JMD|BBD|BZD|KYD|XCD|FJD|SBD|GYD|SRD|LRD|NAD|ZWL|UYU|BRL|DOP|CUC|AWG|BMD)\b",
    re.I | re.S,
)


def _foreign_pay_text(text: str) -> bool:
    blob = text or ""
    return bool(_FOREIGN_PAY_RE.search(blob) or _FOREIGN_DOLLAR_RE.search(blob))


def _foreign_salary(html: str) -> bool:
    """True when the listing states a non-USD salary. Ranking is USD $/hour."""
    posting = _job_posting(html)
    if _posting_foreign(posting):
        return True
    return _foreign_pay_text(_listing_plain_text(html))


_US_COUNTRY_RE = re.compile(
    r"(?i)^(?:the\s+)?(?:united states(?:\s+of\s+america)?|usa|u\.s\.a?\.?|us)$"
)
_US_PLACE_RE = re.compile(
    r"(?i)(?:^|,\s*)(?:the\s+)?(?:united states(?:\s+of\s+america)?|usa|u\.s\.a?\.?|us)$"
)
_NON_US_PLACE_RE = re.compile(
    r"(?i)(?:^|,\s*)(?:the\s+)?(?:"
    r"canada|mexico|united kingdom|great britain|uk|"
    r"australia|new zealand|germany|france|spain|italy|netherlands|"
    r"sweden|norway|denmark|finland|ireland|switzerland|"
    r"india|japan|china|singapore|brazil|"
    r"emea|europe|apac|asia(?:[-\s]pacific)?|"
    r"european union|eu|south america"
    r")$"
)


def _country_label(value) -> str:
    """Scalar or {name,@value,@id,alternateName} country. @id keeps the last path token."""
    if isinstance(value, dict):
        text = (
            _ld_text(value.get("name"))
            or _ld_text(value.get("alternateName"))
            or _ld_text(value)
            or ""
        )
    else:
        text = _ld_text(value) or ""
    text = text.strip()
    if not text:
        return ""
    token = text.rsplit("/", 1)[-1].rsplit("#", 1)[-1].strip().replace("_", " ")
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token)


def _country_from_label(label: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", (label or "").strip())
    if not s:
        return None
    if _US_COUNTRY_RE.fullmatch(s) or _US_PLACE_RE.search(s):
        return "US"
    m = _NON_US_PLACE_RE.search(s)
    if not m:
        return None
    return re.sub(r"^,\s*", "", m.group(0)).strip()


def _posting_countries(posting: dict) -> list[str]:
    rows: list = []
    for key in (
        "jobLocation",
        "job_location",
        "workLocation",
        "work_location",
        "applicantLocationRequirements",
        "applicant_location_requirements",
    ):
        raw = posting.get(key)
        if raw is None:
            continue
        rows.extend(raw if isinstance(raw, list) else [raw])
    countries: list[str] = []

    def push(token: str) -> None:
        t = token.strip()
        if t and t not in countries:
            countries.append(t)

    def add_label(raw: str) -> None:
        inferred = _country_from_label(raw)
        if inferred:
            push(inferred)

    for row in rows:
        if isinstance(row, str):
            add_label(row)
            continue
        if not isinstance(row, dict):
            continue
        add_label(_ld_text(row.get("name")) or "")
        loc_country = _country_label(
            row.get("addressCountry")
            or row.get("address_country")
            or row.get("country")
        )
        if loc_country:
            push(loc_country)
            add_label(loc_country)
        addr = row.get("address")
        addrs = addr if isinstance(addr, list) else [addr]
        for item in addrs:
            if isinstance(item, str):
                add_label(item)
                continue
            if not isinstance(item, dict):
                continue
            raw = (
                item.get("addressCountry")
                or item.get("address_country")
                or item.get("country")
            )
            for val in raw if isinstance(raw, list) else [raw]:
                name = _country_label(val)
                if name:
                    push(name)
                    add_label(name)
            city = (
                _ld_text(item.get("addressLocality") or item.get("address_locality"))
                or ""
            ).strip()
            region = (
                _ld_text(item.get("addressRegion") or item.get("address_region"))
                or ""
            ).strip()
            country = _country_label(
                item.get("addressCountry")
                or item.get("address_country")
                or item.get("country")
            )
            add_label(", ".join(p for p in (city, region, country) if p))
    return countries


_SALARY_TOKEN_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s*k)?\b"
)


_BLOB_UNIT_RE = re.compile(
    r"(?i)(?:/|\bper\s+|\ban?\s+)?\b("
    r"biweekly|bi-weekly|fortnightly|fortnight|"
    r"semi-monthly|semimonthly|"
    r"hourly|hours|hour|hrs|hr|"
    r"yearly|year|yr|"
    r"annually|annual|annum|"
    r"monthly|months|month|"
    r"weekly|weeks|week|"
    r"daily|days|day|diem"
    r")\b"
)


def _unit_from_blob(text: str) -> Optional[str]:
    blob = (text or "").lower().replace("_", " ").replace("-", " ")
    if re.search(r"(?i)every\s+(?:two|2|other)\s+weeks?|bi[-\s]?weekly|fortnight", blob):
        return "biweek"
    if re.search(
        r"(?i)semi[-\s]?monthly|twice\s+(?:a|per)\s+month|twice\s+monthly", blob
    ):
        return "semimonth"
    m = _BLOB_UNIT_RE.search(text or "")
    return _pay_unit(m.group(1) if m else None)


def _span_nums(text: str) -> list[float]:
    """Stated range amounts from a salary string. Ignore years-of-experience."""
    out: list[float] = []
    for m in _SALARY_TOKEN_RE.finditer(text or ""):
        n = float(m.group(1).replace(",", ""))
        if m.group(2):
            n *= 1000
        out.append(n)
    yearly = [n for n in out if n >= 10_000]
    rate = [n for n in out if 10 <= n <= 1000]
    if yearly:
        return yearly
    if re.search(r"(?i)\byears?\b", text or ""):
        return []
    unit = _unit_from_blob(text)
    if unit in {"hour", "day", "week", "biweek", "semimonth", "month"}:
        period = [n for n in out if 10 <= n < 10_000]
        if period:
            return period
    if len(rate) >= 2:
        return rate
    if len(rate) == 1 and unit in {
        "hour",
        "day",
        "week",
        "biweek",
        "semimonth",
        "month",
    }:
        return rate
    return []


_UNIT_TEXT = {
    "hour": "HOUR",
    "day": "DAY",
    "week": "WEEK",
    "month": "MONTH",
    "year": "YEAR",
    "biweek": "BIWEEKLY",
    "semimonth": "SEMIMONTHLY",
}


def _span_pay_ld(blob: str) -> Optional[dict]:
    """JSON-LD baseSalary from a stated range string. Skip bare rates with no period."""
    if not isinstance(blob, str) or not blob.strip():
        return None
    nums = _span_nums(blob)
    if not nums:
        return None
    period = _unit_from_blob(blob)
    if period is None and not any(n >= 10_000 for n in nums):
        return None
    value: dict = {"unitText": _UNIT_TEXT.get(period or "year", "YEAR")}
    if len(nums) >= 2:
        value["minValue"] = int(min(nums))
        value["maxValue"] = int(max(nums))
    else:
        value["value"] = int(nums[0])
    return {
        "currency": "EUR" if _foreign_pay_text(blob) else "USD",
        "value": value,
    }


def _period_annualizes(pay: dict) -> bool:
    """True when JSON-LD unitText annualizes every amount into the yearly band."""
    node = pay.get("value") if isinstance(pay, dict) else None
    if not isinstance(node, dict):
        return False
    unit = _pay_unit(node.get("unitText"))
    if unit in (None, "year"):
        return False
    nums = [
        node[k]
        for k in ("minValue", "maxValue", "value")
        if isinstance(node.get(k), (int, float))
    ]
    return bool(nums) and all(_annualize(float(n), unit, 40) is not None for n in nums)


def _named_pay_ld(name: str, value: str) -> Optional[dict]:
    """Range from a pay-named field. Prefer the name's period when it annualizes."""
    plain = _span_pay_ld(value)
    blob = f"{name} {value}".strip()
    named = (
        _span_pay_ld(blob)
        if name.strip() and blob != (value or "").strip()
        else None
    )
    if named and _period_annualizes(named):
        return named
    return plain or named


def _salary_blob(salary) -> str:
    if isinstance(salary, str):
        return salary
    if isinstance(salary, list):
        return " ".join(_salary_blob(item) for item in salary)
    if isinstance(salary, dict):
        blob = " ".join(
            _salary_blob(salary.get(key))
            for key in (
                "minValue",
                "maxValue",
                "value",
                "min",
                "max",
                "from",
                "to",
                "minimum",
                "maximum",
                "low",
                "high",
                "minSalary",
                "maxSalary",
                "salaryMin",
                "salaryMax",
                "salaryMinimum",
                "salaryMaximum",
                "minimumSalary",
                "maximumSalary",
                "salaryFrom",
                "salaryTo",
                "minCompensation",
                "maxCompensation",
                "salaryRangeMin",
                "salaryRangeMax",
                "min_salary",
                "max_salary",
                "minPay",
                "maxPay",
                "payMin",
                "payMax",
                "min_pay",
                "max_pay",
                "pay_min",
                "pay_max",
                "salary_from",
                "salary_to",
                "compensationMin",
                "compensationMax",
                "salary_min",
                "salary_max",
                "compensation_min",
                "compensation_max",
                "min_compensation",
                "max_compensation",
                "compensation",
                "salary",
                "salaryRange",
                "salary_range",
                "payRange",
                "pay_range",
                "estimatedSalary",
                "estimated_salary",
                "baseCompensation",
                "base_compensation",
                "jobCompensation",
                "job_compensation",
                "offeredSalary",
                "offered_salary",
                "salaryOffered",
                "salary_offered",
                "yearlySalary",
                "yearly_salary",
                "annualSalary",
                "annual_salary",
                "annualPay",
                "annual_pay",
                "yearlyPay",
                "yearly_pay",
                "jobSalary",
                "job_salary",
                "basePay",
                "base_pay",
                "amount",
                "minAmount",
                "maxAmount",
                "min_amount",
                "max_amount",
                "min_value",
                "max_value",
                "rangeStart",
                "rangeEnd",
                "range_start",
                "range_end",
                "lower_bound",
                "upper_bound",
                "lowerBound",
                "upperBound",
            )
            if key in salary
        )
        text = _ld_text(salary)
        if text:
            return f"{blob} {text}".strip()
        return blob
    return ""


_MONEY_NEST_KEYS = (
    "minValue",
    "maxValue",
    "value",
    "min",
    "max",
    "from",
    "to",
    "minimum",
    "maximum",
    "low",
    "high",
    "minSalary",
    "maxSalary",
    "salaryMin",
    "salaryMax",
    "salaryMinimum",
    "salaryMaximum",
    "minimumSalary",
    "maximumSalary",
    "salaryFrom",
    "salaryTo",
    "minCompensation",
    "maxCompensation",
    "salaryRangeMin",
    "salaryRangeMax",
    "min_salary",
    "max_salary",
    "minPay",
    "maxPay",
    "payMin",
    "payMax",
    "min_pay",
    "max_pay",
    "pay_min",
    "pay_max",
    "salary_from",
    "salary_to",
    "compensationMin",
    "compensationMax",
    "salary_min",
    "salary_max",
    "compensation_min",
    "compensation_max",
    "min_compensation",
    "max_compensation",
    "compensation",
    "salary",
    "salaryRange",
    "salary_range",
    "payRange",
    "pay_range",
    "estimatedSalary",
    "estimated_salary",
    "baseCompensation",
    "base_compensation",
    "jobCompensation",
    "job_compensation",
    "offeredSalary",
    "offered_salary",
    "salaryOffered",
    "salary_offered",
    "annualSalary",
    "annual_salary",
    "yearlySalary",
    "yearly_salary",
    "annualPay",
    "annual_pay",
    "yearlyPay",
    "yearly_pay",
    "jobSalary",
    "job_salary",
    "basePay",
    "base_pay",
    "amount",
    "minAmount",
    "maxAmount",
    "min_amount",
    "max_amount",
    "min_value",
    "max_value",
    "rangeStart",
    "rangeEnd",
    "range_start",
    "range_end",
    "lower_bound",
    "upper_bound",
    "lowerBound",
    "upperBound",
)


def _nums(value) -> list[float]:
    """Numbers from a salary node. QuantitativeValue.value may be [min, max]."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        n = _num(value)
        return [n] if n is not None else []
    if isinstance(value, str):
        n = _num(value)
        if n is not None:
            return [n]
        return _span_nums(value)
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_nums(item))
        return out
    if isinstance(value, dict):
        out: list[float] = []
        for key in _MONEY_NEST_KEYS:
            if key in value:
                out.extend(_nums(value.get(key)))
        if out:
            return out
        amount = _num(value)
        if amount is not None:
            out.append(amount)
        out.extend(_span_nums(_ld_text(value) or ""))
        return out
    return []


def _salary_has_amount(salary) -> bool:
    return any(n > 0 for n in _nums(salary))


_DURATION_UNITS = {
    "PT1H": "HOUR",
    "PT60M": "HOUR",
    "P1D": "DAY",
    "P1W": "WEEK",
    "P7D": "WEEK",
    "P2W": "BIWEEKLY",
    "P14D": "BIWEEKLY",
    "P1M": "MONTH",
    "P1Y": "YEAR",
}


def _ld_text(value) -> Optional[str]:
    """String from a JSON-LD scalar, [scalar], or {@value,name,value,@id,text} node."""
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list):
        for item in value:
            text = _ld_text(item)
            if text:
                return text
        return None
    if isinstance(value, dict):
        for key in ("@value", "name", "value", "@id", "text"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw
    return None


def _duration_unit(raw: str) -> Optional[str]:
    token = raw.strip().upper().rsplit("/", 1)[-1]
    token = re.sub(r"(?<!\d)0+[YMDHS]", "", token)
    if token.endswith("T"):
        token = token[:-1]
    return _DURATION_UNITS.get(token)


def _unit_raw(node) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    raw = (
        _ld_text(node.get("unitText"))
        or _ld_text(node.get("unit_text"))
        or _ld_text(node.get("unitCode"))
        or _ld_text(node.get("unit_code"))
        or _ld_text(node.get("unit"))
        or _ld_text(node.get("salaryUnit"))
        or _ld_text(node.get("salary_unit"))
        or _ld_text(node.get("period"))
        or _ld_text(node.get("interval"))
        or _ld_text(node.get("frequency"))
    )
    if raw:
        return raw
    duration = _ld_text(node.get("duration"))
    if duration:
        return _duration_unit(duration)
    return None


def _unit_text(salary) -> Optional[str]:
    if isinstance(salary, list):
        for item in salary:
            raw = _unit_text(item)
            if raw:
                return raw
        return None
    if not isinstance(salary, dict):
        return None
    for key in _MONEY_NEST_KEYS:
        nest = salary.get(key)
        if nest is not salary:
            raw = _unit_text(nest)
            if raw:
                return raw
    return _unit_raw(salary)


def _salary_span(items: list) -> dict:
    nums: list[float] = []
    currency = None
    unit = None
    for item in items:
        nums.extend(_nums(item))
        if isinstance(item, dict):
            currency = currency or _currency_of(item)
            unit = unit or _unit_text(item)
    qv: dict = {}
    if unit:
        qv["unitText"] = unit
    if len(nums) == 1:
        qv["value"] = nums[0]
    else:
        qv["minValue"] = min(nums)
        qv["maxValue"] = max(nums)
    blob: dict = {"value": qv}
    if currency:
        blob["currency"] = currency
    return blob


def _posting_salary(posting: Optional[dict]):
    """baseSalary, then salary aliases, then salaryMin/minValue/amount. Skip empty objects."""
    if not isinstance(posting, dict):
        return None
    for key in (
        "baseSalary",
        "base_salary",
        "salary",
        "estimatedSalary",
        "estimated_salary",
        "baseCompensation",
        "base_compensation",
        "compensation",
        "salaryRange",
        "salary_range",
        "payRange",
        "pay_range",
        "jobCompensation",
        "job_compensation",
        "offeredSalary",
        "offered_salary",
        "salaryOffered",
        "salary_offered",
        "annualSalary",
        "annual_salary",
        "yearlySalary",
        "yearly_salary",
        "annualPay",
        "annual_pay",
        "yearlyPay",
        "yearly_pay",
        "jobSalary",
        "job_salary",
        "basePay",
        "base_pay",
    ):
        raw = posting.get(key)
        items = raw if isinstance(raw, list) else [raw]
        found = [item for item in items if _salary_has_amount(item)]
        if not found:
            continue
        if len(found) == 1 or not all(len(_nums(item)) == 1 for item in found):
            return found[0]
        return _salary_span(found)
    low = high = None
    bound_unit = None
    for a, b in (
        ("salaryMin", "salaryMax"),
        ("salaryMinimum", "salaryMaximum"),
        ("minSalary", "maxSalary"),
        ("salaryRangeMin", "salaryRangeMax"),
        ("min_salary", "max_salary"),
        ("salaryFrom", "salaryTo"),
        ("minCompensation", "maxCompensation"),
        ("minPay", "maxPay"),
        ("payMin", "payMax"),
        ("minimumSalary", "maximumSalary"),
        ("min_pay", "max_pay"),
        ("pay_min", "pay_max"),
        ("salary_from", "salary_to"),
        ("compensationMin", "compensationMax"),
        ("salary_min", "salary_max"),
        ("compensation_min", "compensation_max"),
        ("min_compensation", "max_compensation"),
        ("minAmount", "maxAmount"),
        ("min_amount", "max_amount"),
        ("rangeStart", "rangeEnd"),
        ("range_start", "range_end"),
        ("lower_bound", "upper_bound"),
        ("lowerBound", "upperBound"),
        ("min", "max"),
        ("from", "to"),
        ("minValue", "maxValue"),
        ("min_value", "max_value"),
        ("minimum", "maximum"),
        ("low", "high"),
    ):
        left, right = posting.get(a), posting.get(b)
        nums = _nums(left) + _nums(right)
        if not nums:
            continue
        low, high = min(nums), max(nums)
        bound_unit = _unit_text(left) or _unit_text(right)
        break
    if low is None and high is None:
        amount = posting.get("amount")
        nums = _nums(amount)
        if not nums:
            return None
        low, high = min(nums), max(nums)
        bound_unit = _unit_text(amount)
    unit = (
        _ld_text(posting.get("salaryUnit"))
        or _ld_text(posting.get("salary_unit"))
        or _ld_text(posting.get("unitText"))
        or _ld_text(posting.get("unit_text"))
        or _ld_text(posting.get("unit"))
        or _ld_text(posting.get("unitCode"))
        or _ld_text(posting.get("unit_code"))
        or _ld_text(posting.get("period"))
        or _ld_text(posting.get("interval"))
        or _ld_text(posting.get("frequency"))
    )
    if not unit:
        duration = _ld_text(posting.get("duration"))
        if duration:
            unit = _duration_unit(duration)
    if not unit:
        unit = bound_unit
    value: dict = {}
    if unit:
        value["unitText"] = unit
    if low is not None and high is not None:
        value["minValue"] = low
        value["maxValue"] = high
    else:
        value["value"] = high or low
    return value


def _currency_of(value) -> Optional[str]:
    if isinstance(value, list):
        for item in value:
            cur = _currency_of(item)
            if cur:
                return cur
        return None
    if not isinstance(value, dict):
        return None
    cur = (
        _ld_text(value.get("currency"))
        or _ld_text(value.get("currencyCode"))
        or _ld_text(value.get("currency_code"))
        or _ld_text(value.get("salaryCurrency"))
        or _ld_text(value.get("salary_currency"))
        or _ld_text(value.get("currencyType"))
        or _ld_text(value.get("currency_type"))
    )
    if cur:
        return cur
    for key in _MONEY_NEST_KEYS:
        nested = value.get(key)
        if nested is not value:
            cur = _currency_of(nested)
            if cur:
                return cur
    return None


def _posting_currency(posting: Optional[dict], salary=None) -> Optional[str]:
    """MonetaryAmount.currency, else JobPosting.salaryCurrency, else any stated currency."""
    cur = _currency_of(salary)
    if cur:
        return cur
    if not isinstance(posting, dict):
        return None
    stated = (
        _ld_text(posting.get("salaryCurrency"))
        or _ld_text(posting.get("salary_currency"))
        or _ld_text(posting.get("currency"))
        or _ld_text(posting.get("currencyCode"))
        or _ld_text(posting.get("currency_code"))
        or _ld_text(posting.get("currencyType"))
        or _ld_text(posting.get("currency_type"))
    )
    if stated:
        return stated
    if _salary_has_amount(salary):
        return None
    for key in (
        "baseSalary",
        "base_salary",
        "salary",
        "estimatedSalary",
        "estimated_salary",
        "baseCompensation",
        "base_compensation",
        "compensation",
        "salaryRange",
        "salary_range",
        "payRange",
        "pay_range",
        "jobCompensation",
        "job_compensation",
        "offeredSalary",
        "offered_salary",
        "salaryOffered",
        "salary_offered",
        "annualSalary",
        "annual_salary",
        "yearlySalary",
        "yearly_salary",
        "annualPay",
        "annual_pay",
        "yearlyPay",
        "yearly_pay",
        "jobSalary",
        "job_salary",
        "basePay",
        "base_pay",
    ):
        raw = posting.get(key)
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            cur = _currency_of(item)
            if cur:
                return cur
    return None


def _posting_foreign(posting: Optional[dict]) -> bool:
    if not isinstance(posting, dict):
        return False
    salary = _posting_salary(posting)
    currency = _posting_currency(posting, salary)
    if currency:
        return not _usd(currency)
    if _foreign_pay_text(_salary_blob(salary)):
        return True
    if not _salary_has_amount(salary):
        return False
    countries = _posting_countries(posting)
    return bool(countries) and all(not _US_COUNTRY_RE.fullmatch(c) for c in countries)


def _posting_company(posting: dict) -> Optional[str]:
    org = posting.get("hiringOrganization") or posting.get("hiring_organization")
    if isinstance(org, list) and org:
        org = org[0]
    if isinstance(org, str):
        name = org.strip()
    elif isinstance(org, dict):
        name = (
            _ld_text(org.get("name"))
            or _ld_text(org.get("legalName"))
            or _ld_text(org.get("legal_name"))
            or _ld_text(org.get("alternateName"))
            or _ld_text(org.get("alternate_name"))
            or ""
        ).strip()
    else:
        return None
    if not name or _PLACE_RE.search(name):
        return None
    return name


_HOUR_KEYS = (
    "workHours",
    "work_hours",
    "hoursPerWeek",
    "weeklyHours",
    "hours_per_week",
    "weekly_hours",
    "standardWeeklyHours",
    "scheduledWeeklyHours",
    "standardHoursPerWeek",
    "scheduledHoursPerWeek",
    "standard_weekly_hours",
    "scheduled_weekly_hours",
    "standard_hours_per_week",
    "scheduled_hours_per_week",
    "weeklyHourCount",
    "weekly_hour_count",
    "hoursWeek",
    "hours_week",
    "weekHours",
    "week_hours",
    "hoursAWeek",
    "hours_a_week",
    "fteHoursPerWeek",
    "fte_hours_per_week",
    "minHoursPerWeek",
    "min_hours_per_week",
    "maxHoursPerWeek",
    "max_hours_per_week",
)


def _hours_from_node(work) -> Optional[int]:
    if isinstance(work, list):
        for item in work:
            n = _hours_from_node(item)
            if n is not None:
                return n
        return None
    n = _num(work)
    if n is None and isinstance(work, dict):
        for key in (
            "value",
            "minValue",
            "min_value",
            "min",
            "minimum",
            "low",
            "lowerBound",
            "lower_bound",
            "rangeStart",
            "range_start",
            "amount",
            "maxValue",
            "max_value",
            "max",
            "maximum",
            "high",
            "upperBound",
            "upper_bound",
            "rangeEnd",
            "range_end",
        ):
            nested = work.get(key)
            if nested is work:
                continue
            n = _hours_from_node(nested)
            if n is not None:
                break
        if n is None:
            for key in _HOUR_KEYS:
                if key in ("workHours", "work_hours"):
                    continue
                nested = work.get(key)
                if nested is work:
                    continue
                n = _hours_from_node(nested)
                if n is not None:
                    break
        if n is None:
            work = _ld_text(work)
    if n is None and isinstance(work, str):
        stated = _stated_hours("", work)
        if stated:
            return stated
        m = re.search(r"(?<![\d.])(\d{1,2}(?:\.\d+)?)", work)
        n = float(m.group(1)) if m else None
    if n is not None and 1 <= n <= 80:
        return int(round(n))
    return None


def _copy_hours(posting: dict, raw: dict) -> None:
    """Copy occupied hoursPerWeek aliases onto JobPosting.workHours."""
    if posting.get("workHours") is not None or not isinstance(raw, dict):
        return
    for key in _HOUR_KEYS:
        n = _hours_from_node(raw.get(key))
        if n:
            posting["workHours"] = str(n)
            return


def _posting_hours(posting: dict) -> Optional[int]:
    for key in _HOUR_KEYS:
        n = _hours_from_node(posting.get(key))
        if n:
            return n
    blob = " ".join(
        t.upper().replace("-", " ").replace("_", " ")
        for t in _ld_types(posting.get("employmentType") or posting.get("employment_type"))
    )
    compact = re.sub(r"\s+", "", blob)
    if "PARTTIME" in compact:
        return 20
    if "FULLTIME" in compact:
        return 40
    return None


def _annualize(amount: float, unit: Optional[str], hours: Optional[int]) -> Optional[int]:
    if unit == "hour":
        if not 10 <= amount <= 1000:
            return None
        return int(amount * (hours or 40) * 50)
    if unit == "day":
        return int(amount * 5 * 50)
    if unit == "week":
        if amount < 100:
            amount *= 1000
        annual = int(amount * 50)
        if 10_000 <= annual <= 2_000_000:
            return annual
        return None
    if unit == "biweek":
        if amount < 100:
            amount *= 1000
        annual = int(amount * 25)
        if 10_000 <= annual <= 2_000_000:
            return annual
        return None
    if unit == "semimonth":
        if amount < 100:
            amount *= 1000
        annual = int(amount * 24)
        if 10_000 <= annual <= 2_000_000:
            return annual
        return None
    if unit == "month":
        if amount < 1000:
            amount *= 1000
        annual = int(amount * 12)
        if 10_000 <= annual <= 2_000_000:
            return annual
        return None
    if unit == "year":
        if amount < 1000:
            amount *= 1000
        if 10_000 <= amount <= 2_000_000:
            return int(amount)
        return None
    if unit is None and 10_000 <= amount <= 2_000_000:
        return int(amount)
    return None


def _salary_unit(salary) -> Optional[str]:
    return _pay_unit(_unit_text(salary)) or _unit_from_blob(_salary_blob(salary))


def _posting_pay(
    posting: dict, hours: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    if _posting_foreign(posting):
        return None, None
    salary = _posting_salary(posting)
    if salary is None or not _usd(_posting_currency(posting, salary)):
        return None, None
    nums = _nums(salary)
    if not nums:
        return None, None
    unit = _salary_unit(salary)
    low, high = min(nums), max(nums)
    if low == high:
        low = None
    annual_low = _annualize(low, unit, hours) if low is not None else None
    annual_high = _annualize(high, unit, hours) if high is not None else None
    if annual_low and annual_high and annual_low > annual_high:
        annual_low, annual_high = annual_high, annual_low
    if annual_high is None:
        annual_high, annual_low = annual_low, None
    if not annual_high or not (10_000 <= annual_high <= 2_000_000):
        return None, None
    if annual_low and not (10_000 <= annual_low <= annual_high):
        annual_low = None
    return annual_low, annual_high


def _apply_listing(opp: Opportunity, html: str) -> bool:
    """Fill fields from JobPosting JSON-LD, then visible listing text.

    Visible yearly USD wins over JSON-LD hourly/daily/weekly/monthly rates.
    Returns True when this HTML stated USD pay.
    """
    posting = _job_posting(html, opp.title)
    listed_pay = False
    if posting:
        pt = (
            _ld_text(posting.get("title"))
            or _ld_text(posting.get("jobTitle"))
            or _ld_text(posting.get("job_title"))
            or _ld_text(posting.get("headline"))
            or _ld_text(posting.get("roleName"))
            or _ld_text(posting.get("role_name"))
            or _ld_text(posting.get("positionTitle"))
            or _ld_text(posting.get("position_title"))
            or ""
        ).strip()
        if pt:
            opp.title = _role_title(pt)
        name = _posting_company(posting)
        if name:
            opp.company = name
        hours = _posting_hours(posting)
        if hours:
            opp.hours_per_week = hours
    if not opp.company:
        opp.company = _guess_company(_html_title(html), opp.url)
    visible = _listing_plain_text(html)
    structured_remote = _remote_from_posting(posting) if posting else None
    if structured_remote is not None:
        opp.remote = structured_remote
    else:
        opp.remote = _guess_remote(opp.title, visible)
    stated = _stated_hours(opp.title, visible)
    if stated:
        opp.hours_per_week = stated
    elif opp.hours_per_week is None:
        hours = _guess_hours(opp.title, visible)
        if hours:
            opp.hours_per_week = hours
    if posting:
        low, high = _posting_pay(posting, opp.hours_per_week)
        if high or low:
            opp.pay_low = low
            opp.pay_high = high
            listed_pay = True
    if not listed_pay and not _posting_foreign(posting):
        low, high = _parse_pay(visible, opp.hours_per_week, remote=opp.remote)
        if high or low:
            opp.pay_low = low
            opp.pay_high = high
            listed_pay = True
    elif (
        listed_pay
        and posting
        and not _posting_foreign(posting)
        and _salary_unit(_posting_salary(posting))
        in {"hour", "day", "week", "biweek", "semimonth", "month"}
    ):
        text = _NON_SALARY_MONEY_RE.sub(" ", visible or "")
        if not (_FOREIGN_DOLLAR_RE.search(text) or _FOREIGN_PAY_RE.search(text)):
            yearly = _annual_pay(text)
            if yearly[0] or yearly[1]:
                opp.pay_low, opp.pay_high = yearly
    if opp.remote and not _posting_foreign(posting):
        geo = _remote_geo_pay(visible)
        if geo:
            opp.pay_low, opp.pay_high = geo
            listed_pay = True
    opp.title = _role_title(opp.title)
    opp.efficiency = opp.refined_rate
    return listed_pay


def _html_title(html: str) -> str:
    m = re.search(r"(?is)<title>([^<]+)</title>", html or "")
    return unescape(m.group(1)).strip() if m else ""


_GONE_LISTING_RE = re.compile(
    r"(?i)(?<!once )(?<!after )(?<!when )(?:sorry,\s+)?this\s+job\s+was\s+removed"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"(?:\s+page)?\s+is\s+no\s+longer\s+(?:available|active|posted|listed|live|published|advertised|vacant)"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"(?:\s+page)?\s+is\s+no\s+longer\s+(?:available|active|posted|listed|live|published|advertised|vacant)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"(?:\s+page)?\s+no\s+longer\s+(?:available|active|posted|listed|live|published|advertised|vacant)\b"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"(?:\s+page)?\s+no\s+longer\s+(?:available|active|posted|listed|live|published|advertised|vacant)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+is\s+no\s+longer\s+needed\b(?!\s+for)"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+is\s+no\s+longer\s+needed\b(?!\s+for)"
    r"|(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+does(?:n['’]t|\s+not)\s+exist"
    r"|the\s+job\s+listing\s+no\s+longer\s+exists"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+no\s+longer\s+exists"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+no\s+longer\s+exists"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+has\s+(?:already\s+)?(?:been\s+)?(?:filled|expired|lapsed|closed|withdrawn|cancelled|canceled|taken\s+down|taken\s+offline|ended(?!\s+up)|removed|discontinued|archived|unpublished|deactivated|deleted|unposted|pulled|hired\s+internally|rescinded|abandoned|called\s+off|shelved|tabled|sunset|retired|dropped|released|frozen|killed|scrapped)"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+has\s+(?:already\s+)?(?:been\s+)?(?:filled|expired|lapsed|closed|withdrawn|cancelled|canceled|taken\s+down|taken\s+offline|ended(?!\s+up)|removed|discontinued|archived|unpublished|deactivated|deleted|unposted|pulled|hired\s+internally|rescinded|abandoned|called\s+off|shelved|tabled|sunset|retired|dropped|released|frozen|killed|scrapped)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+was\s+(?:already\s+)?(?:filled|expired|lapsed|closed|withdrawn|cancelled|canceled|taken\s+down|taken\s+offline|ended(?!\s+up)|removed|discontinued|archived|unpublished|deactivated|deleted|unposted|pulled|hired\s+internally|rescinded|abandoned|called\s+off|shelved|tabled|sunset|retired|dropped|released|frozen|killed|scrapped)"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+was\s+(?:already\s+)?(?:filled|expired|lapsed|closed|withdrawn|cancelled|canceled|taken\s+down|taken\s+offline|ended(?!\s+up)|removed|discontinued|archived|unpublished|deactivated|deleted|unposted|pulled|hired\s+internally|rescinded|abandoned|called\s+off|shelved|tabled|sunset|retired|dropped|released|frozen|killed|scrapped)"
    r"|(?<!once )(?<!after )(?<!when )we(?:'ve|\s+have)?\s+(?:decided|chosen|chose|opted|elected)\s+(?:not\s+to\s+(?:fill|(?:(?:proceed|move\s+forward)\s+with|continue(?:\s+with)?)|hire\s+for|pursue)|against(?:\s+(?:filling|(?:(?:proceeding|moving\s+forward)\s+with|continuing(?:\s+with)?)|hiring\s+for|pursuing))?)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )we(?:'ve|\s+have)?\s+declined\s+to\s+(?:fill|(?:(?:proceed|move\s+forward)\s+with|continue(?:\s+with)?)|hire\s+for|pursue)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+(?:moved\s+on|stepped\s+away|walked\s+away|withdrawn|withdrew)\s+from\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+passed\s+on\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+(?:is|has\s+been|was|has)\s+(?:closed|filled|expired|lapsed|withdrawn|removed|taken\s+down|taken\s+offline|unpublished|archived|deactivated|deleted|unposted|pulled|discontinued)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+(?:has\s+)?concluded\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+is\s+(?:now\s+)?over\b(?!\s+(?:budget|the\b))"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+is\s+(?:now\s+)?over\b(?!\s+(?:budget|the\b))"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:has\s+)?concluded\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+(?:has\s+)?(?:expired|lapsed|ended(?!\s+up))\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+(?:is|has\s+been)\s+(?:paused|on\s+hold)\b"
    r"|we(?:'ve|\s+have)?\s+paused\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)\b"
    r"|we(?:'ve|\s+have)?\s+(?:concluded|wrapped\s+up|ended|stopped|completed|finished)\s+this\s+search\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+(?:has\s+)?(?:wrapped\s+up|stopped)\b"
    r"|we(?:'ve|\s+have)?\s+(?:concluded|wrapped\s+up)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:has\s+)?wrapped\s+up\b"
    r"|we(?:'ve|\s+have)?\s+(?:already\s+)?(?:filled|closed|withdrawn|withdrew|cancelled|canceled|removed|unposted|unpublished|deactivated|archived|deleted|discontinued|pulled|rescinded|abandoned|called\s+off|shelved|tabled|sunset|retired|dropped|released|frozen|killed|scrapped)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we\s+(?:already\s+)?filled\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+(?:already\s+)?hired\s+(?:(?:someone|a\s+candidate)\s+)?(?:internally\s+)?for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+(?:already\s+)?made\s+a\s+hire\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:a|the)\s+hire\s+(?:has\s+been|was)\s+made\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+(?:selected|chosen)\s+(?:(?:a|an|the|another|other)\s+)?(?:candidate|applicant)s?\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:a|an|the)\s+(?:candidate|applicant)\s+(?:has\s+been|was)\s+(?:selected|chosen|hired)\s+(?:internally\s+)?for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )someone\s+(?:has\s+been|was)\s+hired\s+(?:internally\s+)?for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:an|the)\s+offer\s+has\s+been\s+accepted\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:a|an|the)\s+(?:candidate|applicant)\s+has\s+accepted\s+(?:an|the|our)\s+offer\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+has\s+an\s+accepted\s+offer\b"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+has\s+an\s+accepted\s+offer\b"
    r"|we(?:'ve|\s+have)?\s+(?:decided\s+to\s+)?moved?\s+forward\s+with\s+(?:other|another)\s+candidates?\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:is|has\s+been)\s+(?:(?:now|currently)\s+)?(?:closed|expired|lapsed|filled|cancelled|canceled|withdrawn|rescinded|paused|shelved|tabled|sunset|retired|dropped|frozen|killed|scrapped)\b(?!\s+with)"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:is|has\s+been)\s+(?:(?:now|currently)\s+)?(?:closed|expired|lapsed|filled|cancelled|canceled|withdrawn|rescinded|paused|shelved|tabled|sunset|retired|dropped|frozen|killed|scrapped)\b(?!\s+with)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:has\s+been|was|is)\s+marked\s+(?:as\s+)?(?:filled|closed)\b"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:has\s+been|was|is)\s+marked\s+(?:as\s+)?(?:filled|closed)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+is\s+(?:unpublished|archived|deactivated)\b"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+is\s+(?:unpublished|archived|deactivated)\b"
    r"|we(?:'re| are)\s+no\s+longer\s+(?:hiring\s+for\s+this|recruiting\s+for\s+this|advertising\s+for\s+this|accepting\s+(?:new\s+)?(?:applications|applicants)\b(?!\s+from)|taking\s+(?:new\s+)?(?:applications|applicants)\b(?!\s+from))"
    r"|we(?:'re| are)\s+no\s+longer\s+(?:recruiting|advertising|posting|listing|publishing)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:(?:'re| are)\s+not| aren't)\s+(?:recruiting|advertising|posting|listing|publishing)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+stopped\s+(?:recruiting|advertising|posting|listing|publishing)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+closed\s+recruiting\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+stopped\s+(?:accepting|taking|reviewing)\s+(?:new\s+)?(?:applications|applicants)\b(?!\s+from)"
    r"|we(?:(?:'re| are)\s+(?:no\s+longer|not)| aren't)\s+considering\s+(?:applications|applicants|candidates)\b(?!\s+from)"
    r"|(?<!once )(?<!after )(?<!when )we(?:(?:'re| are)\s+no\s+longer|(?:'ve|\s+have)?\s+stopped)\s+considering\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)(?!\s+for)"
    r"|(?<!once )(?<!after )(?<!when )(?:applications|applicants|candidates)\s+"
    r"(?:for\s+this\s+(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)\s+)?"
    r"are\s+no\s+longer\s+being\s+(?:reviewed|accepted|considered)\b(?!\s+from)"
    r"|no\s+longer\s+hiring\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|no\s+longer\s+(?:recruiting|advertising|posting|listing|publishing)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)\s+taken\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:down|offline)\b"
    r"|we\s+took\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:down|offline)\b"
    r"|we\s+took\s+down\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)\s+taken\s+down\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+(?:closed|cancelled|canceled|abandoned|called\s+off)\s+this\s+search\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+search\s+(?:is|has\s+been|was|has)\s+(?:cancelled|canceled|abandoned|called\s+off)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+is\s+no\s+longer\s+being\s+(?:recruited|advertised|posted|listed|published)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+is\s+not\s+being\s+(?:recruited|advertised|filled|posted|listed|published|pursued(?!\s+as\b)|considered(?!\s+for\b))\b"
    r"|we(?:(?:'re| are)\s+not| aren't)\s+(?:accepting|taking)\s+(?:new\s+)?(?:applications|applicants)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|no\s+longer\s+(?:accepting|taking)\s+(?:new\s+)?(?:applications|applicants|candidates)\b(?!\s+from)"
    r"|no\s+longer\s+reviewing\s+applications\b"
    r"|we(?:(?:'re| are)\s+not| aren't)\s+reviewing\s+(?:new\s+)?applications\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:applications|applicants|candidates)\s+"
    r"(?:for\s+this\s+(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)\s+)?"
    r"are\s+no\s+longer\s+under\s+review\b"
    r"|(?:new\s+)?(?:applications|applicants)\s+are\s+not\s+being\s+(?:accepted|taken|reviewed)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|applications\s+are\s+no\s+longer\s+open\b"
    r"|(?<!once )(?<!after )(?<!when )we\s+will\s+not\s+be\s+(?:filling|(?:(?:proceeding|moving\s+forward)\s+with|continuing(?:\s+with)?)|pursuing)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )we\s+will\s+not\s+(?:fill|(?:(?:proceed|move\s+forward)\s+with|continue(?:\s+with)?)|pursue)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:will\s+not|won['’]t)\s+be\s+filled\b(?!\s+(?:until|in\b|this|yet|before))"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:will\s+not|won['’]t)\s+be\s+filled\b(?!\s+(?:until|in\b|this|yet|before))"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:will\s+not|won['’]t)\s+(?:be\s+)?proceed(?:ing)?\b(?!\s+(?:until|in\b|this|yet|before))"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:will\s+not|won['’]t)\s+(?:be\s+)?proceed(?:ing)?\b(?!\s+(?:until|in\b|this|yet|before))"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:will\s+not|won['’]t)\s+be\s+hired\b(?!\s+(?:until|in\b|this|yet|before|for\s+[A-Za-z]))"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:will\s+not|won['’]t)\s+be\s+hired\b(?!\s+(?:until|in\b|this|yet|before|for\s+[A-Za-z]))"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:will\s+)?remain(?:s)?\s+unfilled\b(?!\s+(?:until|in\b|this|yet|before))"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:will\s+)?remain(?:s)?\s+unfilled\b(?!\s+(?:until|in\b|this|yet|before))"
    r"|we\s+will\s+not\s+be\s+(?:hiring|recruiting)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we\s+will\s+not\s+(?:hire|recruit)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )we(?:'re| are)\s+no\s+longer\s+(?:filling|(?:(?:proceeding|moving\s+forward)\s+with|continuing(?:\s+with)?)|pursuing)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )we(?:(?:'re| are)\s+not| aren't)\s+(?:(?:filling|(?:(?:proceeding|moving\s+forward)\s+with|continuing(?:\s+with)?)|pursuing)|going\s+to\s+(?:fill|(?:(?:proceed|move\s+forward)\s+with|continue(?:\s+with)?)|hire\s+for|pursue))\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+is\s+no\s+longer\s+being\s+(?:filled|pursued(?!\s+as\b)|considered(?!\s+for\b))\b"
    r"|applications\s+closed\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|applications\s+for\s+(?:the|this)\s+search\s+(?:are|is|have|has)\s+(?:now\s+)?closed"
    r"|we(?:'ve|\s+have)?\s+paused\s+(?:hiring|recruiting)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+(?:cancelled|canceled|discontinued|closed|ended|stopped)\s+(?:hiring|recruiting)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?:hiring|recruiting)\s+has\s+been\s+(?:cancelled|canceled|discontinued|closed|ended|stopped)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:hiring|recruiting)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:is\s+(?:paused|complete(?:d)?)|has\s+(?:ended|finished|completed))\b"
    r"|(?:hiring|recruiting)\s+has\s+closed\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:hiring|recruiting)\s+for\s+this\s+"
    r"(?:has\s+(?:closed|ended|finished|completed)|is\s+(?:closed|paused|complete(?:d)?))\b(?!\s+from)"
    r"|(?<!once )(?<!after )(?<!when )(?:hiring|recruiting)\s+(?:is\s+complete(?:d)?|has\s+(?:finished|completed))\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )we(?:'ve|\s+have)?\s+(?:completed|finished)\s+(?:our\s+)?(?:hiring|recruiting|search)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?<!the )(?:the\s+)?(?:hiring|recruiting)\s+process\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:is\s+complete(?:d)?|has\s+(?:ended|finished|completed))\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"['’]s\s+(?:hiring|recruiting)\s+(?:is\s+(?:paused|complete(?:d)?)|has\s+(?:ended|finished|completed))\b"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"['’]s\s+(?:hiring|recruiting)\s+(?:is\s+(?:paused|complete(?:d)?)|has\s+(?:ended|finished|completed))\b"
    r"|recruiting\s+has\s+ended\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)\s+put\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+on\s+(?:hold|pause)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:has\s+been\s+|is\s+)?(?:put\s+)?on\s+(?:hold|pause)\b"
    r"|we(?:(?:'re|\s+are)\s+not| aren't)\s+(?:currently\s+)?(?:hiring|recruiting|advertising)\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|applications\s+have\s+closed\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|we(?:'ve|\s+have)?\s+closed\s+applications\s+for\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"|(?<!once )(?<!after )(?<!when )(?:sorry,\s+)?(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:expired|lapsed|ended(?!\s+up))\b"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:expired|lapsed|ended(?!\s+up))\b"
    r"|applications\s+for\s+(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:are|is|have|has)\s+(?:now\s+)?closed"
    r"|(?:the\s+)?application\s+(?:window|period)\s+(?:has\s+closed|is\s+closed|has\s+ended|ended(?!\s+up))"
    r"|applications\s+closed\s+for\s+this\s+search\b"
    r"|(?<!once )(?<!after )(?<!when )applications\s+(?:are|have|has)\s+(?:now\s+)?closed\b(?!\s+from)"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+application\s+(?:is|has\s+been)\s+closed\b"
    r"|(?:the\s+)?application\s+deadline\s+has\s+(?:passed|expired|lapsed)\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+is\s+no\s+longer\s+open\s+(?:to|for)\s+(?:new\s+)?(?:applicants|applications|candidates)"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+is\s+no\s+longer\s+open\s+(?:to|for)\s+(?:new\s+)?(?:applicants|applications|candidates)"
    r"|(?:sorry,?\s+)?we\s+(?:could(?:n['’]t|\s+not)|were\s+unable\s+to)\s+(?:locate|find)\s+this\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|listing|opening|opportunity|vacancy|search)\b(?!\s+description)"
    r"|(?:sorry,?\s+)?we\s+could(?:n['’]t|\s+not)\s+find\s+that\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|listing|opening|opportunity|vacancy|search)\b(?!\s+description)"
    r"|(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|listing|opening|search)"
    r"\s+(?:could\s+not|cannot)\s+be\s+found\b"
    r"|the\s+(?:job(?:\s+posting)?|role|position|posting|listing|search)"
    r"\s+you\s+(?:are|were)\s+looking\s+for\s+(?:is\s+no\s+longer\s+"
    r"(?:available|active|posted|listed|live|published|advertised)|(?:could\s+not|cannot)\s+be\s+found)\b"
    r"|the\s+(?:job(?:\s+posting)?|role|position|posting|listing|search)"
    r"\s+you\s+applied\s+for\s+is\s+no\s+longer\s+"
    r"(?:available|active|posted|listed|live|published|advertised)\b"
    r"|the\s+(?:job(?:\s+posting)?|role|position|posting|listing|search)"
    r"\s+you\s+(?:requested|selected|applied\s+for)\s+no\s+longer\s+exists\b"
    r"|the\s+(?:job(?:\s+posting)?|role|position|posting|listing|search)"
    r"\s+you\s+(?:requested|selected|applied\s+for)\s+(?:could\s+not|cannot)\s+be\s+found\b"
    r"|(?<!once )(?<!after )(?<!when )(?:the|this)\s+"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening|search)"
    r"\s+(?:is|has\s+been)\s+inactive\b"
    r"|(?<!once )(?<!after )(?<!when )(?<!this )(?<!the )(?<!a )(?:sorry,\s+)?"
    r"(?:job(?:\s+posting)?|role|position|posting|vacancy|opportunity|requisition|req|listing|opening)"
    r"\s+(?:is|has\s+been)\s+inactive\b"
)


_MONTH_NUM = {
    name.lower(): i
    for names in (calendar.month_name, calendar.month_abbr)
    for i, name in enumerate(names)
    if i and name
}


def _ymd(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _unix_date(raw) -> Optional[date]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    ts = float(raw)
    if ts > 1e12:
        ts /= 1000.0
    if not (1_000_000_000 <= ts < 4_000_000_000):
        return None
    try:
        return datetime.fromtimestamp(ts, timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None


def _posting_date(raw) -> Optional[date]:
    unix = _unix_date(raw)
    if unix:
        return unix
    text = (_ld_text(raw) or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{10,13}", text):
        return _unix_date(int(text))
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        return _ymd(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if first > 12 >= second:
            return _ymd(year, second, first)
        parsed = _ymd(year, first, second)
        if parsed:
            return parsed
    m = re.match(r"(\d{4})(\d{2})(\d{2})\b", text)
    if m:
        return _ymd(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(
        r"(?:[A-Za-z]+,?\s+)?([A-Za-z]+)\.?,?\s*(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
        text,
    )
    if m:
        month = _MONTH_NUM.get(m.group(1).lower())
        if month:
            return _ymd(int(m.group(3)), month, int(m.group(2)))
    m = re.match(
        r"(?:[A-Za-z]+,?\s+)?(\d{1,2})(?:st|nd|rd|th)?[.,]?\s+([A-Za-z]+)\.?,?\s+(\d{4})",
        text,
    )
    if m:
        month = _MONTH_NUM.get(m.group(2).lower())
        if month:
            return _ymd(int(m.group(3)), month, int(m.group(1)))
    m = re.match(r"(\d{1,2})/([A-Za-z]+)/(\d{4})", text)
    if m:
        month = _MONTH_NUM.get(m.group(2).lower())
        if month:
            return _ymd(int(m.group(3)), month, int(m.group(1)))
    m = re.match(r"([A-Za-z]+)/(\d{1,2})/(\d{4})", text)
    if m:
        month = _MONTH_NUM.get(m.group(1).lower())
        if month:
            return _ymd(int(m.group(3)), month, int(m.group(2)))
    m = re.match(r"(\d{4})-([A-Za-z]+)-(\d{1,2})", text)
    if m:
        month = _MONTH_NUM.get(m.group(2).lower())
        if month:
            return _ymd(int(m.group(1)), month, int(m.group(3)))
    m = re.match(r"([A-Za-z]+)-(\d{1,2})-(\d{4})", text)
    if m:
        month = _MONTH_NUM.get(m.group(1).lower())
        if month:
            return _ymd(int(m.group(3)), month, int(m.group(2)))
    m = re.match(r"(\d{1,2})-([A-Za-z]+)-(\d{4})", text)
    if m:
        month = _MONTH_NUM.get(m.group(2).lower())
        if month:
            return _ymd(int(m.group(3)), month, int(m.group(1)))
    m = re.match(
        r"(\d{4})\s+([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?",
        text,
    )
    if m:
        month = _MONTH_NUM.get(m.group(2).lower())
        if month:
            return _ymd(int(m.group(1)), month, int(m.group(3)))
    m = re.match(r"(\d{4})/([A-Za-z]+)/(\d{1,2})", text)
    if m:
        month = _MONTH_NUM.get(m.group(2).lower())
        if month:
            return _ymd(int(m.group(1)), month, int(m.group(3)))
    m = re.match(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", text)
    if m:
        return _ymd(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"(\d{1,2})[-.](\d{1,2})[-.](\d{4})", text)
    if not m:
        return None
    first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if first > 12 >= second:
        return _ymd(year, second, first)
    if second > 12 >= first:
        return _ymd(year, first, second)
    return None


def _posting_expired(posting: Optional[dict]) -> bool:
    """True when JobPosting.validThrough / valid_through / expires is before today."""
    if not isinstance(posting, dict):
        return False
    through = _posting_date(
        posting.get("validThrough")
        or posting.get("valid_through")
        or posting.get("expires")
    )
    return through is not None and through < date.today()


def _html_is_gone(html: str) -> bool:
    """True when the page says the posting was taken down. 200 HTML is not a listing."""
    if _GONE_LISTING_RE.search(_listing_plain_text(html)):
        return True
    return _posting_expired(_job_posting(html))


def _html_is_index(html: str, url: str) -> bool:
    """Fetched board shells. ATS posting URLs still drop when the HTML title is a board."""
    if _cloudflare_challenge(html):
        return True
    if _INDEX_URL_RE.search(url):
        return True
    if not _ats_job_url(url):
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        if path == "/" or _INDEX_PATH_RE.search(parsed.path):
            return True
    title = _html_title(html)
    if not title:
        return False
    return _title_is_index(title)


def _ddg_result_url(href: str) -> str:
    """Unwrap DDG redirect hrefs (`/l/?uddg=`) to the listing URL."""
    url = unescape(href or "")
    if url.startswith("//"):
        url = f"https:{url}"
    if "uddg=" in url:
        target = parse_qs(urlparse(url).query).get("uddg", [""])[0]
        if target:
            url = unquote(target)
    return url


def _parse_ddg_html(html: str) -> list[dict]:
    """Parse DuckDuckGo HTML results."""
    snippets: dict[str, str] = {}
    for match in re.finditer(
        r'class="result__snippet"\s+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    ):
        key = _normalize_url(_ddg_result_url(match.group(1)))
        if key and key not in snippets:
            snippets[key] = _visible_text(match.group(2))
    lite_snips = [
        _visible_text(m.group(1))
        for m in re.finditer(
            r"(?is)class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
            html or "",
        )
    ]

    results: list[dict] = []
    seen: set[str] = set()

    def add(url: str, title: str, description: str) -> None:
        if not url or not title:
            return
        if "duckduckgo.com/y.js" in url or "bing.com/aclick" in url:
            return
        key = _normalize_url(url)
        if not key or key in seen:
            return
        seen.add(key)
        results.append(
            {
                "title": title,
                "url": url,
                "description": description,
                "source": "duckduckgo",
            }
        )

    for match in re.finditer(
        r'class="result__a"\s+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    ):
        url = _ddg_result_url(match.group(1))
        title = _visible_text(match.group(2))
        add(url, title, snippets.get(_normalize_url(url), ""))
    if results:
        return results[:20]
    for i, match in enumerate(
        re.finditer(
            r'(?is)<a[^>]+href="([^"]*uddg=[^"]+)"[^>]*>(.*?)</a>',
            html or "",
        )
    ):
        url = _ddg_result_url(match.group(1))
        title = _visible_text(match.group(2))
        desc = lite_snips[i] if i < len(lite_snips) else ""
        add(url, title, desc)
    return results[:20]


_HOUR_TAIL = r"\s*(?:/\s*h(?:r|our)s?|(?:per|an|a)\s+h(?:r|our)s?|hourly)\b"
_HOURLY_RANGE_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:\.\d+)?)"
    r"\s*(?:[-–—]|to)\s*"
    r"(?:USD|US\$|\$)?\s*(\d{1,3}(?:\.\d+)?)"
    + _HOUR_TAIL
)
_HOURLY_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)" + _HOUR_TAIL
)
_MONTH_TAIL = r"\s*(?:/\s*mo(?:nth)?s?|(?:per|a)\s+mo(?:nth)?s?|monthly)\b"
_MONTHLY_RANGE_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    r"\s*(?:[-–—]|to)\s*"
    r"(?:USD|US\$|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    + _MONTH_TAIL
)
_MONTHLY_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?" + _MONTH_TAIL
)
_WEEK_TAIL = r"\s*(?:/\s*w(?:ee)?ks?|(?:per|a)\s+w(?:ee)?ks?|weekly)\b"
_WEEKLY_RANGE_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    r"\s*(?:[-–—]|to)\s*"
    r"(?:USD|US\$|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    + _WEEK_TAIL
)
_WEEKLY_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?" + _WEEK_TAIL
)
_BIWEEK_TAIL = (
    r"\s*(?:bi[-\s]?weekly|"
    r"every[-\s]+(?:two|2|other)[-\s]+weeks?|"
    r"(?:per|a)\s+fortnight|"
    r"fortnightly)\b"
)
_BIWEEKLY_RANGE_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    r"\s*(?:[-–—]|to)\s*"
    r"(?:USD|US\$|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    + _BIWEEK_TAIL
)
_BIWEEKLY_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?" + _BIWEEK_TAIL
)
_SEMIMONTH_TAIL = r"\s*(?:semi[-\s]?monthly|twice\s+(?:a|per)\s+month|twice\s+monthly)\b"
_SEMIMONTHLY_RANGE_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    r"\s*(?:[-–—]|to)\s*"
    r"(?:USD|US\$|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    + _SEMIMONTH_TAIL
)
_SEMIMONTHLY_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?" + _SEMIMONTH_TAIL
)
_DAY_TAIL = r"\s*(?:/\s*(?:days?|diem)|(?:per|a)\s+(?:days?|diem)|daily)\b"
_DAILY_RANGE_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    r"\s*(?:[-–—]|to)\s*"
    r"(?:USD|US\$|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?"
    + _DAY_TAIL
)
_DAILY_RE = re.compile(
    r"(?i)(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k\b)?" + _DAY_TAIL
)
_NOT_PERIOD_UNIT = (
    r"(?!"
    + _HOUR_TAIL
    + r"|"
    + _DAY_TAIL
    + r"|"
    + _WEEK_TAIL
    + r"|"
    + _BIWEEK_TAIL
    + r"|"
    + _SEMIMONTH_TAIL
    + r"|"
    + _MONTH_TAIL
    + r")"
)
_NOT_RANGE_CONT = r"(?!\s*(?:[-–—]|to|and)\s*\$?\s*\d)"
_RANGE_K_RE = re.compile(
    r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k?\s*(?:[-–—]|to|and)\s*\$?\s*(\d{2,3}(?:\.\d+)?)\s*k(?!\d)"
    + _NOT_PERIOD_UNIT,
    re.I,
)
_RANGE_FULL_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s*(?:USD|US\$)?"
    r"\s*(?:to|-|–|—|and)\s*"
    r"\$?\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)"
    + _NOT_PERIOD_UNIT,
    re.I,
)
_RANGE_SPACE_K_RE = re.compile(
    r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k?\s+\$\s*(\d{2,3}(?:\.\d+)?)\s*k(?!\d)"
    + _NOT_PERIOD_UNIT,
    re.I,
)
_RANGE_SPACE_FULL_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s+\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)"
    + _NOT_PERIOD_UNIT,
    re.I,
)
_RANGE_USD_RE = re.compile(
    r"(?i)(?:USD|US\$)\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s*(?:to|-|–|—|and)\s*(?:USD|US\$)?\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)"
    + _NOT_PERIOD_UNIT
)
_ANNUAL_K_RE = re.compile(
    r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k(?!\d)" + _NOT_RANGE_CONT + _NOT_PERIOD_UNIT,
    re.I,
)
_ANNUAL_FULL_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\b"
    + _NOT_RANGE_CONT
    + _NOT_PERIOD_UNIT,
    re.I,
)
_ANNUAL_USD_RE = re.compile(
    r"(?i)(?:USD|US\$)\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\b"
    + _NOT_RANGE_CONT
    + _NOT_PERIOD_UNIT
)
_NON_SALARY_MONEY_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:without|no|not|nor|except(?:\s+for)?|excluding|instead\s+of|rather\s+than|versus|vs\.?)\s+"
    r"(?:a\s+|an\s+|any\s+|the\s+)?"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*\$?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"\bup\s+to\s+(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?\+"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:in\s+)?(?:equity|stock(?:\s+options?)?|RSUs?|ESPP|employee\s+stock(?:\s+purchase(?:\s+plan)?)?|restricted\s+stock|option\s+grant|severance)\b"
    r"|"
    r"\b(?:equity|stock(?:\s+options?)?|RSUs?|ESPP|employee\s+stock(?:\s+purchase(?:\s+plan)?)?|restricted\s+stock|option\s+grant|severance)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:(?:employee\s+)?(?:signing|sign-on|sign on|signon|relocation|annual|target|performance|retention|referral|spot|cash|year-end|year end|holiday|discretionary|quarterly|incentive|sales|monthly|stay|completion|anniversary|hiring|welcome|joining|new[-\s]hire|baby|new[-\s]parent|peer|patent|tenure|christmas|moving|variable|recognition|employee)\s+bonus|(?:signing|sign-on|sign on|signon)\b|(?:employee\s+)?referral\s+award|employee\s+referral\b|(?:spot|recognition)\s+award|holiday\s+gift|bonus\b)"
    r"|"
    r"\b(?:bonus|(?:employee\s+)?(?:signing|sign-on|sign on|signon|relocation|annual|target|performance|retention|referral|spot|cash|year-end|year end|holiday|discretionary|quarterly|incentive|sales|monthly|stay|completion|anniversary|hiring|welcome|joining|new[-\s]hire|baby|new[-\s]parent|peer|patent|tenure|christmas|moving|variable|recognition|employee)\s+bonus|(?:signing|sign-on|sign on|signon)|(?:employee\s+)?referral\s+award|employee\s+referral|(?:spot|recognition)\s+award|holiday\s+gift)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+relocation(?:\s+(?:bonus|assistance|package|stipend))?\b(?!\s+to)"
    r"|"
    r"\brelocation(?:\s+(?:bonus|assistance|package|stipend))?\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:tuition(?:\s+(?:reimbursement|assistance|benefit))?(?!\s+in)|education\s+(?:reimbursement|assistance|benefit|budget)|student[-\s]loans?(?:\s+repayment)?|professional\s+development(?:\s+(?:budget|reimbursement|allowance|benefit))?|learning(?:\s+and\s+development(?:\s+(?:budget|reimbursement|allowance|benefit))?|\s+(?:budget|reimbursement|allowance|benefit))|continuing\s+education(?:\s+(?:budget|reimbursement|allowance|benefit))?|conference\s+(?:budget|reimbursement|allowance|benefit)|training\s+(?:budget|reimbursement|allowance|benefit)|(?:annual\s+)?wellness\s+(?:budget|reimbursement|allowance|benefit|program)|annual\s+wellness|fertility\s+(?:benefit|coverage|assistance|budget)|adoption\s+(?:assistance|benefit|coverage)|parental\s+leave|family\s+leave|backup\s+care|child(?:[-\s])?care\s+(?:benefit|stipend|allowance|assistance|budget|FSA)|dependent\s+care\s+(?:benefit|stipend|allowance|assistance|budget|FSA)|mental\s+health\s+(?:benefit|stipend|allowance)|life\s+insurance|legal\s+insurance|legal\s+plan|pet\s+insurance|accident\s+insurance|vision\s+insurance|dental\s+insurance|medical\s+insurance|AD&D(?:\s+insurance)?|accidental\s+death(?:\s+and\s+dismemberment)?(?:\s+insurance)?|critical\s+illness(?:\s+insurance)?|hospital\s+indemnity(?:\s+insurance)?|legal\s+benefit|(?:short|long)[-\s]term\s+disability|disability\s+insurance|(?:LTD|STD)\s+insurance|(?:gym|fitness)\s+membership|(?:fitness|gym(?:\s+membership)?|commuter|parking|phone|cell(?:\s+phone)?|internet|home\s+office|mileage|gas|transit)\s+reimbursement|(?:commuter|parking|phone|cell(?:\s+phone)?|internet|transit|vision|dental|medical|EAP|LTD|STD|gym|fitness|caregiver|pet|accident)\s+benefit)\b"
    r"|"
    r"\b(?:tuition(?:\s+(?:reimbursement|assistance|benefit))?|education\s+(?:reimbursement|assistance|benefit|budget)|student[-\s]loans?(?:\s+repayment)?|professional\s+development(?:\s+(?:budget|reimbursement|allowance|benefit))?|learning(?:\s+and\s+development(?:\s+(?:budget|reimbursement|allowance|benefit))?|\s+(?:budget|reimbursement|allowance|benefit))|continuing\s+education(?:\s+(?:budget|reimbursement|allowance|benefit))?|conference\s+(?:budget|reimbursement|allowance|benefit)|training\s+(?:budget|reimbursement|allowance|benefit)|(?:annual\s+)?wellness\s+(?:budget|reimbursement|allowance|benefit|program)|annual\s+wellness|fertility\s+(?:benefit|coverage|assistance|budget)|adoption\s+(?:assistance|benefit|coverage)|parental\s+leave|family\s+leave|backup\s+care|child(?:[-\s])?care\s+(?:benefit|stipend|allowance|assistance|budget|FSA)|dependent\s+care\s+(?:benefit|stipend|allowance|assistance|budget|FSA)|mental\s+health\s+(?:benefit|stipend|allowance)|life\s+insurance|legal\s+insurance|legal\s+plan|pet\s+insurance|accident\s+insurance|vision\s+insurance|dental\s+insurance|medical\s+insurance|AD&D(?:\s+insurance)?|accidental\s+death(?:\s+and\s+dismemberment)?(?:\s+insurance)?|critical\s+illness(?:\s+insurance)?|hospital\s+indemnity(?:\s+insurance)?|legal\s+benefit|(?:short|long)[-\s]term\s+disability|disability\s+insurance|(?:LTD|STD)\s+insurance|(?:gym|fitness)\s+membership|(?:fitness|gym(?:\s+membership)?|commuter|parking|phone|cell(?:\s+phone)?|internet|home\s+office|mileage|gas|transit)\s+reimbursement|(?:commuter|parking|phone|cell(?:\s+phone)?|internet|transit|vision|dental|medical|EAP|LTD|STD|gym|fitness|caregiver|pet|accident)\s+benefit)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:OTE\b|on[- ]target\s+earnings\b|commission\b|TC\b|total\s+comp(?:ensation)?\b|incentive\s+compensation\b|variable\s+(?:pay|compensation)\b|deferred\s+compensation\b|(?:long|short)[-\s]term\s+incentive\b|(?:target|sales|cash|annual|performance)\s+incentive\b)"
    r"|"
    r"\b(?:OTE|on[- ]target\s+earnings|commission|TC|total\s+comp(?:ensation)?|incentive\s+compensation|variable\s+(?:pay|compensation)|deferred\s+compensation|(?:long|short)[-\s]term\s+incentive|(?:target|sales|cash|annual|performance)\s+incentive)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"(?:\s+(?:employer|annual))*"
    r"\s+401\(?k\)?\s+(?:match|matching|contribution)\b"
    r"|"
    r"\b(?:(?:employer|annual)\s+)*401\(?k\)?\s+(?:match|matching|contribution)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+matching\s+401\(?k\)?\b"
    r"|"
    r"\bmatching\s+401\(?k\)?\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:matching\s+gift|charitable\s+match|charitable\s+contribution|donation\s+match|giving\s+match|volunteer\s+grant|COBRA\s+subsidy)\b"
    r"|"
    r"\b(?:matching\s+gift|charitable\s+match|charitable\s+contribution|donation\s+match|giving\s+match|volunteer\s+grant|COBRA\s+subsidy)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:HSA|FSA|HRA|QSEHRA|ICHRA|pension|529|401\(?k\)?|health\s+savings\s+account|health\s+reimbursement\s+arrangement)(?:\s+(?:contribution|benefit))?\b(?!\s+in)"
    r"|"
    r"\b(?:HSA|FSA|HRA|QSEHRA|ICHRA|pension|529|401\(?k\)?|health\s+savings\s+account|health\s+reimbursement\s+arrangement)(?:\s+(?:contribution|benefit))?\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+pension\b"
    r"|"
    r"\bpension\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+wellness\b(?!\s+in)"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:phone|cell|internet|gym|fitness|housing|travel|meals?|food|living|parking|commuter|transit|mileage|clothing|equipment|laptop|moving)\b(?!\s+in)"
    r"|"
    r"\b(?:mileage|clothing|equipment|laptop|moving)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+pet\b(?!\s+in)"
    r"|"
    r"\bpet\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+legal\b(?!\s+in)"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+(?:PTO|vacation|sick)\s+(?:buyback|cash[- ]?out|payout)\b"
    r"|"
    r"\b(?:PTO|vacation|sick)\s+(?:buyback|cash[- ]?out|payout)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+unused\s+(?:PTO|vacation)\b"
    r"|"
    r"\bunused\s+(?:PTO|vacation)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+identity\s+theft(?:\s+protection)?\b"
    r"|"
    r"\bidentity\s+theft(?:\s+protection)?\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"\s+profit[-\s]shar(?:e|ing)\b"
    r"|"
    r"\bprofit[-\s]shar(?:e|ing)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"(?:\s*(?:/\s*mo(?:nth)?s?|(?:per|a)\s+mo(?:nth)?s?|monthly))?"
    r"(?:\s+(?:housing|living|relocation|meal|food|travel|wellness|phone|cell(?:phone)?|mobile|internet|commuter|parking|home\s+office|gym(?:\s+membership)?|fitness|child(?:[-\s])?care|dependent\s+care|health(?:care)?|clothing|uniform|WFH|work\s+from\s+home|remote\s+work|equipment|laptop|volunteer|tools|transit|education|moving|fertility|adoption|caregiver|parental|tech(?:nology)?|coworking|EAP|bike|mileage))*"
    r"\s+stipend\b"
    r"|"
    r"\b(?:monthly\s+)?stipend\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"(?:\s*(?:/\s*mo(?:nth)?s?|(?:per|a)\s+mo(?:nth)?s?|monthly))?"
    r"(?:\s+(?:housing|living|relocation|car|vehicle|auto|phone|cell(?:phone)?|mobile|internet|meal|food|travel|commuter|parking|home\s+office|gym(?:\s+membership)?|fitness|child(?:[-\s])?care|dependent\s+care|health(?:care)?|clothing|uniform|WFH|work\s+from\s+home|remote|equipment|laptop|tech(?:nology)?|coworking|transit|mileage|caregiver))*"
    r"\s+allowance\b"
    r"|"
    r"\b(?:(?:monthly|housing|living|relocation|car|vehicle|auto|phone|cell(?:phone)?|mobile|internet|meal|food|travel|commuter|parking|clothing|uniform|WFH|work\s+from\s+home|remote|equipment|laptop|tech(?:nology)?|coworking|transit|mileage|caregiver)\s+)*allowance\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"(?:" + _HOUR_TAIL + r"|" + _DAY_TAIL + r"|" + _WEEK_TAIL + r"|" + _MONTH_TAIL + r")"
    r"\s+(?:for\s+)?(?:housing|travel|meals?|food|living|phone|cell|internet|parking|gym|commuter|wellness|fitness)\b"
    r"|"
    r"\b(?:housing|travel|meals?|food|living|phone|cell|internet|parking|gym|commuter|wellness|fitness)\s*(?:of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"(?:" + _HOUR_TAIL + r"|" + _DAY_TAIL + r"|" + _WEEK_TAIL + r"|" + _MONTH_TAIL + r")?"
    r"|"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"(?:" + _HOUR_TAIL + r")?"
    r"\s+(?:overtime|on[-\s]call|shift\s+differential)\b"
    r"|"
    r"\b(?:overtime(?:\s+paid)?|on[-\s]call(?:\s+pay)?|shift\s+differential)\s*(?:at|of|:)?\s*"
    r"(?:USD|US\$|\$)\s*[\d,]+(?:\.\d+)?(?:\s*k)?"
    r"(?:\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*[\d,]+(?:\.\d+)?(?:\s*k)?)?"
    r"(?:" + _HOUR_TAIL + r")?"
    r")"
)
_HOURS_RE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)[\s-]*(?:(?:working|scheduled)[\s-]+)?(?:hours?|hrs?|h)\.?\s*"
    r"(?:/|\s*per\s*|\s+a\s+|\s+worked(?:\s+(?:per|/|a))?|\s+working(?:\s+(?:per|/|a))?|\s+scheduled(?:\s+(?:per|/|a))?|\s+work[\s-]*|\s+of\s+(?:the\s+)?(?:work(?:ing)?|scheduled)(?:\s+(?:a|per))?\s*)?\s*"
    r"(?:work[\s-]*weeks?|workweeks?|week(?:ly)?|wk)\b"
    r"(?!\s+(?:meeting|standup|stand-up|sync|call|all-?hands))"
    r"|(?:hours?|hrs?)\s*(?:(?:worked|working|scheduled)\s+)?(?:(?:per|/|a|for(?:\s+the)?)\s*)?(?:wk|weeks?|weekly)\s*(?:scheduled\s*)?[:=\-–—]?\s*(\d{1,2}(?:\.\d+)?)"
    r"|weekly\s+(?:(?:scheduled|work(?:ing)?)\s+)*(?:hours?|hrs?)\s*(?:scheduled\s*)?[:=\-–—]?\s*(\d{1,2}(?:\.\d+)?)"
    r"|(?:hours?|hrs?)\s+weekly\s*[:=\-–—]?\s*(\d{1,2}(?:\.\d+)?)"
    r"|(?<![\d.])(\d{1,2}(?:\.\d+)?)[\s-]*(?:scheduled[\s-]+)?weekly[\s-]+(?:(?:scheduled|work(?:ing)?)[\s-]+)?(?:hours?|hrs?)\b"
    r"(?!\s+(?:meeting|standup|stand-up|sync|call|all-?hands))"
    r"|(?:hours?|hrs?)\s*(?:worked\s*)?[:=\-–—]\s*(\d{1,2}(?:\.\d+)?)\s*(?:per|/|a)\s*(?:wk|weeks?|weekly)\b"
    r"|work[\s-]*weeks?\s*[:=\-–—]\s*(\d{1,2}(?:\.\d+)?)\s*(?:hours?|hrs?)?\b"
    r"|(?:hours?|hrs?)\s+of\s+(?:the\s+)?(?:work(?:ing)?|scheduled)\s*(?:(?:per|/|a)\s*)?(?:wk|weeks?|weekly)\s*[:=\-–—]?\s*(\d{1,2}(?:\.\d+)?)",
    re.I,
)
_DUAL_TIME_RE = re.compile(
    r"(?i)(?:full[\s-]*time\s*(?:and|or|/|&|,)\s*(?:and\s+)?part[\s-]*time"
    r"|part[\s-]*time\s*(?:and|or|/|&|,)\s*(?:and\s+)?full[\s-]*time)"
)
_PART_TIME_RE = re.compile(r"(?i)\bpart[\s-]+time\b")
_FULL_TIME_RE = re.compile(r"(?i)\bfull[\s-]+time\b")
_GEO_PAREN_RANGE_RE = re.compile(
    r"(?is)\(([^)]{0,120})\)\s*:\s*"
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(k\b)?"
    r"\s*(?:[-–—]|to)\s*"
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(k\b)?"
)
_GEO_NAMED_RANGE_RE = re.compile(
    r"(?i)\b("
    r"(?:us\s*[-–]\s*)?all other|"
    r"remote(?:\s+(?:us|usa|united states))?|"
    r"nationwide|"
    r"rest of(?: the)? (?:us|usa|united states)|"
    r"anywhere in (?:the )?(?:us|usa|united states)"
    r")\s*:\s*"
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(k\b)?"
    r"\s*(?:[-–—]|to)\s*"
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(k\b)?"
)
_REMOTE_BAND_RE = re.compile(
    r"(?i)\b(?:all other|remote|nationwide|"
    r"rest of(?: the)? (?:us|usa|united states)|"
    r"anywhere in (?:the )?(?:us|usa|united states))\b"
)


def _money(raw: str) -> float:
    return float(raw.replace(",", ""))


def _labeled_annual(raw: str, k: Optional[str]) -> int:
    n = _money(raw)
    if k or ("," not in raw and n < 1000):
        n *= 1000
    return int(n)


def _month_annual(raw: str, k: Optional[str]) -> Optional[int]:
    n = _money(raw)
    if k:
        n *= 1000
    annual = int(n * 12)
    if 10_000 <= annual <= 2_000_000:
        return annual
    return None


def _week_annual(raw: str, k: Optional[str]) -> Optional[int]:
    n = _money(raw)
    if k:
        n *= 1000
    annual = int(n * 50)
    if 10_000 <= annual <= 2_000_000:
        return annual
    return None


def _biweek_annual(raw: str, k: Optional[str]) -> Optional[int]:
    n = _money(raw)
    if k:
        n *= 1000
    annual = int(n * 25)
    if 10_000 <= annual <= 2_000_000:
        return annual
    return None


def _semimonth_annual(raw: str, k: Optional[str]) -> Optional[int]:
    n = _money(raw)
    if k:
        n *= 1000
    annual = int(n * 24)
    if 10_000 <= annual <= 2_000_000:
        return annual
    return None


def _day_annual(raw: str, k: Optional[str]) -> Optional[int]:
    n = _money(raw)
    if k:
        n *= 1000
    annual = int(n * 5 * 50)
    if 10_000 <= annual <= 2_000_000:
        return annual
    return None


def _labeled_range(
    low_raw: str, low_k: Optional[str], high_raw: str, high_k: Optional[str]
) -> Optional[tuple[int, int]]:
    low, high = _labeled_annual(low_raw, low_k), _labeled_annual(high_raw, high_k)
    if 10_000 <= low <= high <= 2_000_000:
        return low, high
    return None


def _remote_geo_pay(text: str) -> Optional[tuple[int, int]]:
    """Remote / rest-of-US band when a listing posts geo-labeled USD ranges."""
    blob = text or ""
    for match in _GEO_PAREN_RANGE_RE.finditer(blob):
        if not _REMOTE_BAND_RE.search(match.group(1)):
            continue
        pair = _labeled_range(match.group(2), match.group(3), match.group(4), match.group(5))
        if pair:
            return pair
    for match in _GEO_NAMED_RANGE_RE.finditer(blob):
        pair = _labeled_range(match.group(2), match.group(3), match.group(4), match.group(5))
        if pair:
            return pair
    return None


def _period_pay(
    text: str, hours: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """Annual USD from hourly/daily/weekly/monthly units. (None, None) if none."""
    hourly_range = _HOURLY_RANGE_RE.search(text)
    if hourly_range:
        weeks = hours or 40
        low = int(_money(hourly_range.group(1)) * weeks * 50)
        high = int(_money(hourly_range.group(2)) * weeks * 50)
        if 10_000 <= low <= high <= 2_000_000:
            return low, high
    hourly = _HOURLY_RE.search(text)
    if hourly:
        rate = _money(hourly.group(1))
        if 10 <= rate <= 1000:
            return None, int(rate * (hours or 40) * 50)
    daily_range = _DAILY_RANGE_RE.search(text)
    if daily_range:
        low = _day_annual(daily_range.group(1), daily_range.group(2))
        high = _day_annual(daily_range.group(3), daily_range.group(4))
        if low and high and low <= high:
            return low, high
    daily = _DAILY_RE.search(text)
    if daily:
        annual = _day_annual(daily.group(1), daily.group(2))
        if annual:
            return None, annual
    weekly_range = _WEEKLY_RANGE_RE.search(text)
    if weekly_range:
        low = _week_annual(weekly_range.group(1), weekly_range.group(2))
        high = _week_annual(weekly_range.group(3), weekly_range.group(4))
        if low and high and low <= high:
            return low, high
    weekly = _WEEKLY_RE.search(text)
    if weekly:
        annual = _week_annual(weekly.group(1), weekly.group(2))
        if annual:
            return None, annual
    biweekly_range = _BIWEEKLY_RANGE_RE.search(text)
    if biweekly_range:
        low = _biweek_annual(biweekly_range.group(1), biweekly_range.group(2))
        high = _biweek_annual(biweekly_range.group(3), biweekly_range.group(4))
        if low and high and low <= high:
            return low, high
    biweekly = _BIWEEKLY_RE.search(text)
    if biweekly:
        annual = _biweek_annual(biweekly.group(1), biweekly.group(2))
        if annual:
            return None, annual
    semimonth_range = _SEMIMONTHLY_RANGE_RE.search(text)
    if semimonth_range:
        low = _semimonth_annual(semimonth_range.group(1), semimonth_range.group(2))
        high = _semimonth_annual(semimonth_range.group(3), semimonth_range.group(4))
        if low and high and low <= high:
            return low, high
    semimonth = _SEMIMONTHLY_RE.search(text)
    if semimonth:
        annual = _semimonth_annual(semimonth.group(1), semimonth.group(2))
        if annual:
            return None, annual
    monthly_range = _MONTHLY_RANGE_RE.search(text)
    if monthly_range:
        low = _month_annual(monthly_range.group(1), monthly_range.group(2))
        high = _month_annual(monthly_range.group(3), monthly_range.group(4))
        if low and high and low <= high:
            return low, high
    monthly = _MONTHLY_RE.search(text)
    if monthly:
        annual = _month_annual(monthly.group(1), monthly.group(2))
        if annual:
            return None, annual
    return None, None


def _annual_pay(text: str) -> tuple[Optional[int], Optional[int]]:
    """Annual USD from yearly amounts. (None, None) if none."""
    ranged = _RANGE_K_RE.search(text)
    if ranged:
        low, high = int(_money(ranged.group(1)) * 1000), int(_money(ranged.group(2)) * 1000)
        if 10_000 <= low <= high <= 2_000_000:
            return low, high
    ranged_full = _RANGE_FULL_RE.search(text)
    if ranged_full:
        low, high = int(_money(ranged_full.group(1))), int(_money(ranged_full.group(2)))
        if 10_000 <= low <= high <= 2_000_000:
            return low, high
    spaced_k = _RANGE_SPACE_K_RE.search(text)
    if spaced_k:
        low, high = int(_money(spaced_k.group(1)) * 1000), int(_money(spaced_k.group(2)) * 1000)
        if 10_000 <= low <= high <= 2_000_000:
            return low, high
    spaced = _RANGE_SPACE_FULL_RE.search(text)
    if spaced:
        low, high = int(_money(spaced.group(1))), int(_money(spaced.group(2)))
        if 10_000 <= low <= high <= 2_000_000:
            return low, high
    ranged_usd = _RANGE_USD_RE.search(text)
    if ranged_usd:
        low, high = int(_money(ranged_usd.group(1))), int(_money(ranged_usd.group(2)))
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
    usd = _ANNUAL_USD_RE.search(text)
    if usd:
        annual = int(_money(usd.group(1)))
        if 10_000 <= annual <= 2_000_000:
            return None, annual
    return None, None


def _parse_pay(
    text: str, hours: Optional[int] = None, *, remote: bool = False
) -> tuple[Optional[int], Optional[int]]:
    """(pay_low, pay_high) annual USD from listing text. (None, None) if unknown.

    Stated yearly pay wins over on-call/travel period rates on the same page.
    """
    text = _NON_SALARY_MONEY_RE.sub(" ", text or "")
    if _FOREIGN_DOLLAR_RE.search(text) or _FOREIGN_PAY_RE.search(text):
        return None, None
    if remote:
        geo = _remote_geo_pay(text)
        if geo:
            return geo
    yearly = _annual_pay(text)
    if yearly[0] or yearly[1]:
        return yearly
    return _period_pay(text, hours)


def _guess_pay(title: str, description: str, hours: Optional[int] = None) -> Optional[int]:
    """Best annual pay in the listing text, or None. Does not invent a number."""
    low, high = _parse_pay(f"{title} {description}", hours)
    return high or low


def _stated_hours(title: str, description: str) -> Optional[int]:
    """Hours explicitly written as N hours/week. None if the listing does not say."""
    match = _HOURS_RE.search(f"{title} {description}")
    if match:
        raw = next((g for g in match.groups() if g), None)
        if raw:
            n = int(round(float(raw)))
            if 1 <= n <= 80:
                return n
    return None


def _employment_hours(text: str) -> Optional[int]:
    """20/40 from this blob's employment type. None if both or neither."""
    stripped = _DUAL_TIME_RE.sub(" ", text or "")
    part = bool(_PART_TIME_RE.search(stripped))
    full = bool(_FULL_TIME_RE.search(stripped))
    if part and not full:
        return 20
    if full and not part:
        return 40
    return None


def _guess_hours(title: str, description: str) -> Optional[int]:
    """Hours from the listing text, or None. Does not assume full-time."""
    stated = _stated_hours(title, description)
    if stated:
        return stated
    return _employment_hours(title) or _employment_hours(description)


_COUNTRY_ONLY_RE = re.compile(
    r"(?i)^(?:the\s+)?(?:"
    r"united states(?:\s+of\s+america)?|usa|u\.s\.a?\.?|us|"
    r"canada|mexico|united kingdom|great britain|uk|"
    r"australia|new zealand|germany|france|spain|italy|netherlands|"
    r"sweden|norway|denmark|finland|ireland|switzerland|"
    r"india|japan|china|singapore|brazil|"
    r"worldwide|global|anywhere|"
    r"north america|south america|emea|europe|apac|asia(?:[-\s]pacific)?|"
    r"european union|eu"
    r")$"
)
_UNKNOWN_WORKPLACE_RE = re.compile(
    r"(?i)^(?:unspecified|unknown|n/?a|none|null|not\s+(?:specified|set|listed|available))$"
)


def _workplace_remote(place: str) -> Optional[bool]:
    """True/False from an ATS workplace/location string. None if unknown.

    Remote wins when the string offers both (e.g. "Remote or Hybrid").
    """
    p = (place or "").casefold()
    if not p:
        return None
    if re.search(r"\b(?:not remote|no remote|onsite only|on-site only)\b", p):
        return False
    if re.search(r"\b(?:remote|offsite|off-site|telecommute|distributed)\b", p):
        return True
    if re.search(r"\bhybrid\b", p) or re.search(r"\bflex(?:ible)?\b", p):
        return False
    if re.search(r"\b(?:onsite|on-site|on site|in-office|in office|in the office|into the office)\b", p):
        return False
    if re.search(r"\boffice[-\s]?first\b(?!\s+aid)", p):
        return False
    if re.search(
        r"\b(?:site|campus|lab(?:oratory)?|field|headquarters|hq)[-\s]?based\b", p
    ):
        return False
    if re.search(r"\bon[-\s]?campus\b", p):
        return False
    compact = re.sub(r"[\s_-]+", "", p)
    if compact in {"remote", "offsite", "telecommute", "distributed"}:
        return True
    if compact in {
        "hybrid",
        "onsite",
        "office",
        "inoffice",
        "intheoffice",
        "intotheoffice",
        "flex",
        "flexible",
        "officebased",
        "officefirst",
        "sitebased",
        "campusbased",
        "labbased",
        "laboratorybased",
        "fieldbased",
        "headquartersbased",
        "hqbased",
        "oncampus",
    }:
        return False
    return None


def _apply_workplace(posting: dict, *places: str) -> None:
    site = ""
    for raw in places:
        place = (raw or "").strip()
        if not place or _UNKNOWN_WORKPLACE_RE.fullmatch(place):
            continue
        flag = _workplace_remote(place)
        if flag is True:
            posting["jobLocationType"] = "TELECOMMUTE"
            return
        if flag is False:
            posting["jobLocationType"] = "ON_SITE"
            return
        if not site:
            site = place
    if site and not _COUNTRY_ONLY_RE.fullmatch(site):
        posting["jobLocationType"] = "ON_SITE"


def _jsonld_place(posting: dict) -> str:
    """Workplace label from JobPosting jobLocation / job_location. Empty if omitted."""
    loc = (
        posting.get("jobLocation")
        or posting.get("job_location")
        or posting.get("workLocation")
        or posting.get("work_location")
    )
    rows = loc if isinstance(loc, list) else [loc]
    names = []
    for row in rows:
        if isinstance(row, str) and row.strip():
            names.append(row.strip())
            continue
        if not isinstance(row, dict):
            continue
        label = (_ld_text(row.get("name")) or "").strip()
        addr = row.get("address")
        if isinstance(addr, list) and addr:
            addr = addr[0]
        if isinstance(addr, dict):
            city = (
                _ld_text(addr.get("addressLocality") or addr.get("address_locality"))
                or ""
            ).strip()
            region = (
                _ld_text(addr.get("addressRegion") or addr.get("address_region"))
                or ""
            ).strip()
            country = _country_label(
                addr.get("addressCountry")
                or addr.get("address_country")
                or addr.get("country")
            )
            label = (
                label
                or ", ".join(p for p in (city, region) if p)
                or country
                or (_ld_text(addr) or "").strip()
            )
        elif isinstance(addr, str) and addr.strip():
            label = label or addr.strip()
        if label:
            names.append(label)
    if any(_workplace_remote(n) is True for n in names):
        return "remote"
    return next((n for n in names if n and not _UNKNOWN_WORKPLACE_RE.fullmatch(n)), "")


def _remote_from_posting(posting: dict) -> Optional[bool]:
    raw = (
        posting.get("jobLocationType")
        or posting.get("job_location_type")
        or posting.get("workplaceType")
        or posting.get("workplace_type")
        or posting.get("locationType")
        or posting.get("location_type")
    )
    types = {t.upper().replace("-", "_") for t in _ld_types(raw)}
    if any("TELECOMMUTE" in t for t in types):
        return True
    if any("ON_SITE" in t or t == "ONSITE" for t in types):
        return False
    if any("HYBRID" in t or t == "FLEX" for t in types):
        return False
    for t in _ld_types(raw):
        flag = _workplace_remote(t)
        if flag is not None:
            return flag
    place = _jsonld_place(posting)
    if not place:
        return None
    flag = _workplace_remote(place)
    if flag is not None:
        return flag
    if _UNKNOWN_WORKPLACE_RE.fullmatch(place.strip()) or _COUNTRY_ONLY_RE.fullmatch(place.strip()):
        return None
    return False


_REMOTE_OPTION_RE = re.compile(
    r"(?i)\b(?:fully\s+remote|remote(?:-|\s+)?first|work\s+from\s+(?:home|anywhere)"
    r"|hybrid\s*(?:/|,?\s*or)\s+(?:fully\s+)?remote"
    r"|remote\s*(?:/|,?\s*or)\s+hybrid)\b"
)
_HYBRID_WORKPLACE_RE = re.compile(
    r"(?i)\bhybrid\s+(?:work(?:ing)?(?:\s+(?:environment|model|schedule|arrangement|approach))?|"
    r"role|position|schedule|workplace|environment|office|setup|approach)\b"
    r"|\bmust\s+be\s+hybrid\b"
    r"|(?:(?:this|the|a)\s+)?(?:role|position|job)\s+is\s+hybrid\b"
    r"|\(hybrid\)"
    r"|\[hybrid\]"
)
_ONSITE_WORKPLACE_RE = re.compile(
    r"(?i)\b(?:onsite|on-site|on site|in-office|in office)\b"
    r"|office[-\s]based\b"
    r"|office(?:-|\s+)?first\b(?!\s+aid)"
    r"|\b(?:site|campus|lab(?:oratory)?|field|headquarters|hq)[-\s]based\b"
    r"|on[-\s]campus\s+(?:role|position|job)\b"
    r"|work\s+on[-\s]campus\b"
    r"|work\s+from\s+(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}(?:office|campus|lab(?:oratory)?|headquarters|hq|field(?!\s+of\b))\b"
    r"|work\s+out\s+of\s+(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}(?:office|campus|lab(?:oratory)?|headquarters|hq|field(?!\s+of\b))\b"
    r"|based\s+out\s+of\s+(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}(?:office|campus|lab(?:oratory)?|headquarters|hq)\b"
    r"|\b(?:lab(?:oratory)?|field|headquarters|hq|office)\s+(?:role|position|job)\b"
    r"|report\s+to\s+(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}offices?\b"
    r"|commute\s+to\s+(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}offices?\b"
    r"|in(?:to)?\s+the\s+offices?\b"
    r"|in\s+our\s+(?:\S+\s+){0,4}offices?\b"
    r"|come\s+(?:in)?to\s+(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}offices?\b(?!\s+hours)"
    r"|in[-\s]person\s+(?:role|position|job)\b"
    r"|work\s+in[-\s]person\b"
    r"|in[-\s]person\s+in\b"
    r"|(?:(?:this|the|a)\s+)?(?:role|position|job)\s+is\s+(?:based|located)\s+in"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"|(?:(?:this|the|a)\s+)?(?:role|position|job)\s+requires\s+you\s+to\s+be\s+in"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"|(?:(?:this|the|a)\s+)?(?:role|position|job)\s+requires\s+(?:your\s+)?presence\s+in"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"|you\s+will\s+be\s+(?:based|located)\s+in"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"|(?:you|candidates)\s+must\s+be\s+(?:based|located)\s+in"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"|must\s+commute\s+to\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote|interviews?)\b)"
    r"|must\s+relocate\s+to\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote|interviews?)\b)"
    r"|(?<!not )required\s+to\s+relocate\s+to\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"|(?:(?:this|the|a)\s+)?(?:role|position|job)\s+requires\s+relocation"
    r"(?!\s+(?:assistance|package|bonus|stipend))"
    r"(?!\s+to\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"|relocation\s+to\s+(?!(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"(?:\S+\s+){0,4}(?:is\s+)?required\b"
    r"|(?<!no )(?<!not )relocation\s+is\s+required\b"
    r"|must\s+be\s+in\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"(?:\s+(?!two\b|three\b|four\b|five\b|\d+\b)\S+){0,4}"
    r"\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?<!not )required\s+to\s+be\s+in\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"(?:\s+(?!two\b|three\b|four\b|five\b|\d+\b)\S+){0,4}"
    r"\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?:(?:this|the|a)\s+)?(?:role|position|job)\s+is\s+in\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote)\b)"
    r"(?:\s+(?!two\b|three\b|four\b|five\b|\d+\b)\S+){0,4}"
    r"\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\s+in\s+"
    r"(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}(?:office|campus|lab(?:oratory)?|headquarters|hq|hub)\b"
    r"|in[-\s]person\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\s+in[-\s]person\b"
    r"|hybrid\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\s+in\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote|meetings?)\b)"
    r"|must\s+work\s+(?:from|in)\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote|home|meetings?|standups?)\b)"
    r"(?:\s+(?!two\b|three\b|four\b|five\b|\d+\b)\S+){0,4}"
    r"\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?<!home )(?<!back )(?<!front )(?<!microsoft )\boffice\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:(?:a|per)\s+week|weekly)\b"
    r"|(?:office|site|campus|hub|headquarters|HQ|lab(?:oratory)?|field)\s+(?:presence|attendance)\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:(?:a|per)\s+week|weekly)\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:of\s+)?(?:office|site|campus|hub|headquarters|HQ|lab(?:oratory)?|field)\s+(?:presence|attendance)"
    r"(?:\s+(?:a|per)\s+week|\s+weekly)?\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+"
    r"(?<!home )(?<!back )(?<!front )(?<!microsoft )office\s+(?:a|per)\s+week\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\s+from\s+"
    r"(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}(?<!home )(?<!microsoft )office\b(?!\s+hours)"
    r"|hybrid\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+weekly\b"
    r"|hybrid\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+"
    r"(?<!home )office\b(?!\s+hours)"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+from\s+"
    r"(?:the\s+|our\s+|an\s+)?(?:\S+\s+){0,4}(?<!home )(?<!microsoft )office\b(?!\s+hours)"
    r"\s+(?:each|a|per)\s+week\b"
    r"|on[-\s]campus\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+on[-\s]campus\s+(?:a|per)\s+week\b"
    r"|come\s+in(?:to|\s+to)\s+work\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|come\s+to\s+work\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?<!results )(?<!reports )(?<!data )\bcome\s+in\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?<!on )(?<!on-)\bcampus\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|\b(?:headquarters|HQ)\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|\blab(?:oratory)?\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|in\s+the\s+lab(?:oratory)?\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:each|a|per)\s+week\s+from\s+"
    r"(?:the\s+|our\s+|an\s+)?lab(?:oratory)?\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:each|a|per)\s+week\s+from\s+"
    r"(?:the\s+|our\s+|an\s+)?campus\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:each|a|per)\s+week\s+from\s+"
    r"(?:the\s+|our\s+|an\s+)?(?:headquarters|HQ)\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:each|a|per)\s+week\s+at\s+"
    r"(?:the\s+|our\s+|an\s+)?(?:office|campus|lab(?:oratory)?|headquarters|HQ)\b"
    r"|report\s+to\s+(?:the\s+|our\s+|an\s+)?(?:headquarters|HQ)\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+(?:a|per)\s+week\b"
    r"|(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+in\s+"
    r"(?!(?:the\s+|our\s+|an\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote|meetings?|standups?|calls?)\b)"
    r"(?:\S+\s+){0,3}(?:a|per)\s+week\b"
    r"|hybrid\s+(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+in\b"
    r"(?!\s+(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote|meetings?|standups?)\b)"
    r"|(?:(?:this|the|a)\s+)?(?:role|position|job)\s+requires\s+"
    r"(?:two|three|four|five|\d+(?:\s*[-–]\s*\d+)?)\s+days?\s+in\s+"
    r"(?!(?:the\s+)?(?:us|usa|united\s+states|uk|united\s+kingdom|europe|eu|emea|apac|anywhere|worldwide|globally|remote|meetings?|standups?)\b)"
)


def _guess_remote(title: str, description: str) -> bool:
    """Remote unless the listing's own text says office/hybrid workplace.

    Related-job cards and ML 'hybrid retrieval' are not workplace.
    """
    desc = _RELATED_JOBS_RE.sub("", description or "")
    text = f"{title} {desc}"
    if _REMOTE_OPTION_RE.search(text):
        return True
    if _ONSITE_WORKPLACE_RE.search(text) or _HYBRID_WORKPLACE_RE.search(text):
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
