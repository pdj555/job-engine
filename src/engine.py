"""The engine. One class. Does everything."""

import asyncio
import json
import re
from html import unescape
from typing import Optional
from urllib.parse import urlparse

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
        """Fill missing pay/hours/company from the listing page. Never invent."""
        missing = [o for o in opps if o.pay is None or not o.company]
        if missing:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=8.0,
                headers=_LISTING_HEADERS,
            ) as client:
                self._http_client = client
                try:
                    texts = await asyncio.gather(
                        *(self._listing_text(o.url) for o in missing),
                        return_exceptions=True,
                    )
                finally:
                    self._http_client = None
            gone = []
            for o, text in zip(missing, texts):
                if not isinstance(text, str) or not text:
                    continue
                if _html_is_index(text, o.url):
                    gone.append(o)
                    continue
                _apply_listing(o, text)
            if gone:
                opps[:] = [o for o in opps if o not in gone]
        _unify_board_companies(opps)

    async def _listing_text(self, url: str) -> str:
        if not _public_http_url(url):
            return ""
        client = getattr(self, "_http_client", None)

        async def fetch(target: str) -> str:
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
            if raw.lstrip().startswith("#"):
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
        return await fetch(_lever_job_url(url))

    async def _search_all(self, query: str) -> list[dict]:
        """Search all sources in parallel."""
        searches = [self._search_brave(q) for q in _search_angles(query)]

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
        """Free web search fallback. Retry DDG 202s; cap concurrency so site: angles survive."""
        async with self._ddg_sem:
            try:
                async with httpx.AsyncClient() as client:
                    for attempt in range(3):
                        resp = await client.post(
                            "https://html.duckduckgo.com/html/",
                            data={"q": query, "b": ""},
                            headers={"User-Agent": "Mozilla/5.0 (compatible; JobEngine/1.0)"},
                            timeout=30.0,
                            follow_redirects=True,
                        )
                        if resp.status_code == 202:
                            await asyncio.sleep(0.4 * (attempt + 1))
                            continue
                        if resp.status_code >= 400:
                            return []
                        return _parse_ddg_html(resp.text)
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


async def _http_get_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        resp = await client.get(url)
        if resp.status_code >= 400:
            return ""
        return resp.text
    except Exception:
        return ""


_ATS_TITLE_TAIL_RE = re.compile(
    r"(?i)\s*[-–—|]\s*(?:jobs\.(?:lever\.co|ashbyhq\.com|workable\.com)|jobs by workable)\s*$"
)


def _strip_ats_title(title: str) -> str:
    t = _ATS_TITLE_TAIL_RE.sub("", title or "")
    return re.sub(r"(?i)^job application for\s+", "", t)


def _role_title(title: str) -> str:
    t = _strip_ats_title(title).strip()
    return t or (title or "Unknown").strip() or "Unknown"


def _title_key(title: str, company: Optional[str] = None) -> str:
    """Role identity across boards: strip ATS suffixes and company wrappers."""
    t = _strip_ats_title(title)
    if company and company.strip():
        c = re.escape(company.strip())
        t = re.sub(rf"(?i)^{c}\s*[-:|]\s*", "", t)
        t = re.sub(rf"(?i)\s+at\s+{c}\b.*$", "", t)
        t = re.sub(rf"(?i)\s+@\s+{c}\s*$", "", t)
        t = re.sub(rf"(?i)\s*[|\-–—]\s*{c}\s*$", "", t)
    t = re.sub(r"(?i)\s+in remote\b.*$", "", t)
    return re.sub(r"\W+", " ", t).casefold().strip()


def _dedupe_opportunities(opps: list) -> list:
    """Keep the first of each role. Call after sorting so the best score wins."""
    seen: set[str] = set()
    unique = []
    for o in opps:
        key = _title_key(o.title, o.company)
        if not key or key in seen:
            continue
        seen.add(key)
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
            "jobs.ashbyhq.com",
            "jobs.workable.com",
            "apply.workable.com",
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
    if not name or _PLACE_RE.search(name) or _ROLE_START_RE.search(name):
        return None
    return name


def _company_from_title(title: str, url: str = "") -> str | None:
    """Employer from ` at X`, ` @ X`, Lever `Company - Role`, or Workable suffixes."""
    t = _strip_ats_title(title)
    m = re.search(r"(?i)\bat\s+(.+)$", t)
    if m:
        name = m.group(1).strip(" .,-")
        if name and not _PLACE_RE.search(name):
            return name
    m = re.search(r"(?i)\s+@\s+(.+)$", t)
    if m:
        name = m.group(1).strip(" .,-")
        if name and not _PLACE_RE.search(name):
            return name
    host = (urlparse(url).hostname or "").casefold()
    if host.endswith("lever.co"):
        m = re.match(r"^(.+?)\s+[-–—]\s+\S", t)
        if m:
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
    r"(?:indeed\.com/q-|indeed\.com/jobs\?|linkedin\.com/jobs/(?!view/)"
    r"|glassdoor\.com/Job/jobs|simplyhired\.com/search|/search\?q="
    r"|upwork\.com/freelance-jobs/apply/"
    r"|lemon\.io/for-developers/"
    r"|magic\.lemon\.io/share/"
    r"|docs\.lemon\.io/"
    r"|corptocorp\.org/)",
    re.I,
)
_INDEX_TITLE_RE = re.compile(
    r"(?i)\bjobs\b(?!\.)(?! by workable)|^hire a freelance\b|\bcurrent openings\b"
)
_GH_JOB_RE = re.compile(
    r"(?i)https?://(?:job-boards(?:\.[a-z]+)?|boards)\.greenhouse\.io/([^/]+)/jobs/(\d+)",
)


def _greenhouse_api_url(url: str) -> Optional[str]:
    m = _GH_JOB_RE.search(url or "")
    if not m:
        return None
    return f"https://boards-api.greenhouse.io/v1/boards/{m.group(1)}/jobs/{m.group(2)}"


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
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{''.join(parts)}"
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
    page_title = f"{title} at {company}" if company else title
    return (
        f"<title>{page_title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"<pre>{md}</pre>"
    )


_INDEX_PATH_RE = re.compile(
    r"^/(?:category|categories|tag|tags|topics?|major)(?:/|$)|/search",
    re.I,
)


def _is_index_page(raw: dict) -> bool:
    """True for search/board/home/category pages, not a single opportunity."""
    url = raw.get("url") or ""
    title = raw.get("title") or ""
    desc = raw.get("description") or ""
    if _INDEX_URL_RE.search(url):
        return True
    if _INDEX_TITLE_RE.search(title):
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
    if _workable_is_board(url):
        return True
    return False


def _compensation_from_raw(
    raw: dict, title: str, description: str, hours: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """Structured source pay wins; otherwise parse listing text. Never invent."""
    if raw.get("pay") is not None:
        return None, raw["pay"]
    return _parse_pay(f"{title} {description}", hours)


def _visible_text(html: str) -> str:
    return unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html))).strip()


def _listing_plain_text(html: str) -> str:
    """Visible listing copy only — scripts/styles are not compensation."""
    html = re.sub(r"(?is)<script\b[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style\b[^>]*>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript\b[^>]*>.*?</noscript>", " ", html)
    return _visible_text(html)


_LD_SCRIPT_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)
_PAY_UNITS = {
    "HOUR": "hour",
    "HOURLY": "hour",
    "HR": "hour",
    "YEAR": "year",
    "ANNUAL": "year",
    "ANNUM": "year",
    "YR": "year",
    "WEEK": "week",
    "WEEKLY": "week",
    "MONTH": "month",
    "MONTHLY": "month",
}


def _ld_types(value) -> set[str]:
    if isinstance(value, str):
        return {value.rsplit("/", 1)[-1]}
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out |= _ld_types(item)
        return out
    return set()


def _walk_ld(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from _walk_ld(item)
    elif isinstance(obj, dict):
        yield obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from _walk_ld(value)


def _job_posting(html: str) -> Optional[dict]:
    for raw in _LD_SCRIPT_RE.findall(html or ""):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for obj in _walk_ld(data):
            if "JobPosting" in _ld_types(obj.get("@type")):
                return obj
    return None


def _num(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _usd(currency) -> bool:
    if not currency:
        return True
    return str(currency).upper().replace("$", "").strip() in {"USD", "US", "USA"}


def _posting_company(posting: dict) -> Optional[str]:
    org = posting.get("hiringOrganization")
    if isinstance(org, list) and org:
        org = org[0]
    if isinstance(org, str):
        name = org.strip()
    elif isinstance(org, dict):
        name = str(org.get("name") or "").strip()
    else:
        return None
    if not name or _PLACE_RE.search(name):
        return None
    return name


def _posting_hours(posting: dict) -> Optional[int]:
    work = posting.get("workHours")
    n = _num(work)
    if n is None and isinstance(work, str):
        m = re.search(r"\b(\d{1,2})\b", work)
        n = float(m.group(1)) if m else None
    if n is not None and 1 <= n <= 80:
        return int(n)
    types = posting.get("employmentType")
    blob = " ".join(str(t).upper().replace("-", "_") for t in (
        types if isinstance(types, list) else [types]
    ) if t)
    if "PART_TIME" in blob:
        return 20
    if "FULL_TIME" in blob:
        return 40
    return None


def _annualize(amount: float, unit: Optional[str], hours: Optional[int]) -> Optional[int]:
    if unit == "hour":
        if not 10 <= amount <= 1000:
            return None
        return int(amount * (hours or 40) * 50)
    if unit == "week":
        return int(amount * 50)
    if unit == "month":
        return int(amount * 12)
    if unit == "year" or (unit is None and 10_000 <= amount <= 2_000_000):
        return int(amount)
    return None


def _posting_pay(
    posting: dict, hours: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    salary = posting.get("baseSalary") or posting.get("salary")
    if salary is None:
        return None, None
    if isinstance(salary, (int, float, str)):
        annual = _annualize(_num(salary) or 0, None, hours)
        if annual and 10_000 <= annual <= 2_000_000:
            return None, annual
        return None, None
    if not isinstance(salary, dict) or not _usd(salary.get("currency")):
        return None, None
    value = salary.get("value")
    unit = None
    low = high = None
    if isinstance(value, dict):
        unit = _PAY_UNITS.get(str(value.get("unitText") or "").upper())
        low, high = _num(value.get("minValue")), _num(value.get("maxValue"))
        if high is None:
            high = _num(value.get("value"))
    else:
        high = _num(value)
        unit = _PAY_UNITS.get(str(salary.get("unitText") or "").upper())
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


def _apply_listing(opp: Opportunity, html: str) -> None:
    """Fill missing fields from JobPosting JSON-LD, then visible listing text."""
    posting = _job_posting(html)
    if posting:
        pt = str(posting.get("title") or "").strip()
        if pt:
            opp.title = _role_title(pt)
        name = _posting_company(posting)
        if name:
            opp.company = name
        if opp.hours_per_week is None:
            hours = _posting_hours(posting)
            if hours:
                opp.hours_per_week = hours
        if opp.pay is None:
            low, high = _posting_pay(posting, opp.hours_per_week)
            if high or low:
                opp.pay_low = low
                opp.pay_high = high
    if not opp.company:
        opp.company = _guess_company(_html_title(html), opp.url)
    if opp.pay is None:
        visible = _listing_plain_text(html)
        hours = opp.hours_per_week or _guess_hours(opp.title, visible)
        low, high = _parse_pay(visible, hours)
        if high or low:
            opp.pay_low = low
            opp.pay_high = high
            if opp.hours_per_week is None and hours:
                opp.hours_per_week = hours
    opp.title = _role_title(opp.title)
    opp.efficiency = opp.refined_rate


def _html_title(html: str) -> str:
    m = re.search(r"(?is)<title>([^<]+)</title>", html or "")
    return unescape(m.group(1)).strip() if m else ""


def _html_is_index(html: str, url: str) -> bool:
    title = _html_title(html)
    return bool(title) and _is_index_page({"url": url, "title": title, "description": ""})


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


_HOURLY_RANGE_RE = re.compile(
    r"\$\s*(\d{1,3}(?:\.\d+)?)\s*(?:[-–—]|to)\s*\$?\s*(\d{1,3}(?:\.\d+)?)\s*(?:/|\s+per\s+)\s*h(?:r|our)s?\b",
    re.I,
)
_HOURLY_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:/|\s+per\s+)\s*h(?:r|our)s?\b",
    re.I,
)
_RANGE_K_RE = re.compile(
    r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k?\s*(?:[-–—]|to)\s*\$?\s*(\d{2,3}(?:\.\d+)?)\s*k\b",
    re.I,
)
_RANGE_FULL_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s*(?:to|-|–|—)\s*\$?\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)",
    re.I,
)
_RANGE_USD_RE = re.compile(
    r"(?i)(?:USD|US\$)\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s*(?:to|-|–|—)\s*(?:USD|US\$)?\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)"
)
_ANNUAL_K_RE = re.compile(r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k\b", re.I)
_ANNUAL_FULL_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\b")
_ANNUAL_USD_RE = re.compile(
    r"(?i)(?:USD|US\$)\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\b"
)
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
            annual = int(rate * (hours or 40) * 50)
            return None, annual
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
