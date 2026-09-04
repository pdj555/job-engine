"""The engine. One class. Does everything."""

import asyncio
import json
import re
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
                if _html_is_index(text, o.url):
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
        iframe = _icims_iframe_url(url)
        if iframe:
            raw = await fetch(iframe)
            if raw is None:
                return None
            return raw or ""
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
        return await fetch(_lever_job_url(url))

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
_LEVER_PAY_UNITS = (
    ("hour", "HOUR"),
    ("month", "MONTH"),
    ("week", "WEEK"),
    ("year", "YEAR"),
)


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
    if isinstance(rng, dict) and rng.get("currency"):
        interval = str(rng.get("interval") or "").lower()
        unit = "YEAR"
        for needle, name in _LEVER_PAY_UNITS:
            if needle in interval:
                unit = name
                break
        value: dict = {"unitText": unit}
        low, high = rng.get("min"), rng.get("max")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            value["minValue"] = low
            value["maxValue"] = high
        elif isinstance(high, (int, float)):
            value["value"] = high
        elif isinstance(low, (int, float)):
            value["value"] = low
        if "value" in value or "minValue" in value:
            posting["baseSalary"] = {
                "currency": str(rng.get("currency")).upper(),
                "value": value,
            }
    parts = []
    place = str(data.get("workplaceType") or "").strip()
    if place:
        parts.append(f"<p>{place}</p>")
        _apply_workplace(posting, place)
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


async def _http_get_text(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        resp = await client.get(url)
        if resp.status_code in (404, 410):
            return None
        if resp.status_code >= 400:
            return ""
        return resp.text
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
    loc = f"<p>{place}</p>" if place else ""
    _apply_workplace(posting, place)
    return (
        f"<title>{title}</title>"
        f'<script type="application/ld+json">{json.dumps(posting)}</script>'
        f"{loc}<p>{summary}</p>{desc}"
    )


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
    """Role identity across boards: same employer + role, after stripping wrappers."""
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
    if org:
        return f"{org.casefold()}\t{role}"
    return role


def _dedupe_opportunities(opps: list) -> list:
    """Keep the first of each employer+role. Call after sorting so the best score wins."""
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
            "jobs.eu.lever.co",
            "jobs.ashbyhq.com",
            "jobs.workable.com",
            "apply.workable.com",
            "jobs.smartrecruiters.com",
            "myworkdayjobs.com",
            "icims.com",
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
    r"|glassdoor\.com/Job/"
    r"|simplyhired\.com/search|/search\?q="
    r"|upwork\.com/freelance-jobs/apply/"
    r"|lemon\.io/for-developers/"
    r"|magic\.lemon\.io/share/"
    r"|docs\.lemon\.io/"
    r"|corptocorp\.org/"
    r"|karkidi\.com/"
    r"|jobleads\.com/"
    r"|remoterocketship\.com/"
    r"|migratemate\.co/"
    r"|builtin\.com/jobs"
    r"|ziprecruiter\.com/Jobs/"
    r"|ziprecruiter\.com/jobs-search)",
    re.I,
)
_INDEX_TITLE_RE = re.compile(
    r"(?i)^hire a freelance\b|\bcurrent openings\b"
)
_JOBS_WORD_RE = re.compile(r"(?i)\bjobs\b(?!\.)(?! by workable)")
_ROLE_JOBS_AT_RE = re.compile(r"(?i).+\bjobs at \S")


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


def _greenhouse_api_url(url: str) -> Optional[str]:
    ids = _greenhouse_ids(url)
    if not ids:
        return None
    return (
        f"https://boards-api.greenhouse.io/v1/boards/{ids[0]}/jobs/{ids[1]}"
        "?pay_transparency=true"
    )


def _cents_to_annual(cents) -> Optional[int]:
    if not isinstance(cents, (int, float)):
        return None
    annual = int(cents) // 100
    if 10_000 <= annual <= 2_000_000:
        return annual
    return None


def _greenhouse_pay_ld(data: dict) -> Optional[dict]:
    """USD baseSalary from pay_input_ranges. Ignore other currencies."""
    for row in data.get("pay_input_ranges") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("currency_type") or "").upper() not in {"USD", "US", "USA"}:
            continue
        low = _cents_to_annual(row.get("min_cents"))
        high = _cents_to_annual(row.get("max_cents"))
        if not high and not low:
            continue
        value: dict = {"unitText": "YEAR"}
        if low and high:
            value["minValue"] = low
            value["maxValue"] = high
        else:
            value["value"] = high or low
        return {"currency": "USD", "value": value}
    return None


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
    if n is not None and 1 <= n <= 80:
        posting["workHours"] = str(int(n))
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
    place = str(data.get("workplace") or "").strip()
    loc = f"<p>{place}</p>" if place else ""
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
    _apply_workplace(posting, place)
    loc = str(info.get("location") or "").strip()
    desc = str(info.get("jobDescription") or "")
    page_title = f"{title} at {company}" if company else title
    bits = []
    if place:
        bits.append(f"<p>{place}</p>")
    if loc:
        bits.append(f"<p>{loc}</p>")
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
    if _icims_ids(url):
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
    if _workable_is_board(url):
        return True
    if _smartrecruiters_is_board(url):
        return True
    if _workday_is_board(url):
        return True
    if _icims_is_board(url):
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


_FOREIGN_PAY_RE = re.compile(
    r"(?:€|£)\s*\d{1,3}(?:,\d{3}){1,2}"
    r"|(?:€|£)\s*\d{5,7}\b"
    r"|(?:€|£)\s*\d{2,3}(?:\.\d+)?\s*k\b"
    r"|\b(?:EUR|GBP)\s+\d{1,3}(?:,\d{3}){1,2}"
    r"|\b(?:EUR|GBP)\s+\d{5,7}\b"
    r"|\b(?:EUR|GBP)\s+\d{2,3}(?:\.\d+)?\s*k\b",
    re.I,
)
_FOREIGN_DOLLAR_RE = re.compile(
    r"\b(?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|CLP|COP|PEN)\b\s*\$?\s*\d"
    r"|(?<![A-Za-z])(?:C|A)\$\s*\d"
    r"|\b(?:salario|mensual|pesos?)\b.{0,80}\$\s*\d"
    r"|\$\s*\d[\d,]*.{0,80}salary\s+monthly"
    r"|\$\s*\d[\d,]*(?:\.\d+)?\s*(?:k\b)?\s*"
    r"(?:MXN|CAD|AUD|NZD|SGD|HKD|ARS|pesos?)\b",
    re.I | re.S,
)


def _foreign_pay_text(text: str) -> bool:
    blob = text or ""
    return bool(_FOREIGN_PAY_RE.search(blob) or _FOREIGN_DOLLAR_RE.search(blob))


def _foreign_salary(html: str) -> bool:
    """True when the listing states a non-USD salary. Ranking is USD $/hour."""
    posting = _job_posting(html)
    if posting:
        salary = posting.get("baseSalary") or posting.get("salary")
        if isinstance(salary, dict) and salary.get("currency") and not _usd(salary.get("currency")):
            return True
    return _foreign_pay_text(_listing_plain_text(html))


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


def _apply_listing(opp: Opportunity, html: str) -> bool:
    """Fill fields from JobPosting JSON-LD, then visible listing text. Listing wins.

    Returns True when this HTML stated USD pay.
    """
    posting = _job_posting(html)
    listed_pay = False
    if posting:
        pt = str(posting.get("title") or "").strip()
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
    if not listed_pay:
        low, high = _parse_pay(visible, opp.hours_per_week, remote=opp.remote)
        if high or low:
            opp.pay_low = low
            opp.pay_high = high
            listed_pay = True
    if opp.remote:
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


def _html_is_index(html: str, url: str) -> bool:
    title = _html_title(html)
    return bool(title) and _is_index_page({"url": url, "title": title, "description": ""})


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


_HOURLY_RANGE_RE = re.compile(
    r"\$\s*(\d{1,3}(?:\.\d+)?)\s*(?:[-–—]|to)\s*\$?\s*(\d{1,3}(?:\.\d+)?)\s*(?:/|\s+per\s+)\s*h(?:r|our)s?\b",
    re.I,
)
_HOURLY_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:/|\s+per\s+)\s*h(?:r|our)s?\b",
    re.I,
)
_RANGE_K_RE = re.compile(
    r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k?\s*(?:[-–—]|to|and)\s*\$?\s*(\d{2,3}(?:\.\d+)?)\s*k(?!\d)",
    re.I,
)
_RANGE_FULL_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s*(?:USD|US\$)?"
    r"\s*(?:to|-|–|—|and)\s*"
    r"\$?\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)",
    re.I,
)
_RANGE_SPACE_K_RE = re.compile(
    r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k?\s+\$\s*(\d{2,3}(?:\.\d+)?)\s*k(?!\d)",
    re.I,
)
_RANGE_SPACE_FULL_RE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s+\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)",
    re.I,
)
_RANGE_USD_RE = re.compile(
    r"(?i)(?:USD|US\$)\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\s*(?:to|-|–|—|and)\s*(?:USD|US\$)?\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})(?!\d)"
)
_ANNUAL_K_RE = re.compile(r"\$\s*(\d{2,3}(?:\.\d+)?)\s*k(?!\d)", re.I)
_ANNUAL_FULL_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\b")
_ANNUAL_USD_RE = re.compile(
    r"(?i)(?:USD|US\$)\s*(\d{1,3}(?:,\d{3}){1,2}|\d{5,7})\b"
)
_HOURS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:hrs?|hours?)\s*(?:/|\s*per\s*|\s+a\s+)?\s*(?:wk|week)\b",
    re.I,
)
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


def _parse_pay(
    text: str, hours: Optional[int] = None, *, remote: bool = False
) -> tuple[Optional[int], Optional[int]]:
    """(pay_low, pay_high) annual USD from listing text. (None, None) if unknown."""
    if _FOREIGN_DOLLAR_RE.search(text):
        return None, None
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
    if remote:
        geo = _remote_geo_pay(text)
        if geo:
            return geo
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


def _guess_pay(title: str, description: str, hours: Optional[int] = None) -> Optional[int]:
    """Best annual pay in the listing text, or None. Does not invent a number."""
    low, high = _parse_pay(f"{title} {description}", hours)
    return high or low


def _stated_hours(title: str, description: str) -> Optional[int]:
    """Hours explicitly written as N hours/week. None if the listing does not say."""
    match = _HOURS_RE.search(f"{title} {description}")
    if match:
        n = int(match.group(1))
        if 1 <= n <= 80:
            return n
    return None


def _guess_hours(title: str, description: str) -> Optional[int]:
    """Hours from the listing text, or None. Does not assume full-time."""
    stated = _stated_hours(title, description)
    if stated:
        return stated
    lower = f"{title} {description}".lower()
    if "part-time" in lower or "part time" in lower:
        return 20
    if "full-time" in lower or "full time" in lower:
        return 40
    return None


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
    if re.search(r"\b(?:onsite|on-site|on site|in-office|in office)\b", p):
        return False
    compact = re.sub(r"[\s_-]+", "", p)
    if compact in {"remote", "offsite", "telecommute", "distributed"}:
        return True
    if compact in {"hybrid", "onsite", "office", "inoffice", "flex", "flexible"}:
        return False
    return None


def _apply_workplace(posting: dict, place: str) -> None:
    flag = _workplace_remote(place)
    if flag is True:
        posting["jobLocationType"] = "TELECOMMUTE"
    elif flag is False:
        posting["jobLocationType"] = "ON_SITE"


def _remote_from_posting(posting: dict) -> Optional[bool]:
    jlt = str(posting.get("jobLocationType") or "").upper().replace("-", "_")
    if "TELECOMMUTE" in jlt:
        return True
    if "ON_SITE" in jlt or "ONSITE" in jlt:
        return False
    return None


_REMOTE_OPTION_RE = re.compile(
    r"(?i)\b(?:fully\s+remote|remote(?:-|\s+)?first|work\s+from\s+(?:home|anywhere)"
    r"|hybrid\s*[,/]?\s*(?:or\s+)?(?:fully\s+)?remote"
    r"|remote\s*[,/]?\s*(?:or\s+)?hybrid)\b"
)


def _guess_remote(title: str, description: str) -> bool:
    text = f"{title} {description}"
    if _REMOTE_OPTION_RE.search(text):
        return True
    lower = text.lower()
    if any(
        w in lower
        for w in ("onsite", "on-site", "on site", "in-office", "in office", "hybrid")
    ):
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
