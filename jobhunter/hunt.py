"""Strict plugin policy around the existing discovery, enrichment and ranker."""

import io
from contextlib import redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
from urllib.parse import urlsplit

from jobhunter.profile import Profile
from jobhunter.registry import Registry, posting_key
from src.compensation import (ats_json_url, canonicalize_url, is_search_serp,
                              is_syndicated_listing, parse_job_posting, parse_listing_pay)
from src.engine import Engine, _apply_comp
from src.models import Opportunity

VERIFIED = {"ats", "schema", "posted"}
ATS_HOSTS = {"boards-api.greenhouse.io", "api.lever.co", "api.ashbyhq.com",
             "api.eu.lever.co", "api.smartrecruiters.com"}
ATS_LISTING_HOSTS = {"job-boards.greenhouse.io", "boards.greenhouse.io", "jobs.lever.co",
                     "jobs.eu.lever.co", "jobs.ashbyhq.com", "jobs.smartrecruiters.com"}


class SameOriginClient:
    """Never label redirected mirror content as evidence from the requested host."""
    def __init__(self, client, final_urls):
        self.client = client
        self.final_urls = final_urls

    async def get(self, url, **kwargs):
        response = await self.client.get(url, **kwargs)
        requested, fetched = urlsplit(url), urlsplit(str(response.url))
        if fetched.scheme != "https" or fetched.hostname != requested.hostname:
            raise ValueError("Employer evidence redirected to a different origin.")
        self.final_urls[canonicalize_url(url)] = canonicalize_url(str(response.url))
        return response


class HuntEngine(Engine):
    """Keep the actual job URL when the core fetches a board-wide ATS endpoint."""
    def __init__(self):
        super().__init__()
        self._resolved_listing = ContextVar("jobhunter_resolved_listing", default=None)
        self._identities = {}
        self._field_urls = {}
        self._final_urls = {}
        self._search_failed = False

    async def _search_all(self, query):
        self._search_failed = False
        diagnostics = io.StringIO()
        with redirect_stdout(diagnostics):
            hits = await super()._search_all(query)
        if not hits and (self._search_failed or diagnostics.getvalue().strip()):
            raise RuntimeError("Job discovery failed. Check search connectivity.")
        return hits

    async def _search_brave(self, query):
        try:
            return await super()._search_brave(query)
        except Exception:
            self._search_failed = True
            raise

    async def _enrich_one(self, opp, client):
        token = self._resolved_listing.set(None)
        try:
            await super()._enrich_one(opp, client)
            resolved = self._resolved_listing.get()
            if resolved:
                self._identities[opp.url] = resolved
            origins = {field: resolved if getattr(opp, field + "_source") == "ats"
                       else self._final_urls.get(opp.url, opp.url)
                       for field in ("hours", "remote")}
            # Core can return after ATS pay+remote, before fetching explicit hours.
            # Supplement from that employer's actual job page without changing core.
            if opp.hours_source not in VERIFIED:
                listing = resolved or opp.url
                html = await self._fetch_listing(listing, client)
                if html:
                    for parsed, source in ((parse_job_posting(html), "schema"),
                                           (parse_listing_pay(html), "posted")):
                        before = {"hours": (opp.hours_per_week, opp.hours_source),
                                  "remote": (opp.remote, opp.remote_source)}
                        provenance = "unverified" if is_syndicated_listing(listing) else source
                        _apply_comp(opp, parsed, provenance, listing)
                        after = {"hours": (opp.hours_per_week, opp.hours_source),
                                 "remote": (opp.remote, opp.remote_source)}
                        for field in before:
                            if before[field] != after[field]:
                                origins[field] = self._final_urls.get(listing, listing)
            self._field_urls[opp.url] = origins
            if opp.pay_source_url:
                opp.pay_source_url = self._final_urls.get(canonicalize_url(opp.pay_source_url), opp.pay_source_url)
        finally:
            self._resolved_listing.reset(token)

    async def _fetch_ats(self, url, client):
        parsed = await super()._fetch_ats(url, SameOriginClient(client, self._final_urls))
        if parsed and parsed.posted:
            self._resolved_listing.set(canonicalize_url(url))
        return parsed

    async def _fetch_listing(self, url, client):
        return await super()._fetch_listing(url, SameOriginClient(client, self._final_urls))

    def evidence_identity(self, opp):
        if opp.pay_source != "ats":
            return opp.pay_source_url
        listing = self._identities.get(opp.url)
        if not listing:
            return opp.pay_source_url
        # Ashby exposes the entire board at one API URL; preserve the job ID.
        if urlsplit(listing).hostname == "jobs.ashbyhq.com":
            return self._final_urls.get(listing, listing)
        endpoint = canonicalize_url(ats_json_url(listing) or listing)
        return self._final_urls.get(endpoint, endpoint)

    def field_urls(self, opp):
        return self._field_urls.get(opp.url, {})


def employer_url(url: str, profile: Profile) -> bool:
    if not url or is_syndicated_listing(url):
        return False
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (
        host in profile.employer_domains or host in ATS_HOSTS or host in ATS_LISTING_HOSTS
        or host.endswith(".myworkdayjobs.com"))


def employer_origin(o: Opportunity, profile: Profile) -> bool:
    source = o.pay_source_url or ""
    if not source or is_syndicated_listing(source) or is_syndicated_listing(o.url):
        return False
    host = urlsplit(source).hostname or ""
    if host in profile.employer_domains:
        return True
    if o.pay_source == "ats" and (host in ATS_HOSTS or host.endswith(".myworkdayjobs.com")):
        return True
    return host in ATS_LISTING_HOSTS and bool(ats_json_url(source))


def evaluate(o: Opportunity, profile: Profile, field_urls=None) -> tuple[str, str, float | None]:
    field_urls = field_urls or {}
    missing = []
    if o.pay_source not in VERIFIED or o.pay is None or o.pay <= 0 or not employer_origin(o, profile):
        missing.append("employer-posted pay")
    if o.pay_is_annualized:
        missing.append("pay conversion basis (core annualization can assume hours)")
    if o.hours_source not in VERIFIED or not o.hours_per_week or not 0 < o.hours_per_week <= 80:
        missing.append("listing hours")
    elif not employer_url(field_urls.get("hours") or (o.pay_source_url if o.hours_source == "ats" else o.url), profile):
        missing.append("employer origin for hours")
    if o.remote_source not in VERIFIED or o.remote is None:
        missing.append("remote evidence")
    elif not employer_url(field_urls.get("remote") or (o.pay_source_url if o.remote_source == "ats" else o.url), profile):
        missing.append("employer origin for remote evidence")
    if missing:
        return "unverified", "Missing verified " + ", ".join(missing), None

    failed = []
    if profile.remote_only and o.remote is not True:
        failed.append("remote only")
    if (o.pay_low if o.pay_low is not None else o.pay_high) < profile.min_annual_pay:
        failed.append("minimum annual pay (posted floor)")
    if o.hours_per_week > profile.max_hours_per_week:
        failed.append("maximum hours")
    if any(term not in o.title.lower() for term in profile.title_terms):
        failed.append("required title terms")
    if any(company in (o.company or "").lower() for company in profile.exclude_companies):
        failed.append("excluded company")
    if profile.exclude_companies and not o.company:
        return "unverified", "Missing employer identity for company exclusions", None
    rate = o.dollars_per_hour
    if rate is not None and o.remote is False:
        rate *= 0.7
    if rate is None or rate < profile.min_dollars_per_hour:
        failed.append("minimum dollars per hour")
    if failed:
        return "filtered", "; ".join(failed), None
    return "qualified", "All gates cleared", rate


async def hunt(profile: Profile, registry: Registry, sheets, engine=None, *, dry_run=False):
    engine = engine or HuntEngine()
    with registry.lock():
        # Flush cached outcomes first. No re-evaluation after delivery failures.
        pending = registry.pending()
        if pending:
            sheets.sync(pending)
            registry.synced(pending)
        with redirect_stdout(io.StringIO()):
            hits = await engine._search_all(profile.query)
            for hit in hits:
                url = canonicalize_url(hit.get("url", ""))
                parsed = urlsplit(url)
                if parsed.scheme != "https" or not parsed.hostname or is_search_serp(url):
                    continue
                if not registry.claim(url):
                    continue
                # Fresh models prevent snippet/model pay, hours or remote guesses
                # surviving into the evidence gate. Core enrich fetches the listing.
                o = Opportunity(title=hit.get("title") or "Unknown", url=url,
                                company=hit.get("company"), source="jobhunter")
                await engine.enrich([o])
                field_urls = engine.field_urls(o) if hasattr(engine, "field_urls") else {}
                status, reason, score = evaluate(o, profile, field_urls)
                if status == "qualified":
                    identity = engine.evidence_identity(o) if hasattr(engine, "evidence_identity") else o.pay_source_url
                    if identity and not registry.claim_evidence(identity, url):
                        registry.alias(url)
                        continue
                    # Commit the qualified employer identity before calling the
                    # unchanged ranker. Unverified mirrors cannot reserve it.
                    score = o.score()
                registry.finish({
                    "key": posting_key(url), "first_seen": datetime.now(timezone.utc).isoformat(),
                    "title": o.title, "company": o.company, "url": url,
                    "pay_low": o.pay_low, "pay_high": o.pay_high,
                    "hours_per_week": o.hours_per_week, "dollars_per_hour": score,
                    "gate_status": status, "reason": reason,
                    "pay_source": o.pay_source, "pay_source_url": o.pay_source_url,
                    "hours_source": o.hours_source, "remote": o.remote,
                    "remote_source": o.remote_source,
                    "pay_is_annualized": o.pay_is_annualized,
                    "hours_source_url": field_urls.get("hours") or o.url,
                    "remote_source_url": field_urls.get("remote") or o.url,
                    "profile_signature": profile.signature(),
                    "mode": "dry_run" if dry_run else "live",
                    "pipeline_state": "shortlisted" if status == "qualified" else "discovered",
                    "notes": "Synthetic verification fixture" if dry_run else "",
                })
        pending = registry.pending()
        if pending:
            sheets.sync(pending)
            registry.synced(pending)
        return [r for r in registry.notifications() if r.get("profile_signature") == profile.signature()]


class DryEngine(Engine):
    """Synthetic discovery and employer HTML; all parsing/ranking is real core code."""
    def __init__(self):
        self.openai = None

    async def _search_all(self, query):
        return [{"title": "Python engineer [DRY RUN]", "url": f"https://careers.example.com/jobs/{i}"}
                for i in range(1, 4)]

    async def _fetch_ats(self, url, client=None):
        return None

    async def _fetch_listing(self, url, client=None):
        if url.endswith("/1"):
            return "<p>Fully remote. Base salary: $140,000-$180,000 annually. 40 hours/week.</p>"
        if url.endswith("/2"):
            return "<p>Fully remote. 40 hours/week. Competitive pay.</p>"
        return "<p>Base salary: $160,000 annually. Hours and location to be discussed.</p>"
