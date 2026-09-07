"""Posted pay/hours from listing text, JobPosting JSON-LD, and ATS URL identity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = {
    "fbclid",
    "gclid",
    "gbraid",
    "wbraid",
    "dclid",
    "msclkid",
    "twclid",
    "yclid",
    "ttclid",
    "li_fat_id",
    "_ga",
    "_gl",
    "mc_cid",
    "mc_eid",
    "_hsenc",
    "_hsmi",
    "mkt_tok",
    "igshid",
    "gh_src",
    "lever-source",
    "lever-source[]",
    "lever-origin",
    "source",
}

_BENEFIT = re.compile(
    r"(?i)401\s*\(?k\)?|403\s*\(?b\)?|hsa|fsa|hra|rsu|espp|equity|"
    r"bonus|commission|\bote\b|signing|health|dental|vision|insurance|"
    r"stipend|relocation|tuition|wellness|pto"
)
_FOREIGN = re.compile(r"(?i)(?:£|€|¥|(?<![A-Z])(?:CAD|AUD|GBP|EUR))\s*[\d$]")
_USD_MARK = re.compile(r"(?:USD|US\$|\$)\s*\d")
_BARE_USD_RANGE = re.compile(
    r"(?i)(?<![\d$])(\d{2,3})\s*k\s*(?:[-–—]|to)\s*(\d{2,3})\s*k\s*USD\b"
)
_BARE_USD = re.compile(r"(?i)(?<![\d$])(\d{2,3})\s*k\s*USD\b")
_HOURS = re.compile(
    r"(?i)(\d{1,2}(?:\.\d)?)\s*(?:hours?|hrs?|h)\s*(?:/|per)?\s*(?:week|wk)"
)
_AMOUNT = r"(?:USD|US\$|\$)\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*[kK])?"
_RANGE = re.compile(
    rf"{_AMOUNT}\s*(?:[-–—]|to)\s*(?:USD|US\$|\$)?\s*"
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s*[kK])?",
    re.I,
)
_SINGLE = re.compile(_AMOUNT, re.I)
_HOUR_TAIL = re.compile(r"(?i)(?:/|\bper\b|\b)\s*(?:hr|hour|hourly)\b")
_ANNUAL_TAIL = re.compile(r"(?i)(?:/|\bper\b)?\s*(?:yr|year|annual(?:ly)?)\b")
_LD_SCRIPT = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.S,
)
_SCHEMA_HOURS = re.compile(r"(?i)(\d{1,2}(?:\.\d)?)\s*(?:hours?|hrs?)\b")
_USD = {"", "USD", "US", "USA"}


@dataclass(frozen=True)
class Compensation:
    pay_low: int | None = None
    pay_high: int | None = None
    hours: int | None = None
    remote: bool | None = None
    company: str | None = None
    title: str | None = None

    @property
    def posted(self) -> bool:
        return self.pay_low is not None or self.pay_high is not None


def parse_compensation(text: str) -> Compensation:
    """Extract explicit USD pay and weekly hours. Invents nothing."""
    blob = _scrub_benefits(text or "")
    hours = _parse_hours(blob)
    if _FOREIGN.search(blob) and not _USD_MARK.search(blob) and not _BARE_USD.search(blob):
        return Compensation(hours=hours)
    annual = _parse_annual(blob, hours)
    return Compensation(pay_low=annual[0], pay_high=annual[1], hours=hours)


def parse_job_posting(html: str) -> Compensation:
    """Employer-posted USD pay from schema.org JobPosting JSON-LD. Invents nothing."""
    for posting in _job_postings(html):
        hours = _schema_hours(posting)
        pay_low, pay_high = _schema_salary(posting.get("baseSalary"), hours)
        if pay_low is None and pay_high is None and hours is None:
            remote = _schema_remote(posting)
            company = _schema_company(posting)
            if remote is None and not company:
                continue
            return Compensation(hours=hours, remote=remote, company=company, title=_text(posting.get("title")))
        return Compensation(
            pay_low=pay_low,
            pay_high=pay_high,
            hours=hours,
            remote=_schema_remote(posting),
            company=_schema_company(posting),
            title=_text(posting.get("title")),
        )
    return Compensation()


_LEVER_UNIT = {
    "per-year-salary": "YEAR",
    "year": "YEAR",
    "per-month-salary": "MONTH",
    "month": "MONTH",
    "per-week-salary": "WEEK",
    "week": "WEEK",
    "per-day-wage": "DAY",
    "day": "DAY",
    "per-hour-wage": "HOUR",
    "hour": "HOUR",
}


def ats_json_url(url: str) -> str | None:
    """Public board JSON for a canonical Greenhouse, Lever, or Ashby listing."""
    parts = urlsplit(canonicalize_url(url))
    host = parts.hostname or ""
    segs = [p for p in parts.path.split("/") if p]
    if host.endswith("greenhouse.io") and len(segs) >= 3 and segs[-2] == "jobs":
        board, job_id = segs[0], segs[-1]
        if job_id.isdigit():
            return (
                f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
                "?pay_transparency=true"
            )
    if host.endswith("lever.co") and len(segs) >= 2:
        api = "api.eu.lever.co" if ".eu." in host else "api.lever.co"
        return f"https://{api}/v0/postings/{segs[0]}/{segs[1]}?mode=json"
    if host == "jobs.ashbyhq.com" and len(segs) >= 2:
        return f"https://api.ashbyhq.com/posting-api/job-board/{segs[0]}?includeCompensation=true"
    return None


def parse_ats_json(url: str, payload) -> Compensation:
    """Employer-posted USD pay from an ATS board JSON payload. Invents nothing."""
    host = (urlsplit(canonicalize_url(url)).hostname or "")
    if host.endswith("greenhouse.io"):
        return _greenhouse_pay(payload)
    if host.endswith("lever.co"):
        return _lever_pay(payload)
    if host == "jobs.ashbyhq.com":
        segs = [p for p in urlsplit(canonicalize_url(url)).path.split("/") if p]
        job_id = segs[1] if len(segs) >= 2 else ""
        return _ashby_pay(payload, job_id)
    return Compensation()


def is_search_serp(url: str) -> bool:
    """True for job-board search pages, not individual listings."""
    parts = urlsplit(canonicalize_url(url))
    host = parts.hostname or ""
    path = (parts.path or "").lower()
    if host.endswith("indeed.com"):
        if "/viewjob" in path or "/rc/clk" in path or "/pagead/" in path:
            return False
        return "/q-" in path or path.endswith("-jobs.html") or path.rstrip("/") == "/jobs"
    if host.endswith("linkedin.com"):
        return "/jobs/search" in path
    if host.endswith("ziprecruiter.com"):
        return "jobs-search" in path or path.rstrip("/") == "/jobs"
    if host.endswith("google.com") or host.endswith("duckduckgo.com") or host.endswith("bing.com"):
        return True
    return False


def _greenhouse_pay(payload) -> Compensation:
    if not isinstance(payload, dict):
        return Compensation()
    job = payload.get("job") if isinstance(payload.get("job"), dict) else payload
    title = _text(job.get("title"))
    remote = None
    offices = job.get("offices") or []
    if isinstance(offices, list):
        names = " ".join(_text(o.get("name") if isinstance(o, dict) else o) for o in offices)
        if "remote" in names.lower():
            remote = True
    usd = []
    for row in job.get("pay_input_ranges") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("currency_type") or "USD").upper() not in _USD:
            continue
        low = _cents(row.get("min_cents"))
        high = _cents(row.get("max_cents"))
        if low is None and high is None:
            continue
        usd.append((low or high, high or low))
    if not usd:
        return Compensation(remote=remote, title=title or None)
    lows, highs = zip(*usd)
    annual = _clamp_annual(min(lows), max(highs))
    if not annual:
        return Compensation(remote=remote, title=title or None)
    return Compensation(pay_low=annual[0], pay_high=annual[1], remote=remote, title=title or None)


def _cents(value) -> float | None:
    amount = _number(value)
    if amount is None:
        return None
    return amount / 100.0


def _lever_pay(payload) -> Compensation:
    if not isinstance(payload, dict):
        return Compensation()
    title = _text(payload.get("text") or payload.get("title"))
    company = _text((payload.get("categories") or {}).get("team")) or None
    workplace = str(payload.get("workplaceType") or "").upper()
    remote = True if workplace in {"REMOTE", "TELECOMMUTE"} else None
    sr = payload.get("salaryRange")
    if not isinstance(sr, dict):
        return Compensation(remote=remote, company=company, title=title or None)
    if str(sr.get("currency") or "USD").upper() not in _USD:
        return Compensation(remote=remote, company=company, title=title or None)
    unit = _LEVER_UNIT.get(str(sr.get("interval") or "per-year-salary").lower(), "YEAR")
    low, high = _number(sr.get("min")), _number(sr.get("max"))
    if low is None and high is None:
        return Compensation(remote=remote, company=company, title=title or None)
    annual = _to_annual(low if low is not None else high, high if high is not None else low, unit, None)
    if not annual:
        return Compensation(remote=remote, company=company, title=title or None)
    return Compensation(
        pay_low=annual[0], pay_high=annual[1], remote=remote, company=company, title=title or None
    )


def _ashby_pay(payload, job_id: str) -> Compensation:
    jobs = []
    if isinstance(payload, dict):
        jobs = payload.get("jobs") or payload.get("jobPostings") or []
        if isinstance(payload.get("id"), str):
            jobs = [payload]
    elif isinstance(payload, list):
        jobs = payload
    match = None
    for job in jobs:
        if isinstance(job, dict) and str(job.get("id") or "") == job_id:
            match = job
            break
    if match is None and len(jobs) == 1 and isinstance(jobs[0], dict):
        match = jobs[0]
    if not isinstance(match, dict):
        return Compensation()
    title = _text(match.get("title"))
    company = _text(match.get("departmentName")) or None
    remote = True if match.get("isRemote") is True else None
    comp = match.get("compensation") if isinstance(match.get("compensation"), dict) else {}
    salary = None
    for row in comp.get("summaryComponents") or []:
        if isinstance(row, dict) and str(row.get("compensationType") or "") == "Salary":
            salary = row
            break
    if salary is None:
        return Compensation(remote=remote, company=company, title=title or None)
    if str(salary.get("currencyCode") or "USD").upper() not in _USD:
        return Compensation(remote=remote, company=company, title=title or None)
    interval = str(salary.get("interval") or "1 YEAR").upper().replace("1 ", "")
    unit = {"YEAR": "YEAR", "MONTH": "MONTH", "WEEK": "WEEK", "DAY": "DAY", "HOUR": "HOUR"}.get(
        interval, "YEAR"
    )
    low, high = _number(salary.get("minValue")), _number(salary.get("maxValue"))
    if low is None and high is None:
        return Compensation(remote=remote, company=company, title=title or None)
    annual = _to_annual(low if low is not None else high, high if high is not None else low, unit, None)
    if not annual:
        return Compensation(remote=remote, company=company, title=title or None)
    return Compensation(
        pay_low=annual[0], pay_high=annual[1], remote=remote, company=company, title=title or None
    )


def canonicalize_url(url: str) -> str:
    """Identity key: https, lowercase host, ATS rewrite, tracking stripped."""
    raw = (url or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = f"https:{raw}"
    parts = urlsplit(raw)
    host = (parts.hostname or "").lower()
    if not host:
        return raw.rstrip("/")
    path = parts.path or "/"
    query = parse_qsl(parts.query, keep_blank_values=True)
    host, path, query = _ats_shape(host, path, query)
    keep = []
    drop_all = host.endswith(
        ("greenhouse.io", "lever.co", "ashbyhq.com", "myworkdayjobs.com")
    )
    for key, value in query:
        low = key.lower()
        if drop_all or low.startswith("utm_") or low in _TRACKING:
            continue
        keep.append((key, value))
    keep.sort(key=lambda kv: kv[0].lower())
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit(("https", host, path, urlencode(keep, doseq=True), ""))


def _scrub_benefits(text: str) -> str:
    def drop(match: re.Match[str]) -> str:
        start, end = match.span()
        left = text[max(0, start - 24) : start]
        right = text[end : end + 24]
        nearer = None
        for found in _BENEFIT.finditer(left):
            nearer = found
        if nearer and not _USD_MARK.search(left[nearer.end() :]):
            return " "
        right_hit = _BENEFIT.search(right)
        if right_hit and not re.search(
            r"(?i)\b(?:plus|and|with|includes?|including)\b", right[: right_hit.start()]
        ):
            return " "
        return match.group(0)

    return _SINGLE.sub(drop, text)


def _parse_hours(text: str) -> int | None:
    match = _HOURS.search(text)
    if not match:
        return None
    hours = int(round(float(match.group(1))))
    return hours if 1 <= hours <= 80 else None


def _money(num: str, thousand: str | None) -> float:
    value = float(num.replace(",", ""))
    if thousand:
        value *= 1000
    return value


def _clamp_annual(low: float, high: float) -> tuple[int, int] | None:
    lo, hi = int(round(min(low, high))), int(round(max(low, high)))
    if 10_000 <= lo <= hi <= 2_000_000:
        return lo, hi
    return None


def _parse_annual(text: str, hours: int | None) -> tuple[int | None, int | None]:
    week = hours or 40
    bare_range = _BARE_USD_RANGE.search(text)
    if bare_range:
        annual = _clamp_annual(int(bare_range.group(1)) * 1000, int(bare_range.group(2)) * 1000)
        if annual:
            return annual[0], annual[1]
    bare = _BARE_USD.search(text)
    if bare:
        annual = _clamp_annual(int(bare.group(1)) * 1000, int(bare.group(1)) * 1000)
        if annual:
            return None, annual[1]
    ranged = _RANGE.search(text)
    if ranged:
        left = _money(ranged.group(1), ranged.group(2))
        right = _money(ranged.group(3), ranged.group(4))
        if ranged.group(2) or ranged.group(4):
            if not ranged.group(2) and left < 1000:
                left *= 1000
            if not ranged.group(4) and right < 1000:
                right *= 1000
        tail = text[ranged.end() : ranged.end() + 16].lstrip()
        if _HOUR_TAIL.match(tail):
            if 10 <= left <= 1000 and 10 <= right <= 1000:
                annual = _clamp_annual(left * week * 50, right * week * 50)
                return (annual[0], annual[1]) if annual else (None, None)
        elif not _period_other(tail):
            if left < 1000 and right < 1000:
                left, right = left * 1000, right * 1000
            annual = _clamp_annual(left, right)
            if annual:
                return annual[0], annual[1]
    for match in _SINGLE.finditer(text):
        amount = _money(match.group(1), match.group(2))
        tail = text[match.end() : match.end() + 16].lstrip()
        if _HOUR_TAIL.match(tail):
            if 10 <= amount <= 1000:
                annual = _clamp_annual(amount * week * 50, amount * week * 50)
                return (None, annual[1]) if annual else (None, None)
            continue
        if _period_other(tail) and not _ANNUAL_TAIL.match(tail):
            continue
        if not match.group(2) and amount < 1000:
            continue
        annual = _clamp_annual(amount, amount)
        if annual:
            return None, annual[1]
    return None, None


def _period_other(tail: str) -> bool:
    return bool(re.match(r"(?i)\s*(?:/|\bper\b)\s*(?:day|wk|week|mo|month)", tail))


def _ats_shape(
    host: str, path: str, query: list[tuple[str, str]]
) -> tuple[str, str, list[tuple[str, str]]]:
    params = {k.lower(): v for k, v in query}
    if host in {"boards.greenhouse.io", "job-boards.greenhouse.io"}:
        host = "job-boards.greenhouse.io"
        if path.rstrip("/").endswith("/embed/job_app") and params.get("for") and params.get(
            "token"
        ):
            path = f"/{params['for']}/jobs/{params['token']}"
        return host, path, []
    if host.endswith("lever.co"):
        trimmed = re.sub(r"(?i)/(?:apply|thanks)/?$", "", path)
        return host, trimmed or path, []
    if host == "jobs.ashbyhq.com":
        trimmed = re.sub(r"(?i)/applications?/?$", "", path)
        return host, trimmed or path, []
    if host.endswith("myworkdayjobs.com"):
        parts = [p for p in path.split("/") if p]
        if parts and re.fullmatch(r"[a-z]{2}-[A-Z]{2}", parts[0]):
            parts = parts[1:]
        return host, "/" + "/".join(parts) if parts else "/", []
    kept = []
    if "gh_jid" in params:
        kept.append(("gh_jid", params["gh_jid"]))
    if "ashby_jid" in params:
        kept.append(("ashby_jid", params["ashby_jid"]))
    return host, path, kept if kept else query


def _job_postings(html: str):
    for match in _LD_SCRIPT.finditer(html or ""):
        raw = re.sub(r"^\s*<!--|-->\s*$", "", match.group(1)).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        yield from _walk_postings(data)


def _walk_postings(node):
    if isinstance(node, list):
        for item in node:
            yield from _walk_postings(item)
        return
    if not isinstance(node, dict):
        return
    types = node.get("@type")
    names = types if isinstance(types, list) else [types]
    cleaned = [str(n).rsplit("/", 1)[-1] for n in names if n]
    if "JobPosting" in cleaned:
        yield node
    if "@graph" in node:
        yield from _walk_postings(node["@graph"])


def _schema_hours(posting: dict) -> int | None:
    raw = posting.get("workHours")
    if raw is None:
        return None
    text = _text(raw)
    if not text:
        return None
    hours = _parse_hours(text)
    if hours:
        return hours
    if text.isdigit():
        value = int(text)
        return value if 1 <= value <= 80 else None
    match = _SCHEMA_HOURS.search(text)
    if not match:
        return None
    value = int(round(float(match.group(1))))
    return value if 1 <= value <= 80 else None


def _schema_salary(block, hours: int | None) -> tuple[int | None, int | None]:
    if isinstance(block, list) and block:
        block = block[0]
    if not isinstance(block, dict):
        return None, None
    currency = str(block.get("currency") or "").upper().replace("$", "")
    if currency not in _USD:
        return None, None
    value = block.get("value")
    if isinstance(value, dict):
        unit = str(value.get("unitText") or "YEAR").upper()
        low = _number(value.get("minValue", value.get("value")))
        high = _number(value.get("maxValue", value.get("value")))
    else:
        unit = "YEAR"
        low = high = _number(value)
    if low is None and high is None:
        return None, None
    if low is None:
        low = high
    if high is None:
        high = low
    annual = _to_annual(low, high, unit, hours)
    return (annual[0], annual[1]) if annual else (None, None)


def _to_annual(
    low: float, high: float, unit: str, hours: int | None
) -> tuple[int, int] | None:
    week = hours or 40
    factor = {
        "YEAR": 1,
        "MONTH": 12,
        "WEEK": 50,
        "DAY": 250,
        "HOUR": week * 50,
    }.get(unit)
    if factor is None:
        return None
    return _clamp_annual(low * factor, high * factor)


def _schema_remote(posting: dict) -> bool | None:
    location_type = _text(posting.get("jobLocationType")).upper()
    if "TELECOMMUTE" in location_type:
        return True
    return None


def _schema_company(posting: dict) -> str | None:
    org = posting.get("hiringOrganization")
    if isinstance(org, list) and org:
        org = org[0]
    if isinstance(org, dict):
        return _text(org.get("name")) or None
    return _text(org) or None


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return _text(value[0]) if value else ""
    if isinstance(value, dict):
        return str(value.get("name") or value.get("value") or "")
    return str(value).strip()


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None
