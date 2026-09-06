"""Posted pay/hours from listing text, plus ATS URL identity."""

from __future__ import annotations

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


@dataclass(frozen=True)
class Compensation:
    pay_low: int | None = None
    pay_high: int | None = None
    hours: int | None = None

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
