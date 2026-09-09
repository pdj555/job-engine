"""Autonomous opportunity agent.

The OpenAI Agents SDK is the brain when OPENAI_API_KEY is set: it plans web
searches via a tool, then returns candidates. With no key, the same Engine
search path as `find` runs so agent mode never depends on an extra process.
Opportunity.score() owns the $/hour either way. See docs/AGENT.md.
"""

import json
from dataclasses import dataclass, field
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from config.settings import settings
from src.compensation import canonicalize_url
from src.engine import Engine, opportunity_from_raw, search_angles
from src.models import Opportunity

INSTRUCTIONS = """You are an opportunity scout. Use search_web to research the
open web across remote roles, contracts/freelance, grants, cofounder/equity,
and ATS boards (Greenhouse, Lever, Ashby, Workday). Prefer listing URLs on
those hosts — they often expose posted pay.

Use read_listing on promising search hits to confirm posted pay before
including them. Then return searches you actually ran plus opportunities.

Copy title, url, company, remote from search hits. Do not invent pay or hours.
Only include http(s) listing URLs that appeared in search_web results.
Copy those URLs exactly."""


@dataclass
class AgentRun:
    """Angles researched + the ranked shortlist."""

    searches: list[str] = field(default_factory=list)
    ranked: list[Opportunity] = field(default_factory=list)


class ScoutHit(BaseModel):
    title: str = "Unknown"
    url: str
    company: str | None = None
    description: str | None = None
    remote: bool = True


class ScoutResult(BaseModel):
    searches: list[str] = Field(default_factory=list)
    opportunities: list[ScoutHit] = Field(default_factory=list)


def _angles(query: str) -> list[str]:
    return search_angles(query)


def _http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _ground_to_search_hits(items: list[dict], search_hits: list[dict]) -> list[dict]:
    """Keep only opportunities whose URL appeared in search_web hits."""
    by_url = {canonicalize_url(r["url"]): r for r in search_hits if r.get("url")}
    grounded: list[dict] = []
    seen: set[str] = set()
    for item in items:
        url = item.get("url") or ""
        if not _http_url(url):
            continue
        key = canonicalize_url(url)
        raw = by_url.get(key)
        if not raw or key in seen:
            continue
        seen.add(key)
        grounded.append(
            {
                "title": raw.get("title") or item.get("title") or "Unknown",
                "url": raw["url"],
                "company": item.get("company") or raw.get("company"),
                "description": raw.get("description") or "",
                "remote": item.get("remote", raw.get("remote", True)),
                "source": "agent",
            }
        )
    return grounded


def _rank(items: list[dict], *, from_search: bool = True) -> list[Opportunity]:
    """Build Opportunity models and order by $/hour (highest first). Deterministic."""
    opportunities = []
    for o in items:
        parsed = opportunity_from_raw(
            {
                "title": o.get("title") or "Unknown",
                "url": o.get("url") or "",
                "company": o.get("company"),
                "description": o.get("description") or "",
                "remote": o.get("remote", True),
                "source": o.get("source") or "agent",
            },
            listing_text=None if from_search else "",
        )
        if parsed:
            opportunities.append(parsed)
    return sorted(opportunities, key=lambda o: o.score(), reverse=True)


def _parse(content: str) -> dict:
    """Pull JSON out of a model reply, tolerating wrapping prose. Always a dict."""
    raw = None
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start, end = content.find(open_c), content.rfind(close_c) + 1
            if 0 <= start < end:
                try:
                    raw = json.loads(content[start:end])
                    break
                except json.JSONDecodeError:
                    continue
    if isinstance(raw, list):
        return {"opportunities": raw}
    return raw if isinstance(raw, dict) else {}


def _from_scout(
    out: ScoutResult,
    searches: list[str],
    limit: int,
    search_hits: list[dict],
) -> AgentRun:
    items = [o.model_dump() for o in out.opportunities]
    if search_hits:
        items = _ground_to_search_hits(items, search_hits)
        ranked = _rank(items, from_search=True)
    else:
        ranked = _rank(items, from_search=False)
    return AgentRun(searches=out.searches or searches, ranked=ranked[:limit])


async def _search_run(query: str, limit: int) -> AgentRun:
    """Open-web research when no LLM is configured."""
    ranked = await Engine().find(query, limit)
    return AgentRun(searches=_angles(query), ranked=ranked)


async def _sdk_run(query: str, limit: int) -> AgentRun:
    from agents import Agent, Runner, function_tool

    engine = Engine()
    searches: list[str] = []
    search_hits: list[dict] = []

    @function_tool
    async def search_web(q: str) -> str:
        """Search the open web for roles, contracts, grants, equity, or ATS boards."""
        searches.append(q)
        hits = await engine.search_web(q)
        search_hits.extend(hits[:10])
        return json.dumps(hits[:10])

    @function_tool
    async def read_listing(url: str) -> str:
        """Fetch posted pay/hours for a listing URL from ATS JSON or JobPosting schema."""
        return json.dumps(await engine.read_listing(url))

    agent = Agent(
        name="OpportunityScout",
        instructions=INSTRUCTIONS,
        tools=[search_web, read_listing],
        output_type=ScoutResult,
        model=settings.fast_model,
    )
    result = await Runner.run(agent, query, max_turns=10)
    out = result.final_output
    if isinstance(out, ScoutResult):
        run = _from_scout(out, searches, limit, search_hits)
    else:
        data = _parse(str(out or ""))
        items = data.get("opportunities", [])
        if search_hits:
            items = _ground_to_search_hits(items, search_hits)
            ranked = _rank(items, from_search=True)
        else:
            ranked = _rank(items, from_search=False)
        run = AgentRun(searches=data.get("searches") or searches, ranked=ranked[:limit])
    await engine.enrich(run.ranked)
    run.ranked = sorted(run.ranked, key=lambda o: o.score(), reverse=True)[:limit]
    return run


async def agent_run(query: str, limit: int = 20) -> AgentRun:
    """Research the goal; rank what comes back by $/hour."""
    if settings.openai_api_key:
        return await _sdk_run(query, limit)
    return await _search_run(query, limit)


async def agent_find(query: str, limit: int = 20) -> list[Opportunity]:
    """Autonomously find + rank opportunities for a goal."""
    return (await agent_run(query, limit)).ranked
