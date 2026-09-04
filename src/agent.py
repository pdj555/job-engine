"""Autonomous opportunity agent.

The OpenAI Agents SDK is the brain when OPENAI_API_KEY is set: it plans web
searches via a tool, then returns candidates. With no key, the same Engine
search path as `find` runs so agent mode never depends on an extra process.
Opportunity.score() owns the $/hour either way. See docs/AGENT.md.
"""

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from config.settings import settings
from src.engine import Engine
from src.models import Opportunity

INSTRUCTIONS = """You are an opportunity scout. Use search_web to research the
open web across remote roles, contracts/freelance, grants, and cofounder/equity.
Then return searches you actually ran plus opportunities you found.

pay = annual USD (null if unknown). hours_per_week = number (null if unknown).
Only include http(s) listing URLs, never search-result homepages."""


@dataclass
class AgentRun:
    """Angles researched + the ranked shortlist."""

    searches: list[str] = field(default_factory=list)
    ranked: list[Opportunity] = field(default_factory=list)


class ScoutHit(BaseModel):
    title: str = "Unknown"
    url: str
    company: str | None = None
    pay: int | None = None
    hours_per_week: int | None = None
    remote: bool = True


class ScoutResult(BaseModel):
    searches: list[str] = Field(default_factory=list)
    opportunities: list[ScoutHit] = Field(default_factory=list)


def _angles(query: str) -> list[str]:
    return [
        f"{query} remote job hiring",
        f"{query} freelance contract",
        f"{query} grant funding opportunity",
        f"{query} startup equity cofounder",
    ]


def _rank(items: list[dict]) -> list[Opportunity]:
    """Build Opportunity models and order by $/hour (highest first). Deterministic."""
    opportunities = [
        Opportunity(
            title=o.get("title", "Unknown"),
            url=o["url"],
            company=o.get("company"),
            pay_high=o.get("pay"),
            hours_per_week=o.get("hours_per_week"),
            remote=o.get("remote", True),
            source="agent",
        )
        for o in items
        if o.get("url")
    ]
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


def _from_scout(out: ScoutResult, searches: list[str], limit: int) -> AgentRun:
    ranked = _rank([o.model_dump() for o in out.opportunities])[:limit]
    return AgentRun(searches=out.searches or searches, ranked=ranked)


async def _search_run(query: str, limit: int) -> AgentRun:
    """Open-web research when no LLM is configured."""
    ranked = await Engine().find(query, limit)
    return AgentRun(searches=_angles(query), ranked=ranked)


async def _sdk_run(query: str, limit: int) -> AgentRun:
    from agents import Agent, Runner, function_tool

    engine = Engine()
    searches: list[str] = []

    @function_tool
    async def search_web(q: str) -> str:
        """Search the open web for roles, contracts, grants, or equity."""
        searches.append(q)
        hits = await engine.search_web(q)
        return json.dumps(hits[:10])

    agent = Agent(
        name="OpportunityScout",
        instructions=INSTRUCTIONS,
        tools=[search_web],
        output_type=ScoutResult,
        model=settings.fast_model,
    )
    result = await Runner.run(agent, query, max_turns=8)
    out = result.final_output
    if isinstance(out, ScoutResult):
        return _from_scout(out, searches, limit)
    data = _parse(str(out or ""))
    ranked = _rank(data.get("opportunities", []))[:limit]
    return AgentRun(searches=data.get("searches") or searches, ranked=ranked)


async def agent_run(query: str, limit: int = 20) -> AgentRun:
    """Research the goal; rank what comes back by $/hour."""
    if settings.openai_api_key:
        return await _sdk_run(query, limit)
    return await _search_run(query, limit)


async def agent_find(query: str, limit: int = 20) -> list[Opportunity]:
    """Autonomously find + rank opportunities for a goal."""
    return (await agent_run(query, limit)).ranked
