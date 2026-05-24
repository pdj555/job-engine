"""Autonomous opportunity agent.

Hermes Agent (Nous Research — github.com/NousResearch/hermes-agent) is the brain.
Given a goal it plans and runs its OWN web research with its built-in toolset, then
returns structured opportunities. We reach it over its OpenAI-compatible API server
and rank the results deterministically: Hermes decides WHAT, Opportunity.score()
owns the $/hour. See docs/AGENT.md.
"""

import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from config.settings import settings
from src.models import Opportunity

PROMPT = """You are an autonomous opportunity scout. Goal: {query}

Research the open web across remote roles, contracts/freelance, grants, and
cofounder/equity. Then return ONLY this JSON, no prose:

{{
  "searches": ["the angles you actually researched"],
  "opportunities": [
    {{"title": "...", "url": "https://...", "company": "...",
      "pay": 180000, "hours_per_week": 40, "remote": true}}
  ]
}}

pay = annual USD number (null if unknown). hours_per_week = number (null if unknown)."""


@dataclass
class AgentRun:
    """What a run produced: the angles Hermes researched + the ranked shortlist."""

    searches: list[str] = field(default_factory=list)
    ranked: list[Opportunity] = field(default_factory=list)


def _client() -> AsyncOpenAI:
    """Hermes Agent over its OpenAI-compatible API server."""
    return AsyncOpenAI(base_url=settings.hermes_base_url, api_key=settings.hermes_api_key)


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
    """Pull the JSON out of Hermes' reply, tolerating wrapping prose. Always a dict."""
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


async def agent_run(query: str, limit: int = 20) -> AgentRun:
    """Hermes autonomously researches the goal; we rank what it returns by $/hour."""
    resp = await _client().chat.completions.create(
        model=settings.hermes_model,
        messages=[{"role": "user", "content": PROMPT.format(query=query)}],
    )
    data = _parse(resp.choices[0].message.content or "")
    ranked = _rank(data.get("opportunities", []))[:limit]
    return AgentRun(searches=data.get("searches", []), ranked=ranked)


async def agent_find(query: str, limit: int = 20) -> list[Opportunity]:
    """Autonomously find + rank opportunities for a goal, via the Hermes Agent brain."""
    return (await agent_run(query, limit)).ranked
