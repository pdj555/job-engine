import asyncio
import types

from src.agent import ScoutHit, ScoutResult, _parse, _rank, agent_run
from src.engine import Engine
from src.models import Opportunity


# --- deterministic ranking (the math the brain is never trusted with) ---


def test_rank_orders_by_dollars_per_hour():
    items = [
        {"title": "Low", "url": "u1", "pay": 100_000, "hours_per_week": 40},   # $50/hr
        {"title": "High", "url": "u2", "pay": 200_000, "hours_per_week": 20},  # $200/hr
    ]
    assert [o.title for o in _rank(items)] == ["High", "Low"]


def test_rank_skips_items_without_url():
    assert _rank([{"title": "no url", "pay": 100_000, "hours_per_week": 10}]) == []


def test_rank_builds_opportunity_models_with_fields():
    ranked = _rank(
        [{"title": "X", "url": "u", "company": "Acme", "pay": 120_000,
          "hours_per_week": 30, "remote": False}]
    )
    assert isinstance(ranked[0], Opportunity)
    assert ranked[0].company == "Acme"
    assert ranked[0].remote is False


# --- parsing model replies -----------------------------------------------


def test_parse_clean_object():
    out = _parse('{"searches": ["a"], "opportunities": [{"url": "u"}]}')
    assert out["searches"] == ["a"]
    assert out["opportunities"] == [{"url": "u"}]


def test_parse_tolerates_wrapping_prose():
    out = _parse('Here you go:\n{"opportunities": [{"url": "u"}]}\nThanks!')
    assert out["opportunities"] == [{"url": "u"}]


def test_parse_bare_array_becomes_opportunities():
    assert _parse('[{"url": "u"}]') == {"opportunities": [{"url": "u"}]}


def test_parse_garbage_is_empty_dict():
    assert _parse("not json at all") == {}


# --- agent_run: Engine fallback and SDK ----------------------------------


def test_agent_run_without_openai_uses_engine(monkeypatch):
    monkeypatch.setattr("src.agent.settings.openai_api_key", "")

    async def fake_find(self, query, limit=20):
        return [
            Opportunity(title="Lush", url="u2", pay_high=200_000, hours_per_week=20),
            Opportunity(title="Cheap", url="u1", pay_high=100_000, hours_per_week=40),
        ]

    monkeypatch.setattr(Engine, "find", fake_find)

    run = asyncio.run(agent_run("ml contract"))

    assert [o.title for o in run.ranked] == ["Lush", "Cheap"]
    assert run.ranked[0].score() == 200.0
    assert any("ml contract" in s for s in run.searches)


def test_agent_run_with_openai_uses_agents_sdk(monkeypatch):
    monkeypatch.setattr("src.agent.settings.openai_api_key", "sk-test")

    out = ScoutResult(
        searches=["remote ml contract", "ai grants"],
        opportunities=[
            ScoutHit(title="Cheap", url="u1", pay=100_000, hours_per_week=40),
            ScoutHit(title="Lush", url="u2", pay=200_000, hours_per_week=20),
        ],
    )

    async def fake_run(agent, input, max_turns=None):
        return types.SimpleNamespace(final_output=out)

    monkeypatch.setattr("agents.Runner.run", fake_run)

    run = asyncio.run(agent_run("find me work"))

    assert run.searches == ["remote ml contract", "ai grants"]
    assert [o.title for o in run.ranked] == ["Lush", "Cheap"]
    assert run.ranked[0].score() == 200.0
