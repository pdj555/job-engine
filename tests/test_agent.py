import asyncio
import types

from src.agent import ScoutHit, ScoutResult, _ground_to_search_hits, _parse, _rank, agent_run
from src.engine import Engine
from src.models import Opportunity


def test_rank_orders_by_dollars_per_hour():
    items = [
        {"title": "Low $100k 40 hours/week", "url": "u1"},
        {"title": "High $200k 20 hours/week", "url": "u2"},
    ]
    assert [o.title for o in _rank(items)] == ["High $200k 20 hours/week", "Low $100k 40 hours/week"]


def test_rank_skips_items_without_url():
    assert _rank([{"title": "no url", "pay": 100_000, "hours_per_week": 10}]) == []


def test_rank_ignores_invented_pay_fields():
    ranked = _rank(
        [
            {"title": "Stated $200k", "url": "https://jobs.lever.co/acme/abc/apply"},
            {
                "title": "Silent",
                "url": "u2",
                "pay": 100_000,
                "hours_per_week": 40,
                "description": "no numbers here",
            },
        ]
    )
    assert ranked[0].url == "https://jobs.lever.co/acme/abc"
    assert ranked[0].pay_source == "posted"
    assert ranked[1].pay is None
    assert ranked[1].pay_source is None
    assert ranked[1].score() == 0


def test_rank_reads_pay_from_description():
    ranked = _rank(
        [{"title": "Engineer", "url": "u", "description": "$180k · 40 hours/week"}]
    )
    assert ranked[0].pay == 180_000
    assert ranked[0].hours_per_week == 40
    assert ranked[0].pay_source == "posted"


def test_rank_builds_opportunity_models_with_fields():
    ranked = _rank(
        [{"title": "X $120k 30 hours/week", "url": "u", "company": "Acme", "remote": False}]
    )
    assert isinstance(ranked[0], Opportunity)
    assert ranked[0].company == "Acme"
    assert ranked[0].remote is False
    assert ranked[0].pay == 120_000
    assert ranked[0].hours_per_week == 30


def test_ground_drops_urls_that_were_not_searched():
    hits = [{"url": "https://jobs.lever.co/acme/real", "title": "Real", "description": "$90k"}]
    items = [
        {"title": "Fake", "url": "https://evil.example/fake"},
        {"title": "Invented $500k", "url": "https://jobs.lever.co/acme/real/apply", "description": "$500k"},
    ]
    grounded = _ground_to_search_hits(items, hits)
    assert [g["url"] for g in grounded] == ["https://jobs.lever.co/acme/real"]
    assert grounded[0]["title"] == "Real"
    assert grounded[0]["description"] == "$90k"
    ranked = _rank(grounded)
    assert ranked[0].pay == 90_000


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
            ScoutHit(title="Cheap $100k 40 hours/week", url="u1"),
            ScoutHit(title="Lush $200k 20 hours/week", url="u2"),
        ],
    )

    async def fake_run(agent, input, max_turns=None):
        return types.SimpleNamespace(final_output=out)

    async def no_enrich(self, opportunities):
        return None

    monkeypatch.setattr("agents.Runner.run", fake_run)
    monkeypatch.setattr(Engine, "enrich", no_enrich)

    run = asyncio.run(agent_run("find me work"))

    assert run.searches == ["remote ml contract", "ai grants"]
    assert [o.title for o in run.ranked] == [
        "Lush $200k 20 hours/week",
        "Cheap $100k 40 hours/week",
    ]
    assert run.ranked[0].score() == 200.0
