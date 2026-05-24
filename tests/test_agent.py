import asyncio
import types

from src.agent import _parse, _rank, agent_run
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


# --- parsing Hermes' reply -----------------------------------------------


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


# --- end to end: Hermes brain mocked, real parse + rank ------------------


def _fake_client(content: str):
    """Stub the AsyncOpenAI surface agent_run touches: .chat.completions.create."""

    async def create(**kwargs):
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_agent_run_parses_and_ranks_hermes_reply(monkeypatch):
    reply = (
        '{"searches": ["remote ml contract", "ai grants"],'
        ' "opportunities": ['
        '   {"title": "Cheap", "url": "u1", "pay": 100000, "hours_per_week": 40},'
        '   {"title": "Lush", "url": "u2", "pay": 200000, "hours_per_week": 20}'
        ' ]}'
    )
    monkeypatch.setattr("src.agent._client", lambda: _fake_client(reply))

    run = asyncio.run(agent_run("find me work"))

    assert run.searches == ["remote ml contract", "ai grants"]
    assert [o.title for o in run.ranked] == ["Lush", "Cheap"]  # ranked by $/hr
    assert run.ranked[0].score() == 200.0
