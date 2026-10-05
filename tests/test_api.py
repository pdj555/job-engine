from src.api.routes import _payload, health
from src.models import Opportunity


def test_payload_exposes_refined_rate_when_hours_missing():
    opp = Opportunity(title="x", url="https://u", pay_high=100_000)
    row = _payload([opp])["results"][0]
    assert row["dollars_per_hour"] is None
    assert row["refined_rate"] == 50.0
    assert row["rate_imputed"] is True
    assert row["score"] == 50.0
    assert row["pay_source"] is None


def test_payload_known_rate_is_not_imputed():
    opp = Opportunity(
        title="x",
        url="https://u",
        pay_high=100_000,
        hours_per_week=20,
        remote=False,
        pay_source="schema",
        hours_source="schema",
    )
    row = _payload([opp])["results"][0]
    assert row["dollars_per_hour"] == 100.0
    assert row["refined_rate"] == 100.0
    assert row["rate_imputed"] is False
    assert row["score"] == 70.0
    assert row["pay_source"] == "schema"
    assert row["hours_source"] == "schema"


def test_payload_includes_honesty_fields():
    opp = Opportunity(title="x", url="https://u", pay_high=80_000, pay_source="posted")
    body = _payload([opp])
    assert set(body["results"][0]) >= {
        "pay",
        "pay_low",
        "pay_high",
        "refined_rate",
        "rate_imputed",
        "dollars_per_hour",
        "score",
        "pay_source",
        "hours_source",
    }


def test_payload_pay_is_range_midpoint():
    opp = Opportunity(title="x", url="https://u", pay_low=120_000, pay_high=180_000, hours_per_week=40)
    row = _payload([opp])["results"][0]
    assert row["pay"] == 150_000
    assert row["pay_low"] == 120_000
    assert row["pay_high"] == 180_000
    assert row["refined_rate"] == 75.0
    assert row["score"] == 75.0


def test_health_reports_agent_ready():
    import asyncio

    body = asyncio.run(health())
    assert body["status"] == "ok"
    assert "agent_ready" in body
    assert body["agent_ready"] is bool(body["apis"]["openai"])
