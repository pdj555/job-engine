from fastapi.testclient import TestClient

from src.api import routes
from src.api.routes import _payload
from src.models import Opportunity


def test_payload_exposes_refined_rate_when_hours_missing():
    opp = Opportunity(title="x", url="https://u", pay_high=100_000)
    row = _payload([opp])["results"][0]
    assert row["dollars_per_hour"] is None
    assert row["refined_rate"] == 50.0
    assert row["rate_imputed"] is True
    assert row["score"] == 50.0


def test_payload_exposes_pay_range():
    opp = Opportunity(title="x", url="https://u", pay_low=143_000, pay_high=197_000)
    row = _payload([opp])["results"][0]
    assert row["pay"] == 197_000
    assert row["pay_low"] == 143_000
    assert row["pay_high"] == 197_000


def test_payload_known_rate_is_not_imputed():
    opp = Opportunity(title="x", url="https://u", pay_high=100_000, hours_per_week=20, remote=False)
    row = _payload([opp])["results"][0]
    assert row["dollars_per_hour"] == 100.0
    assert row["refined_rate"] == 100.0
    assert row["rate_imputed"] is False
    assert row["score"] == 70.0


def test_health_reports_hermes_configured():
    res = TestClient(routes.app).get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["hermes_configured"] is True
    assert body["apis"]["hermes"] is True


def test_agent_timeout_returns_504(monkeypatch):
    from openai import APITimeoutError

    async def boom(*_args, **_kwargs):
        raise APITimeoutError(request=None)

    monkeypatch.setattr("src.agent.agent_run", boom)
    res = TestClient(routes.app).post("/agent", json={"q": "ai"})
    assert res.status_code == 504


def test_agent_other_error_returns_503(monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr("src.agent.agent_run", boom)
    res = TestClient(routes.app).post("/agent", json={"q": "ai"})
    assert res.status_code == 503


def test_search_http_returns_refined_fields(monkeypatch):
    async def fake_find(query, limit=20):
        return [
            Opportunity(title="x", url="https://u", pay_low=143_000, pay_high=197_000)
        ]

    monkeypatch.setattr(routes.engine, "find", fake_find)
    res = TestClient(routes.app).get("/search", params={"q": "ai"})
    assert res.status_code == 200
    row = res.json()["results"][0]
    assert row["refined_rate"] == 98.5
    assert row["rate_imputed"] is True
    assert row["dollars_per_hour"] is None
    assert row["pay"] == 197_000
    assert row["pay_low"] == 143_000
    assert row["pay_high"] == 197_000


def test_cli_pay_label_shows_range():
    from src.cli import _pay_label

    ranged = Opportunity(title="x", url="https://u", pay_low=143_000, pay_high=197_000)
    assert _pay_label(ranged) == "$143,000–$197,000"
    single = Opportunity(title="x", url="https://u", pay_high=90_000)
    assert _pay_label(single) == "$90,000"
    empty = Opportunity(title="x", url="https://u")
    assert _pay_label(empty) == "?"
