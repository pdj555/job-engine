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


def test_payload_known_rate_is_not_imputed():
    opp = Opportunity(title="x", url="https://u", pay_high=100_000, hours_per_week=20, remote=False)
    row = _payload([opp])["results"][0]
    assert row["dollars_per_hour"] == 100.0
    assert row["refined_rate"] == 100.0
    assert row["rate_imputed"] is False
    assert row["score"] == 70.0


def test_search_http_returns_refined_fields(monkeypatch):
    async def fake_find(query, limit=20):
        return [Opportunity(title="x", url="https://u", pay_high=100_000)]

    monkeypatch.setattr(routes.engine, "find", fake_find)
    res = TestClient(routes.app).get("/search", params={"q": "ai"})
    assert res.status_code == 200
    row = res.json()["results"][0]
    assert row["refined_rate"] == 50.0
    assert row["rate_imputed"] is True
    assert row["dollars_per_hour"] is None
