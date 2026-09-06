from src.api.routes import _payload
from src.models import Opportunity


def test_payload_exposes_pay_provenance():
    posted = Opportunity(
        title="Listed",
        url="https://example.com/a",
        pay_high=180_000,
        hours_per_week=40,
        pay_source="posted",
        hours_source="posted",
    )
    thin = Opportunity(title="Thin", url="https://example.com/b")
    body = _payload([posted, thin])
    first, second = body["results"]
    assert first["pay"] == 180_000
    assert first["pay_source"] == "posted"
    assert first["hours_source"] == "posted"
    assert first["dollars_per_hour"] == 90.0
    assert second["pay"] is None
    assert second["pay_source"] is None
    assert second["score"] == 0
