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
