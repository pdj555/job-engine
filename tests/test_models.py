from src.models import Opportunity


def test_dollars_per_hour():
    opp = Opportunity(
        title="ML Engineer",
        url="https://example.com/job",
        pay_high=100_000,
        hours_per_week=40,
        remote=True,
    )
    assert opp.dollars_per_hour == 50.0


def test_office_penalty():
    remote = Opportunity(
        title="Remote",
        url="https://example.com/remote",
        pay_high=100_000,
        hours_per_week=40,
        remote=True,
    )
    office = Opportunity(
        title="Office",
        url="https://example.com/office",
        pay_high=100_000,
        hours_per_week=40,
        remote=False,
    )
    assert office.score() < remote.score()


def test_pay_prefers_high_over_low():
    opp = Opportunity(title="x", url="u", pay_low=80_000, pay_high=120_000)
    assert opp.pay == 120_000


def test_dollars_per_hour_none_when_missing_data():
    no_pay = Opportunity(title="x", url="u", hours_per_week=40)
    no_hours = Opportunity(title="x", url="u", pay_high=100_000)
    assert no_pay.dollars_per_hour is None
    assert no_hours.dollars_per_hour is None


def test_score_unknown_pay_is_zero():
    opp = Opportunity(title="x", url="u", hours_per_week=20)
    assert opp.score() == 0


def test_score_unknown_hours_assumes_full_time():
    # pay 100k, no hours -> assumes 40h/wk -> 100000 / (40*50) = 50.0
    opp = Opportunity(title="x", url="u", pay_high=100_000)
    assert opp.score() == 50.0
