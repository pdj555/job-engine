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
