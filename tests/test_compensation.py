from src.compensation import canonicalize_url, parse_compensation


def test_parse_annual_range_and_single():
    ranged = parse_compensation("Staff Engineer $120k-$150k")
    assert (ranged.pay_low, ranged.pay_high) == (120_000, 150_000)
    assert ranged.posted

    single = parse_compensation("Role pays $180,000 a year")
    assert single.pay_low is None
    assert single.pay_high == 180_000


def test_parse_shorthand_range_applies_k_to_both():
    parsed = parse_compensation("$120-150k remote")
    assert (parsed.pay_low, parsed.pay_high) == (120_000, 150_000)


def test_parse_hourly_annualizes_at_stated_or_40h():
    stated = parse_compensation("$75/hr · 20 hours/week")
    assert stated.hours == 20
    assert stated.pay_high == 75_000

    assumed = parse_compensation("$75 per hour")
    assert assumed.hours is None
    assert assumed.pay_high == 150_000


def test_parse_hourly_range():
    parsed = parse_compensation("$50-$75/hr")
    assert (parsed.pay_low, parsed.pay_high) == (100_000, 150_000)


def test_parse_refuses_benefit_and_foreign_amounts():
    assert parse_compensation("401k match $10,000 plus $180k salary").pay_high == 180_000
    assert parse_compensation("$6,000 health insurance stipend").pay_high is None
    assert parse_compensation("Up to $250k OTE").pay_high is None
    assert parse_compensation("£120,000 plus £10k bonus").pay_high is None
    assert parse_compensation("Pay is $180k (CAD $240k)").pay_high == 180_000


def test_parse_bare_k_requires_usd():
    assert parse_compensation("120k-150k USD remote").pay_low == 120_000
    assert parse_compensation("180k USD").pay_high == 180_000
    assert parse_compensation("180k users loved this").pay_high is None


def test_parse_keeps_salary_when_benefits_follow():
    assert parse_compensation("$180k salary plus health benefits").pay_high == 180_000
    assert parse_compensation("$120k-$150k plus health").pay_low == 120_000
    assert parse_compensation("Base salary $140k, benefits include 401k").pay_high == 140_000
    assert parse_compensation("$90k/yr plus bonus").pay_high == 90_000


def test_parse_invents_nothing_from_seniority():
    parsed = parse_compensation("Senior Staff Principal Lead Engineer")
    assert parsed.pay_low is None
    assert parsed.pay_high is None
    assert parsed.hours is None


def test_canonicalize_greenhouse_lever_ashby_workday():
    assert canonicalize_url(
        "https://boards.greenhouse.io/Example/jobs/12345?gh_src=abc&utm_source=li"
    ) == "https://job-boards.greenhouse.io/Example/jobs/12345"
    assert canonicalize_url(
        "https://job-boards.greenhouse.io/embed/job_app?for=acme&token=99&gh_src=x"
    ) == "https://job-boards.greenhouse.io/acme/jobs/99"
    assert canonicalize_url(
        "https://JOBS.LEVER.CO/leverdemo/681fbc53-1e34-4a46-8677-3a78118674eb/apply?lever-source=LinkedIn#ok"
    ) == "https://jobs.lever.co/leverdemo/681fbc53-1e34-4a46-8677-3a78118674eb"
    assert canonicalize_url(
        "https://jobs.ashbyhq.com/ashby/45134452-f53b-4d4c-915e-4a4615fb6c93/application?utm_source=x"
    ) == "https://jobs.ashbyhq.com/ashby/45134452-f53b-4d4c-915e-4a4615fb6c93"
    assert canonicalize_url(
        "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Israel-Tel-Aviv/Software-Engineer_JR2025162/?source=LinkedIn"
    ) == "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/Israel-Tel-Aviv/Software-Engineer_JR2025162"


def test_canonicalize_generic_and_custom_ats_embed():
    assert canonicalize_url("https://Example.com/jobs/foo/?utm_campaign=spring&gclid=x#section") == (
        "https://example.com/jobs/foo"
    )
    assert canonicalize_url("https://careers.acme.com/jobs?gh_jid=12345&gh_src=abcdef&fbclid=1") == (
        "https://careers.acme.com/jobs?gh_jid=12345"
    )


def test_canonicalize_is_idempotent():
    url = canonicalize_url("https://boards.greenhouse.io/Acme/jobs/1/?gh_src=x")
    assert canonicalize_url(url) == url
