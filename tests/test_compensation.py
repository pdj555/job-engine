from src.compensation import (
    ats_json_url,
    canonicalize_url,
    is_search_serp,
    parse_ats_json,
    parse_compensation,
    parse_job_posting,
)


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


_JOBPOSTING = """
<html><head>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "JobPosting",
  "title": "Staff Engineer",
  "hiringOrganization": {"@type": "Organization", "name": "Acme"},
  "jobLocationType": "TELECOMMUTE",
  "workHours": "32 hours a week",
  "baseSalary": {
    "@type": "MonetaryAmount",
    "currency": "USD",
    "value": {
      "@type": "QuantitativeValue",
      "minValue": 140000,
      "maxValue": 180000,
      "unitText": "YEAR"
    }
  }
}
</script>
</head></html>
"""


def test_parse_job_posting_reads_usd_range_hours_and_remote():
    parsed = parse_job_posting(_JOBPOSTING)
    assert (parsed.pay_low, parsed.pay_high) == (140_000, 180_000)
    assert parsed.hours == 32
    assert parsed.remote is True
    assert parsed.company == "Acme"
    assert parsed.title == "Staff Engineer"


def test_parse_job_posting_annualizes_hourly_and_walks_graph():
    html = """
    <script type="application/ld+json">
    {"@graph": [{"@type": "Organization", "name": "Skip"}, {
      "@type": ["JobPosting"],
      "baseSalary": {
        "@type": "MonetaryAmount",
        "currency": "USD",
        "value": {"@type": "QuantitativeValue", "value": 75, "unitText": "HOUR"}
      }
    }]}
    </script>
    """
    parsed = parse_job_posting(html)
    assert parsed.pay_high == 150_000
    assert parsed.hours is None


def test_parse_job_posting_rejects_foreign_and_estimated():
    cad = """
    <script type="application/ld+json">
    {"@type": "JobPosting", "baseSalary": {
      "@type": "MonetaryAmount", "currency": "CAD",
      "value": {"@type": "QuantitativeValue", "value": 180000, "unitText": "YEAR"}
    }}
    </script>
    """
    estimated = """
    <script type="application/ld+json">
    {"@type": "JobPosting", "estimatedSalary": {
      "@type": "MonetaryAmount", "currency": "USD",
      "value": {"@type": "QuantitativeValue", "value": 180000, "unitText": "YEAR"}
    }}
    </script>
    """
    assert parse_job_posting(cad).pay_high is None
    assert parse_job_posting(estimated).pay_high is None


def test_parse_job_posting_empty_html():
    assert parse_job_posting("").pay_high is None
    assert parse_job_posting("<html></html>").posted is False


def test_ats_json_url_maps_greenhouse_lever_ashby():
    assert ats_json_url(
        "https://job-boards.greenhouse.io/acme/jobs/12345"
    ) == "https://boards-api.greenhouse.io/v1/boards/acme/jobs/12345?pay_transparency=true"
    assert ats_json_url(
        "https://jobs.lever.co/acme/681fbc53-1e34-4a46-8677-3a78118674eb"
    ) == "https://api.lever.co/v0/postings/acme/681fbc53-1e34-4a46-8677-3a78118674eb?mode=json"
    assert ats_json_url(
        "https://jobs.ashbyhq.com/luminary/84c74ea8-20b1-4e0d-9aa5-731da2cb1bf3"
    ) == "https://api.ashbyhq.com/posting-api/job-board/luminary?includeCompensation=true"
    assert ats_json_url("https://example.com/jobs/1") is None


def test_parse_greenhouse_pay_transparency_cents():
    parsed = parse_ats_json(
        "https://job-boards.greenhouse.io/acme/jobs/1",
        {
            "title": "Staff Engineer",
            "pay_input_ranges": [
                {"min_cents": 15_000_000, "max_cents": 18_000_000, "currency_type": "USD"}
            ],
            "offices": [{"name": "Remote"}],
        },
    )
    assert (parsed.pay_low, parsed.pay_high) == (150_000, 180_000)
    assert parsed.remote is True
    assert parsed.title == "Staff Engineer"


def test_parse_lever_salary_range_and_ashby_salary_component():
    lever = parse_ats_json(
        "https://jobs.lever.co/acme/abc",
        {
            "text": "Engineer",
            "workplaceType": "remote",
            "salaryRange": {
                "currency": "USD",
                "interval": "per-year-salary",
                "min": 140000,
                "max": 170000,
            },
        },
    )
    assert (lever.pay_low, lever.pay_high) == (140_000, 170_000)
    assert lever.remote is True

    ashby = parse_ats_json(
        "https://jobs.ashbyhq.com/acme/job-1",
        {
            "jobs": [
                {
                    "id": "job-1",
                    "title": "Applied AI",
                    "isRemote": True,
                    "compensation": {
                        "summaryComponents": [
                            {
                                "compensationType": "Salary",
                                "minValue": 170000,
                                "maxValue": 225000,
                                "currencyCode": "USD",
                                "interval": "1 YEAR",
                            }
                        ]
                    },
                }
            ]
        },
    )
    assert (ashby.pay_low, ashby.pay_high) == (170_000, 225_000)
    assert ashby.remote is True

    tiers = parse_ats_json(
        "https://jobs.ashbyhq.com/acme/job-2",
        {
            "jobs": [
                {
                    "id": "job-2",
                    "title": "EM",
                    "compensation": {
                        "summaryComponents": None,
                        "compensationTiers": [
                            {
                                "components": [
                                    {
                                        "compensationType": "Salary",
                                        "minValue": 110000,
                                        "maxValue": 185000,
                                        "currencyCode": "USD",
                                        "interval": "1 YEAR",
                                    }
                                ]
                            }
                        ],
                    },
                }
            ]
        },
    )
    assert (tiers.pay_low, tiers.pay_high) == (110_000, 185_000)


def test_parse_ats_rejects_foreign_and_missing():
    cad = parse_ats_json(
        "https://jobs.lever.co/acme/abc",
        {"salaryRange": {"currency": "CAD", "interval": "per-year-salary", "min": 180000, "max": 180000}},
    )
    assert cad.pay_high is None
    assert parse_ats_json("https://example.com/x", {"pay_input_ranges": []}).posted is False


def test_is_search_serp_drops_indeed_query_pages_not_viewjob():
    assert is_search_serp("https://www.indeed.com/q-python-engineer-jobs.html")
    assert is_search_serp("https://www.linkedin.com/jobs/search/?keywords=python")
    assert not is_search_serp("https://www.indeed.com/viewjob?jk=abc")
    assert not is_search_serp("https://job-boards.greenhouse.io/acme/jobs/1")
