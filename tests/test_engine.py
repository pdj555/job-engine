import asyncio
import types

from src.compensation import Compensation, parse_ats_json
from src.engine import (
    Engine,
    _guess_remote,
    _parse_ddg_html,
    opportunity_from_raw,
    search_angles,
)
from src.models import Opportunity


def test_search_angles_include_ats_hosts():
    angles = search_angles("ml engineer")
    assert any("boards.greenhouse.io" in a for a in angles)
    assert any("jobs.ashbyhq.com" in a for a in angles)
    assert any("jobs.smartrecruiters.com" in a for a in angles)
    assert any("remote job hiring" in a for a in angles)


def test_read_listing_returns_ats_pay(monkeypatch):
    engine = Engine()

    async def fake_ats(url, client):
        return Compensation(pay_low=150_000, pay_high=180_000, hours=40, company="Acme")

    async def fake_html(url, client):
        return None

    monkeypatch.setattr(engine, "_fetch_ats", fake_ats)
    monkeypatch.setattr(engine, "_fetch_listing", fake_html)

    out = asyncio.run(engine.read_listing("https://boards.greenhouse.io/acme/jobs/1"))
    assert out["pay"] == 165_000
    assert out["pay_low"] == 150_000
    assert out["pay_high"] == 180_000
    assert out["pay_source"] == "ats"
    assert out["company"] == "Acme"
    assert out["hours_per_week"] == 40


def test_extract_does_not_invent_pay_from_title():
    opp = opportunity_from_raw(
        {"title": "Senior ML Engineer", "url": "https://example.com/j", "description": "great team"}
    )
    assert opp is not None
    assert opp.pay is None
    assert opp.pay_source is None
    assert opp.score() == 0


def test_extract_parses_posted_pay_and_hours():
    opp = opportunity_from_raw(
        {
            "title": "Engineer $180k",
            "url": "https://example.com/j",
            "description": "40 hours/week remote",
        }
    )
    assert opp is not None
    assert opp.pay == 180_000
    assert opp.hours_per_week == 40
    assert opp.pay_source == "snippet"
    assert opp.hours_source == "snippet"
    assert opp.dollars_per_hour is None
    assert opp.refined_rate is None


def test_extract_ignores_perplexity_prose_pay():
    opp = opportunity_from_raw(
        {
            "title": "Staff Engineer $180k",
            "url": "https://example.com/j",
            "description": "pays $200k · 40 hours/week",
            "source": "perplexity",
        }
    )
    assert opp is not None
    assert opp.pay is None
    assert opp.hours_per_week is None
    assert opp.pay_source is None


def test_extract_ignores_search_provider_estimated_pay():
    opp = opportunity_from_raw(
        {
            "title": "Staff Engineer",
            "url": "https://example.com/j",
            "description": "build systems",
            "pay": 180_000,
            "hours": 40,
            "source": "perplexity",
        }
    )
    assert opp is not None
    assert opp.pay is None
    assert opp.hours_per_week is None


def test_extract_canonicalizes_ats_url():
    opp = opportunity_from_raw(
        {
            "title": "Eng",
            "url": "https://boards.greenhouse.io/Acme/jobs/12345?gh_src=li",
            "description": "",
        }
    )
    assert opp is not None
    assert opp.url == "https://job-boards.greenhouse.io/Acme/jobs/12345"


def test_extract_drops_indeed_search_pages():
    assert (
        opportunity_from_raw(
            {
                "title": "Python $180k Jobs",
                "url": "https://www.indeed.com/q-python-$180k-jobs.html",
                "description": "$180k",
            }
        )
        is None
    )


def test_extract_drops_career_now_index():
    assert (
        opportunity_from_raw(
            {
                "title": "Python Engineer Remote Jobs - Median $105/hr | Career.now",
                "url": "https://career.now/remote-jobs/python-engineer",
                "description": "",
            }
        )
        is None
    )


def test_extract_keeps_real_listing_pay_with_board_noise():
    linkedin = opportunity_from_raw(
        {
            "title": "Staff Engineer $180k | LinkedIn",
            "url": "https://www.linkedin.com/jobs/view/12345",
            "description": "Acme · Remote · 12 jobs",
        }
    )
    assert linkedin is not None
    assert linkedin.pay == 180_000
    assert linkedin.pay_source == "snippet"

    greenhouse = opportunity_from_raw(
        {
            "title": "Staff Engineer $180k",
            "url": "https://job-boards.greenhouse.io/acme/jobs/1",
            "description": "The estimated salary range for this role is $150,000-$180,000. 12 jobs.",
        }
    )
    assert greenhouse is not None
    assert greenhouse.pay == 165_000
    assert (greenhouse.pay_low, greenhouse.pay_high) == (150_000, 180_000)
    assert greenhouse.pay_source == "snippet"


def test_extract_ignores_seo_title_pay_on_unknown_host():
    opp = opportunity_from_raw(
        {
            "title": "Python Engineer Salary: $95k-$140k (Glassdoor estimate)",
            "url": "https://careers.acme.com/python-engineer",
            "description": "",
        }
    )
    assert opp is not None
    assert opp.pay is None
    assert opp.pay_source is None
    assert opp.score() == 0


def test_guess_remote_penalizes_onsite_signals():
    assert _guess_remote("Engineer", "hybrid schedule") is False
    assert _guess_remote("Engineer", "must be onsite") is False
    assert _guess_remote("Engineer", "fully distributed team") is None


DDG_HTML = """
<div class="links_main">
  <a class="result__a" href="https://example.com/job1">Senior ML Engineer</a>
  <a class="result__snippet" href="https://example.com/job1">Remote role, great pay</a>
</div>
<div class="links_main">
  <a class="result__a" href="//example.org/job2">Data Scientist</a>
</div>
<div class="links_main">
  <a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Sponsored</a>
</div>
"""


def test_parse_ddg_extracts_title_url_and_snippet():
    results = _parse_ddg_html(DDG_HTML)
    assert len(results) == 2
    first = results[0]
    assert first["url"] == "https://example.com/job1"
    assert first["title"] == "Senior ML Engineer"
    assert first["description"] == "Remote role, great pay"
    assert first["source"] == "duckduckgo"


def test_parse_ddg_normalizes_protocol_relative_url():
    results = _parse_ddg_html(DDG_HTML)
    assert results[1]["url"] == "https://example.org/job2"


def test_parse_ddg_empty_input():
    assert _parse_ddg_html("") == []


def test_search_all_dedupes_by_url():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {"url": "https://a.com/x", "title": "A"},
            {"url": "https://b.com/y", "title": "B"},
            {"url": "https://a.com/x", "title": "A duplicate"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity

    results = asyncio.run(engine._search_all("anything"))
    assert [r["url"] for r in results] == ["https://a.com/x", "https://b.com/y"]


def test_search_all_dedupes_canonical_ats_urls():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {"url": "https://boards.greenhouse.io/Acme/jobs/1?utm_source=li", "title": "A"},
            {"url": "https://job-boards.greenhouse.io/Acme/jobs/1", "title": "A again"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity

    results = asyncio.run(engine._search_all("anything"))
    assert [r["url"] for r in results] == ["https://job-boards.greenhouse.io/Acme/jobs/1"]


def test_find_drops_aggregator_seo_pay_below_real_listings():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Python Engineer Remote Jobs - Median $105/hr | Career.now",
                "url": "https://career.now/remote-jobs/python-engineer",
                "description": "",
            },
            {
                "title": "Python Developer Remote Jobs $200k",
                "url": "https://www.linkedin.com/jobs/python-developer-remote-jobs",
                "description": "",
            },
            {
                "title": "Engineer $90k",
                "url": "https://jobs.lever.co/acme/abc",
                "description": "40 hours/week",
            },
            {
                "title": "Python Engineer Salary: $95k-$140k (Glassdoor estimate)",
                "url": "https://careers.acme.com/python-engineer",
                "description": "",
            },
        ]

    async def no_fetch(url, client=None):
        return None

    engine._search_all = fake_search
    engine._fetch_listing = no_fetch
    async def verified_ats(url, client=None):
        return Compensation(pay_low=90_000, pay_high=90_000, remote=True) if "lever.co" in url else None

    engine._fetch_ats = verified_ats
    ranked = asyncio.run(engine.find("eng", limit=10))
    urls = [o.url for o in ranked]
    assert ranked[0].url == "https://jobs.lever.co/acme/abc"
    assert ranked[0].pay_source == "ats"
    assert all("career.now" not in u and "linkedin.com/jobs/python" not in u for u in urls)
    acme = next(o for o in ranked if "careers.acme.com" in o.url)
    assert acme.pay is None
    assert acme.score() == 0


def test_find_ranks_posted_pay_above_thin_listings():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {"title": "Senior Staff Engineer", "url": "https://example.com/thin", "description": ""},
            {
                "title": "Engineer $90k",
                "url": "https://example.com/listed",
                "description": "40 hours/week",
            },
        ]

    async def no_fetch(url, client=None):
        return "<p>Base salary: $90k annually.</p>" if "listed" in url else None

    engine._search_all = fake_search
    engine._fetch_listing = no_fetch
    ranked = asyncio.run(engine.find("eng", limit=10))
    assert [o.url for o in ranked] == ["https://example.com/listed", "https://example.com/thin"]
    assert ranked[0].pay_source == "posted"
    assert ranked[1].score() == 0


def test_enrich_applies_jobposting_schema_when_snippet_has_no_pay():
    engine = Engine()
    html = """
    <script type="application/ld+json">
    {"@type": "JobPosting", "hiringOrganization": {"name": "Acme"},
     "jobLocationType": "TELECOMMUTE",
     "baseSalary": {"@type": "MonetaryAmount", "currency": "USD",
       "value": {"@type": "QuantitativeValue", "value": 160000, "unitText": "YEAR"}}}
    </script>
    """

    async def fake_fetch(url, client=None):
        return html if "listed" in url else None

    engine._fetch_listing = fake_fetch
    thin = Opportunity(title="Senior Staff", url="https://example.com/thin")
    listed = Opportunity(title="Engineer", url="https://example.com/listed")
    asyncio.run(engine.enrich([thin, listed]))
    assert thin.pay is None
    assert listed.pay == 160_000
    assert listed.pay_source == "schema"
    assert listed.company == "Acme"
    assert listed.remote is True
    assert listed.score() == 80.0


def test_enrich_skips_already_verified_non_ats_listing():
    engine = Engine()
    fetched = []

    async def boom(url, client=None):
        fetched.append(url)

    engine._fetch_listing = boom
    engine._fetch_ats = boom
    opp = Opportunity(
        title="Eng",
        url="https://example.com/j",
        pay_high=90_000,
        pay_source="posted",
        hours_per_week=40,
        hours_source="posted",
        remote=True, remote_source="posted",
    )
    asyncio.run(engine.enrich([opp]))
    assert fetched == []
    assert opp.pay_source == "posted"
    assert opp.pay == 90_000


def test_enrich_ats_json_overrides_snippet_ceiling():
    engine = Engine()
    html_urls = []

    async def fake_ats(url, client=None):
        return parse_ats_json(
            url,
            {
                "title": "Staff Software Engineer",
                "offices": [{"name": "Remote - US"}],
                "pay_input_ranges": [
                    {
                        "min_cents": 20_000_000,
                        "max_cents": 24_500_000,
                        "currency_type": "USD",
                        "title": "Base Pay Range",
                    }
                ],
            },
        )

    async def fake_html(url, client=None):
        html_urls.append(url)
        return "<html></html>"

    engine._fetch_ats = fake_ats
    engine._fetch_listing = fake_html
    opp = Opportunity(
        title="Staff Software Engineer $245k",
        url="https://job-boards.greenhouse.io/engine/jobs/7994750003",
        pay_high=245_000,
        pay_source="posted",
    )
    asyncio.run(engine.enrich([opp]))
    assert html_urls == []
    assert opp.pay == 222_500
    assert (opp.pay_low, opp.pay_high) == (200_000, 245_000)
    assert opp.pay_source == "ats"
    assert opp.score() == 111.25


def test_enrich_gh_jid_overrides_snippet_ceiling():
    engine = Engine()
    html = """
    <html><body>
      <script src="https://boards.greenhouse.io/embed/job_board/js?for=datadog"></script>
    </body></html>
    """

    async def fake_ats(url, client=None):
        if "job-boards.greenhouse.io/datadog/jobs/6572669" in url:
            return Compensation(pay_low=320_000, pay_high=400_000, title="AI Research Scientist")
        return None

    async def fake_html(url, client=None):
        return html

    engine._fetch_ats = fake_ats
    engine._fetch_listing = fake_html
    opp = Opportunity(
        title="AI Research Scientist $400k",
        url="https://careers.datadoghq.com/detail/6572669/?gh_jid=6572669",
        pay_high=400_000,
        pay_source="posted",
    )
    asyncio.run(engine.enrich([opp]))
    assert opp.pay == 360_000
    assert (opp.pay_low, opp.pay_high) == (320_000, 400_000)
    assert opp.pay_source == "ats"
    assert opp.score() == 180.0


def test_enrich_reads_workday_cxs_json():
    engine = Engine()
    html_urls = []

    async def fake_ats(url, client=None):
        return Compensation(pay_low=207_000, pay_high=351_225, remote=True, title="Manager")

    async def fake_html(url, client=None):
        html_urls.append(url)
        return "<html></html>"

    engine._fetch_ats = fake_ats
    engine._fetch_listing = fake_html
    opp = Opportunity(
        title="Manager",
        url="https://adobe.wd5.myworkdayjobs.com/external_experienced/job/Remote-California/Role_R1",
    )
    asyncio.run(engine.enrich([opp]))
    assert html_urls == []
    assert opp.pay == (207_000 + 351_225) // 2
    assert opp.pay_source == "ats"
    assert opp.remote is True
    assert opp.score() == ((207_000 + 351_225) // 2) / (40 * 50)


def test_enrich_prefers_ats_json_over_html_schema():
    engine = Engine()
    html_urls = []

    async def fake_ats(url, client=None):
        return Compensation(pay_low=150_000, pay_high=180_000, remote=True, title="Staff")

    async def fake_html(url, client=None):
        html_urls.append(url)
        return "<html></html>"

    engine._fetch_ats = fake_ats
    engine._fetch_listing = fake_html
    opp = Opportunity(
        title="Engineer",
        url="https://job-boards.greenhouse.io/acme/jobs/1",
    )
    asyncio.run(engine.enrich([opp]))
    assert html_urls == []
    assert opp.pay == 165_000
    assert opp.pay_source == "ats"
    assert opp.remote is True


def test_extract_batch_drops_ungrounded_urls():
    engine = Engine()

    async def fake_create(**_kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"opportunities": ['
                        '{"url": "https://evil.example/fake", "title": "Fake $200k"},'
                        '{"url": "https://example.com/real", "title": "Real $90k"}]}'
                    )
                )
            ]
        )

    engine.openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )
    batch = [
        {"title": "Real", "url": "https://example.com/real", "description": "$90k · 40 hours/week"}
    ]
    opps = asyncio.run(engine._extract_batch(batch, "eng"))
    assert [o.url for o in opps] == ["https://example.com/real"]
    assert opps[0].pay == 90_000
    assert opps[0].pay_source == "snippet"


def test_extract_batch_does_not_trust_llm_title_pay():
    engine = Engine()

    async def fake_create(**_kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"opportunities": ['
                        '{"url": "https://example.com/real", "title": "Staff Engineer $180k"}]}'
                    )
                )
            ]
        )

    engine.openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )
    batch = [
        {"title": "Staff Engineer", "url": "https://example.com/real", "description": "build systems"}
    ]
    opps = asyncio.run(engine._extract_batch(batch, "eng"))
    assert opps[0].title == "Staff Engineer"
    assert opps[0].pay is None
    assert opps[0].score() == 0


def test_enrich_reads_labeled_listing_body_when_schema_missing():
    engine = Engine()
    html = """
    <html><body>
      <p>Team budget is $80,000 -- $120,000.</p>
      <p>The pay range for this position is $160,000 -- $200,000 annually.</p>
    </body></html>
    """

    async def no_ats(url, client=None):
        return None

    async def fake_html(url, client=None):
        return html

    engine._fetch_ats = no_ats
    engine._fetch_listing = fake_html
    opp = Opportunity(title="Engineer", url="https://careers.acme.com/eng")
    asyncio.run(engine.enrich([opp]))
    assert opp.pay == 180_000
    assert (opp.pay_low, opp.pay_high) == (160_000, 200_000)
    assert opp.pay_source == "posted"
    assert opp.score() == 90.0


def test_enrich_engine_greenhouse_range_midpoint():
    engine = Engine()
    payload = {
        "title": "Staff Software Engineer",
        "pay_input_ranges": [
            {
                "min_cents": 20_000_000,
                "max_cents": 24_500_000,
                "currency_type": "USD",
                "title": "Base Pay Range",
            }
        ],
        "offices": [{"name": "Remote - US"}],
    }

    async def fake_ats(url, client=None):
        return parse_ats_json(url, payload)

    engine._fetch_ats = fake_ats
    opp = Opportunity(
        title="Unknown",
        url="https://job-boards.greenhouse.io/engine/jobs/7994750003",
    )
    asyncio.run(engine.enrich([opp]))
    assert opp.pay == 222_500
    assert (opp.pay_low, opp.pay_high) == (200_000, 245_000)
    assert opp.pay_source == "ats"
    assert opp.refined_rate == 111.25
    assert opp.score() == 111.25
    assert opp.title == "Staff Software Engineer"


def test_enrich_discovers_greenhouse_embed_from_career_html():
    engine = Engine()
    html = """
    <html><body>
      <script src="https://boards.greenhouse.io/embed/job_board/js?for=datadog"></script>
      <div id="grnhse_app"></div>
    </body></html>
    """

    async def fake_ats(url, client=None):
        if "job-boards.greenhouse.io/datadog/jobs/6572669" in url:
            return Compensation(pay_low=320_000, pay_high=400_000, title="AI Research Scientist")
        return None

    async def fake_html(url, client=None):
        return html

    engine._fetch_ats = fake_ats
    engine._fetch_listing = fake_html
    opp = Opportunity(
        title="AI Research Scientist",
        url="https://careers.datadoghq.com/detail/6572669/?gh_jid=6572669",
    )
    asyncio.run(engine.enrich([opp]))
    assert opp.pay == 360_000
    assert opp.pay_source == "ats"
    assert opp.title == "AI Research Scientist"


def test_find_ranks_range_midpoint_not_ceiling():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Wide $120k-$200k",
                "url": "https://example.com/wide",
                "description": "40 hours/week",
            },
            {
                "title": "Point $170k",
                "url": "https://example.com/point",
                "description": "40 hours/week",
            },
        ]

    async def no_fetch(url, client=None):
        return "<p>Base salary: $120k-$200k annually.</p>" if "wide" in url else "<p>Base salary: $170k annually.</p>"

    engine._search_all = fake_search
    engine._fetch_listing = no_fetch
    async def no_ats(url, client=None):
        return None

    engine._fetch_ats = no_ats
    ranked = asyncio.run(engine.find("eng", limit=10))
    assert [o.url for o in ranked] == ["https://example.com/point", "https://example.com/wide"]
    assert ranked[0].pay == 170_000
    assert ranked[1].pay == 160_000
    assert ranked[1].score() == 80.0


def test_find_ats_midpoint_outranks_snippet_ceiling():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Staff Software Engineer $245k",
                "url": "https://job-boards.greenhouse.io/engine/jobs/7994750003",
                "description": "40 hours/week",
            },
            {
                "title": "Point $230k",
                "url": "https://example.com/point",
                "description": "40 hours/week",
            },
        ]

    async def fake_ats(url, client=None):
        if "greenhouse.io/engine/jobs/7994750003" in url:
            return parse_ats_json(
                url,
                {
                    "pay_input_ranges": [
                        {
                            "min_cents": 20_000_000,
                            "max_cents": 24_500_000,
                            "currency_type": "USD",
                            "title": "Base Pay Range",
                        }
                    ]
                },
            )
        return None

    async def no_html(url, client=None):
        return "<p>Base salary: $230k annually.</p>" if "point" in url else None

    engine._search_all = fake_search
    engine._fetch_ats = fake_ats
    engine._fetch_listing = no_html
    ranked = asyncio.run(engine.find("eng", limit=10))
    assert [o.url for o in ranked] == [
        "https://example.com/point",
        "https://job-boards.greenhouse.io/engine/jobs/7994750003",
    ]
    assert ranked[0].pay == 230_000
    assert ranked[0].score() == 115.0
    assert ranked[1].pay == 222_500
    assert (ranked[1].pay_low, ranked[1].pay_high) == (200_000, 245_000)
    assert ranked[1].pay_source == "ats"
    assert ranked[1].score() == 111.25


def test_search_all_drops_failed_sources():
    engine = Engine()

    async def fake_brave(_query: str):
        raise RuntimeError("source down")

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity
    assert asyncio.run(engine._search_all("anything")) == []
