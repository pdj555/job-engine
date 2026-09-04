import asyncio
import json
import types

from src.engine import (
    Engine,
    _guess_hours,
    _guess_pay,
    _guess_remote,
    _heuristic_opportunity,
    _parse_ddg_html,
    _search_angles,
)
from src.models import Opportunity


# --- compensation from listing text (never invented) --------------------


def test_guess_pay_parses_real_numbers_and_refuses_to_invent():
    assert _guess_pay("Senior ML Engineer", "$180k") == 180_000
    assert _guess_pay("Staff Engineer $150,000", "") == 150_000
    assert _guess_pay("Engineer", "$120k-$180k") == 180_000
    assert _guess_pay("Engineer", "$143,000 to 197,000") == 197_000
    assert _guess_pay("Software Engineer", "") is None
    assert _guess_pay("Senior Staff Principal Lead", "junior intern") is None


def test_guess_pay_annualizes_hourly():
    assert _guess_pay("Contract", "$80/hr") == 160_000  # 80 * 40 * 50
    assert _guess_pay("Contract", "$80/hr", hours=20) == 80_000
    assert _guess_pay("", "$80 - $100 / Hour") == 200_000
    from src.engine import _parse_pay
    assert _parse_pay("$80 - $100 / Hour") == (160_000, 200_000)
    assert _parse_pay("$80–$100/hr") == (160_000, 200_000)


def test_guess_pay_reads_description_not_just_title():
    assert _guess_pay("Engineer", "comp $175k plus equity") == 175_000


def test_guess_hours_from_text_not_job_type():
    assert _guess_hours("Engineer", "20 hrs/week") == 20
    assert _guess_hours("Part-time role", "") == 20
    assert _guess_hours("Full-time Engineer", "") == 40
    assert _guess_hours("Contract Engineer", "") is None
    assert _guess_hours("Engineer", "") is None


def test_guess_remote_penalizes_onsite_signals():
    assert _guess_remote("Engineer", "hybrid schedule") is False
    assert _guess_remote("Engineer", "must be onsite") is False
    assert _guess_remote("Engineer", "fully distributed team") is True  # default


# --- DuckDuckGo HTML parsing -------------------------------------------


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

    # ad link (y.js) filtered out -> 2 real results
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


DDG_LIVE_SHAPE = """
<a class="result__a" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">Flexible Ml Engineer Remote $150,000 Jobs - Indeed</a>
<div class="result__extras">
  <a rel="nofollow" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">
    <img class="result__icon__img" width="16" height="16" alt="" src="//external-content.duckduckgo.com/ip3/www.indeed.com.ico" />
  </a>
  <a class="result__url" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">
    www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html
  </a>
</div>
<a class="result__snippet" href="https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html">Browse 568 <b>Ml</b> <b>Engineer</b> <b>Remote</b> $150,000 job openings. Discover flexible, work-from-home opportunities.</a>
<a class="result__a" href="https://jobs.example/onsite">Office Role</a>
<a class="result__snippet" href="https://jobs.example/onsite">Must be <b>hybrid</b>, $80/hr, 20 hrs/week</a>
"""


def test_parse_ddg_strips_bold_and_does_not_need_a_tiny_window():
    results = _parse_ddg_html(DDG_LIVE_SHAPE)
    assert len(results) == 2
    assert results[0]["description"] == (
        "Browse 568 Ml Engineer Remote $150,000 job openings. "
        "Discover flexible, work-from-home opportunities."
    )
    assert results[1]["description"] == "Must be hybrid, $80/hr, 20 hrs/week"


def test_heuristic_uses_ddg_snippet_pay_hours_and_remote():
    results = _parse_ddg_html(DDG_LIVE_SHAPE)
    office = _heuristic_opportunity(results[1])
    assert office.pay_high == 80_000
    assert office.hours_per_week == 20
    assert office.remote is False
    assert office.score() == 56.0  # 80k / (20*50) * 0.7 office


# --- search aggregation -------------------------------------------------


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
    urls = [r["url"] for r in results]

    assert urls == ["https://a.com/x", "https://b.com/y"]


def test_search_all_dedupes_normalized_urls():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {"url": "https://a.com/x/", "title": "A slash"},
            {"url": "HTTPS://A.COM/X", "title": "A case"},
            {"url": "https://b.com/y", "title": "B"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity

    results = asyncio.run(engine._search_all("anything"))
    assert [r["url"] for r in results] == ["https://a.com/x/", "https://b.com/y"]


def test_search_all_dedupes_lever_apply_to_job_url():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {
                "title": "Apply",
                "url": "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply",
            },
            {
                "title": "Job",
                "url": "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff",
            },
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity
    results = asyncio.run(engine._search_all("ml"))
    assert [r["url"] for r in results] == [
        "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply"
    ]


def test_search_all_drops_failed_sources():
    engine = Engine()

    async def fake_brave(_query: str):
        raise RuntimeError("source down")

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity

    # gather(return_exceptions=True) -> exceptions ignored, no crash
    assert asyncio.run(engine._search_all("anything")) == []


def _fake_client(content: str, captured: dict | None = None):
    async def create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def _fake_client_raises(exc: Exception):
    async def create(**kwargs):
        raise exc

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def test_heuristic_opportunity_requires_url():
    assert _heuristic_opportunity({"title": "No url", "pay": 100_000}) is None


def test_heuristic_opportunity_prefers_raw_then_guesses():
    raw = {
        "title": "Staff Engineer",
        "company": "Acme",
        "url": "https://example.com/job",
        "description": "onsite hybrid",
        "pay": 200_000,
        "hours": 25,
        "remote": False,
        "source": "brave",
    }
    opp = _heuristic_opportunity(raw)
    assert isinstance(opp, Opportunity)
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 25
    assert opp.remote is False
    assert opp.efficiency == opp.refined_rate == 160.0

    guessed = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer $180k",
            "url": "https://example.com/senior",
            "description": "must be onsite, 20 hrs/week",
            "source": "ddg",
        }
    )
    assert guessed.pay_high == 180_000
    assert guessed.hours_per_week == 20
    assert guessed.remote is False
    assert guessed.rate_is_imputed is False

    thin = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer",
            "url": "https://example.com/thin",
            "description": "must be onsite",
            "source": "ddg",
        }
    )
    assert thin.pay_high is None
    assert thin.hours_per_week is None
    assert thin.score() == 0
    assert thin.remote is False


def test_index_pages_are_not_opportunities():
    assert (
        _heuristic_opportunity(
            {
                "title": "Flexible Ml Engineer Remote $150,000 Jobs - Indeed",
                "url": "https://www.indeed.com/q-ml-engineer-remote-$150,000-jobs.html",
                "description": "Browse 568 Ml Engineer Remote $150,000 job openings.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Remote Machine Learning Engineer Jobs ($104K-$225K)",
                "url": "https://www.remoterocketship.com/jobs/machine-learning-engineer/",
                "description": "Search 546 remote jobs.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "RemoteFront | 100,000+ Remote Jobs from 20,000+ Vetted Companies",
                "url": "https://www.remotefront.com/remote-ml-engineer-jobs",
                "description": "median $190k (most $150k-$215k)",
            }
        )
        is None
    )
    kept = _heuristic_opportunity(
        {
            "title": "Senior AI/ML Engineer",
            "url": "https://www.gravityer.com/jobs/ctg-senior-ai-ml-engineer",
            "description": "Remote (US Only) | $150K-$200K",
        }
    )
    assert kept is not None
    assert kept.pay_high == 200_000
    lever = _heuristic_opportunity(
        {
            "title": "Lyra Health - Senior ML Engineer (ML/AI) - jobs.lever.co",
            "url": "https://jobs.lever.co/lyrahealth/d33ddfed-8c69-4e29-966b-0e190190cd6a",
            "description": "Remote role.",
        }
    )
    assert lever is not None
    assert (
        _heuristic_opportunity(
            {
                "title": "Home | Grants.gov",
                "url": "https://www.grants.gov/",
                "description": "Find grants",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "AI/ML federal funding",
                "url": "https://nondilute.com/category/aiml/",
                "description": "52 open in 2026",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Search | Simpler.Grants.gov",
                "url": "https://www.grants.gov/search-grants/?keywords=intelligence",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Hire a Freelance Machine Learning Engineer — No Agency Fees",
                "url": "https://remoteai.io/v2/freelance/machine-learning-engineers",
                "description": "Browse freelance ML engineers.",
            }
        )
        is None
    )


def test_search_all_drops_index_pages():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {
                "title": "Jobs - Indeed",
                "url": "https://www.indeed.com/q-ml-jobs.html",
            },
            {"title": "Real role", "url": "https://jobs.example/ml"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity
    results = asyncio.run(engine._search_all("ml"))
    assert [r["url"] for r in results] == ["https://jobs.example/ml"]


def test_search_angles_omit_grants_and_equity_unless_asked():
    job = _search_angles("senior ML engineer remote")
    assert job == [
        "senior ML engineer remote",
        "senior ML engineer remote job hiring",
        "senior ML engineer remote freelance contract",
        "senior ML engineer remote site:greenhouse.io",
        "senior ML engineer remote site:jobs.lever.co",
    ]
    assert _search_angles("ml site:example.com") == [
        "ml site:example.com",
        "ml site:example.com remote job hiring",
        "ml site:example.com freelance contract",
    ]
    grant = _search_angles("AI grant funding")
    assert "AI grant funding opportunity" in grant
    assert any("hiring" in q for q in grant)
    equity = _search_angles("startup cofounder")
    assert "startup cofounder equity" in equity


def test_find_ranks_and_limits_without_llm():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Low",
                "url": "https://a.example/low",
                "description": "",
                "pay": 100_000,
                "hours": 40,
                "remote": True,
            },
            {
                "title": "High",
                "url": "https://a.example/high",
                "description": "",
                "pay": 200_000,
                "hours": 20,
                "remote": True,
            },
            {
                "title": "Office",
                "url": "https://a.example/office",
                "description": "",
                "pay": 200_000,
                "hours": 40,
                "remote": False,
            },
            {"title": "dropped — no url", "pay": 999_999, "hours": 1},
        ]

    engine._search_all = fake_search
    ranked = asyncio.run(engine.find("ml", limit=2))
    assert [o.title for o in ranked] == ["High", "Office"]
    assert ranked[0].score() == 200.0
    assert ranked[1].score() == 70.0


def test_find_dedupes_same_title_keeps_higher_score():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Senior ML Engineer",
                "url": "https://board-a.example/1",
                "description": "$100k",
            },
            {
                "title": "Senior ML Engineer",
                "url": "https://board-b.example/2",
                "description": "$180k",
            },
            {
                "title": "Other Role $90k",
                "url": "https://board-c.example/3",
                "description": "",
            },
        ]

    engine._search_all = fake_search
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.title for o in ranked] == ["Senior ML Engineer", "Other Role $90k"]
    assert ranked[0].url == "https://board-b.example/2"
    assert ranked[0].pay_high == 180_000


def test_find_llm_grounds_urls_and_drops_hallucinations():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Cheap LLM",
                        "company": "CoA",
                        "url": "https://jobs.example/a",
                        "pay_high": 100_000,
                        "hours_per_week": 40,
                        "remote": True,
                    },
                    {
                        "title": "Lush LLM",
                        "company": "CoB",
                        "url": "HTTPS://JOBS.EXAMPLE/B/",
                        "pay_high": 200_000,
                        "hours_per_week": 20,
                        "remote": True,
                    },
                    {
                        "title": "Hallucinated",
                        "url": "https://evil.example/nope",
                        "pay_high": 9_999_999,
                        "hours_per_week": 1,
                    },
                ]
            }
        )
    )

    async def fake_search(_query: str):
        return [
            {"title": "Raw A", "url": "https://jobs.example/a", "description": "$100k"},
            {"title": "Raw B", "url": "https://jobs.example/b", "description": "$200k, 20 hrs/week"},
        ]

    engine._search_all = fake_search
    ranked = asyncio.run(engine.find("contracts", limit=20))
    assert [o.title for o in ranked] == ["Lush LLM", "Cheap LLM"]
    assert ranked[0].url == "https://jobs.example/b"
    assert ranked[0].score() == 200.0


def test_extract_batch_prompt_asks_for_opportunities_object():
    captured: dict = {}
    engine = Engine()
    engine.openai = _fake_client('{"opportunities": []}', captured)
    batch = [{"title": "Senior Engineer", "url": "https://example.com/job", "description": "remote"}]
    asyncio.run(engine._extract_batch(batch, "ai engineer"))
    assert captured.get("response_format") == {"type": "json_object"}
    prompt = captured["messages"][0]["content"]
    assert "opportunities" in prompt
    assert "ai engineer" in prompt
    assert "Do not estimate pay or hours" in prompt


def test_extract_batch_keeps_raw_then_heuristics_when_llm_pay_hours_missing():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {"title": "LLM title", "company": "LLM Co", "url": "https://keep-raw.example/a"},
                    {"title": "Needs guesses", "url": "https://guess.example/b"},
                ]
            }
        )
    )
    batch = [
        {
            "title": "Raw A",
            "url": "https://keep-raw.example/a",
            "description": "",
            "pay": 160_000,
            "hours": 20,
            "remote": True,
        },
        {
            "title": "Junior Developer",
            "url": "https://guess.example/b",
            "description": "hybrid office",
        },
    ]
    out = {o.url: o for o in asyncio.run(engine._extract_batch(batch, "q"))}
    kept = out["https://keep-raw.example/a"]
    assert kept.title == "LLM title"
    assert kept.pay_high == 160_000
    assert kept.hours_per_week == 20
    assert kept.efficiency == 160.0
    guessed = out["https://guess.example/b"]
    assert guessed.pay_high is None
    assert guessed.hours_per_week is None
    assert guessed.score() == 0
    assert guessed.remote is False


def test_extract_batch_falls_back_on_error_or_ungrounded_llm():
    boom = Engine()
    boom.openai = _fake_client_raises(RuntimeError("boom"))
    batch = [
        {
            "title": "Senior ML Engineer",
            "url": "https://fallback.example/1",
            "description": "contract",
        }
    ]
    failed = asyncio.run(boom._extract_batch(batch, "q"))
    assert failed[0].pay_high is None
    assert failed[0].hours_per_week is None
    assert failed[0].score() == 0
    assert failed[0].efficiency == failed[0].refined_rate

    ghost = Engine()
    ghost.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Hallucinated",
                        "url": "https://not-in-batch.example/x",
                        "pay_high": 500_000,
                        "hours_per_week": 10,
                    }
                ]
            }
        )
    )
    grounded = asyncio.run(
        ghost._extract_batch(
            [
                {
                    "title": "Staff Engineer",
                    "url": "https://real.example/job",
                    "description": "fully remote",
                    "pay": 180_000,
                    "hours": 30,
                }
            ],
            "q",
        )
    )
    assert grounded[0].url == "https://real.example/job"
    assert grounded[0].title == "Staff Engineer"
    assert grounded[0].pay_high == 180_000


def test_extract_ignores_llm_invented_pay_and_hours():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Inflated",
                        "url": "https://jobs.example/a",
                        "pay_high": 9_999_999,
                        "hours_per_week": 1,
                    }
                ]
            }
        )
    )
    out = asyncio.run(
        engine._extract_batch(
            [
                {
                    "title": "Engineer",
                    "url": "https://jobs.example/a",
                    "description": "no compensation listed",
                }
            ],
            "q",
        )
    )
    assert out[0].title == "Inflated"
    assert out[0].pay_high is None
    assert out[0].hours_per_week is None
    assert out[0].score() == 0


def test_find_ranks_parsed_pay_above_unknown():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Unknown pay",
                "url": "https://a.example/thin",
                "description": "Senior staff role",
            },
            {
                "title": "Priced",
                "url": "https://a.example/paid",
                "description": "$90k",
            },
        ]

    engine._search_all = fake_search

    async def no_page(_url: str) -> str:
        return ""

    engine._listing_text = no_page
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.title for o in ranked] == ["Priced", "Unknown pay"]
    assert ranked[0].score() == 45.0
    assert ranked[0].rate_is_imputed is True
    assert ranked[1].score() == 0


def test_heuristic_company_from_title_at():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer at Lyra Health",
            "url": "https://job-boards.greenhouse.io/lyrahealth/jobs/123",
            "description": "Role in San Francisco.",
        }
    )
    assert h.company == "Lyra Health"


def test_heuristic_skips_at_remote():
    h = _heuristic_opportunity(
        {
            "title": "ML Engineer at Remote",
            "url": "https://example.com/jobs/1",
            "description": "Work from home.",
        }
    )
    assert h.company is None


def test_merge_company_from_raw_title_when_llm_omits_it():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "Senior Machine Learning Engineer",
                        "url": "https://job-boards.greenhouse.io/lyrahealth/jobs/123",
                    }
                ]
            }
        )
    )
    out = asyncio.run(
        engine._extract_batch(
            [
                {
                    "title": "Senior Machine Learning Engineer at Lyra Health",
                    "url": "https://job-boards.greenhouse.io/lyrahealth/jobs/123",
                    "description": "San Francisco",
                }
            ],
            "q",
        )
    )
    assert out[0].company == "Lyra Health"


def test_heuristic_range_and_imputed_hours():
    ranged = _heuristic_opportunity(
        {
            "title": "Eng $120k-$180k",
            "url": "https://example.com/range",
            "description": "",
        }
    )
    assert ranged.pay_low == 120_000
    assert ranged.pay_high == 180_000
    assert ranged.pay == 180_000
    assert ranged.hours_per_week is None
    assert ranged.rate_is_imputed is True
    assert ranged.refined_rate == 90.0


def test_enrich_pay_from_listing_html():
    engine = Engine()

    async def page(_url: str) -> str:
        return "for this full-time position is $143,000 to 197,000."

    engine._listing_text = page
    opp = Opportunity(title="Senior ML Engineer", url="https://careers.example/x")
    asyncio.run(engine._enrich_pay([opp]))
    assert opp.pay_low == 143_000
    assert opp.pay_high == 197_000
    assert opp.hours_per_week == 40
    assert opp.score() == 98.5


def test_apply_listing_json_ld_company_and_hourly_pay():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Senior ML Engineer",
     "hiringOrganization":{"@type":"Organization","name":"Braintrust"},
     "employmentType":"FULL_TIME",
     "baseSalary":{"@type":"MonetaryAmount","currency":"USD",
       "value":{"@type":"QuantitativeValue","minValue":80,"maxValue":100,"unitText":"HOUR"}}}
    </script>
    """
    opp = Opportunity(title="Senior ML Engineer", url="https://karkidi.example/x")
    _apply_listing(opp, html)
    assert opp.company == "Braintrust"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 40
    assert opp.score() == 100.0


def test_apply_listing_empty_json_ld_salary_falls_back_to_visible_text():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Lyra Health"},
     "baseSalary":{"@type":"MonetaryAmount","currency":"","value":{"unitText":""}}}
    </script>
    <p>for this full-time position is $143,000 to 197,000.</p>
    """
    opp = Opportunity(title="Senior ML Engineer", url="https://careers.example/x")
    _apply_listing(opp, html)
    assert opp.company == "Lyra Health"
    assert opp.pay_low == 143_000
    assert opp.pay_high == 197_000


def test_apply_listing_ignores_non_usd_salary():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
     "baseSalary":{"currency":"EUR","value":{"minValue":120000,"maxValue":180000,"unitText":"YEAR"}}}
    </script>
    """
    opp = Opportunity(title="Engineer", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.company == "Acme"
    assert opp.pay_high is None


def test_enrich_fetches_when_company_missing_even_if_paid():
    engine = Engine()
    captured: list[str] = []

    async def page(url: str) -> str:
        captured.append(url)
        return """
        <script type="application/ld+json">
        {"@type":"JobPosting","hiringOrganization":{"name":"Braintrust"}}
        </script>
        """

    engine._listing_text = page
    opp = Opportunity(
        title="Senior ML Engineer",
        url="https://karkidi.example/x",
        pay_low=160_000,
        pay_high=200_000,
    )
    asyncio.run(engine._enrich_pay([opp]))
    assert captured == ["https://karkidi.example/x"]
    assert opp.company == "Braintrust"
    assert opp.pay_high == 200_000


def test_apply_listing_reads_json_ld_past_first_80k():
    from src.engine import _apply_listing

    html = (
        "<html><head><title>Role</title></head><body>"
        + ("x" * 81_000)
        + """<script type="application/ld+json">
        {"@type":"JobPosting","hiringOrganization":{"name":"Sword Health"}}
        </script></body></html>"""
    )
    opp = Opportunity(title="Senior ML Engineer", url="https://jobs.lever.co/swordhealth/1")
    _apply_listing(opp, html)
    assert opp.company == "Sword Health"


def test_apply_listing_pay_not_blocked_by_css_dollar_prefix():
    from src.engine import _apply_listing

    html = (
        "<style>" + ("$iconThumbnailMarginX;" * 5000) + "</style>"
        "<p>for this full-time position is $143,000 to 197,000.</p>"
    )
    opp = Opportunity(title="Senior ML Engineer", url="https://jobs.lever.co/lyrahealth/x")
    _apply_listing(opp, html)
    assert opp.pay_low == 143_000
    assert opp.pay_high == 197_000


def test_listing_text_fetches_lever_job_not_apply_form(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        return "<title>Provectus - Senior ML</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    asyncio.run(
        engine._listing_text(
            "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply"
        )
    )
    assert seen == [
        "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"
    ]


def test_apply_listing_company_from_html_title():
    from src.engine import _apply_listing

    html = "<title>Job Application for Senior ML Engineer I // II at Signifyd</title><p>Apply</p>"
    opp = Opportunity(title="Senior Machine Learning Engineer I // II", url="https://job-boards.greenhouse.io/signifyd95/jobs/1")
    _apply_listing(opp, html)
    assert opp.company == "Signifyd"


def test_enrich_drops_fetched_board_index_html():
    engine = Engine()

    async def page(_url: str) -> str:
        return "<title>Jobs at Grafana Labs</title><p>Current openings</p>"

    engine._listing_text = page
    keep = Opportunity(title="Real", url="https://jobs.example/real", pay_high=100_000, company="Acme")
    ghost = Opportunity(
        title="Senior Machine Learning Engineer, Developer Advocacy | US | Remote",
        url="https://job-boards.greenhouse.io/grafanalabs/jobs/1",
    )
    opps = [keep, ghost]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Real"]


def test_greenhouse_api_url_from_job_board_link():
    from src.engine import _greenhouse_api_url

    assert _greenhouse_api_url(
        "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    ) == "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831"
    assert _greenhouse_api_url(
        "https://job-boards.eu.greenhouse.io/jetbrains/jobs/4713663101"
    ) == "https://boards-api.greenhouse.io/v1/boards/jetbrains/jobs/4713663101"
    assert _greenhouse_api_url("https://jobs.lever.co/lyrahealth/abc") is None


def test_greenhouse_api_html_fills_company_and_pay_range():
    from src.engine import _apply_listing, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Reddit",
            "title": "Senior Machine Learning Engineer",
            "location": {"name": "Remote - United States"},
            "content": (
                "&lt;div class=&quot;pay-range&quot;&gt;"
                "&lt;span&gt;$216,700&lt;/span&gt;&lt;span&gt;&amp;mdash;&lt;/span&gt;"
                "&lt;span&gt;$303,400 USD&lt;/span&gt;&lt;/div&gt;"
            ),
        }
    )
    opp = Opportunity(
        title="Senior Machine Learning Engineer, ML Efficiency",
        url="https://job-boards.greenhouse.io/reddit/jobs/6960831",
    )
    _apply_listing(opp, html)
    assert opp.company == "Reddit"
    assert opp.pay_low == 216_700
    assert opp.pay_high == 303_400


def test_listing_text_prefers_greenhouse_api_over_board_shell(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return json.dumps(
                {
                    "company_name": "Reddit",
                    "title": "Senior ML",
                    "content": "$180,000",
                    "location": {"name": "Remote - United States"},
                }
            )
        return "<title>Jobs at Reddit</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://job-boards.greenhouse.io/reddit/jobs/6960831")
    )
    assert seen[0] == "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831"
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://job-boards.greenhouse.io/reddit/jobs/6960831")
    _apply_listing(opp, html)
    assert opp.company == "Reddit"
    assert opp.pay_high == 180_000


def test_listing_plain_text_ignores_script_salaries():
    from src.engine import _listing_plain_text, _parse_pay, _visible_text

    html = '<script>budget = "$500,000"</script><p>Apply now. No salary listed.</p>'
    assert _parse_pay(_visible_text(html)) == (None, 500_000)
    assert _parse_pay(_listing_plain_text(html)) == (None, None)


def test_public_http_url_rejects_localhost():
    from src.engine import _public_http_url

    assert _public_http_url("https://careers.example/x") is True
    assert _public_http_url("http://127.0.0.1/secret") is False
    assert _public_http_url("javascript:alert(1)") is False


def test_extract_batch_fills_rows_llm_omitted_and_dedupes_url_aliases():
    engine = Engine()
    engine.openai = _fake_client(
        json.dumps(
            {
                "opportunities": [
                    {
                        "title": "From LLM",
                        "url": "https://jobs.example/a",
                        "pay_high": 200_000,
                        "hours_per_week": 20,
                    },
                    {
                        "title": "Alias of A",
                        "url": "HTTPS://JOBS.EXAMPLE/A/",
                        "pay_high": 1,
                        "hours_per_week": 1,
                    },
                ]
            }
        )
    )
    batch = [
        {"title": "Raw A", "url": "https://jobs.example/a", "description": "", "pay": 90_000, "hours": 40},
        {"title": "Junior Developer", "url": "https://jobs.example/b", "description": "hybrid"},
    ]
    out = asyncio.run(engine._extract_batch(batch, "q"))
    assert [o.url for o in out] == ["https://jobs.example/a", "https://jobs.example/b"]
    assert out[0].title == "From LLM"
    assert out[0].pay_high == 90_000
    assert out[0].hours_per_week == 40
    assert out[1].title == "Junior Developer"
    assert out[1].pay_high is None
    assert out[1].score() == 0
