"""Exercise real search parsing, HTTP enrichment, filtering, ranking, and API/SDK paths."""

import asyncio
import json
import types

import httpx
import pytest
from fastapi.testclient import TestClient

from src.agent import ScoutHit, ScoutResult, agent_run
from src.api import routes
from src.compensation import (
    is_syndicated_listing,
    parse_ats_json,
    parse_compensation,
    parse_job_posting,
    parse_listing_pay,
)
from src.engine import Engine, opportunity_from_raw


@pytest.fixture
def listing_network(monkeypatch):
    hits = [
        {
            "title": "AI Engineer $600k",
            "url": "https://jobs.lever.co/acme/hybrid",
            "description": "Remote, 10 hours/week",
        },
        {
            "title": "AI Engineer $245k",
            "url": "https://job-boards.greenhouse.io/acme/jobs/1",
            "description": "Remote, 20 hours/week",
        },
        {"title": "AI Engineer $230k", "url": "https://jobs.lever.co/acme/narrow"},
        {"title": "AI Engineer $500k", "url": "https://careers.acme.com/embedded"},
        {
            "title": "AI Engineer $700k",
            "url": "https://careers.acme.com/unavailable",
            "description": "Remote",
        },
        {"title": "AI Engineer", "url": "https://jobs.lever.co/acme/missing"},
        {
            "title": "AI Engineer $400k",
            "url": "https://jobs.lever.co/acme/unknown",
            "description": "Remote",
        },
    ]
    requests = []

    def serve(request):
        requests.append(str(request.url))
        host, path = request.url.host, request.url.path
        if host == "api.search.brave.com":
            return httpx.Response(200, json={"web": {"results": hits}})
        if host == "boards-api.greenhouse.io":
            assert request.url.params["pay_transparency"] == "true"
            return httpx.Response(
                200,
                json={
                    "title": "AI Engineer",
                    "location": {"name": "Remote - US"},
                    "pay_input_ranges": [
                        {
                            "title": "Base salary",
                            "currency_type": "USD",
                            "min_cents": 20_000_000,
                            "max_cents": 24_500_000,
                        }
                    ],
                },
            )
        if host == "api.lever.co":
            job_id = path.rsplit("/", 1)[-1]
            payload = {"text": "AI Engineer", "categories": {"team": "Engineering"}}
            if job_id != "unknown":
                payload["workplaceType"] = "hybrid" if job_id == "hybrid" else "remote"
            if job_id != "missing":
                payload["salaryRange"] = {
                    "currency": "USD",
                    "interval": "per-year-salary",
                    "min": 300_000 if job_id == "hybrid" else 230_000,
                    "max": 600_000 if job_id == "hybrid" else 230_000,
                }
            return httpx.Response(200, json=payload)
        if path == "/embedded":
            return httpx.Response(
                200, text='<a href="https://jobs.ashbyhq.com/acme/embed">Apply</a>'
            )
        if host == "api.ashbyhq.com":
            return httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": "embed",
                            "title": "AI Engineer",
                            "isRemote": True,
                            "compensation": {
                                "summaryComponents": [
                                    {
                                        "compensationType": "Salary",
                                        "currencyCode": "USD",
                                        "interval": "1 YEAR",
                                        "minValue": 160_000,
                                        "maxValue": 240_000,
                                    }
                                ]
                            },
                        }
                    ]
                },
            )
        # An unrelated remote navigation link must not verify an unknown arrangement.
        if path.endswith("/unknown"):
            return httpx.Response(
                200,
                text="<nav>Locations Remote All Jobs</nav><main>"
                + "Build production AI systems. " * 15
                + "</main>",
            )
        return httpx.Response(503)

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(serve)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(**{**kwargs, "transport": transport, "trust_env": False}),
    )
    monkeypatch.setattr("src.engine.settings.brave_api_key", "test-search")
    monkeypatch.setattr("src.engine.settings.perplexity_api_key", "")
    monkeypatch.setattr("src.engine.settings.openai_api_key", "")
    return hits, requests


def test_search_to_rank_verifies_pay_and_filters_remote_before_limit(listing_network):
    _, requests = listing_network
    engine = Engine()
    ranked = asyncio.run(engine.find("fully remote AI engineer", limit=10))
    assert [o.url for o in ranked] == [
        "https://jobs.lever.co/acme/narrow",
        "https://job-boards.greenhouse.io/acme/jobs/1",
        "https://careers.acme.com/embedded",
        "https://jobs.lever.co/acme/missing",
    ]
    gh = ranked[1]
    assert (gh.pay_low, gh.pay_high, gh.pay) == (200_000, 245_000, 222_500)
    assert gh.pay_source == "ats" and "pay_transparency=true" in gh.pay_source_url
    # Unverified 20-hour snippet cannot double the rate from the posted annual salary.
    assert gh.refined_rate == 111.25 and gh.rate_is_imputed
    assert gh.dollars_per_hour is None
    assert all(o.remote is True and o.remote_source == "ats" for o in ranked)
    assert all(o.company != "Engineering" for o in ranked)
    assert ranked[-1].pay is None and ranked[-1].score() == 0
    assert len([u for u in requests if "api.search.brave.com" in u]) == 7
    assert len([u for u in requests if "boards-api.greenhouse.io" in u]) == 1
    assert asyncio.run(engine.find("fully remote AI engineer", limit=1))[0].pay == 230_000


def test_search_api_exposes_verified_midpoint_and_evidence(listing_network, monkeypatch):
    monkeypatch.setattr(routes, "engine", Engine())
    response = TestClient(routes.app).post("/search", json={"q": "fully remote AI engineer"})
    assert response.status_code == 200
    rows = response.json()["results"]
    assert len(rows) == 4
    assert rows[1]["pay"] == 222_500
    assert rows[1]["pay_source"] == "ats" and rows[1]["remote_source"] == "ats"
    assert rows[1]["rate_imputed"] is True
    assert rows[-1]["pay"] is None


def test_agent_api_fallback_uses_the_verified_search_path(listing_network):
    response = TestClient(routes.app).post(
        "/agent", json={"q": "fully remote AI engineer", "limit": 1}
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["pay"] == 230_000
    assert response.json()["count"] == 1


@pytest.mark.parametrize("typed_output", [True, False])
def test_sdk_search_tools_keep_candidates_until_after_verification(
    listing_network,
    monkeypatch,
    typed_output,
):
    hits, requests = listing_network
    monkeypatch.setattr("src.agent.settings.openai_api_key", "sk-test")

    async def fake_run(agent, input, max_turns=None):
        from agents.tool_context import ToolContext

        tool = next(t for t in agent.tools if t.name == "search_web")
        args = json.dumps({"q": input})
        ctx = ToolContext(
            context=None, tool_name="search_web", tool_call_id="test", tool_arguments=args
        )
        await tool.on_invoke_tool(ctx, args)
        out = ScoutResult(opportunities=[ScoutHit(**{**h, "remote": True}) for h in hits])
        return types.SimpleNamespace(final_output=out if typed_output else out.model_dump_json())

    monkeypatch.setattr("agents.Runner.run", fake_run)
    run = asyncio.run(agent_run("fully remote AI engineer", limit=1))
    assert len(run.ranked) == 1 and run.ranked[0].url.endswith("/narrow")
    assert run.ranked[0].pay == 230_000 and run.ranked[0].remote_source == "ats"
    # Later, initially unpriced candidates were enriched even with limit=1.
    assert any("posting-api/job-board/acme" in u for u in requests)


def test_llm_remote_guess_cannot_override_onsite_search_hit():
    engine = Engine()

    async def fake_create(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"opportunities":[{"url":"https://example.com/job","remote":true}]}'
                    )
                )
            ]
        )

    engine.openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )
    results = asyncio.run(
        engine._extract_batch(
            [
                {
                    "url": "https://example.com/job",
                    "title": "AI Engineer",
                    "description": "Hybrid in Dallas",
                }
            ],
            "fully remote AI engineer",
        )
    )
    assert results[0].remote is False and results[0].remote_source == "snippet"


@pytest.mark.parametrize("workplace", ["onsite", "hybrid"])
def test_ats_negative_remote_overrides_snippet(workplace):
    lever = parse_ats_json("https://jobs.lever.co/acme/1", {"workplaceType": workplace})
    ashby = parse_ats_json(
        "https://jobs.ashbyhq.com/acme/1",
        {"jobs": [{"id": "1", "workplaceType": workplace, "isRemote": False}]},
    )
    assert lever.remote is False and ashby.remote is False


def test_smartrecruiters_hybrid_description_overrides_remote_location_flag():
    parsed = parse_ats_json(
        "https://jobs.smartrecruiters.com/acme/123-role",
        {
            "location": {"remote": True},
            "jobAd": {
                "sections": {
                    "jobDescription": {
                        "text": "This is a hybrid role requiring three days in office."
                    }
                }
            },
        },
    )
    assert parsed.remote is False
    assert (
        parse_ats_json(
            "https://jobs.smartrecruiters.com/acme/123-role",
            {
                "location": {"remote": True, "hybrid": True},
            },
        ).remote
        is False
    )


def test_foreign_unknown_interval_estimates_and_ceiling_do_not_become_salary():
    assert not parse_compensation("Up to $500k potential earnings").posted
    assert not parse_ats_json(
        "https://jobs.lever.co/acme/1",
        {
            "salaryRange": {
                "min": 200_000,
                "max": 300_000,
                "currency": "CAD",
                "interval": "per-year-salary",
            }
        },
    ).posted
    assert not parse_ats_json(
        "https://jobs.lever.co/acme/1",
        {"salaryRange": {"min": 200_000, "max": 300_000, "currency": "USD", "interval": "unknown"}},
    ).posted
    assert not parse_listing_pay("<p>Glassdoor estimated salary range: $200k-$300k.</p>").posted


def test_hourly_salary_is_marked_as_annualized_in_schema_and_listing():
    parsed = parse_listing_pay("<p>Base salary: $100-$120 per hour.</p>")
    assert (parsed.pay_low, parsed.pay_high) == (200_000, 240_000) and parsed.annualized
    schema = parse_job_posting(
        '<script type="application/ld+json">'
        + json.dumps(
            {
                "@type": "JobPosting",
                "jobLocationType": "TELECOMMUTE",
                "baseSalary": {
                    "currency": "USD",
                    "value": {"minValue": 100, "maxValue": 120, "unitText": "HOUR"},
                },
            }
        )
        + "</script>"
    )
    assert schema.annualized and schema.remote is True


def test_remote_navigation_does_not_verify_the_job():
    text = (
        "<nav>Remote All Jobs</nav><main>" + "AI research in New York and Paris. " * 10 + "</main>"
    )
    assert parse_listing_pay(text).remote is None


@pytest.mark.parametrize(
    "url",
    [
        "https://remoteai.io/v2/jobs",
        "https://remoteai.io/find-work/united-states",
        "https://aitrainingjobs.org/freelance-ai-jobs",
        "https://remowork.life/salaries/ai-engineer",
    ],
)
def test_live_discovered_category_pages_are_not_job_opportunities(url):
    assert opportunity_from_raw({"title": "Remote AI jobs $300k", "url": url}) is None


def test_syndicated_schema_does_not_verify_employer_compensation():
    engine = Engine()

    async def html(url, client):
        return (
            '<script type="application/ld+json">'
            + json.dumps(
                {
                    "@type": "JobPosting",
                    "jobLocationType": "TELECOMMUTE",
                    "baseSalary": {
                        "currency": "USD",
                        "value": {"value": 700_000, "unitText": "YEAR"},
                    },
                }
            )
            + "</script>"
        )

    engine._fetch_listing = html
    result = asyncio.run(engine.read_listing("https://remoteai.io/v2/jobs/example-1"))
    assert result["pay_source"] == "unverified" and result["remote_source"] == "unverified"


@pytest.mark.parametrize(
    "url,payload",
    [
        (
            "https://job-boards.greenhouse.io/acme/jobs/1",
            {"pay_input_ranges": [{"min_cents": 20_000_000, "max_cents": 24_500_000}]},
        ),
        (
            "https://jobs.lever.co/acme/1",
            {"salaryRange": {"min": 200_000, "max": 245_000, "interval": "per-year-salary"}},
        ),
        (
            "https://jobs.ashbyhq.com/acme/1",
            {
                "jobs": [
                    {
                        "id": "1",
                        "compensation": {
                            "summaryComponents": [
                                {
                                    "compensationType": "Salary",
                                    "interval": "1 YEAR",
                                    "minValue": 200_000,
                                    "maxValue": 245_000,
                                }
                            ]
                        },
                    }
                ]
            },
        ),
    ],
)
def test_unknown_currency_does_not_default_to_usd(url, payload):
    assert not parse_ats_json(url, payload).posted


@pytest.mark.parametrize("board", ["PhillyTechCo", "NextStepSystems", "ParallelPartners1"])
def test_known_recruiter_ats_boards_are_not_direct_employer_evidence(board):
    assert is_syndicated_listing(f"https://jobs.smartrecruiters.com/{board}/123-role")
    assert not is_syndicated_listing("https://jobs.smartrecruiters.com/NBCUniversal3/123-role")


def test_html_schema_cannot_replace_more_authoritative_ats_pay():
    engine = Engine()

    async def ats(url, client):
        from src.compensation import Compensation

        return Compensation(pay_low=200_000, pay_high=245_000)

    async def html(url, client):
        return (
            '<script type="application/ld+json">'
            + json.dumps(
                {
                    "@type": "JobPosting",
                    "jobLocationType": "TELECOMMUTE",
                    "baseSalary": {
                        "currency": "USD",
                        "value": {"value": 300_000, "unitText": "YEAR"},
                    },
                }
            )
            + "</script>"
        )

    engine._fetch_ats, engine._fetch_listing = ats, html
    result = asyncio.run(engine.read_listing("https://job-boards.greenhouse.io/acme/jobs/1"))
    assert result["pay"] == 222_500 and result["pay_source"] == "ats"
    assert result["remote"] is True and result["remote_source"] == "schema"
