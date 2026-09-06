import asyncio
import types

from src.engine import (
    Engine,
    _guess_remote,
    _parse_ddg_html,
    opportunity_from_raw,
)


# --- posted pay vs invented seniority -----------------------------------


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
    assert opp.pay_source == "posted"
    assert opp.hours_source == "posted"
    assert opp.dollars_per_hour == 90.0


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

    engine._search_all = fake_search
    ranked = asyncio.run(engine.find("eng", limit=10))
    assert [o.url for o in ranked] == ["https://example.com/listed", "https://example.com/thin"]
    assert ranked[0].pay_source == "posted"
    assert ranked[1].score() == 0


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
    assert opps[0].pay_source == "posted"


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
    assert opps[0].title == "Staff Engineer $180k"
    assert opps[0].pay is None
    assert opps[0].score() == 0


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
