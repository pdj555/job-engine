import asyncio

from src.engine import (
    Engine,
    _guess_hours,
    _guess_pay,
    _guess_remote,
    _parse_ddg_html,
)


# --- heuristic guessers -------------------------------------------------


def test_guess_pay_seniority_tiers():
    assert _guess_pay("Senior ML Engineer", "") == 180_000
    assert _guess_pay("Staff Engineer", "") == 180_000
    assert _guess_pay("Junior Developer", "") == 90_000
    assert _guess_pay("Freelance Designer", "") == 130_000
    assert _guess_pay("Software Engineer", "") == 120_000  # default


def test_guess_pay_reads_description_not_just_title():
    assert _guess_pay("Engineer", "principal level role") == 180_000


def test_guess_hours_part_time_vs_full():
    assert _guess_hours("Contract Engineer", "") == 30
    assert _guess_hours("Part-time role", "") == 30
    assert _guess_hours("Engineer", "") == 40  # default full-time


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
