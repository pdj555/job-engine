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
)
from src.models import Opportunity


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
            "title": "Senior ML Engineer",
            "url": "https://example.com/senior",
            "description": "must be onsite",
            "source": "ddg",
        }
    )
    assert guessed.pay_high == _guess_pay("Senior ML Engineer", "must be onsite") == 180_000
    assert guessed.hours_per_week is None
    assert guessed.rate_is_imputed is True
    assert guessed.refined_rate == 90.0
    assert guessed.remote is False


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
            {"title": "Raw A", "url": "https://jobs.example/a", "description": "a"},
            {"title": "Raw B", "url": "https://jobs.example/b", "description": "b"},
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
    assert guessed.pay_high == 90_000
    assert guessed.hours_per_week is None
    assert guessed.rate_is_imputed is True
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
    assert failed[0].pay_high == 180_000
    assert failed[0].hours_per_week is None
    assert failed[0].rate_is_imputed is True
    assert failed[0].efficiency == failed[0].refined_rate == 90.0

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
