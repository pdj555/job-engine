import asyncio
import types

from src.agent import _parse, _rank, agent_run
from src.models import Opportunity


# --- deterministic ranking (the math the brain is never trusted with) ---


def test_rank_orders_by_dollars_per_hour():
    items = [
        {"title": "Low", "url": "https://a.example/1", "pay": 100_000, "hours_per_week": 40},
        {"title": "High", "url": "https://a.example/2", "pay": 200_000, "hours_per_week": 20},
    ]
    assert [o.title for o in _rank(items)] == ["High", "Low"]


def test_rank_dedupes_same_title():
    ranked = _rank(
        [
            {
                "title": "Senior ML Engineer",
                "url": "https://a.example/1",
                "pay": 100_000,
                "hours_per_week": 40,
            },
            {
                "title": "Senior ML Engineer",
                "url": "https://b.example/2",
                "pay": 200_000,
                "hours_per_week": 20,
            },
        ]
    )
    assert [o.url for o in ranked] == ["https://b.example/2"]


def test_rank_keeps_same_title_at_different_companies():
    ranked = _rank(
        [
            {
                "title": "Senior ML Engineer",
                "url": "https://jobs.ashbyhq.com/quilter/aaa",
                "company": "Quilter",
                "pay": 200_000,
                "hours_per_week": 40,
            },
            {
                "title": "Senior ML Engineer",
                "url": "https://jobs.ashbyhq.com/coralai/bbb",
                "company": "Coral AI",
                "pay": 150_000,
                "hours_per_week": 40,
            },
        ]
    )
    assert [o.company for o in ranked] == ["Quilter", "Coral AI"]


def test_rank_dedupes_same_role_across_boards():
    ranked = _rank(
        [
            {
                "title": "Senior ML Engineer (ML/AI) in Remote at Lyra Health",
                "url": "https://careers.example/lyra",
                "company": "Lyra Health",
                "pay": 197_000,
                "hours_per_week": 40,
            },
            {
                "title": "Lyra Health - Senior ML Engineer (ML/AI) - jobs.lever.co",
                "url": "https://jobs.lever.co/lyrahealth/abc",
                "pay": 100_000,
                "hours_per_week": 40,
            },
        ]
    )
    assert [o.url for o in ranked] == ["https://careers.example/lyra"]


def test_rank_skips_items_without_url():
    assert _rank([{"title": "no url", "pay": 100_000, "hours_per_week": 10}]) == []


def test_rank_skips_non_http_urls():
    ranked = _rank(
        [
            {"title": "bare", "url": "u1", "pay": 400_000, "hours_per_week": 10},
            {"title": "js", "url": "javascript:alert(1)", "pay": 400_000, "hours_per_week": 10},
            {"title": "ftp", "url": "ftp://files.example/x", "pay": 400_000, "hours_per_week": 10},
            {"title": "ok", "url": "https://jobs.example/x", "pay": 100_000, "hours_per_week": 40},
        ]
    )
    assert [o.title for o in ranked] == ["ok"]


def test_rank_skips_index_pages():
    ranked = _rank(
        [
            {
                "title": "Jobs - Indeed",
                "url": "https://www.indeed.com/q-ml-jobs.html",
                "pay": 400_000,
                "hours_per_week": 10,
            },
            {
                "title": "Real",
                "url": "https://jobs.example/x",
                "pay": 100_000,
                "hours_per_week": 40,
            },
        ]
    )
    assert [o.title for o in ranked] == ["Real"]


def test_rank_skips_upwork_apply_gate():
    ranked = _rank(
        [
            {
                "title": "AI/ML Engineer - Freelance Job",
                "url": "https://www.upwork.com/freelance-jobs/apply/Engineer_~022084959075748613623/",
                "pay": 400_000,
                "hours_per_week": 10,
            },
            {
                "title": "Real",
                "url": "https://jobs.example/x",
                "pay": 100_000,
                "hours_per_week": 40,
            },
        ]
    )
    assert [o.title for o in ranked] == ["Real"]


def test_rank_canonicalizes_lever_apply_url():
    ranked = _rank(
        [
            {
                "title": "Provectus ML",
                "url": "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply",
            }
        ]
    )
    assert ranked[0].url == "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"


def test_rank_company_from_lever_slug():
    ranked = _rank(
        [
            {
                "title": "Senior ML Engineer (Portugal Based Remote/Hybrid)",
                "url": "https://jobs.lever.co/swordhealth/770e2ca0-a6a4-4ca9-9c0f-ce419284ddbe",
            }
        ]
    )
    assert ranked[0].company == "Swordhealth"


def test_rank_company_from_title_when_field_missing():
    ranked = _rank(
        [
            {
                "title": "Senior Machine Learning Engineer at Lyra Health",
                "url": "https://job-boards.greenhouse.io/lyrahealth/jobs/123",
            }
        ]
    )
    assert ranked[0].company == "Lyra Health"


def test_rank_strips_job_application_title():
    ranked = _rank(
        [
            {
                "title": "Job Application for Senior AI/ML Engineer at Dragos",
                "url": "https://job-boards.greenhouse.io/dragos/jobs/5364876008",
            }
        ]
    )
    assert ranked[0].title == "Senior AI/ML Engineer at Dragos"
    assert ranked[0].company == "Dragos"


def test_rank_skips_hire_a_freelance_directory():
    ranked = _rank(
        [
            {
                "title": "Hire a Freelance Machine Learning Engineer — No Agency Fees",
                "url": "https://remoteai.io/v2/freelance/machine-learning-engineers",
                "pay": 400_000,
                "hours_per_week": 10,
            },
            {
                "title": "Real",
                "url": "https://jobs.example/x",
                "pay": 100_000,
                "hours_per_week": 40,
            },
        ]
    )
    assert [o.title for o in ranked] == ["Real"]


def test_rank_builds_opportunity_models_with_fields():
    ranked = _rank(
        [{"title": "X", "url": "https://acme.example/x", "company": "Acme", "pay": 120_000,
          "hours_per_week": 30, "remote": False}]
    )
    assert isinstance(ranked[0], Opportunity)
    assert ranked[0].company == "Acme"
    assert ranked[0].remote is False


# --- parsing Hermes' reply -----------------------------------------------


def test_parse_clean_object():
    out = _parse('{"searches": ["a"], "opportunities": [{"url": "u"}]}')
    assert out["searches"] == ["a"]
    assert out["opportunities"] == [{"url": "u"}]


def test_parse_tolerates_wrapping_prose():
    out = _parse('Here you go:\n{"opportunities": [{"url": "u"}]}\nThanks!')
    assert out["opportunities"] == [{"url": "u"}]


def test_parse_bare_array_becomes_opportunities():
    assert _parse('[{"url": "u"}]') == {"opportunities": [{"url": "u"}]}


def test_parse_garbage_is_empty_dict():
    assert _parse("not json at all") == {}


# --- end to end: Hermes brain mocked, real parse + rank ------------------


def _fake_client(content: str):
    """Stub the AsyncOpenAI surface agent_run touches: .chat.completions.create."""

    async def create(**kwargs):
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_agent_run_parses_and_ranks_hermes_reply(monkeypatch):
    reply = (
        '{"searches": ["remote ml contract", "ai grants"],'
        ' "opportunities": ['
        '   {"title": "Cheap", "url": "https://a.example/1", "pay": 100000, "hours_per_week": 40},'
        '   {"title": "Lush", "url": "https://a.example/2", "pay": 200000, "hours_per_week": 20}'
        ' ]}'
    )
    monkeypatch.setattr("src.agent._client", lambda: _fake_client(reply))

    run = asyncio.run(agent_run("find me work"))

    assert run.searches == ["remote ml contract", "ai grants"]
    assert [o.title for o in run.ranked] == ["Lush", "Cheap"]  # ranked by $/hr
    assert run.ranked[0].score() == 200.0


def test_agent_run_enriches_missing_pay_from_listing(monkeypatch):
    reply = (
        '{"searches": ["x"], "opportunities": ['
        '  {"title": "Senior ML", "url": "https://careers.example/x"}'
        ']}'
    )
    monkeypatch.setattr("src.agent._client", lambda: _fake_client(reply))

    from src.engine import Engine

    engine = Engine()

    async def page(_url: str) -> str:
        return "for this full-time position is $143,000 to 197,000."

    engine._listing_text = page
    monkeypatch.setattr("src.agent.get_engine", lambda: engine)

    run = asyncio.run(agent_run("ml"))
    assert run.ranked[0].pay_high == 197_000
    assert run.ranked[0].score() == 98.5


def test_agent_run_enriches_company_from_json_ld(monkeypatch):
    reply = (
        '{"searches": ["x"], "opportunities": ['
        '  {"title": "Senior ML Engineer", "url": "https://karkidi.example/x",'
        '   "pay": 200000}'
        ']}'
    )
    monkeypatch.setattr("src.agent._client", lambda: _fake_client(reply))

    from src.engine import Engine

    engine = Engine()

    async def page(_url: str) -> str:
        return (
            '<script type="application/ld+json">'
            '{"@type":"JobPosting","hiringOrganization":{"name":"Braintrust"},'
            '"baseSalary":{"currency":"USD","value":{"minValue":80,"maxValue":100,"unitText":"HOUR"}}}'
            "</script>"
        )

    engine._listing_text = page
    monkeypatch.setattr("src.agent.get_engine", lambda: engine)

    run = asyncio.run(agent_run("ml"))
    assert run.ranked[0].company == "Braintrust"
    assert run.ranked[0].pay_high == 200_000
