import asyncio
import json

import pytest
from pydantic import ValidationError

from jobhunter.hunt import hunt
from jobhunter.profile import Profile
from jobhunter.registry import Registry
from src.models import Opportunity


class Sink:
    def __init__(self):
        self.rows = {}
        self.fail = False

    def sync(self, records):
        if self.fail:
            raise RuntimeError("sheet unavailable")
        self.rows.update({r["key"]: r for r in records})


class Scout:
    def __init__(self, opportunities):
        self.opportunities = opportunities
        self.enriched = []

    async def _search_all(self, query):
        print("provider diagnostic must stay off stdout")
        return [{"title": o.title, "url": o.url} for o in self.opportunities]

    async def enrich(self, opportunities):
        for o in opportunities:
            self.enriched.append(o.url)
            source = next(x for x in self.opportunities if x.url.split("?")[0] == o.url)
            for name, value in source.model_dump().items():
                if name != "url":
                    setattr(o, name, value)


def role(**changes):
    return Opportunity.model_validate({
        "title": "Python engineer", "company": "Example",
        "url": "https://careers.example.com/jobs/1",
        "pay_low": 140000, "pay_high": 180000, "pay_source": "posted",
        "pay_source_url": changes.get("url", "https://careers.example.com/jobs/1"),
        "hours_per_week": 40, "hours_source": "posted",
        "remote": True, "remote_source": "posted", **changes,
    })


def profile(**changes):
    return Profile.model_validate({"query": "Python engineer", "employer_domains": ["careers.example.com"], **changes})


def test_hunt_scores_only_verified_matches_and_never_twice(tmp_path, monkeypatch, capsys):
    opportunities = [role(), role(url="https://careers.example.com/jobs/2", pay_source="snippet"),
                     role(url="https://careers.example.com/jobs/3", hours_source=None),
                     role(url="https://careers.example.com/jobs/4", remote_source="snippet"),
                     role(url="https://careers.example.com/jobs/5", remote=False)]
    scout, sink, registry = Scout(opportunities), Sink(), Registry(tmp_path)
    calls = []
    original = Opportunity.score
    monkeypatch.setattr(Opportunity, "score", lambda o: calls.append(o.url) or original(o))
    first = asyncio.run(hunt(profile(min_dollars_per_hour=70), registry, sink, scout))
    second = asyncio.run(hunt(profile(), registry, sink, scout))
    assert len(first) == 1 and first[0]["dollars_per_hour"] == 80
    assert second == []
    assert calls == [opportunities[0].url]
    assert len(scout.enriched) == 5
    assert len(sink.rows) == 5
    assert sum(r["gate_status"] == "unverified" for r in sink.rows.values()) == 3
    assert all(r["dollars_per_hour"] is None for r in sink.rows.values() if r["gate_status"] != "qualified")
    assert capsys.readouterr().out == ""


def test_tracking_urls_and_duplicates_share_one_claim(tmp_path):
    scout = Scout([role(), role(url=role().url + "?utm_source=board")])
    sink = Sink()
    assert len(asyncio.run(hunt(profile(), Registry(tmp_path), sink, scout))) == 1
    assert len(scout.enriched) == 1


def test_failed_sheet_retries_cached_result_without_rescoring(tmp_path, monkeypatch):
    sink, scout, registry = Sink(), Scout([role()]), Registry(tmp_path)
    sink.fail = True
    with pytest.raises(RuntimeError):
        asyncio.run(hunt(profile(), registry, sink, scout))
    sink.fail = False
    monkeypatch.setattr(Opportunity, "score", lambda _: pytest.fail("rescored"))
    assert len(asyncio.run(hunt(profile(), registry, sink, scout))) == 1
    assert len(sink.rows) == 1
    assert len(scout.enriched) == 1


@pytest.mark.parametrize("changes", [
    {"min_annual_pay": 170000}, {"max_hours_per_week": 30},
    {"min_dollars_per_hour": 90}, {"title_terms": ["rust"]},
    {"exclude_companies": ["example"]},
])
def test_user_gates_are_enforced_before_score(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(Opportunity, "score", lambda _: pytest.fail("filtered role scored"))
    sink = Sink()
    assert asyncio.run(hunt(profile(**changes), Registry(tmp_path), sink, Scout([role()]))) == []
    assert next(iter(sink.rows.values()))["gate_status"] == "filtered"


def test_unknown_employer_origin_is_unverified(tmp_path):
    sink = Sink()
    assert asyncio.run(hunt(profile(employer_domains=[]), Registry(tmp_path), sink, Scout([role()]))) == []
    assert next(iter(sink.rows.values()))["gate_status"] == "unverified"


def test_minimum_pay_uses_posted_floor_not_midpoint(tmp_path):
    assert asyncio.run(hunt(profile(min_annual_pay=150000), Registry(tmp_path), Sink(), Scout([role()]))) == []


def test_nonremote_rate_gate_includes_engine_penalty(tmp_path):
    assert asyncio.run(hunt(profile(remote_only=False, min_dollars_per_hour=70), Registry(tmp_path),
                            Sink(), Scout([role(remote=False)]))) == []


def test_employer_evidence_alias_is_not_scored_twice(tmp_path, monkeypatch):
    first = role(url="https://careers.example.com/jobs/first",
                 pay_source="ats", pay_source_url="https://boards-api.greenhouse.io/v1/boards/example/jobs/123")
    second = first.model_copy(update={"url": "https://job-boards.greenhouse.io/example/jobs/123"})
    calls = []
    monkeypatch.setattr(Opportunity, "score", lambda o: calls.append(o.url) or 80)
    assert len(asyncio.run(hunt(profile(), Registry(tmp_path), Sink(), Scout([first, second])))) == 1
    assert len(calls) == 1


def test_ashby_board_endpoint_does_not_collapse_distinct_jobs(tmp_path, monkeypatch):
    from jobhunter.hunt import HuntEngine
    from src.compensation import Compensation
    from src.engine import Engine
    engine = HuntEngine()
    async def search(_query):
        return [{"title": "Python engineer", "url": f"https://jobs.ashbyhq.com/example/{i}"}
                for i in ("job-one", "job-two")]
    async def ats(self, url, client=None):
        return Compensation(pay_low=160000, pay_high=160000, hours=40, remote=True)
    monkeypatch.setattr(engine, "_search_all", search)
    monkeypatch.setattr(Engine, "_fetch_ats", ats)
    assert len(asyncio.run(hunt(profile(), Registry(tmp_path), Sink(), engine))) == 2


def test_deceptive_ats_hostname_is_not_employer_evidence(tmp_path):
    fake = role(url="https://evilgreenhouse.io/example/jobs/123", pay_source="posted")
    assert asyncio.run(hunt(profile(), Registry(tmp_path), Sink(), Scout([fake]))) == []


def test_mirror_hours_cannot_be_combined_with_genuine_ats_pay(tmp_path):
    mirrored = role(url="https://untrusted.example/jobs/1", pay_source="ats",
                    pay_source_url="https://api.lever.co/v0/postings/example/123?mode=json",
                    hours_per_week=10, hours_source="posted", remote_source="posted")
    sink = Sink()
    assert asyncio.run(hunt(profile(), Registry(tmp_path), sink, Scout([mirrored]))) == []
    assert next(iter(sink.rows.values()))["gate_status"] == "unverified"


def test_unverified_mirror_cannot_claim_qualified_native_identity(tmp_path):
    mirror = role(url="https://untrusted.example/jobs/1", pay_source="ats",
                  pay_source_url="https://api.lever.co/v0/postings/example/123?mode=json",
                  hours_source="posted", remote_source="posted")
    native = role(url="https://jobs.lever.co/example/123", pay_source="ats",
                  pay_source_url=mirror.pay_source_url, hours_source="ats", remote_source="ats")
    matches = asyncio.run(hunt(profile(), Registry(tmp_path), Sink(), Scout([mirror, native])))
    assert len(matches) == 1 and matches[0]["url"] == native.url


def test_ats_early_return_still_verifies_hours_from_employer_html(tmp_path, monkeypatch):
    from jobhunter.hunt import HuntEngine
    from src.compensation import Compensation
    from src.engine import Engine
    engine = HuntEngine()
    async def search(_query):
        return [{"title": "Python engineer", "url": "https://jobs.lever.co/example/123"}]
    async def ats(self, url, client=None):
        return Compensation(pay_low=160000, pay_high=160000, remote=True)
    async def html(url, client=None):
        return "<p>Fully remote. 40 hours/week.</p>"
    monkeypatch.setattr(engine, "_search_all", search)
    monkeypatch.setattr(Engine, "_fetch_ats", ats)
    monkeypatch.setattr(engine, "_fetch_listing", html)
    matches = asyncio.run(hunt(profile(), Registry(tmp_path), Sink(), engine))
    assert len(matches) == 1 and matches[0]["dollars_per_hour"] == 80


def test_changed_gates_suppress_cached_notifications(tmp_path):
    registry, sink, scout = Registry(tmp_path), Sink(), Scout([role()])
    sink.fail = True
    with pytest.raises(RuntimeError):
        asyncio.run(hunt(profile(), registry, sink, scout))
    sink.fail = False
    assert asyncio.run(hunt(profile(min_annual_pay=200000), registry, sink, scout)) == []
    assert len(sink.rows) == 1


@pytest.mark.parametrize("printed", [True, False])
def test_total_discovery_failure_is_reported(tmp_path, monkeypatch, printed):
    from jobhunter.hunt import HuntEngine
    from src.engine import Engine
    engine = HuntEngine()
    engine.perplexity_key = ""
    async def fail(self, query):
        if printed:
            print("provider error: must not leak")
            return []
        raise RuntimeError("provider error: must not leak")
    monkeypatch.setattr(Engine, "_search_brave", fail)
    with pytest.raises(RuntimeError, match="discovery"):
        asyncio.run(hunt(profile(), Registry(tmp_path), Sink(), engine))


def test_annualized_pay_with_unknown_conversion_basis_never_scores(tmp_path):
    annualized = role(pay_low=200000, pay_high=200000, pay_is_annualized=True,
                      hours_per_week=20, pay_source="ats",
                      pay_source_url="https://api.lever.co/v0/postings/example/123?mode=json")
    sink = Sink()
    assert asyncio.run(hunt(profile(), Registry(tmp_path), sink, Scout([annualized]))) == []
    assert "conversion" in next(iter(sink.rows.values()))["reason"]


def test_allowlisted_host_cannot_launder_cross_host_redirect_evidence(monkeypatch):
    import httpx
    from jobhunter.hunt import HuntEngine, evaluate
    def respond(request):
        if request.url.host == "careers.example.com":
            return httpx.Response(302, headers={"location": "https://untrusted.example/job/1"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text=
                              "<p>Fully remote. Base salary: $160,000 annually. 40 hours/week.</p>")
    engine = HuntEngine()
    async def verify():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
            o = Opportunity(title="Python engineer", url=role().url)
            await engine._enrich_one(o, client)
            return evaluate(o, profile(), engine.field_urls(o))
    assert asyncio.run(verify())[0] == "unverified"


def test_same_host_redirect_alias_is_scored_once(tmp_path, monkeypatch):
    import httpx
    from jobhunter.hunt import HuntEngine
    def respond(request):
        if request.url.path.endswith("/old"):
            return httpx.Response(302, headers={"location": "https://careers.example.com/jobs/new"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text=
                              "<p>Fully remote. Base salary: $160,000 annually. 40 hours/week.</p>")
    engine = HuntEngine()
    async def search(query):
        return [{"title": "Python engineer", "url": f"https://careers.example.com/jobs/{s}"}
                for s in ("old", "new")]
    async def enrich(opportunities):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
            for o in opportunities:
                await engine._enrich_one(o, client)
    monkeypatch.setattr(engine, "_search_all", search)
    monkeypatch.setattr(engine, "enrich", enrich)
    assert len(asyncio.run(hunt(profile(), Registry(tmp_path), Sink(), engine))) == 1


def test_registry_lock_prevents_parallel_hunts(tmp_path):
    registry = Registry(tmp_path)
    with registry.lock():
        with pytest.raises(RuntimeError, match="already running"):
            with Registry(tmp_path).lock():
                pass


def test_incomplete_claim_is_not_rescored(tmp_path):
    registry, sink, scout = Registry(tmp_path), Sink(), Scout([role()])
    assert registry.claim(role().url)
    assert asyncio.run(hunt(profile(), registry, sink, scout)) == []
    assert scout.enriched == []


@pytest.mark.parametrize("changes", [{"max_hours_per_week": 0}, {"min_annual_pay": -1},
                                    {"remote_only": "false"}, {"unknown_gate": True}])
def test_invalid_profile_fails_closed(changes):
    with pytest.raises(ValidationError):
        profile(**changes)


def test_fixture_hunt_uses_engine_parsers(tmp_path):
    from jobhunter.hunt import DryEngine
    sink = Sink()
    matches = asyncio.run(hunt(profile(), Registry(tmp_path), sink, DryEngine(), dry_run=True))
    assert len(matches) == 1
    assert matches[0]["mode"] == "dry_run"
    assert matches[0]["dollars_per_hour"] == 80
    assert len(sink.rows) == 3
