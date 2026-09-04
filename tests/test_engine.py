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
    assert _guess_pay("Engineer", "USD 200,000–240,000") == 240_000
    from src.engine import _parse_pay
    assert _parse_pay("**Salary:** USD 160,000–190,000") == (160_000, 190_000)
    assert _parse_pay("Base Salary: $126,000 - $180,000Diversity") == (126_000, 180_000)
    assert _parse_pay("proposed band b/t US$175k and $250k annually") == (175_000, 250_000)
    assert _parse_pay("$160,000 and $190,000") == (160_000, 190_000)
    assert _parse_pay("$180,000 and $5,000 signing bonus") == (None, 180_000)
    assert _parse_pay("Salary range: $190,000 $250,000 + performance-based bonus") == (
        190_000,
        250_000,
    )
    assert _parse_pay("$180K $200K") == (180_000, 200_000)
    assert _parse_pay("Salary: $157-200kApplicants must be authorized") == (
        157_000,
        200_000,
    )
    assert _parse_pay("Base Pay Range: $160,000 USD - $240,000 USD") == (
        160_000,
        240_000,
    )
    assert _guess_pay("Software Engineer", "") is None
    assert _guess_pay("Senior Staff Principal Lead", "junior intern") is None


_SIGNIFYD_GEO_PAY = """
Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000
Tier 2 (DC Metro/Austin/Boston/Los Angeles): $150,000 - $180,000
Tier 3 (US - All Other): $140,000 - $170,000
"""


def test_parse_pay_prefers_remote_geo_band():
    from src.engine import _parse_pay, _remote_geo_pay

    assert _parse_pay(_SIGNIFYD_GEO_PAY) == (160_000, 190_000)
    assert _parse_pay(_SIGNIFYD_GEO_PAY, remote=False) == (160_000, 190_000)
    assert _parse_pay(_SIGNIFYD_GEO_PAY, remote=True) == (140_000, 170_000)
    assert _remote_geo_pay(_SIGNIFYD_GEO_PAY) == (140_000, 170_000)
    assert _parse_pay("Tier 3 (US - All Other): $140k - $170k", remote=True) == (
        140_000,
        170_000,
    )
    assert _parse_pay(
        "NYC: $160,000 - $190,000\nRemote: $140,000 - $170,000", remote=True
    ) == (140_000, 170_000)
    assert _parse_pay(
        "We're a remote company. Salary: $160,000 - $190,000", remote=True
    ) == (160_000, 190_000)
    assert _parse_pay("$80 - $100 / Hour", remote=True) == (160_000, 200_000)


def test_foreign_salary_detects_k_suffix_gbp_and_eur():
    from src.engine import _foreign_salary, _parse_pay

    for blob in ("£60k", "£60K - £80K", "€85k", "GBP 60k", "EUR 85k"):
        assert _parse_pay(blob) == (None, None)
        assert _foreign_salary(f"<p>{blob} a year</p>") is True
    assert _foreign_salary("<p>$60k a year</p>") is False
    assert _foreign_salary("<p>Apply now. No salary listed.</p>") is False


def test_foreign_salary_detects_mxn_cad_and_salario_dollars():
    from src.engine import _foreign_salary, _parse_pay

    mx = "Salario bruto mensual entre $20,000 y $25,000"
    assert _parse_pay(mx) == (None, None)
    assert _foreign_salary(f"<p>{mx}</p>") is True
    assert _parse_pay("CAD $160,000 - $180,000") == (None, None)
    assert _foreign_salary("<p>CAD $160,000 - $180,000</p>") is True
    assert _parse_pay("C$90,000") == (None, None)
    assert _foreign_salary("<p>Pay is $180,000 CAD a year</p>") is True
    assert _parse_pay("$160,000 - 200,000 (CAD)") == (None, None)
    assert _foreign_salary("<p>The salary range for this role is $160,000 - 200,000 (CAD)</p>") is True
    assert _parse_pay("$180,000 a year") == (None, 180_000)
    assert _foreign_salary("<p>$180,000 a year</p>") is False
    assert _parse_pay("$15000 to $17000 gross Salary Monthly") == (None, None)
    assert _foreign_salary("<p>$15000 to $17000 gross Salary Monthly</p>") is True


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
    assert _guess_hours("Engineer", "32 hours a week") == 32
    assert _guess_hours("Engineer", "32 hours a week. This is a full-time role.") == 32
    assert _guess_hours("Engineer", "12 weeks of parental leave") is None
    assert _guess_hours("Part-time role", "") == 20
    assert _guess_hours("Full-time Engineer", "") == 40
    assert _guess_hours("Contract Engineer", "") is None
    assert _guess_hours("Engineer", "") is None


def test_apply_listing_reads_hours_a_week_for_rate():
    from src.engine import _apply_listing

    html = "<title>Engineer at Acme</title><p>$160,000 a year. 32 hours a week.</p>"
    opp = Opportunity(title="Engineer", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.pay_high == 160_000
    assert opp.hours_per_week == 32
    assert opp.rate_is_imputed is False
    assert opp.score() == 100.0


def test_apply_listing_stated_hours_beat_part_time_default():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Therapist","employmentType":"PART_TIME",
     "baseSalary":{"currency":"USD","value":{"minValue":80000,"maxValue":80000,"unitText":"YEAR"}}}
    </script>
    <p>Approximately 24 hours per week.</p>
    """
    opp = Opportunity(title="Therapist", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.hours_per_week == 24
    assert opp.pay_high == 80_000
    assert opp.rate_is_imputed is False
    assert opp.score() == 80_000 / (24 * 50)
    assert _guess_remote("Engineer", "hybrid schedule") is False
    assert _guess_remote("Engineer", "must be onsite") is False
    assert _guess_remote("Engineer", "must work on site") is False
    assert _guess_remote("Engineer", "fully distributed team") is True  # default
    assert _guess_remote("Engineer", "This role can be hybrid, or fully remote/virtually.") is True


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


def test_search_ddg_retries_202_then_parses(monkeypatch):
    import httpx

    hits: list[int] = []

    class FakeResp:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            hits.append(1)
            if len(hits) == 1:
                return FakeResp(202, "<html>challenge</html>")
            return FakeResp(200, DDG_HTML)

        async def get(self, _url, **_kwargs):
            return FakeResp(202, "<html>challenge</html>")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    rows = asyncio.run(Engine()._search_ddg("ml"))
    assert len(hits) == 2
    assert [r["url"] for r in rows] == [
        "https://example.com/job1",
        "https://example.org/job2",
    ]


def test_search_ddg_gives_up_after_202s(monkeypatch):
    import httpx

    hits: list[int] = []

    class FakeResp:
        status_code = 202
        text = "<html>challenge</html>"

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            hits.append(1)
            return FakeResp()

        async def get(self, _url, **_kwargs):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert asyncio.run(Engine()._search_ddg("ml")) == []
    assert len(hits) == 4


def test_search_ddg_retries_200_without_results(monkeypatch):
    import httpx

    hits: list[int] = []

    class FakeResp:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            hits.append(1)
            if len(hits) == 1:
                return FakeResp(200, "<html>challenge</html>")
            return FakeResp(200, DDG_HTML)

        async def get(self, _url, **_kwargs):
            return FakeResp(202, "<html>challenge</html>")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    rows = asyncio.run(Engine()._search_ddg("ml"))
    assert len(hits) == 2
    assert rows[0]["url"] == "https://example.com/job1"


DDG_LITE_HTML = """
<table border="0">
  <tr>
    <td>1.&nbsp;</td>
    <td>
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fjobs.ashbyhq.com%2Fquilter%2F2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1&amp;rut=abc" class='result-link'>Senior ML Engineer @ Quilter</a>
    </td>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td class='result-snippet'><b>Senior</b> <b>ML</b> Engineer. Remote. $180K – $200K.</td>
  </tr>
  <tr>
    <td>2.&nbsp;</td>
    <td>
      <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fjobs.lever.co%2Fswordhealth%2F50945411-2f43-421a-8bb8-86aa1de6d890&amp;rut=def" class='result-link'>Sword Health - Senior ML</a>
    </td>
  </tr>
</table>
"""


def test_parse_ddg_lite_unwraps_uddg_and_snippets():
    results = _parse_ddg_html(DDG_LITE_HTML)
    assert [r["url"] for r in results] == [
        "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
        "https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
    ]
    assert results[0]["title"] == "Senior ML Engineer @ Quilter"
    assert results[0]["description"] == "Senior ML Engineer. Remote. $180K – $200K."
    assert results[1]["description"] == ""


def test_search_ddg_falls_back_to_lite_when_html_202s(monkeypatch):
    import httpx

    posts: list[int] = []
    gets: list[str] = []

    class FakeResp:
        def __init__(self, status_code: int, text: str):
            self.status_code = status_code
            self.text = text

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def post(self, _url, **_kwargs):
            posts.append(1)
            return FakeResp(202, "<html>challenge</html>")

        async def get(self, url, **kwargs):
            gets.append(url)
            assert kwargs.get("params", {}).get("q") == "ml"
            return FakeResp(200, DDG_LITE_HTML)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_k: FakeClient())
    rows = asyncio.run(Engine()._search_ddg("ml"))
    assert posts == [1]
    assert gets == ["https://lite.duckduckgo.com/lite/"]
    assert [r["url"] for r in rows] == [
        "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
        "https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
    ]


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
    assert lever.title == "Lyra Health - Senior ML Engineer (ML/AI)"
    gh_app = _heuristic_opportunity(
        {
            "title": "Job Application for Senior, ML Engineer - VLM at Torc Robotics",
            "url": "https://job-boards.greenhouse.io/torcrobotics/jobs/8572505002",
            "description": "",
        }
    )
    assert gh_app is not None
    assert gh_app.title == "Senior, ML Engineer - VLM at Torc Robotics"
    assert gh_app.company == "Torc Robotics"
    workable = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer | Canopy | Jobs By Workable",
            "url": "https://jobs.workable.com/view/7mMjfHgS93LyPeHLK2XeMV/remote-senior-machine-learning-engineer-in-detroit-at-canopy",
            "description": "Remote role.",
        }
    )
    assert workable is not None
    assert workable.company == "Canopy"
    assert workable.title == "Senior ML Engineer | Canopy"
    assert (
        _heuristic_opportunity(
            {
                "title": "Intuition Machines, Inc. - Current Openings",
                "url": "https://apply.workable.com/imachines",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "A2Z Sync - Current Openings",
                "url": "https://apply.workable.com/a2z-sync/",
                "description": "",
            }
        )
        is None
    )
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
    assert (
        _heuristic_opportunity(
            {
                "title": "AI/ML Engineer - Freelance Job in AI & Machine Learning - Upwork",
                "url": "https://www.upwork.com/freelance-jobs/apply/Engineer_~022084959075748613623/",
                "description": "Senior Machine Learning Engineer contract.",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "ML Engineer - Lemon.io",
                "url": "https://lemon.io/for-developers/ml-engineer-jobs/",
                "description": "ML Engineer on an oncology KOL analytics backend $35-$100/hr",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Ilias - Senior Machine Learning Engineer expert on Lemon.io",
                "url": "https://magic.lemon.io/share/ilias-s-gabgcvgom",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior AI/ML Developer : Remote : Contract - Corp to Corp",
                "url": "https://corptocorp.org/senior-ai-ml-developer-remote-contract/",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning (ML) Engineer - Freelance [Remote]",
                "url": "https://www.karkidi.com/job-details/76760-senior-machine-learning-ml-engineer-freelance-remote-job",
                "description": "Braintrust $80 - $100 / Hour. Posted on: 17 Apr 2024",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning (ML) Engineer - Freelance [Remote]",
                "url": "https://www.jobleads.com/us/job/senior-machine-learning-ml-engineer-freelance-remote-job",
                "description": "",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.glassdoor.com/Job/remote-us-machine-learning-engineer-jobs-SRCH_IL.0,9_IS1_KO10,36.htm",
                "description": "$160K–$240K",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.remoterocketship.com/company/acme/jobs/senior-ml-engineer",
                "description": "$160k remote",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://migratemate.co/jobs/senior-machine-learning-engineer",
                "description": "United States $180k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.builtin.com/jobs/remote/ml",
                "description": "$160k–$200k",
            }
        )
        is None
    )
    assert (
        _heuristic_opportunity(
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.ziprecruiter.com/Jobs/Senior-Machine-Learning-Engineer",
                "description": "$160k–$200k",
            }
        )
        is None
    )
    kept_listing = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://www.glassdoor.com/job-listing/senior-machine-learning-engineer-acme-JV_IC1147401_KO0,32.htm",
            "description": "$180k–$220k",
        }
    )
    assert kept_listing is not None
    assert kept_listing.pay_high == 220_000
    kept_builtin = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://www.builtin.com/job/senior-machine-learning-engineer/12345",
            "description": "$180k–$220k",
        }
    )
    assert kept_builtin is not None
    amgen = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer Jobs at Amgen in United States - Remote",
            "url": "https://careers.amgen.com/en/job/washington-d-c/senior-machine-learning-engineer/87/99808047504",
            "description": "Remote",
        }
    )
    assert amgen is not None
    from src.engine import _is_index_page

    assert _is_index_page(
        {
            "url": "https://job-boards.greenhouse.io/grafanalabs/jobs/1",
            "title": "Jobs at Grafana Labs",
            "description": "",
        }
    )
    assert _is_index_page(
        {"url": "https://jobs.ashbyhq.com/acme", "title": "Jobs", "description": ""}
    )
    yelp = _heuristic_opportunity(
        {
            "title": "Careers at Yelp | Yelp Jobs",
            "url": "https://uscareers-yelp.icims.com/jobs/13815/senior-machine-learning-engineer/job",
            "description": "Remote United States",
        }
    )
    assert yelp is not None


def test_heuristic_stores_lever_job_url_not_apply():
    h = _heuristic_opportunity(
        {
            "title": "Provectus - Senior AI/ML Engineer (GenAI, AWS)",
            "url": "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply",
            "description": "",
        }
    )
    assert h.url == "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"


def test_search_all_drops_index_pages():
    engine = Engine()

    async def fake_brave(_query: str):
        return [
            {
                "title": "Jobs - Indeed",
                "url": "https://www.indeed.com/q-ml-jobs.html",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.glassdoor.com/Job/remote-us-machine-learning-engineer-jobs-SRCH_IL.0,9_IS1_KO10,36.htm",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.remoterocketship.com/company/acme/jobs/senior-ml-engineer",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://migratemate.co/jobs/senior-machine-learning-engineer",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.builtin.com/jobs/remote/ml",
            },
            {
                "title": "Senior Machine Learning Engineer",
                "url": "https://www.ziprecruiter.com/jobs-search?search=ml",
            },
            {"title": "Real role", "url": "https://jobs.example/ml"},
        ]

    async def fake_perplexity(_query: str):
        return []

    engine._search_brave = fake_brave
    engine._search_perplexity = fake_perplexity
    results = asyncio.run(engine._search_all("ml"))
    assert [r["url"] for r in results] == ["https://jobs.example/ml"]


def test_search_all_runs_site_angles_before_generic():
    engine = Engine()
    seen: list[str] = []

    async def fake_brave(query: str):
        seen.append(query)
        return [{"title": "R", "url": f"https://jobs.example/{len(seen)}"}]

    engine._search_brave = fake_brave
    asyncio.run(engine._search_all("ml"))
    generic = [q for q in seen if "site:" not in q]
    sites = [q for q in seen if "site:" in q]
    assert generic
    assert sites
    assert seen == sites + generic


def test_search_all_retries_empty_site_angles_after_generic():
    engine = Engine()
    seen: list[str] = []

    async def fake_brave(query: str):
        seen.append(query)
        if "ashbyhq.com" in query and seen.count(query) == 1:
            return []
        return [{"title": "R", "url": f"https://jobs.example/{len(seen)}"}]

    engine._search_brave = fake_brave
    results = asyncio.run(engine._search_all("ml"))
    ashby = "ml site:jobs.ashbyhq.com"
    assert seen.count(ashby) == 2
    assert seen.index(ashby) < seen.index("ml")
    assert seen[-1] == ashby
    assert "https://jobs.example/13" in [r["url"] for r in results]


def test_search_angles_omit_grants_and_equity_unless_asked():
    job = _search_angles("senior ML engineer remote")
    assert job == [
        "senior ML engineer remote",
        "senior ML engineer remote job hiring",
        "senior ML engineer remote freelance contract",
        "senior ML engineer remote site:greenhouse.io",
        "senior ML engineer remote site:jobs.lever.co",
        "senior ML engineer remote site:jobs.eu.lever.co",
        "senior ML engineer remote site:jobs.ashbyhq.com",
        "senior ML engineer remote site:jobs.workable.com",
        "senior ML engineer remote site:apply.workable.com",
        "senior ML engineer remote site:jobs.smartrecruiters.com",
        "senior ML engineer remote site:myworkdayjobs.com",
        "senior ML engineer remote site:icims.com",
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


def test_dedupe_keeps_same_title_at_different_companies():
    from src.engine import _dedupe_opportunities

    quilter_low = Opportunity(
        title="Senior ML Engineer @ Quilter",
        url="https://jobs.ashbyhq.com/quilter/low",
        company="Quilter",
        pay_high=100_000,
        hours_per_week=40,
    )
    quilter = Opportunity(
        title="Senior ML Engineer",
        url="https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
        company="Quilter",
        pay_low=180_000,
        pay_high=200_000,
        hours_per_week=40,
    )
    coral = Opportunity(
        title="Senior ML Engineer",
        url="https://jobs.ashbyhq.com/coralai/1ce17887-c305-4d77-a659-f75cf74bf8af",
        company="Coral AI",
    )
    ranked = sorted(
        [coral, quilter_low, quilter],
        key=lambda o: o.score(),
        reverse=True,
    )
    out = _dedupe_opportunities(ranked)
    assert [o.company for o in out] == ["Quilter", "Coral AI"]
    assert out[0].url == quilter.url
    assert out[0].pay_high == 200_000


def test_heuristic_company_from_lever_prefix():
    h = _heuristic_opportunity(
        {
            "title": "Lyra Health - Senior ML Engineer (ML/AI) - jobs.lever.co",
            "url": "https://jobs.lever.co/lyrahealth/d33ddfed-8c69-4e29-966b-0e190190cd6a",
            "description": "",
        }
    )
    assert h.company == "Lyra Health"


def test_heuristic_lever_requisition_suffix_is_not_company():
    h = _heuristic_opportunity(
        {
            "title": "IT Network Engineer II - 936",
            "url": "https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7",
            "description": "",
        }
    )
    assert h.company == "Quantinuum"


def test_heuristic_company_from_ashby_at():
    h = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer @ Quilter",
            "url": "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1/application",
            "description": "",
        }
    )
    assert h.company == "Quilter"
    assert h.url == "https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1"


def test_apply_listing_ashby_json_ld_pay():
    from src.engine import _apply_listing

    html = """
    <title>Senior ML Engineer @ Quilter</title>
    <script type="application/ld+json">
    {"@type":"JobPosting","hiringOrganization":{"name":"Quilter"},
     "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
    </script>
    """
    opp = Opportunity(
        title="Senior ML Engineer @ Quilter",
        url="https://jobs.ashbyhq.com/quilter/2b0f95cb-7c8b-4b62-8bcb-b9993344f2f1",
    )
    _apply_listing(opp, html)
    assert opp.company == "Quilter"
    assert opp.pay_low == 180_000
    assert opp.pay_high == 200_000


def test_heuristic_company_from_workable_apply_title():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer - Multi Media LLC",
            "url": "https://apply.workable.com/multimediallc/j/73CB637EE8",
            "description": "",
        }
    )
    assert h.company == "Multi Media LLC"
    assert h.url == "https://apply.workable.com/multimediallc/j/73CB637EE8"


def test_heuristic_stores_workable_job_url_not_markdown():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://apply.workable.com/runware/jobs/view/B0A0A14125.md",
            "description": "",
        }
    )
    assert h.url == "https://apply.workable.com/runware/j/B0A0A14125"
    assert h.company == "Runware"


def test_apply_listing_workable_markdown_pay():
    from src.engine import _apply_listing, _workable_to_html

    md = """# Senior Machine Learning Engineer

> Multi Media LLC · United States (Remote) · Full-time · Posted 2026-06-01

**Salary:** USD 200,000–240,000

**Workplace:** remote
"""
    opp = Opportunity(
        title="Senior Machine Learning Engineer - Multi Media LLC",
        url="https://apply.workable.com/multimediallc/j/73CB637EE8",
    )
    _apply_listing(opp, _workable_to_html(md))
    assert opp.company == "Multi Media LLC"
    assert opp.pay_low == 200_000
    assert opp.pay_high == 240_000
    assert opp.hours_per_week == 40


def test_listing_text_prefers_workable_markdown_over_spa_shell(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if url.endswith(".md"):
            return (
                "# Senior Engineer, AI/ML\n\n"
                "> A2Z Sync · United States (Remote) · Full-time\n\n"
                "**Salary:** USD 160,000–190,000\n"
            )
        return "<title>Senior Engineer, AI/ML - A2Z Sync</title><p>Apply</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://apply.workable.com/a2z-sync/j/C95E51CDDA")
    )
    assert seen[0] == "https://apply.workable.com/a2z-sync/jobs/view/C95E51CDDA.md"
    from src.engine import _apply_listing

    opp = Opportunity(
        title="Senior Engineer, AI/ML - A2Z Sync",
        url="https://apply.workable.com/a2z-sync/j/C95E51CDDA",
    )
    _apply_listing(opp, html)
    assert opp.company == "A2Z Sync"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 190_000


def test_find_dedupes_workable_apply_and_jobs_board():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Senior Machine Learning Engineer - Multi Media LLC",
                "url": "https://apply.workable.com/multimediallc/j/73CB637EE8",
                "description": "USD 200,000–240,000",
            },
            {
                "title": "Senior Machine Learning Engineer | Multi Media LLC | Jobs By Workable",
                "url": "https://jobs.workable.com/view/bqkqSAJN2W35yHL1WmQ5C9/remote-machine-learning-engineer-in-united-states-at-multi-media-llc",
                "description": "",
            },
        ]

    engine._search_all = fake_search

    async def no_page(_url: str) -> str:
        return ""

    engine._listing_text = no_page
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.url for o in ranked] == [
        "https://apply.workable.com/multimediallc/j/73CB637EE8"
    ]
    assert ranked[0].pay_high == 240_000


def test_unify_workable_slug_with_real_name():
    from src.engine import _unify_board_companies

    named = Opportunity(
        title="Senior Machine Learning Engineer - Multi Media LLC",
        url="https://apply.workable.com/multimediallc/j/73CB637EE8",
        company="Multi Media LLC",
    )
    slugged = Opportunity(
        title="Other Role",
        url="https://apply.workable.com/multimediallc/j/AAAAAAAAAA",
        company="Multimediallc",
    )
    _unify_board_companies([named, slugged])
    assert slugged.company == "Multi Media LLC"


def test_heuristic_company_from_lever_slug():
    h = _heuristic_opportunity(
        {
            "title": "Senior ML Engineer (Portugal Based Remote/Hybrid)",
            "url": "https://jobs.lever.co/swordhealth/770e2ca0-a6a4-4ca9-9c0f-ce419284ddbe",
            "description": "",
        }
    )
    assert h.company == "Swordhealth"


def test_heuristic_title_company_wins_over_url_slug():
    h = _heuristic_opportunity(
        {
            "title": "Senior, ML Engineer - VLM at Torc Robotics",
            "url": "https://job-boards.greenhouse.io/torcrobotics/jobs/8572505002",
            "description": "",
        }
    )
    assert h.company == "Torc Robotics"


def test_heuristic_canonicalizes_greenhouse_embed_url():
    h = _heuristic_opportunity(
        {
            "title": "Senior Machine Learning Engineer",
            "url": "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831",
            "description": "$216,700",
        }
    )
    assert h.url == "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    assert h.company == "Reddit"


def test_find_dedupes_same_role_across_boards():
    engine = Engine()
    engine.openai = None

    async def fake_search(_query: str):
        return [
            {
                "title": "Senior ML Engineer (ML/AI) in Remote at Lyra Health",
                "url": "https://careers.example/lyra",
                "description": "$143,000 to 197,000",
            },
            {
                "title": "Lyra Health - Senior ML Engineer (ML/AI) - jobs.lever.co",
                "url": "https://jobs.lever.co/lyrahealth/abc",
                "description": "$100k",
            },
            {
                "title": "Lyra Health - Sr. ML Engineer (MLOps)",
                "url": "https://jobs.lever.co/lyrahealth/def",
                "description": "$90k",
            },
        ]

    engine._search_all = fake_search

    async def no_page(_url: str) -> str:
        return ""

    engine._listing_text = no_page
    ranked = asyncio.run(engine.find("ml", limit=20))
    assert [o.url for o in ranked] == [
        "https://careers.example/lyra",
        "https://jobs.lever.co/lyrahealth/def",
    ]


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


def test_enrich_drops_foreign_salary_keeps_unknown_usd():
    engine = Engine()

    async def page(url: str) -> str:
        if "eur" in url:
            return """
            <title>Senior ML Engineer</title>
            <p>€60,000 - €85,000 a year</p>
            """
        if "jsonld" in url:
            return """
            <script type="application/ld+json">
            {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
             "baseSalary":{"currency":"EUR","value":{"minValue":120000,"maxValue":180000,"unitText":"YEAR"}}}
            </script>
            """
        if "usd" in url:
            return "<title>Senior ML</title><p>$180,000 a year</p>"
        return "<title>Staff Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    usd = Opportunity(title="USD", url="https://jobs.example/usd")
    eur = Opportunity(title="EUR", url="https://jobs.example/eur")
    jsonld = Opportunity(title="JSON", url="https://jobs.example/jsonld")
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown")
    opps = [usd, eur, jsonld, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["USD", "Unknown"]
    assert usd.pay_high == 180_000
    assert unknown.pay_high is None


def test_enrich_drops_foreign_listing_even_when_snippet_has_dollars():
    engine = Engine()

    async def page(url: str) -> str:
        if "eur" in url:
            return """
            <title>Senior ML Engineer</title>
            <p>€60,000 - €85,000 a year</p>
            """
        if "jsonld" in url:
            return """
            <script type="application/ld+json">
            {"@type":"JobPosting","hiringOrganization":{"name":"Acme"},
             "baseSalary":{"currency":"EUR","value":{"minValue":120000,"maxValue":180000,"unitText":"YEAR"}}}
            </script>
            """
        return "<title>Staff Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    eur = Opportunity(
        title="EUR",
        url="https://jobs.example/eur",
        company="Acme",
        pay_high=180_000,
        hours_per_week=40,
    )
    jsonld = Opportunity(
        title="JSON",
        url="https://jobs.example/jsonld",
        company="Acme",
        pay_high=180_000,
        hours_per_week=40,
    )
    unknown = Opportunity(
        title="Unknown",
        url="https://jobs.example/unknown",
        company="Acme",
        pay_high=90_000,
    )
    opps = [eur, jsonld, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]
    assert unknown.pay_high == 90_000


def test_enrich_drops_foreign_k_suffix_pay():
    engine = Engine()

    async def page(url: str) -> str:
        if "gbp" in url:
            return "<title>Engineer</title><p>£60K - £80K plus equity</p>"
        return "<title>Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    gbp = Opportunity(title="GBP", url="https://jobs.example/gbp", company="Acme")
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown", company="Acme")
    opps = [gbp, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]
    assert unknown.pay_high is None


def test_enrich_drops_salario_dollar_pay_even_when_snippet_has_dollars():
    engine = Engine()

    async def page(url: str) -> str:
        if "mx" in url:
            return (
                "<title>Account Manager Lead</title>"
                "<p>Salario bruto mensual entre $20,000 y $25,000</p>"
            )
        return "<title>Engineer</title><p>Apply now. No salary listed.</p>"

    engine._listing_text = page
    mx = Opportunity(
        title="MX",
        url="https://jobs.example/mx",
        company="Lyra",
        pay_high=180_000,
        hours_per_week=40,
    )
    unknown = Opportunity(title="Unknown", url="https://jobs.example/unknown", company="Lyra")
    opps = [mx, unknown]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Unknown"]


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


def test_enrich_fetches_paid_named_listings_for_hours_and_gone_jobs():
    engine = Engine()

    async def page(url: str):
        if "gone" in url:
            return None
        if "thin" in url:
            return ""
        return """
        <script type="application/ld+json">
        {"@type":"JobPosting","title":"Senior ML Engineer",
         "hiringOrganization":{"name":"Quilter"},
         "employmentType":"FULL_TIME",
         "baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}}
        </script>
        """

    engine._listing_text = page
    priced = Opportunity(
        title="Senior ML Engineer @ Quilter",
        url="https://jobs.ashbyhq.com/quilter/live",
        company="Quilter",
        pay_high=100_000,
        hours_per_week=None,
    )
    ghost = Opportunity(
        title="Expired",
        url="https://jobs.ashbyhq.com/azx/gone",
        company="AZX",
        pay_high=140_000,
        hours_per_week=40,
    )
    thin = Opportunity(
        title="Timeout",
        url="https://jobs.ashbyhq.com/weave/thin",
        company="Weave",
        pay_high=90_000,
    )
    opps = [priced, ghost, thin]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.url for o in opps] == [
        "https://jobs.ashbyhq.com/quilter/live",
        "https://jobs.ashbyhq.com/weave/thin",
    ]
    assert priced.pay_low == 180_000
    assert priced.pay_high == 200_000
    assert priced.hours_per_week == 40
    assert priced.rate_is_imputed is False
    assert thin.pay_high == 90_000
    assert thin.hours_per_week is None


def test_unify_board_companies_prefers_real_name_over_slug():
    from src.engine import _unify_board_companies

    named = Opportunity(
        title="Sword Health - Senior ML Engineer (Europe-based/Remote)",
        url="https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
        company="Sword Health",
    )
    slugged = Opportunity(
        title="Senior ML Engineer (Portugal Based Remote/Hybrid)",
        url="https://jobs.lever.co/swordhealth/770e2ca0-a6a4-4ca9-9c0f-ce419284ddbe",
        company="Swordhealth",
    )
    other = Opportunity(
        title="Egen - Senior AI Engineer",
        url="https://jobs.lever.co/egen/1b870652-5768-45e9-b55b-4420e6402314",
        company="Egen",
    )
    _unify_board_companies([named, slugged, other])
    assert slugged.company == "Sword Health"
    assert named.company == "Sword Health"
    assert other.company == "Egen"


def test_enrich_unifies_slug_company_when_listings_already_priced():
    engine = Engine()

    async def page(_url: str) -> str:
        return ""

    engine._listing_text = page
    named = Opportunity(
        title="Sword Health - Senior ML",
        url="https://jobs.lever.co/swordhealth/aaa",
        company="Sword Health",
        pay_high=100_000,
    )
    slugged = Opportunity(
        title="Senior ML Engineer (Portugal)",
        url="https://jobs.lever.co/swordhealth/bbb",
        company="Swordhealth",
        pay_high=100_000,
    )
    asyncio.run(engine._enrich_pay([named, slugged]))
    assert slugged.company == "Sword Health"


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
        if "api.lever.co" in url:
            return json.dumps(
                {
                    "id": "0bf1decc-002c-4b0a-b97b-6407d2930fff",
                    "text": "Senior AI/ML Engineer (GenAI, AWS)",
                    "categories": {"commitment": "Full-time"},
                    "salaryRange": {
                        "min": 159300,
                        "max": 219245,
                        "currency": "USD",
                        "interval": "per-year-salary",
                    },
                    "description": "<p>Build GenAI systems.</p>",
                }
            )
        return "<title>Provectus - Senior ML</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply"
        )
    )
    assert seen == [
        "https://api.lever.co/v0/postings/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff")
    _apply_listing(opp, html)
    assert opp.title == "Senior AI/ML Engineer (GenAI, AWS)"
    assert opp.pay_low == 159_300
    assert opp.pay_high == 219_245
    assert opp.hours_per_week == 40


def test_lever_api_url_uses_eu_host():
    from src.engine import _lever_api_url

    assert _lever_api_url(
        "https://jobs.lever.co/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff/apply"
    ) == "https://api.lever.co/v0/postings/provectus/0bf1decc-002c-4b0a-b97b-6407d2930fff"
    assert _lever_api_url(
        "https://jobs.eu.lever.co/prima/cc0b6018-ef61-453f-8201-ab5e6db53e31"
    ) == "https://api.eu.lever.co/v0/postings/prima/cc0b6018-ef61-453f-8201-ab5e6db53e31"
    assert _lever_api_url(
        "https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7/apply"
    ) == "https://api.eu.lever.co/v0/postings/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7"
    assert _lever_api_url("https://jobs.eu.lever.co/prima") is None


def test_listing_text_reads_lever_eu_api(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "api.eu.lever.co" in url:
            return json.dumps(
                {
                    "id": "753dc869-e097-4ae9-89d1-81cf56de46a7",
                    "text": "IT Network Engineer II",
                    "workplaceType": "remote",
                    "categories": {"commitment": "Full-time"},
                    "salaryRange": {
                        "min": 86000,
                        "max": 108000,
                        "currency": "USD",
                        "interval": "per-year-salary",
                    },
                    "description": "<p>Run the network.</p>",
                }
            )
        return "<title>Jobs at Quantinuum</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7"
        )
    )
    assert seen == [
        "https://api.eu.lever.co/v0/postings/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(
        title="x",
        url="https://jobs.eu.lever.co/quantinuum/753dc869-e097-4ae9-89d1-81cf56de46a7",
    )
    _apply_listing(opp, html)
    assert opp.title == "IT Network Engineer II"
    assert opp.pay_low == 86_000
    assert opp.pay_high == 108_000
    assert opp.remote is True
    assert opp.score() == 54.0
    assert opp.company == "Quantinuum"


def test_apply_listing_company_from_html_title():
    from src.engine import _apply_listing

    html = "<title>Job Application for Senior ML Engineer I // II at Signifyd</title><p>Apply</p>"
    opp = Opportunity(title="Senior Machine Learning Engineer I // II", url="https://job-boards.greenhouse.io/signifyd95/jobs/1")
    _apply_listing(opp, html)
    assert opp.company == "Signifyd"


def test_enrich_drops_fetched_board_index_html():
    engine = Engine()

    async def page(url: str) -> str:
        if "grafanalabs" in url:
            return "<title>Jobs at Grafana Labs</title><p>Current openings</p>"
        return ""

    engine._listing_text = page
    keep = Opportunity(title="Real", url="https://jobs.example/real", pay_high=100_000, company="Acme")
    ghost = Opportunity(
        title="Senior Machine Learning Engineer, Developer Advocacy | US | Remote",
        url="https://job-boards.greenhouse.io/grafanalabs/jobs/1",
    )
    opps = [keep, ghost]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Real"]


def test_http_get_text_none_on_404_empty_on_403():
    from src.engine import _http_get_text

    class _Resp:
        def __init__(self, status: int, body: str):
            self.status_code = status
            self.text = body

    class _Client:
        def __init__(self, status: int):
            self.status = status

        async def get(self, _url: str):
            return _Resp(self.status, "x" * 1000)

    assert asyncio.run(_http_get_text(_Client(404), "https://jobs.lever.co/x")) is None
    assert asyncio.run(_http_get_text(_Client(410), "https://jobs.lever.co/x")) is None
    assert asyncio.run(_http_get_text(_Client(403), "https://jobs.lever.co/x")) == ""


def test_listing_text_none_when_canonical_page_is_gone(monkeypatch):
    engine = Engine()

    async def fake_get(_client, _url: str):
        return None

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.lever.co/provectus/76640225-4aa7-45a3-bcdc-cb156271057b"
        )
    )
    assert html is None


def test_listing_text_lever_api_404_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "api.lever.co" in url:
            return None
        return "<title>Jobs at Provectus</title><p>Current openings</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.lever.co/provectus/76640225-4aa7-45a3-bcdc-cb156271057b"
        )
    )
    assert seen == [
        "https://api.lever.co/v0/postings/provectus/76640225-4aa7-45a3-bcdc-cb156271057b"
    ]
    assert html is None


def test_ashby_ids_strips_application():
    from src.engine import _ashby_ids

    assert _ashby_ids(
        "https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc/application"
    ) == ("quilter", "9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc")
    assert _ashby_ids("https://jobs.ashbyhq.com/quilter") is None


def test_ashby_to_html_pay_from_scrapeable_summary():
    from src.engine import _apply_listing, _ashby_to_html

    html = _ashby_to_html(
        {
            "title": "Machine Learning Engineer",
            "employmentType": "FullTime",
            "workplaceType": "Remote",
            "descriptionHtml": "<p>Build ML systems.</p>",
            "scrapeableCompensationSalarySummary": "$180K - $200K",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
    )
    _apply_listing(opp, html)
    assert opp.title == "Machine Learning Engineer"
    assert opp.company == "Quilter"
    assert opp.pay_low == 180_000
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 40
    assert opp.rate_is_imputed is False
    assert opp.remote is True
    assert opp.score() == 100


def test_apply_listing_reads_workplace_from_listing():
    from src.engine import _apply_listing, _ashby_to_html, _lever_to_html, _workable_jobs_to_html

    hybrid = _ashby_to_html(
        {
            "title": "Engineer",
            "employmentType": "FullTime",
            "workplaceType": "Hybrid",
            "descriptionHtml": "<p>Build systems.</p>",
            "scrapeableCompensationSalarySummary": "$180K - $200K",
        }
    )
    ashby = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/acme/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
        remote=True,
    )
    _apply_listing(ashby, hybrid)
    assert ashby.remote is False
    assert ashby.pay_high == 200_000
    assert ashby.score() == 70.0

    lever = Opportunity(title="x", url="https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", remote=True)
    _apply_listing(
        lever,
        _lever_to_html(
            {
                "text": "Engineer",
                "workplaceType": "onsite",
                "categories": {"commitment": "Full-time"},
                "description": "<p>Build systems. $160,000 - $180,000</p>",
            }
        ),
    )
    assert lever.remote is False
    assert lever.pay_low == 160_000
    assert lever.pay_high == 180_000

    workable = Opportunity(title="x", url="https://jobs.workable.com/view/abc", remote=True)
    _apply_listing(
        workable,
        _workable_jobs_to_html(
            {
                "title": "Engineer",
                "workplace": "hybrid",
                "employmentType": "Full-time",
                "description": "<p>$140,000 - $160,000</p>",
                "company": {"title": "Acme"},
            }
        ),
    )
    assert workable.remote is False
    assert workable.company == "Acme"
    assert workable.pay_low == 140_000
    assert workable.hours_per_week == 40

    body = Opportunity(title="Engineer", url="https://jobs.example/x", remote=True)
    _apply_listing(
        body,
        "<title>Engineer at Acme</title><p>This is a hybrid role in NYC. $120,000 - $140,000</p>",
    )
    assert body.remote is False
    assert body.pay_high == 140_000

    remote = Opportunity(title="x", url="https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    _apply_listing(
        remote,
        _lever_to_html(
            {
                "text": "Staff Software Engineer",
                "workplaceType": "remote",
                "categories": {"commitment": "Full-time"},
                "description": (
                    "<p>This role can be hybrid, or fully remote/virtually. $180,000 - $200,000</p>"
                ),
            }
        ),
    )
    assert remote.remote is True
    assert remote.pay_high == 200_000
    assert remote.score() == 100.0

    offered = Opportunity(title="Engineer", url="https://jobs.example/x")
    _apply_listing(
        offered,
        (
            "<title>Engineer at Acme</title>"
            "<p>This role can be hybrid, or fully remote/virtually. $180,000 - $200,000</p>"
        ),
    )
    assert offered.remote is True
    assert offered.pay_high == 200_000
    assert offered.score() == 100.0


def test_workplace_remote_or_hybrid_is_remote():
    from src.engine import _apply_listing, _greenhouse_to_html, _workplace_remote

    assert _workplace_remote("Remote or Hybrid") is True
    assert _workplace_remote("Hybrid / Remote") is True
    assert _workplace_remote("Distributed; Hybrid") is True
    assert _workplace_remote("hybrid") is False
    assert _workplace_remote("Flex") is False
    assert _workplace_remote("New York, NY (Hybrid)") is False
    assert _workplace_remote("Remote - United States") is True
    assert _workplace_remote("onsite only") is False

    html = _greenhouse_to_html(
        {
            "company_name": "Acme",
            "title": "Engineer",
            "location": {"name": "Remote or Hybrid"},
            "content": (
                "<p>Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000</p>"
                "<p>Tier 3 (US - All Other): $140,000 - $170,000</p>"
            ),
        }
    )
    opp = Opportunity(
        title="Engineer",
        url="https://job-boards.greenhouse.io/acme/jobs/1",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.remote is True
    assert opp.pay_low == 140_000
    assert opp.pay_high == 170_000
    assert opp.score() == 85.0


def test_ashby_to_html_foreign_summary_is_not_usd():
    from src.engine import _apply_listing, _ashby_to_html, _foreign_salary

    html = _ashby_to_html(
        {
            "title": "Engineer",
            "employmentType": "FullTime",
            "scrapeableCompensationSalarySummary": "€60,000 - €80,000",
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.ashbyhq.com/acme/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
    )
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_ashby_posting_null_is_gone():
    from src.engine import _ashby_posting

    class _Resp:
        status_code = 200
        text = '{"data":{"jobPosting":null}}'

    class _Client:
        def __init__(self):
            self.url = None
            self.payload = None

        async def post(self, url, **kwargs):
            self.url = url
            self.payload = kwargs.get("json")
            return _Resp()

    client = _Client()
    jid = "23ce794a-4aa7-45a3-bcdc-cb156271057b"
    assert asyncio.run(_ashby_posting(client, "azx", jid)) is None
    assert client.url == "https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobPosting"
    assert client.payload["variables"] == {
        "organizationHostedJobsPageName": "azx",
        "jobPostingId": jid,
    }


def test_ashby_posting_http_error_is_empty():
    from src.engine import _ashby_posting

    class _Resp:
        status_code = 500
        text = "nope"

    class _Client:
        async def post(self, _url, **_kwargs):
            return _Resp()

    assert asyncio.run(_ashby_posting(_Client(), "azx", "x")) == {}


def test_listing_text_ashby_graphql_null_is_gone(monkeypatch):
    engine = Engine()
    seen: list[tuple[str, str]] = []

    async def fake_ashby(_client, board: str, jid: str):
        seen.append((board, jid))
        return None

    async def fake_get(_client, _url: str):
        raise AssertionError("SPA HTML must not be fetched when GraphQL says gone")

    monkeypatch.setattr("src.engine._ashby_posting", fake_ashby)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    jid = "23ce794a-4aa7-45a3-bcdc-cb156271057b"
    html = asyncio.run(
        engine._listing_text(f"https://jobs.ashbyhq.com/azx/{jid}/application")
    )
    assert html is None
    assert seen == [("azx", jid)]


def test_listing_text_ashby_graphql_timeout_falls_back_to_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_ashby(_client, _board: str, _jid: str):
        return {}

    async def fake_get(_client, url: str):
        seen.append(url)
        return (
            "<html><script type='application/ld+json'>{"
            '"@type":"JobPosting","title":"ML Engineer",'
            '"hiringOrganization":{"name":"Quilter"},'
            '"baseSalary":{"currency":"USD","value":{"minValue":180000,"maxValue":200000,"unitText":"YEAR"}}'
            "}</script></html>"
        )

    monkeypatch.setattr("src.engine._ashby_posting", fake_ashby)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc"
    html = asyncio.run(engine._listing_text(url))
    assert seen == [url]
    assert html and "JobPosting" in html


def test_listing_text_ashby_graphql_pay_from_posting(monkeypatch):
    engine = Engine()

    async def fake_ashby(_client, board: str, jid: str):
        assert (board, jid) == (
            "quilter",
            "9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc",
        )
        return {
            "title": "Machine Learning Engineer",
            "employmentType": "FullTime",
            "descriptionHtml": "<p>Build ML systems.</p>",
            "compensationTierSummary": "$180K - $200K • Offsite",
            "scrapeableCompensationSalarySummary": "$180K - $200K",
        }

    async def fake_get(_client, _url: str):
        raise AssertionError("SPA HTML must not be fetched when GraphQL has the posting")

    monkeypatch.setattr("src.engine._ashby_posting", fake_ashby)
    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    url = "https://jobs.ashbyhq.com/quilter/9a15ed0b-1a0e-4c00-b7c8-8a0c4e8e9abc"
    html = asyncio.run(engine._listing_text(url))
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url=url)
    _apply_listing(opp, html)
    assert opp.pay_low == 180_000
    assert opp.pay_high == 200_000
    assert opp.hours_per_week == 40
    assert opp.score() == 100


def test_lever_eur_salary_range_is_foreign():
    from src.engine import _apply_listing, _foreign_salary, _lever_to_html

    html = _lever_to_html(
        {
            "text": "Senior ML Engineer (Europe-based/Remote)",
            "salaryRange": {
                "min": 60000,
                "max": 85000,
                "currency": "EUR",
                "interval": "per-year-salary",
            },
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.lever.co/swordhealth/50945411-2f43-421a-8bb8-86aa1de6d890",
    )
    _apply_listing(opp, html)
    assert opp.pay_high is None
    assert _foreign_salary(html) is True


def test_listing_text_greenhouse_api_404_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return None
        return "<title>Jobs at Reddit</title><p>Current openings</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://job-boards.greenhouse.io/reddit/jobs/8084032")
    )
    assert seen == [
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/8084032?pay_transparency=true"
    ]
    assert html is None


def test_listing_text_greenhouse_api_timeout_falls_back_to_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "boards-api.greenhouse.io" in url:
            return ""
        return "<title>Senior ML at Reddit</title><p>$180,000</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://job-boards.greenhouse.io/reddit/jobs/6960831")
    )
    assert seen[0] == (
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    )
    assert seen[1] == "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    assert html and "$180,000" in html


def test_enrich_drops_http_404_listings_keeps_empty_fetches():
    engine = Engine()

    async def page(url: str):
        if "gone" in url:
            return None
        if "thin" in url:
            return ""
        return "<title>Senior ML</title><p>$180,000 a year</p>"

    engine._listing_text = page
    priced = Opportunity(title="Priced", url="https://jobs.example/paid")
    ghost = Opportunity(title="Ghost", url="https://jobs.lever.co/gone/abc")
    thin = Opportunity(title="Thin", url="https://jobs.example/thin")
    opps = [priced, ghost, thin]
    asyncio.run(engine._enrich_pay(opps))
    assert [o.title for o in opps] == ["Priced", "Thin"]
    assert priced.pay_high == 180_000
    assert thin.pay_high is None


def test_greenhouse_api_url_from_job_board_link():
    from src.engine import _greenhouse_api_url, _lever_job_url, _normalize_url

    api = "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    assert _greenhouse_api_url("https://job-boards.greenhouse.io/reddit/jobs/6960831") == api
    assert _greenhouse_api_url(
        "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
    ) == api
    assert _greenhouse_api_url(
        "https://job-boards.greenhouse.io/embed/job_app?token=6960831&for=reddit"
    ) == api
    assert _greenhouse_api_url(
        "https://job-boards.eu.greenhouse.io/jetbrains/jobs/4713663101"
    ) == "https://boards-api.greenhouse.io/v1/boards/jetbrains/jobs/4713663101?pay_transparency=true"
    assert _greenhouse_api_url(
        "https://boards.eu.greenhouse.io/jetbrains/jobs/4713663101"
    ) == "https://boards-api.greenhouse.io/v1/boards/jetbrains/jobs/4713663101?pay_transparency=true"
    assert _greenhouse_api_url("https://jobs.lever.co/lyrahealth/abc") is None
    assert _lever_job_url(
        "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
    ) == "https://job-boards.greenhouse.io/reddit/jobs/6960831"
    assert _normalize_url(
        "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
    ) == _normalize_url("https://job-boards.greenhouse.io/reddit/jobs/6960831")


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
    assert opp.title == "Senior Machine Learning Engineer"
    assert opp.pay_low == 216_700
    assert opp.pay_high == 303_400


def test_greenhouse_pay_transparency_fills_json_ld_without_content_dollars():
    from src.engine import _apply_listing, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Reddit",
            "title": "Senior ML",
            "content": "<p>Apply now. No figures in the body.</p>",
            "pay_input_ranges": [
                {
                    "min_cents": 21670000,
                    "max_cents": 30340000,
                    "currency_type": "USD",
                    "title": "The base salary range for this position is:",
                },
                {
                    "min_cents": 10000000,
                    "max_cents": 12000000,
                    "currency_type": "EUR",
                },
            ],
        }
    )
    opp = Opportunity(title="x", url="https://job-boards.greenhouse.io/reddit/jobs/6960831")
    _apply_listing(opp, html)
    assert opp.pay_low == 216_700
    assert opp.pay_high == 303_400
    assert opp.hours_per_week is None

    eur_only = _greenhouse_to_html(
        {
            "company_name": "Acme",
            "title": "Engineer",
            "content": "<p>Apply now.</p>",
            "pay_input_ranges": [
                {"min_cents": 12000000, "max_cents": 18000000, "currency_type": "EUR"}
            ],
        }
    )
    skipped = Opportunity(title="Engineer", url="https://job-boards.greenhouse.io/acme/jobs/1")
    _apply_listing(skipped, eur_only)
    assert skipped.pay_high is None


def test_greenhouse_metadata_scheduled_hours_and_time_type():
    from src.engine import _apply_listing, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Reddit",
            "title": "Senior Machine Learning Engineer",
            "content": "<p>Apply now. No hours in the body.</p>",
            "pay_input_ranges": [
                {
                    "min_cents": 21670000,
                    "max_cents": 30340000,
                    "currency_type": "USD",
                }
            ],
            "metadata": [
                {"name": "Time Type", "value": "Full time", "value_type": "single_select"},
                {"name": "Scheduled Weekly Hours", "value": "40.0", "value_type": "number"},
                {"name": "Worker Sub-Type", "value": "Regular", "value_type": "single_select"},
            ],
        }
    )
    opp = Opportunity(title="x", url="https://job-boards.greenhouse.io/reddit/jobs/6960831")
    _apply_listing(opp, html)
    assert opp.pay_low == 216_700
    assert opp.pay_high == 303_400
    assert opp.hours_per_week == 40
    assert opp.rate_is_imputed is False
    assert opp.score() == 151.7


def test_apply_listing_prefers_remote_geo_band_over_json_ld():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"currency":"USD","value":{"minValue":160000,"maxValue":190000,"unitText":"YEAR"}}}
    </script>
    <p>Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000</p>
    <p>Tier 2 (DC Metro/Austin/Boston/Los Angeles): $150,000 - $180,000</p>
    <p>Tier 3 (US - All Other): $140,000 - $170,000</p>
    """
    remote = Opportunity(title="x", url="https://jobs.example/x", remote=True)
    _apply_listing(remote, html)
    assert remote.pay_low == 140_000
    assert remote.pay_high == 170_000

    office = Opportunity(title="x", url="https://jobs.example/x", remote=True)
    office_html = html.replace(
        "<p>Tier 1",
        "<p>This is a hybrid role.</p><p>Tier 1",
    )
    _apply_listing(office, office_html)
    assert office.remote is False
    assert office.pay_low == 160_000
    assert office.pay_high == 190_000


def test_greenhouse_geo_bands_use_all_other_when_remote():
    from src.engine import _apply_listing, _greenhouse_to_html

    html = _greenhouse_to_html(
        {
            "company_name": "Signifyd",
            "title": "Senior Machine Learning Engineer",
            "location": {"name": "Remote, USA"},
            "content": (
                "<p>Tier 1 (NYC/SF Bay Area/Seattle): $160,000 - $190,000</p>"
                "<p>Tier 2 (DC Metro/Austin/Boston/Los Angeles): $150,000 - $180,000</p>"
                "<p>Tier 3 (US - All Other): $140,000 - $170,000</p>"
            ),
        }
    )
    opp = Opportunity(
        title="Senior Machine Learning Engineer",
        url="https://job-boards.greenhouse.io/signifyd/jobs/1",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.company == "Signifyd"
    assert opp.pay_low == 140_000
    assert opp.pay_high == 170_000
    assert opp.score() == 85.0


def test_apply_listing_guesses_hours_when_json_ld_already_has_pay():
    from src.engine import _apply_listing

    html = """
    <script type="application/ld+json">
    {"@type":"JobPosting","title":"Engineer",
     "baseSalary":{"currency":"USD","value":{"minValue":160000,"maxValue":190000,"unitText":"YEAR"}}}
    </script>
    <p>This is a full-time position.</p>
    """
    opp = Opportunity(title="x", url="https://jobs.example/x")
    _apply_listing(opp, html)
    assert opp.pay_low == 160_000
    assert opp.pay_high == 190_000
    assert opp.hours_per_week == 40


def test_workable_jobs_api_url_from_view_link():
    from src.engine import _workable_jobs_api_url

    assert _workable_jobs_api_url(
        "https://jobs.workable.com/view/3wwPqWr4G8nzLWnxfEAKur/remote-senior-engineer-ai-ml"
    ) == "https://jobs.workable.com/api/v1/jobs/3wwPqWr4G8nzLWnxfEAKur"
    assert _workable_jobs_api_url("https://apply.workable.com/a2z-sync/j/C95E51CDDA") is None


def test_workable_jobs_api_html_fills_company_and_pay_range():
    from src.engine import _apply_listing, _workable_jobs_to_html

    html = _workable_jobs_to_html(
        {
            "title": "Senior Engineer, AI/ML",
            "company": {"title": "A2Z Sync"},
            "employmentType": "Full-time",
            "description": "<p>Build agents.</p>",
            "requirementsSection": (
                "<p>The expected salary range for this role is "
                "<strong>$160,000 to $190,000 annually</strong>.</p>"
            ),
        }
    )
    opp = Opportunity(
        title="Senior Engineer, AI/ML | A2Z Sync | Jobs By Workable",
        url="https://jobs.workable.com/view/3wwPqWr4G8nzLWnxfEAKur/x",
    )
    _apply_listing(opp, html)
    assert opp.company == "A2Z Sync"
    assert opp.title == "Senior Engineer, AI/ML"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 190_000
    assert opp.hours_per_week == 40


def test_listing_text_prefers_workable_jobs_api_over_spa_shell(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "/api/v1/jobs/" in url:
            return json.dumps(
                {
                    "title": "Senior Machine Learning Engineer",
                    "company": {"title": "Canopy"},
                    "requirementsSection": "<p>Base Salary: $126,000 - $180,000</p>",
                }
            )
        return "<title>Senior Machine Learning Engineer | Canopy | Jobs By Workable</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.workable.com/view/7mMjfHgS93LyPeHLK2XeMV/remote-senior-machine-learning-engineer"
        )
    )
    assert seen[0] == "https://jobs.workable.com/api/v1/jobs/7mMjfHgS93LyPeHLK2XeMV"
    from src.engine import _apply_listing

    opp = Opportunity(
        title="Senior Machine Learning Engineer | Canopy | Jobs By Workable",
        url="https://jobs.workable.com/view/7mMjfHgS93LyPeHLK2XeMV/x",
    )
    _apply_listing(opp, html)
    assert opp.company == "Canopy"
    assert opp.pay_low == 126_000
    assert opp.pay_high == 180_000


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
    assert seen[0] == (
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    )
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://job-boards.greenhouse.io/reddit/jobs/6960831")
    _apply_listing(opp, html)
    assert opp.company == "Reddit"
    assert opp.pay_high == 180_000


def test_listing_text_reads_greenhouse_embed_via_api(monkeypatch):
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
        engine._listing_text(
            "https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831"
        )
    )
    assert seen[0] == (
        "https://boards-api.greenhouse.io/v1/boards/reddit/jobs/6960831?pay_transparency=true"
    )
    from src.engine import _apply_listing

    opp = Opportunity(
        title="x",
        url="https://boards.greenhouse.io/embed/job_app?for=reddit&token=6960831",
    )
    _apply_listing(opp, html)
    assert opp.company == "Reddit"
    assert opp.pay_high == 180_000
    assert opp.remote is True


def test_smartrecruiters_api_url_from_job_link():
    from src.engine import (
        _is_index_page,
        _lever_job_url,
        _smartrecruiters_api_url,
    )

    api = "https://api.smartrecruiters.com/v1/companies/Socotec/postings/744000141322430"
    assert (
        _smartrecruiters_api_url(
            "https://jobs.smartrecruiters.com/Socotec/744000141322430-applied-ai-engineer"
        )
        == api
    )
    assert _lever_job_url(
        "https://jobs.smartrecruiters.com/Socotec/744000141322430-applied-ai-engineer"
    ) == "https://jobs.smartrecruiters.com/Socotec/744000141322430"
    assert _is_index_page(
        {"url": "https://jobs.smartrecruiters.com/Socotec", "title": "SOCOTEC", "description": ""}
    )
    assert not _is_index_page(
        {
            "url": "https://jobs.smartrecruiters.com/Socotec/744000141322430",
            "title": "Applied AI Engineer",
            "description": "",
        }
    )
    assert _smartrecruiters_api_url("https://jobs.lever.co/acme/x") is None


def test_smartrecruiters_to_html_fills_company_pay_and_remote():
    from src.engine import _apply_listing, _smartrecruiters_to_html

    html = _smartrecruiters_to_html(
        {
            "name": "Applied AI Engineer",
            "company": {"name": "SOCOTEC"},
            "typeOfEmployment": {"id": "permanent", "label": "Full-time"},
            "location": {
                "city": "New York",
                "remote": False,
                "hybrid": False,
                "fullLocation": "New York, United States",
            },
            "jobAd": {
                "sections": {
                    "additionalInformation": {"text": "<p>Salary: $157-200k</p>"},
                }
            },
        }
    )
    opp = Opportunity(
        title="x",
        url="https://jobs.smartrecruiters.com/Socotec/744000141322430",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.company == "SOCOTEC"
    assert opp.title == "Applied AI Engineer"
    assert opp.pay_low == 157_000
    assert opp.pay_high == 200_000
    assert opp.remote is False
    assert opp.hours_per_week == 40
    assert opp.score() == 70.0

    remote = Opportunity(title="x", url="https://jobs.smartrecruiters.com/mirantis/1")
    _apply_listing(
        remote,
        _smartrecruiters_to_html(
            {
                "name": "Senior Software Engineer (Golang)",
                "company": {"name": "Mirantis"},
                "typeOfEmployment": {"label": "Full-time"},
                "location": {"city": "Remote", "remote": True, "hybrid": False},
                "jobAd": {"sections": {"jobDescription": {"text": "<p>Go systems.</p>"}}},
            }
        ),
    )
    assert remote.company == "Mirantis"
    assert remote.remote is True
    assert remote.pay_high is None


def test_listing_text_reads_smartrecruiters_api(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str) -> str:
        seen.append(url)
        if "api.smartrecruiters.com" in url:
            return json.dumps(
                {
                    "name": "Applied AI Engineer",
                    "company": {"name": "SOCOTEC"},
                    "typeOfEmployment": {"label": "Full-time"},
                    "location": {"remote": False, "hybrid": False, "city": "New York"},
                    "jobAd": {
                        "sections": {
                            "additionalInformation": {"text": "<p>Salary: $157-200k</p>"}
                        }
                    },
                }
            )
        return "<title>Jobs at SOCOTEC</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://jobs.smartrecruiters.com/Socotec/744000141322430-applied-ai-engineer"
        )
    )
    assert seen == [
        "https://api.smartrecruiters.com/v1/companies/Socotec/postings/744000141322430"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://jobs.smartrecruiters.com/Socotec/744000141322430")
    _apply_listing(opp, html)
    assert opp.company == "SOCOTEC"
    assert opp.pay_high == 200_000
    assert opp.remote is False


def test_workday_api_url_from_job_link():
    from src.engine import _is_index_page, _workday_api_url

    assert _workday_api_url(
        "https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/Machine-Learning-Engineer_JR-0106147"
    ) == (
        "https://workday.wd5.myworkdayjobs.com/wday/cxs/workday/Workday/job/"
        "Machine-Learning-Engineer_JR-0106147"
    )
    assert _workday_api_url(
        "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/"
        "US-CA-Santa-Clara/Machine-Learning-Engineer--AI-Safety_JR2021784-1"
    ) == (
        "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/job/"
        "Machine-Learning-Engineer--AI-Safety_JR2021784-1"
    )
    assert _is_index_page(
        {
            "url": "https://workday.wd5.myworkdayjobs.com/en-US/Workday",
            "title": "Workday Careers",
            "description": "",
        }
    )
    assert not _is_index_page(
        {
            "url": "https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/Machine-Learning-Engineer_JR-0106147",
            "title": "Machine Learning Engineer III",
            "description": "",
        }
    )
    assert _workday_api_url("https://jobs.lever.co/acme/x") is None


def test_workday_to_html_fills_company_pay_and_flex():
    from src.engine import _apply_listing, _workday_to_html

    html = _workday_to_html(
        {
            "hiringOrganization": {"name": "Workday, Inc."},
            "jobPostingInfo": {
                "title": "Machine Learning Engineer III",
                "timeType": "Full Time",
                "remoteType": "Flex",
                "location": "USA, CA, Pleasanton",
                "jobDescription": "<p>Base Pay Range: $160,000 USD - $240,000 USD</p>",
            },
        }
    )
    opp = Opportunity(
        title="x",
        url="https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/x_JR-1",
        remote=True,
    )
    _apply_listing(opp, html)
    assert opp.company == "Workday, Inc."
    assert opp.title == "Machine Learning Engineer III"
    assert opp.pay_low == 160_000
    assert opp.pay_high == 240_000
    assert opp.remote is False
    assert opp.hours_per_week == 40
    assert opp.score() == 84.0


def test_listing_text_reads_workday_cxs(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "/wday/cxs/" in url:
            return json.dumps(
                {
                    "hiringOrganization": {"name": "Workday, Inc."},
                    "jobPostingInfo": {
                        "title": "Machine Learning Engineer III",
                        "timeType": "Full Time",
                        "remoteType": "Remote",
                        "jobDescription": "<p>Base Pay Range: $160,000 USD - $240,000 USD</p>",
                    },
                }
            )
        return "<title>Jobs at Workday</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/Machine-Learning-Engineer_JR-0106147"
        )
    )
    assert seen == [
        "https://workday.wd5.myworkdayjobs.com/wday/cxs/workday/Workday/job/"
        "Machine-Learning-Engineer_JR-0106147"
    ]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://workday.wd5.myworkdayjobs.com/en-US/Workday/job/x")
    _apply_listing(opp, html)
    assert opp.company == "Workday, Inc."
    assert opp.pay_high == 240_000
    assert opp.remote is True
    assert opp.score() == 120.0


def test_listing_text_workday_cxs_404_falls_back_to_html(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "/wday/cxs/" in url:
            return None
        return (
            "<title>Engineer at Motorola</title>"
            "<p>This is a full-time remote role. $180,000 - $200,000</p>"
        )

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://motorolasolutions.wd5.myworkdayjobs.com/en-US/Careers/job/Machine-Learning-Engineer_R64440"
        )
    )
    assert seen[0].startswith("https://motorolasolutions.wd5.myworkdayjobs.com/wday/cxs/")
    assert seen[1] == (
        "https://motorolasolutions.wd5.myworkdayjobs.com/en-US/Careers/job/"
        "Machine-Learning-Engineer_R64440"
    )
    assert html and "$180,000" in html


def test_icims_iframe_url_from_job_link():
    from src.engine import _icims_iframe_url, _is_index_page, _lever_job_url

    pretty = (
        "https://uscareers-yelp.icims.com/jobs/13815/"
        "senior-machine-learning-engineer---content/job"
    )
    assert (
        _icims_iframe_url(pretty)
        == "https://uscareers-yelp.icims.com/jobs/13815/job?in_iframe=1"
    )
    assert _lever_job_url(pretty) == "https://uscareers-yelp.icims.com/jobs/13815/job"
    assert not _is_index_page(
        {"url": pretty, "title": "Careers at Yelp | Yelp Jobs", "description": ""}
    )
    assert _is_index_page(
        {
            "url": "https://careers-mci.icims.com/jobs/intro",
            "title": "Careers Center | Welcome",
            "description": "",
        }
    )
    assert _icims_iframe_url("https://jobs.lever.co/acme/x") is None


def test_listing_text_reads_icims_iframe(monkeypatch):
    engine = Engine()
    seen: list[str] = []
    iframe_html = (
        "<title>Senior ML Engineer</title>"
        '<script type="application/ld+json">'
        '{"@type":"JobPosting","title":"Senior ML Engineer",'
        '"hiringOrganization":{"name":"Yelp, Inc"},'
        '"jobLocationType":"TELECOMMUTE"}'
        "</script>"
        "<p>Compensation range for this role to be between $112,000 and $269,000.</p>"
    )

    async def fake_get(_client, url: str):
        seen.append(url)
        if "in_iframe=1" in url:
            return iframe_html
        return "<title>Careers at Yelp | Yelp Jobs</title>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text(
            "https://uscareers-yelp.icims.com/jobs/13815/senior-machine-learning-engineer/job"
        )
    )
    assert seen == ["https://uscareers-yelp.icims.com/jobs/13815/job?in_iframe=1"]
    from src.engine import _apply_listing

    opp = Opportunity(title="x", url="https://uscareers-yelp.icims.com/jobs/13815/job")
    _apply_listing(opp, html)
    assert opp.company == "Yelp, Inc"
    assert opp.pay_high == 269_000
    assert opp.remote is True


def test_listing_text_icims_410_is_gone(monkeypatch):
    engine = Engine()
    seen: list[str] = []

    async def fake_get(_client, url: str):
        seen.append(url)
        if "in_iframe=1" in url:
            return None
        return "<title>Careers at Acme | Acme Jobs</title><p>Search jobs</p>"

    monkeypatch.setattr("src.engine._http_get_text", fake_get)
    html = asyncio.run(
        engine._listing_text("https://careers-americas.icims.com/jobs/26849/principal-ml/job")
    )
    assert seen == ["https://careers-americas.icims.com/jobs/26849/job?in_iframe=1"]
    assert html is None


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
