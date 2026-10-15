import httpx
import pytest

from jobhunter.sheets import COLUMNS, PIPELINE_STATES, Sheets


class Session:
    def __init__(self):
        self.calls = []
        self.keys = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if method == "GET" and "/values/" not in url:
            data = {"sheets": [{"properties": {"title": "Jobhunter", "sheetId": 7,
                     "gridProperties": {"rowCount": 1000}}}]}
        elif method == "GET" and "A1:" in url:
            data = {"values": [COLUMNS]}
        elif method == "GET":
            data = {"values": [[key] for key in self.keys]}
        else:
            data = {}
        return httpx.Response(200, json=data, request=httpx.Request(method, url))


def record(key="abc"):
    return {"key": key, "title": "=IMPORTXML(\"bad\")", "company": "Example", "url": "https://example.com/1",
            "first_seen": "2026-10-15", "gate_status": "qualified", "dollars_per_hour": 80,
            "pipeline_state": "shortlisted", "mode": "live"}


def test_sheet_reconciles_keys_and_uses_raw_to_prevent_formula_injection():
    session = Session()
    session.keys = ["abc"]
    sheets = Sheets(session, "sheet123")
    sheets.sync([record(), record("def")])
    writes = [c for c in session.calls if c[0] == "POST" and ":append" in c[1]]
    assert len(writes) == 1
    assert writes[0][2]["params"]["valueInputOption"] == "RAW"
    assert len(writes[0][2]["json"]["values"]) == 1
    assert writes[0][2]["json"]["values"][0][COLUMNS.index("Title")] == record()["title"]


def test_tracker_changes_only_pipeline_and_notes():
    session = Session()
    session.keys = ["abc"]
    Sheets(session, "sheet123").track("abc", "applied", "Applied myself")
    write = next(c for c in session.calls if c[0] == "PUT")
    assert write[2]["json"]["values"] == [["applied", "Applied myself"]]
    assert "Q2:R2" in write[1]


def test_invalid_pipeline_is_rejected_before_writing():
    with pytest.raises(ValueError):
        Sheets(Session(), "sheet123").track("abc", "sent_email")
    assert "offer" in PIPELINE_STATES


def test_sheet_header_mismatch_never_overwrites_user_data():
    session = Session()
    original = session.request
    def wrong_headers(method, url, **kwargs):
        if method == "GET" and "A1:" in url:
            return httpx.Response(200, json={"values": [["My unrelated data"]]}, request=httpx.Request(method, url))
        return original(method, url, **kwargs)
    session.request = wrong_headers
    with pytest.raises(ValueError, match="headers"):
        Sheets(session, "sheet123").sync([record()])
    assert not any(c[0] in {"POST", "PUT"} for c in session.calls)
