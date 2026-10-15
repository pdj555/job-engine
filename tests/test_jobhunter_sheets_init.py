import httpx

from jobhunter.sheets import COLUMNS, PIPELINE_STATES, Sheets


def test_new_sheet_tab_gets_headers_and_pipeline_dropdown():
    calls = []
    class Session:
        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            if method == "GET":
                data = {"sheets": []} if "/values/" not in url else {}
            elif "addSheet" in kwargs.get("json", {}).get("requests", [{}])[0]:
                data = {"replies": [{"addSheet": {"properties": {
                    "sheetId": 9, "title": "Jobhunter", "gridProperties": {"rowCount": 1000}}}}]}
            else:
                data = {}
            return httpx.Response(200, json=data, request=httpx.Request(method, url))
    Sheets(Session(), "sheet123").sync([])
    header = next(c for c in calls if c[0] == "PUT")
    assert header[2]["json"]["values"] == [COLUMNS]
    requests = calls[-2][2]["json"]["requests"]
    validation = next(r["setDataValidation"] for r in requests if "setDataValidation" in r)
    assert validation["range"]["sheetId"] == 9
    assert [v["userEnteredValue"] for v in validation["rule"]["condition"]["values"]] == PIPELINE_STATES


def test_google_permission_failure_is_not_treated_as_delivery_success():
    class Session:
        def request(self, method, url, **kwargs):
            return httpx.Response(403, json={"error": "denied"}, request=httpx.Request(method, url))
    import pytest
    with pytest.raises(httpx.HTTPStatusError):
        Sheets(Session(), "sheet123").sync([])
