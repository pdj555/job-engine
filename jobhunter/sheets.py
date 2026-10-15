"""Results and application states in the installer's own Google spreadsheet."""

from urllib.parse import quote

COLUMNS = ["Key", "First seen", "Title", "Company", "Listing URL", "Pay low (USD/year)",
           "Pay high (USD/year)", "Hours/week", "Rank ($/hour)", "Gate status", "Evidence / gate reason",
           "Pay source", "Employer evidence URL", "Hours source", "Remote", "Mode",
           "Pipeline state", "Notes", "Remote source", "Hours evidence URL", "Remote evidence URL",
           "Annualized projection"]
FIELDS = ["key", "first_seen", "title", "company", "url", "pay_low", "pay_high", "hours_per_week",
          "dollars_per_hour", "gate_status", "reason", "pay_source", "pay_source_url", "hours_source",
          "remote", "mode", "pipeline_state", "notes", "remote_source", "hours_source_url",
          "remote_source_url", "pay_is_annualized"]
PIPELINE_STATES = ["discovered", "shortlisted", "applied", "interviewing", "offer", "rejected", "withdrawn"]


class Sheets:
    def __init__(self, session, sheet_id: str, tab="Jobhunter"):
        if not sheet_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in sheet_id):
            raise ValueError("sheet_id must be a Google spreadsheet ID, not a URL")
        self.session, self.sheet_id, self.tab = session, sheet_id, tab
        self.base = f"https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}"

    def request(self, method, path="", **kwargs):
        response = self.session.request(method, self.base + path, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def create(session, title="Jobhunter") -> str:
        response = session.request("POST", "https://sheets.googleapis.com/v4/spreadsheets",
                                   json={"properties": {"title": title}}, timeout=30)
        response.raise_for_status()
        return response.json()["spreadsheetId"]

    def range_path(self, cells):
        name = self.tab.replace("'", "''")
        return "/values/" + quote(f"'{name}'!{cells}", safe="!:')")

    def prepare(self):
        metadata = self.request("GET", params={"fields": "sheets.properties"})
        props = next((s["properties"] for s in metadata.get("sheets", [])
                      if s["properties"]["title"] == self.tab), None)
        if props is None:
            data = self.request("POST", ":batchUpdate", json={"requests": [
                {"addSheet": {"properties": {"title": self.tab,
                  "gridProperties": {"rowCount": 1000, "columnCount": len(COLUMNS)}}}}]})
            props = data["replies"][0]["addSheet"]["properties"]
        headers = self.request("GET", self.range_path("A1:V1")).get("values", [])
        if headers and headers[0] != COLUMNS:
            raise ValueError(f"{self.tab} headers differ. Choose a new tab; user data was preserved.")
        if not headers:
            self.request("PUT", self.range_path("A1:V1"), params={"valueInputOption": "RAW"},
                         json={"values": [COLUMNS]})
        sheet_id = props["sheetId"]
        self.request("POST", ":batchUpdate", json={"requests": [
            {"updateSheetProperties": {"properties": {"sheetId": sheet_id,
              "gridProperties": {"frozenRowCount": 1}}, "fields": "gridProperties.frozenRowCount"}},
            {"repeatCell": {"range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
              "cell": {"userEnteredFormat": {"textFormat": {"bold": True},
              "backgroundColor": {"red": 0.85, "green": 0.92, "blue": 0.98}}},
              "fields": "userEnteredFormat"}},
            {"setDataValidation": {"range": {"sheetId": sheet_id, "startRowIndex": 1,
              "startColumnIndex": 16, "endColumnIndex": 17}, "rule": {"condition": {
              "type": "ONE_OF_LIST", "values": [{"userEnteredValue": s} for s in PIPELINE_STATES]},
              "strict": True, "showCustomUi": True}}},
        ]})
        return props

    def keys(self, props):
        rows = props.get("gridProperties", {}).get("rowCount", 1000)
        if rows < 2:
            return {}
        values = self.request("GET", self.range_path(f"A2:A{rows}")).get("values", [])
        return {row[0]: i for i, row in enumerate(values, 2) if row and row[0]}

    def sync(self, records):
        props = self.prepare()
        existing = self.keys(props)
        values = []
        for record in records:
            if record["key"] not in existing:
                values.append([record.get(field) if record.get(field) is not None else "" for field in FIELDS])
                existing[record["key"]] = True
        if values:
            # RAW stops listing titles or employer strings from executing formulas.
            # Existing keyed rows (including user-edited states/notes) are untouched.
            self.request("POST", self.range_path("A:V") + ":append",
                         params={"valueInputOption": "RAW", "insertDataOption": "INSERT_ROWS"},
                         json={"values": values})

    def track(self, key, state, notes=None):
        if state not in PIPELINE_STATES:
            raise ValueError("Pipeline state must be one of: " + ", ".join(PIPELINE_STATES))
        row = self.keys(self.prepare()).get(key)
        if row is None:
            raise ValueError("Posting key not found in this sheet.")
        cells, values = (f"Q{row}:Q{row}", [[state]]) if notes is None else (f"Q{row}:R{row}", [[state, notes]])
        self.request("PUT", self.range_path(cells), params={"valueInputOption": "RAW"}, json={"values": values})
