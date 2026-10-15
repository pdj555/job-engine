import json
import sys
from types import SimpleNamespace

import pytest

from jobhunter.auth import FILE_SCOPE, SHEETS_SCOPE, authorize


@pytest.fixture
def oauth(monkeypatch):
    events = []
    credentials = SimpleNamespace(valid=True, refresh_token="test-only", to_json=lambda: '{"token":"test-only"}')
    credentials.refresh = lambda request: (events.append("refresh"), setattr(credentials, "valid", True))
    class Flow:
        @classmethod
        def from_client_config(cls, config, scopes):
            events.append(scopes)
            return cls()
        def run_local_server(self, **kwargs):
            events.append(kwargs)
            return credentials
    class Credentials:
        @classmethod
        def from_authorized_user_file(cls, path):
            events.append("load-own-token")
            return credentials
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests",
                        SimpleNamespace(AuthorizedSession=lambda c: c, Request=lambda: object()))
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", SimpleNamespace(Credentials=Credentials))
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", SimpleNamespace(InstalledAppFlow=Flow))
    return events, credentials


def test_oauth_uses_installer_desktop_client_and_minimal_scope(tmp_path, oauth):
    client = tmp_path / "client.json"
    client.write_text(json.dumps({"installed": {"client_id": "installer-specific"}}))
    authorize(tmp_path, client_secrets=client)
    assert oauth[0][0] == [FILE_SCOPE]
    assert oauth[0][1]["port"] == 0
    assert (tmp_path / "google-token.json").stat().st_mode & 0o777 == 0o600


def test_existing_sheet_requires_explicit_broader_scope(tmp_path, oauth):
    client = tmp_path / "client.json"
    client.write_text('{"installed": {}}')
    authorize(tmp_path, client_secrets=client, existing_sheet=True)
    assert oauth[0][0] == [SHEETS_SCOPE]


def test_unattended_hunt_refreshes_its_own_token(tmp_path, oauth):
    (tmp_path / "google-token.json").write_text('{"token":"test-only"}')
    oauth[1].valid = False
    authorize(tmp_path)
    assert oauth[0] == ["load-own-token", "refresh"]


def test_hunt_does_not_launch_oauth_without_connection(tmp_path, oauth):
    with pytest.raises(ValueError, match="Connect your own"):
        authorize(tmp_path)
    assert oauth[0] == []


def test_service_account_key_is_rejected(tmp_path, oauth):
    client = tmp_path / "client.json"
    client.write_text('{"type":"service_account"}')
    with pytest.raises(ValueError, match="Desktop"):
        authorize(tmp_path, client_secrets=client)
    assert oauth[0] == []
