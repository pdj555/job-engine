"""Installer-owned desktop OAuth. No bundled tokens or service accounts."""

import json
from pathlib import Path

# drive.file permits only files this OAuth app creates/opens. Existing arbitrary
# Sheets require explicit broader opt-in at connect time.
FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"


def save_token(path: Path, text: str):
    temporary = path.with_suffix(".tmp")
    # Set permissions at creation, before any token bytes touch disk.
    import os
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    temporary.chmod(0o600)
    temporary.replace(path)


def authorize(home: Path, *, client_secrets: Path | None = None, existing_sheet=False):
    try:
        from google.auth.transport.requests import AuthorizedSession, Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise ValueError('Install the Sheets extra: pip install ".[sheets]"') from exc

    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = home / "google-token.json"
    credentials = None
    if client_secrets:
        config = json.loads(client_secrets.read_text())
        if "installed" not in config:
            raise ValueError("Use a Google Desktop app OAuth client, never a service-account key.")
        scopes = [SHEETS_SCOPE if existing_sheet else FILE_SCOPE]
        flow = InstalledAppFlow.from_client_config(config, scopes=scopes)
        credentials = flow.run_local_server(host="localhost", port=0,
                                            authorization_prompt_message="Open the browser to connect your Google account.")
        save_token(token, credentials.to_json())
    elif token.exists():
        credentials = Credentials.from_authorized_user_file(str(token))
        if not credentials.valid and credentials.refresh_token:
            credentials.refresh(Request())
            save_token(token, credentials.to_json())
    if not credentials or not credentials.valid:
        raise ValueError("Connect your own Google account first: jobhunter connect --client-secrets PATH")
    return AuthorizedSession(credentials)
