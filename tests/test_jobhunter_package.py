import json
from pathlib import Path

from jobhunter.auth import save_token

ROOT = Path(__file__).resolve().parents[1]


def test_plugin_is_listed_and_points_to_root_skill():
    manifest = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    assert manifest["name"] == "jobhunter"
    assert manifest["skills"] == ["./"]
    assert market["plugins"][0]["name"] == "jobhunter"
    assert market["plugins"][0]["source"] == "./"
    assert (ROOT / "SKILL.md").is_file()
    assert (ROOT / "commands/hunt.md").is_file()
    assert (ROOT / "scripts/install.sh").stat().st_mode & 0o111


def test_token_files_are_private_and_replaced_atomically(tmp_path):
    path = tmp_path / "google-token.json"
    save_token(path, '{"refresh_token":"test-only"}')
    assert path.stat().st_mode & 0o777 == 0o600
    save_token(path, '{"refresh_token":"replacement-test-only"}')
    assert json.loads(path.read_text())["refresh_token"] == "replacement-test-only"
    assert not path.with_suffix(".tmp").exists()
