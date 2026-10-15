import json

from typer.testing import CliRunner

from jobhunter.cli import cli, run_schedule


def test_hunt_is_silent_without_matches(tmp_path, monkeypatch):
    from jobhunter import cli as module
    (tmp_path / "profile.json").write_text(json.dumps({"query": "Python", "sheet_id": "sheet123"}))
    async def no_matches(*args, **kwargs):
        return []
    monkeypatch.setattr(module, "hunt", no_matches)
    monkeypatch.setattr(module, "sheet_connection", lambda *a, **k: object())
    result = CliRunner().invoke(cli, ["hunt", "--home", str(tmp_path)])
    assert result.exit_code == 0 and result.stdout == ""


def test_schedule_repeats_and_stops_on_interrupt():
    runs = []
    def sleep(seconds):
        assert seconds == 60
        if len(runs) == 3:
            raise KeyboardInterrupt
    run_schedule(lambda: runs.append(1), every_minutes=1, sleep=sleep)
    assert runs == [1, 1, 1]


def test_dry_run_cannot_be_scheduled(tmp_path):
    result = CliRunner().invoke(cli, ["hunt", "--dry-run", "--every-minutes", "60", "--home", str(tmp_path)])
    assert result.exit_code != 0


def test_profile_validation_explains_missing_setup(tmp_path):
    result = CliRunner().invoke(cli, ["hunt", "--home", str(tmp_path)])
    assert result.exit_code != 0
    assert "profile" in result.output.lower()


def test_scheduled_hunt_reloads_user_profile_each_run(tmp_path, monkeypatch):
    from jobhunter import cli as module
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"query": "Python", "sheet_id": "sheet123"}))
    queries = []
    async def no_matches(profile, *args, **kwargs):
        queries.append(profile.query)
        return []
    def schedule(run, every_minutes):
        run()
        path.write_text(json.dumps({"query": "Rust", "sheet_id": "sheet123"}))
        run()
    monkeypatch.setattr(module, "hunt", no_matches)
    monkeypatch.setattr(module, "sheet_connection", lambda *a, **k: object())
    monkeypatch.setattr(module, "run_schedule", schedule)
    result = CliRunner().invoke(cli, ["hunt", "--home", str(tmp_path), "--every-minutes", "60"])
    assert result.exit_code == 0
    assert queries == ["Python", "Rust"]
