"""At-most-once scoring and a durable Sheets outbox, isolated per installer."""

import fcntl
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.compensation import canonicalize_url


def posting_key(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode()).hexdigest()


class Registry:
    def __init__(self, home: Path):
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.home = home
        self.path = home / "seen.sqlite"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS seen (
                key TEXT PRIMARY KEY, url TEXT NOT NULL, result TEXT,
                synced INTEGER NOT NULL DEFAULT 0, notified INTEGER NOT NULL DEFAULT 0
            )""")
            db.execute("CREATE TABLE IF NOT EXISTS evidence (key TEXT PRIMARY KEY, posting TEXT NOT NULL)")
        self.path.chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @contextmanager
    def lock(self):
        with (self.home / "hunt.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("A hunt is already running for this registry.") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def claim(self, url: str) -> bool:
        # Commit BEFORE evaluating/scoring. A killed process leaves an incomplete
        # claim rather than permitting the same posting to be scored twice.
        with self.connect() as db:
            return db.execute("INSERT OR IGNORE INTO seen (key, url) VALUES (?, ?)",
                              (posting_key(url), canonicalize_url(url))).rowcount == 1

    def finish(self, record: dict):
        with self.connect() as db:
            db.execute("UPDATE seen SET result=? WHERE key=?", (json.dumps(record), record["key"]))

    def claim_evidence(self, url: str, posting: str) -> bool:
        # Careers wrappers and direct ATS URLs can resolve to the same listing.
        identity = canonicalize_url(url)
        with self.connect() as db:
            return db.execute("INSERT OR IGNORE INTO evidence (key, posting) VALUES (?, ?)",
                              (posting_key(identity), posting_key(posting))).rowcount == 1

    def alias(self, url: str):
        record = {"key": posting_key(url), "url": url, "gate_status": "duplicate",
                  "reason": "Resolved employer listing already scored"}
        with self.connect() as db:
            db.execute("UPDATE seen SET result=?, synced=1, notified=1 WHERE key=?",
                       (json.dumps(record), record["key"]))

    def pending(self) -> list[dict]:
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT result FROM seen WHERE synced=0 AND result IS NOT NULL ORDER BY rowid")]

    def synced(self, records: list[dict]):
        with self.connect() as db:
            db.executemany("UPDATE seen SET synced=1 WHERE key=?", [(r["key"],) for r in records])

    def notifications(self) -> list[dict]:
        with self.connect() as db:
            records = [json.loads(row[0]) for row in db.execute(
                "SELECT result FROM seen WHERE synced=1 AND notified=0 AND result IS NOT NULL")]
            db.executemany("UPDATE seen SET notified=1 WHERE key=?", [(r["key"],) for r in records])
        return sorted((r for r in records if r["gate_status"] == "qualified"),
                      key=lambda r: r["dollars_per_hour"], reverse=True)
