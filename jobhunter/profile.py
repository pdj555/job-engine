"""Explicit, fail-closed gates translated from the user's plain-language request."""

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1)
    sheet_id: str = ""
    remote_only: bool = True
    min_annual_pay: int = Field(default=0, ge=0)
    min_dollars_per_hour: float = Field(default=0, ge=0)
    max_hours_per_week: int = Field(default=80, ge=1, le=80)
    title_terms: list[str] = Field(default_factory=list)
    exclude_companies: list[str] = Field(default_factory=list)
    employer_domains: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value):
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()

    @field_validator("title_terms", "exclude_companies", "employer_domains")
    @classmethod
    def nonempty_items(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("gate entries cannot be blank")
        return [value.strip().lower() for value in values]

    @field_validator("employer_domains")
    @classmethod
    def hosts_only(cls, values):
        if any("/" in v or ":" in v or "." not in v or " " in v for v in values):
            raise ValueError("employer_domains must contain verified hostnames, not URLs")
        return values

    def save(self, home: Path):
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = home / "profile.json.tmp"
        temporary.write_text(self.model_dump_json(indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(home / "profile.json")

    def signature(self):
        gates = json.dumps(self.model_dump(exclude={"sheet_id"}), sort_keys=True)
        return hashlib.sha256(gates.encode()).hexdigest()

    @classmethod
    def load(cls, home: Path):
        path = home / "profile.json"
        if not path.exists():
            raise ValueError(f"Create {path} with your query and gates before hunting.")
        return cls.model_validate_json(path.read_text())


def state_home() -> Path:
    return Path(os.environ.get("JOBHUNTER_HOME", Path.home() / ".local/share/jobhunter")).expanduser()
