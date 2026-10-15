# Jobhunter implementation report

The plugin wraps the existing engine. No files under `src/` were changed for
Jobhunter. Before implementation, the active branch was merged with
`origin/main` at `d2207666`, resolving conflicts in favor of main's current engine,
tests and UI while preserving the branch history. The merged baseline passed all
127 existing tests.

| File | Addition or change |
| --- | --- |
| `.claude-plugin/plugin.json` | Names and versions the `jobhunter` plugin; discovers the root skill. |
| `.claude-plugin/marketplace.json` | Lists the repository-root plugin for GitHub and local installation. |
| `SKILL.md` | Plain-language interface, profile translation, evidence requirements, connection, scheduling and tracking instructions. |
| `commands/hunt.md` | `/jobhunter:hunt` entry point into the skill and repeatable hunt command. |
| `jobhunter/__init__.py` | Plugin adapter package. |
| `jobhunter/profile.py` | Strict typed user gates, private profile persistence and a gate signature for cached notifications. |
| `jobhunter/hunt.py` | Uses existing search, listing parsers and ranker; checks evidence/gates before scoring; captures resolved ATS identity and per-field origin; supplies missing hours evidence; detects discovery outages; includes synthetic dry listings. |
| `jobhunter/registry.py` | SQLite seen registry, lock against overlapping hunts, claim before score, resolved-listing deduplication, durable Sheets outbox and notification bookkeeping. |
| `jobhunter/auth.py` | Each installer's Desktop OAuth consent, minimal default `drive.file` scope, explicit existing-sheet scope, automatic refresh, private atomic token persistence. |
| `jobhunter/sheets.py` | Creates/prepares the user's sheet, writes results as RAW values, reconciles posting keys, validates headers, adds pipeline dropdowns, and preserves existing rows/states/notes. |
| `jobhunter/cli.py` | `jobhunter connect`, `hunt`, interval scheduling, dry-run isolation, and `track`; reloads profile/connection every cycle. |
| `scripts/install.sh` | Builds a private Python runtime from the installed plugin, including the Sheets extra. |
| `scripts/jobhunter` | Runs the installer's persistent runtime from any installed plugin cache location. |
| `docs/JOBHUNTER.md` | Installation, profile schema, Google OAuth, cron/interval setup, tracker and verification instructions. |
| `docs/JOBHUNTER_CHANGES.md` | This file-by-file implementation report. |
| `tests/test_jobhunter.py` | Evidence, salary/rate/hour/title/company gates, ranking, URL/ATS alias deduplication, crashes, outbox recovery, profile changes and search-failure regressions. |
| `tests/test_jobhunter_auth.py` | Installer OAuth scopes, loopback callback, refresh, noninteractive runs and service-account rejection. |
| `tests/test_jobhunter_cli.py` | Silent empty hunts, repeat cadence, interruption, profile reload and dry-run scheduling rejection. |
| `tests/test_jobhunter_package.py` | Plugin/marketplace discovery structure, executable installer and private token replacement. |
| `tests/test_jobhunter_sheets.py` | Idempotent keyed writes, RAW formula protection, tracker-only updates, state validation and header safety. |
| `tests/test_jobhunter_sheets_init.py` | New-tab headers/dropdowns and permission-failure propagation. |
| `pyproject.toml` | Adds the `jobhunter` executable, optional Sheets dependencies, and includes `config` and `jobhunter` in built wheels. |
| `uv.lock` | Locks the added OAuth dependency graph. |
| `.gitignore` | Excludes plugin state and Google OAuth credential filenames. |
| `.github/workflows/ci.yml` | Adds a `jobhunter --help` smoke check alongside the existing CLI check. |
| `README.md` | Adds plugin install commands and links to setup documentation. |

## Verification

- Merged baseline: 127 existing tests passed.
- Current suite: 175 tests passed, including 48 new plugin tests.
- Claude Code strict marketplace and plugin manifest validation passed.
- The skill validator passed. Shell syntax and Python compilation checks passed.
- Local marketplace install succeeded in an isolated Claude configuration. Claude
  discovered `jobhunter` and `hunt` as the two skill/command components.
- The plugin runtime installed from the Claude cache as a wheel. Both CLIs and
  engine/config/plugin imports worked from outside the repository. The final
  installed build passed all 175 tests on Python 3.14 from outside the checkout.
- Live dry run: the locally installed plugin wrote three synthetic roles to a
  separate Google Sheet in the user's selected Personal account. Readback matched
  the adapter's output: one qualified role at $80/hour, and two unverified roles
  with blank score cells. A repeat was silent, made no Sheets requests, and left
  exactly three rows. The live seen registry was untouched.
- The installed tracker command changed a fixture's state to `interviewing` and
  its notes; Google readback confirmed the change. The fixture was then restored
  to `shortlisted`. The native dropdown contained all seven pipeline states.
- Live verification used a session-boundary adapter to the user's connected
  Google account. Search fixtures, CLI behavior, gates, ranking, deduplication,
  registry, Sheets serialization and tracker code ran from the installed package.
  Desktop OAuth consent/refresh is unit-tested; a new desktop consent flow was
  not performed during this integration check. No credentials were copied.

## Deliberate limits

Non-annual pay projections remain unverified and unscored because the existing
core can assume hours while annualizing and does not preserve the original unit.
This prevents inflated salary/rate gates without rewriting the core. Custom
careers domains need independent employer-origin verification. Geography,
authorization, benefits and employer quality are not machine-verified gates.

The interval runner requires a live foreground process; use cron or a service
supervisor for reboot persistence. Deduplication is per installer's state directory.
Interrupted scoring claims never automatically retry. Plugin upgrades require
rerunning the runtime installer. No applications or messages are submitted.
