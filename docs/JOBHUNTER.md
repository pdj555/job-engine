# Jobhunter plugin

Jobhunter wraps Job Engine. It adds strict user gates, an installer-owned seen
registry, repeat hunts, and a Google Sheets results/application tracker. The
existing `job-engine`, API, website, search adapters and ranker remain available.

## Install

Requires Claude Code, Python 3.11+, and macOS or Linux.

```sh
claude plugin marketplace add pdj555/job-engine
claude plugin install jobhunter@jobhunter-marketplace
```

For local development:

```sh
claude plugin validate .
claude plugin marketplace add .
claude plugin install jobhunter@jobhunter-marketplace
```

Restart Claude Code after installation. Invoke `/jobhunter:jobhunter` with a
plain-language goal, or `/jobhunter:hunt`. The root SKILL.md carries the interface.
Run the installed plugin's `scripts/install.sh` once, then its `scripts/jobhunter`
wrapper. Upgrading the plugin requires rerunning install.sh to upgrade the runtime.
For Python-only development: `pip install -e ".[dev,sheets]"`.

The plugin cache is code only. `JOBHUNTER_HOME` defaults to
`~/.local/share/jobhunter` and contains `runtime/`, `profile.json`,
`google-token.json`, `seen.sqlite`, and `hunt.lock`. Set it to a different directory
for a separate installer/profile. Keep the same directory to retain deduplication
across upgrades. Never commit this state or OAuth client files.

## Profile

Create the directory and profile before connecting. The skill writes this from
plain language. Example:

```json
{
  "query": "Python engineer",
  "sheet_id": "",
  "remote_only": true,
  "min_annual_pay": 150000,
  "min_dollars_per_hour": 80,
  "max_hours_per_week": 40,
  "title_terms": ["engineer"],
  "exclude_companies": [],
  "employer_domains": []
}
```

All gates apply together. Annual minimum uses the posted salary floor. Ranking
uses the posted midpoint and verified hours, retaining the core's 30% penalty if
office roles are allowed. Missing pay, hours, remote evidence or first-party
origin means **unverified**, with no score and no alert. No hours are imputed.
Custom careers domains must be independently verified and explicitly allowlisted.
Hourly and other non-annual projections are currently unverified and unscored:
the core can annualize at assumed hours and does not retain the original pay unit.
Jobhunter refuses to rank these projections rather than inflate their rate.
Recognized employer ATS hosts are supported without adding domains. Geography,
work authorization and benefits require additional user review; they are not
verified gates. Do not present query terms as eligibility verification.

## Connect your Google account

1. In your Google Cloud project, enable the Google Sheets API. Configure the OAuth
   consent screen. If the app is in testing, add your Google account as a test user.
2. Create an OAuth client with application type **Desktop app**. Download the
   client JSON outside this repository. Each installer supplies their own client;
   the plugin ships no shared application credentials or service-account key.
3. Run `jobhunter connect --client-secrets /absolute/path/to/client_secret.json`.
   Finish Google's browser consent. The loopback callback uses an ephemeral port.
4. The command creates a sheet in your account, prepares its Jobhunter tab, saves
   its ID to your profile and prints its URL. The default scope is `drive.file`,
   limited to files this app creates or opens. To use an existing sheet, pass
   `--sheet-id ID`; this explicitly requests `spreadsheets` access instead.

Refresh tokens are stored locally with mode 0600 and automatically refreshed for
unattended runs. Google test-mode refresh tokens can expire; reconnect when
required. To change accounts, rerun connect and finish consent with the intended
account. Use a separate JOBHUNTER_HOME to maintain independent accounts/registries.

## Hunt and schedule

```sh
jobhunter hunt
jobhunter hunt --every-minutes 360
jobhunter hunt --dry-run
```

The interval loop runs in the foreground, waits after each completed hunt, and
stops with Ctrl-C. It survives empty runs. Operational errors exit nonzero and go
to stderr. Use cron or your service supervisor for restart/reboot persistence.
The interval loop reloads your profile and connection before every hunt, so saved
gate changes take effect on the next cycle.
For example, after installing the runtime, add this with `crontab -e` for every
six hours (replace both absolute paths):

```cron
0 */6 * * * JOBHUNTER_HOME=/absolute/path/to/jobhunter /absolute/path/to/jobhunter/runtime/bin/jobhunter hunt
```

Cron sees an empty stdout on successful hunts with no new qualifying role. Match
JSON goes to stdout for a user-chosen notification sink. Errors go to stderr.
The plugin never sends email, applies to roles, or messages employers. Set up
your own service supervisor/logging if you need background uptime or alerts.

Deduplication uses canonical listing URLs before enrichment/scoring, then resolved
employer evidence URLs to catch ATS aliases. All outcomes are permanent per
registry even if gates change. Overlapping hunts cannot acquire the same lock.
The claim is committed before scoring, so crashes cannot cause a second score.
An interrupted claim remains in seen.sqlite with a null result for manual audit;
it is deliberately not retried. Delivery failures retain a durable outbox; later
hunts publish it without re-evaluating and read sheet keys to avoid duplicate rows.
Cached matches evaluated under an older profile are written as historical outcomes
but cannot trigger alerts after the query or gates change.
Deduplication is local to one installer's state directory, not global across devices.

Dry mode uses three synthetic employer listings with the real engine parsers and
ranker. It writes to **Jobhunter Dry Run** in your sheet and uses `dry-run/seen.sqlite`.
It performs no live search, submits no application, and does not pollute live seen
state. Rows and stdout are marked `dry_run`. Repeating it in the same state is
silent and adds no duplicate rows.

## Application tracker

Each result row contains its posting key, listing/evidence URLs, pay range,
verified hours, rank, gate status/reason, provenance and pipeline state. Unverified
and filtered rows have blank rank cells. The tracker state dropdown supports
`discovered`, `shortlisted`, `applied`, `interviewing`, `offer`, `rejected`, `withdrawn`.

```sh
jobhunter track POSTING_KEY applied --notes "Submitted through employer site"
```

You can edit pipeline states and notes in Sheets. Hunts append new keyed rows and
preserve existing rows. A tab with incompatible headers fails instead of
overwriting your data. Values are written as RAW so job titles cannot execute
spreadsheet formulas.

## Development verification

```sh
pip install -e ".[dev,sheets]"
pytest -q
claude plugin validate . --strict
jobhunter --help
jobhunter hunt --dry-run
```

The final command requires a connected Google account and a configured sheet.
Read back the Jobhunter Dry Run rows to verify the write. Unit tests use mocked
transport; a real sheet write is separate integration evidence.

Manifest and install conventions follow the [Claude Code plugin reference](https://code.claude.com/docs/en/plugins-reference)
and [marketplace guide](https://code.claude.com/docs/en/plugin-marketplaces).
OAuth follows the [Google Sheets Python quickstart](https://developers.google.com/workspace/sheets/api/quickstart/python).
