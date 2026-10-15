---
name: jobhunter
description: Find and track jobs from a plain-language request, with employer-verified pay, remote evidence, strict user gates, and a personal Google Sheets pipeline. Use for one-off or recurring job hunts and application tracking.
---

# Jobhunter

The user describes what they want. Translate that into a profile, run the existing
engine through the strict hunt adapter, and surface only new qualifying roles.

## Turn the request into gates

Use the user's stated constraints. For example, “Remote Python jobs paying at
least $150k, no more than 35 hours a week, check every six hours” becomes:

```json
{
  "query": "Python engineer",
  "sheet_id": "",
  "remote_only": true,
  "min_annual_pay": 150000,
  "min_dollars_per_hour": 0,
  "max_hours_per_week": 35,
  "title_terms": ["python"],
  "exclude_companies": [],
  "employer_domains": []
}
```

Save it as `$JOBHUNTER_HOME/profile.json`, defaulting to
`~/.local/share/jobhunter/profile.json`. Preserve the existing sheet ID when
updating gates. The CLI validates types and rejects unknown gates. Never soften a
user constraint to produce results. Ask one focused question if a requirement
would materially change eligibility and has no supported gate. Geography, work
authorization, benefits and employer quality are not currently machine-verified
gates. Do not claim those constraints were cleared or schedule a hunt that ignores
them. Search terms alone are not gates. Title terms require every listed term in
the title; company exclusions are case-insensitive substrings.

Employer domains are an allowlist for custom careers sites. Add an exact hostname
only after verifying the employer's own website links to that careers site. Do
not add an aggregator to increase results. Recognized employer ATS endpoints are
accepted directly; known syndicated/recruiter boards remain unverified.

## Install and connect

Run `"${CLAUDE_PLUGIN_ROOT}/scripts/install.sh"` once. It installs a private Python
runtime outside the plugin cache. Then invoke
`"${CLAUDE_PLUGIN_ROOT}/scripts/jobhunter"` for commands below. Python 3.11+ and
macOS/Linux are required. See [docs/JOBHUNTER.md](docs/JOBHUNTER.md) for OAuth setup.

Each installer connects **their own Google account** with their own Desktop OAuth
client. Use `connect --client-secrets /absolute/path/to/client_secret.json` to
create a personal sheet using `drive.file`, or add `--sheet-id ID` to explicitly
authorize the broader Sheets scope for an existing sheet. Let the user complete
Google consent in the browser. Never copy another installer's token, use the
author's account, request a service-account key, or print credentials. Tokens,
profile and seen registry stay outside the repository and plugin cache.

## Hunt and repeat

- `hunt` runs once, verifies fresh employer evidence, saves all outcomes to the
  sheet, and prints only newly qualified roles.
- `hunt --every-minutes 360` repeats every six hours while running. For persistence,
  use the cron recipe in the documentation. Do not start a recurring process until
  the user requests a cadence. Do not send email, applications or messages.
- `hunt --dry-run` runs synthetic listings through the actual engine's parsers and
  ranker and writes to **Jobhunter Dry Run**. Its registry is separate from live
  hunts. Clearly identify these as synthetic. Do not schedule dry runs.
- `track POSTING_KEY applied --notes "Submitted on employer site"` updates the
  sheet. Supported states: discovered, shortlisted, applied, interviewing, offer,
  rejected, withdrawn. The user can also edit the sheet's dropdowns and notes.

The local seen registry is per installer and permanent across gate changes.
Deduplication happens before ranking. Canonical URLs remove tracking; resolved
employer evidence catches careers-site/ATS aliases. A crashed scoring attempt is
never automatically retried. An interrupted claim can remain without a result.
Sheets failures retry cached results and reconcile keys, preserving user-edited
pipeline states and notes. Never delete the seen registry to make old jobs rescore.

## Evidence and output contract

Pay must come from the employer's listing, ATS or JobPosting schema. Remote status
and weekly hours must also have listing evidence. Snippets, model estimates,
salary guides, missing hours and unknown arrangements cannot establish evidence.
Every unverified result is marked **unverified**, has a blank score, stays in the
sheet for review, and cannot trigger a notification. Known nonmatching roles are
marked **filtered** with a blank score.

Rank only roles clearing every gate. The unchanged engine uses salary-range
midpoints divided by verified weekly hours × 50. The minimum annual pay gate uses
the posted lower bound. Core non-annual pay conversions can assume hours, so those
projections stay unverified and unscored until their conversion basis is supported.
If the user permits office roles, the existing engine's 30% office penalty applies
to ranking and the rate gate. Remote-only is the default.

Report qualified roles in descending rank with the full posted range, hours,
employer evidence URL and sheet link. If no new role clears the gates, stay silent.
Report operational failures concisely so the user can restore the connection;
silence must not imply a broken hunt succeeded. A dry run is installation evidence,
not evidence that a real employer is hiring.
