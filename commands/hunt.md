---
description: Hunt once or on the user's requested schedule, with strict evidence gates.
argument-hint: [plain-language job goal or --dry-run]
---

Use the jobhunter skill at `${CLAUDE_PLUGIN_ROOT}/SKILL.md`.
Translate the user's request into the existing profile without relaxing gates.
Run `"${CLAUDE_PLUGIN_ROOT}/scripts/jobhunter" hunt` after setup. Only use
`--every-minutes N` if the user asked for repeated hunts. For `--dry-run`, use the
dedicated dry-run mode and label synthetic results. Stay silent when no new role
clears all gates; report operational failures.

User request: $ARGUMENTS
