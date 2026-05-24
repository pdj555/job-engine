# Autonomous Opportunity Agent

job-engine's deterministic core is `search → extract → rank by $/hour`. The agent
path hands the *finding* to an autonomous brain and keeps the *ranking* in Python.

```
goal ─▶ HERMES AGENT (brain) ─▶ {searches, opportunities}  JSON
            self-orchestrates                 │
       (own web / terminal / memory)          ▼
                                       _rank → Opportunity.score()
                                       deterministic  $ / hr
                                              │
                                       ranked shortlist
```

## Why Hermes-direct (not a client-side tool loop)

Hermes Agent is a **full agent runtime**, not a raw model. Its OpenAI-compatible
server "handles requests with its full toolset (terminal, file operations, web
search, memory, skills) and returns the final response" — it **does not honor
client-supplied tool/function definitions** and returns no `tool_calls`
([docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server)).

So an orchestrator that hands Hermes *our* tools (the OpenAI Agents SDK pattern)
would have nothing to drive — Hermes ignores them and self-orchestrates. The
correct integration is to treat Hermes as the autonomous researcher: give it a
goal, let it run its own research, and consume its structured output.

`src/agent.py` does exactly that: one call to Hermes' `/v1/chat/completions`, then
deterministic ranking. The brain decides *what* to surface; `Opportunity.score()`
owns *the $/hour* — the model never invents a number it's graded on.

## The stack, and the frameworks evaluated

| Role | Choice | Why |
|------|--------|-----|
| **Brain / runtime** | [**Hermes Agent**](https://github.com/NousResearch/hermes-agent) | Autonomous web research over its OpenAI-compatible server. The whole point. |
| **Transport** | `openai` async client | Hermes is OpenAI-compatible, so no bespoke client. |

Frameworks we weighed and did **not** wire, with the honest reason:

- [**OpenAI Agents SDK**](https://openai.github.io/openai-agents-python/) — a
  client-side tool-calling loop. Adds nothing here because Hermes won't call
  client tools; it self-orchestrates. (It would fit a *raw* tool-calling model.)
- [**LangGraph**](https://langchain-ai.github.io/langgraph/) — explicit state-graph
  loops. Overkill when the loop lives inside Hermes.
- [**deepagents**](https://github.com/langchain-ai/deepagents) — planner + research
  subagents. Hermes *is* the research subagent; we just consume its result.
- [**Claude Agent SDK**](https://docs.claude.com/en/api/agent-sdk/overview) — its
  principle still holds: keep the scoring math in trusted code, not the model.
  That's our `_rank` boundary.

## Running it

The brain is a separate process — Nous's `hermes-agent` exposing its
OpenAI-compatible server (default `http://127.0.0.1:8642/v1`, model `hermes-agent`).

1. Start Hermes Agent (see its repo); note the port + API key.
2. Point job-engine at it (defaults match Hermes' defaults):

   ```bash
   export HERMES_BASE_URL=http://127.0.0.1:8642/v1
   export HERMES_API_KEY=change-me-local-dev
   export HERMES_MODEL=hermes-agent
   ```

3. Run the agent:

   ```bash
   job-engine agent "senior ML contract, remote"      # CLI
   curl -X POST localhost:8000/agent -d '{"q":"..."}'  # API (job-engine serve)
   ```

If Hermes isn't reachable, the CLI prints the error and `/agent` returns `503` —
the deterministic `/search` path is unaffected.

## What's verified

`tests/test_agent.py` covers the deterministic half end-to-end: `_rank` ordering
and url-filtering, `_parse` against clean / prose-wrapped / array / garbage
replies, and `agent_run` parsing-and-ranking a mocked Hermes response. A live run
needs a running Hermes Agent server.
