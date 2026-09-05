# Autonomous Opportunity Agent

job-engine's core is `search → extract → rank by $/hour`. The agent path
hands *finding* to an in-process brain and keeps *ranking* in Python.

```
goal ─▶ OPENAI AGENTS SDK (or Engine fallback) ─▶ {searches, opportunities}
            search_web tool / open-web engine              │
                                                           ▼
                                                   ground URLs → _rank → score()
                                                   refined $/hr · ~ if hours imputed
```

## Why a client-side tool loop

The previous brain was an external [Hermes Agent](https://github.com/NousResearch/hermes-agent)
server. It is not running in Cloud Agents or on Vercel, so `/agent` failed with a
connection error. [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
is the in-process tool loop: the model calls our `search_web` tool (Brave, or
DuckDuckGo with no key). `Opportunity.score()` still owns the $/hour.

With no `OPENAI_API_KEY`, agent mode uses the same Engine search as `find` and
returns the search angles as the trace — it does not require a sidecar process.

## Running it

```bash
job-engine agent "senior ML contract, remote"      # CLI
curl -X POST localhost:8000/agent -d '{"q":"..."}'  # API (job-engine serve)
```

`OPENAI_API_KEY` enables the Agents SDK planner. Without it, results still come
from open-web search.

## What's verified

`tests/test_agent.py` covers `_rank`, `_parse`, Engine fallback, and a mocked
SDK run. A live SDK run needs `OPENAI_API_KEY`.
