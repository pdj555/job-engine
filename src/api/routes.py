"""API - one endpoint that matters."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.engine import Engine
from src.models import Opportunity
from config.settings import settings

app = FastAPI(
    title="Job Engine",
    description="Find opportunities. Ranked by $/hour.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Engine()


class SearchRequest(BaseModel):
    q: str
    limit: int = 20


def _payload(results: list[Opportunity]) -> dict:
    return {
        "results": [
            {
                "title": o.title,
                "company": o.company,
                "url": o.url,
                "pay": o.pay,
                "hours_per_week": o.hours_per_week,
                "dollars_per_hour": o.dollars_per_hour,
                "remote": o.remote,
                "score": o.score(),
            }
            for o in results
        ],
        "count": len(results),
    }


@app.get("/")
async def root():
    return {"status": "ok", "usage": "POST /search with {q: 'your query'}"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "apis": {
            "openai": bool(settings.openai_api_key),
            "brave": bool(settings.brave_api_key),
            "perplexity": bool(settings.perplexity_api_key),
        },
        "search_ready": True,
    }


@app.post("/search")
async def search(req: SearchRequest):
    """
    Search for opportunities.

    Returns ranked by $/hour (highest first).
    """
    results = await engine.find(req.q, req.limit)
    return _payload(results)


@app.get("/search")
async def search_get(q: str, limit: int = 20):
    """GET version for easy testing."""
    return await search(SearchRequest(q=q, limit=limit))


@app.post("/agent")
async def agent_search(req: SearchRequest):
    """
    Autonomous agent search.

    The Hermes Agent brain plans searches, extracts opportunities, and ranks
    them by $/hour. Returns the same shape as /search.
    """
    from src.agent import agent_run

    try:
        run = await agent_run(req.q, req.limit)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Agent unavailable (is Hermes Agent running?): {e}",
        ) from e
    return {**_payload(run.ranked), "searches": run.searches}
