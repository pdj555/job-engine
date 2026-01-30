"""FastAPI routes - lean API for the opportunity engine."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models import UserProfile, OpportunityType, EffortLevel, SearchQuery
from src.agents.orchestrator import OpportunityFinder
from src.memory.vector_store import OpportunityMemory
from src.ranking.scorer import OpportunityScorer
from config.settings import settings


# Global instances
finder: Optional[OpportunityFinder] = None
memory: Optional[OpportunityMemory] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global memory
    memory = OpportunityMemory()
    yield
    # Cleanup on shutdown
    if finder:
        await finder.close()


app = FastAPI(
    title="Job Engine",
    description="AI-powered opportunity finder: minimum effort, maximum return",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ProfileCreate(BaseModel):
    """Create or update user profile."""
    min_income: int = Field(default=100000, description="Minimum annual income target")
    max_hours_weekly: int = Field(default=20, description="Maximum hours per week")
    remote_only: bool = Field(default=True)
    skills: list[str] = Field(default_factory=list)
    experience_years: int = Field(default=5)
    industries: list[str] = Field(default_factory=list)
    opportunity_types: list[str] = Field(
        default_factory=lambda: ["job", "freelance", "contract"]
    )
    interested_in_equity: bool = True
    interested_in_founding: bool = True
    interested_in_grants: bool = True


class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(description="What are you looking for?")
    max_results: int = Field(default=20, ge=1, le=100)


class FeedbackRequest(BaseModel):
    """Feedback on an opportunity."""
    opportunity_id: str
    liked: bool
    reason: str = ""


# Routes
@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "running",
        "service": "job-engine",
        "description": "AI opportunity finder - min effort, max return"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "memory": memory.get_stats() if memory else {},
        "config": {
            "openai_configured": bool(settings.openai_api_key),
            "brave_configured": bool(settings.brave_api_key),
            "perplexity_configured": bool(settings.perplexity_api_key),
        }
    }


@app.post("/profile")
async def create_profile(profile_data: ProfileCreate):
    """Create or update your profile."""
    global finder

    # Convert string types to enums
    opp_types = [OpportunityType(t) for t in profile_data.opportunity_types]

    profile = UserProfile(
        min_income=profile_data.min_income,
        max_hours_weekly=profile_data.max_hours_weekly,
        remote_only=profile_data.remote_only,
        skills=profile_data.skills,
        experience_years=profile_data.experience_years,
        industries=profile_data.industries,
        opportunity_types=opp_types,
        interested_in_equity=profile_data.interested_in_equity,
        interested_in_founding=profile_data.interested_in_founding,
        interested_in_grants=profile_data.interested_in_grants,
    )

    # Store profile in memory
    if memory:
        await memory.learn_preferences(profile)

    # Create finder with profile
    if finder:
        await finder.close()
    finder = OpportunityFinder(profile)

    return {
        "status": "profile_created",
        "profile": profile.model_dump()
    }


@app.post("/search")
async def search(request: SearchRequest):
    """Search for opportunities."""
    if not finder:
        raise HTTPException(
            status_code=400,
            detail="Profile not set. POST to /profile first."
        )

    try:
        results = await finder.find(request.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/quick")
async def quick_search(request: SearchRequest):
    """Quick search without full AI analysis."""
    if not finder:
        raise HTTPException(
            status_code=400,
            detail="Profile not set. POST to /profile first."
        )

    try:
        opportunities = await finder.quick_search(request.query)
        return {
            "opportunities": [opp.model_dump() for opp in opportunities],
            "count": len(opportunities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/opportunities")
async def list_opportunities(
    limit: int = 20,
    min_score: float = 0.0
):
    """List stored opportunities from memory."""
    if not memory:
        return {"opportunities": []}

    results = await memory.find_similar("opportunity job work", n_results=limit)
    return {
        "opportunities": [r for r in results if r.get("similarity", 0) >= min_score],
        "count": len(results)
    }


@app.get("/opportunities/{opp_id}")
async def get_opportunity(opp_id: str):
    """Get details on a specific opportunity."""
    if not memory:
        raise HTTPException(status_code=404, detail="Not found")

    # Search for this specific ID
    results = await memory.find_similar(opp_id, n_results=1)
    if not results:
        raise HTTPException(status_code=404, detail="Opportunity not found")

    return results[0]


@app.post("/opportunities/{opp_id}/feedback")
async def submit_feedback(opp_id: str, feedback: FeedbackRequest):
    """Submit feedback on an opportunity."""
    if not memory:
        raise HTTPException(status_code=500, detail="Memory not initialized")

    await memory.mark_feedback(opp_id, feedback.liked, feedback.reason)
    return {"status": "feedback_recorded"}


@app.post("/opportunities/{opp_id}/applied")
async def mark_applied(opp_id: str):
    """Mark that you applied to an opportunity."""
    if not memory:
        raise HTTPException(status_code=500, detail="Memory not initialized")

    await memory.mark_applied(opp_id)
    return {"status": "marked_applied"}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    stats = memory.get_stats() if memory else {}
    return {
        "memory": stats,
        "profile_active": finder is not None
    }


# Run with: uvicorn src.api.routes:app --reload
