"""Core data models - the language of opportunities."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class OpportunityType(str, Enum):
    """Types of opportunities we hunt."""
    JOB = "job"
    FREELANCE = "freelance"
    GRANT = "grant"
    VC_FUNDING = "vc_funding"
    ANGEL = "angel"
    CONTRACT = "contract"
    EQUITY = "equity"
    BOUNTY = "bounty"


class EffortLevel(str, Enum):
    """How much work is required."""
    MINIMAL = "minimal"      # < 10 hrs/week
    LIGHT = "light"          # 10-20 hrs/week
    MODERATE = "moderate"    # 20-30 hrs/week
    FULL = "full"            # 40+ hrs/week
    VARIABLE = "variable"    # Project-based


class IncomeType(str, Enum):
    """How money flows."""
    SALARY = "salary"
    HOURLY = "hourly"
    PROJECT = "project"
    EQUITY = "equity"
    GRANT = "grant"
    REVENUE_SHARE = "revenue_share"


class UserProfile(BaseModel):
    """Your preferences - what you want from work and life."""

    # Core preferences
    min_income: int = Field(default=100000, description="Minimum annual income target")
    max_hours_weekly: int = Field(default=20, description="Maximum hours willing to work per week")
    remote_only: bool = Field(default=True, description="Only remote opportunities")

    # Skills and experience
    skills: list[str] = Field(default_factory=list, description="Your technical/professional skills")
    experience_years: int = Field(default=5, description="Years of relevant experience")
    industries: list[str] = Field(default_factory=list, description="Industries you're interested in")

    # Opportunity preferences
    opportunity_types: list[OpportunityType] = Field(
        default_factory=lambda: [OpportunityType.JOB, OpportunityType.FREELANCE, OpportunityType.CONTRACT],
        description="Types of opportunities to search for"
    )
    effort_levels: list[EffortLevel] = Field(
        default_factory=lambda: [EffortLevel.MINIMAL, EffortLevel.LIGHT],
        description="Acceptable effort levels"
    )

    # Deal breakers
    excluded_companies: list[str] = Field(default_factory=list)
    excluded_industries: list[str] = Field(default_factory=list)
    min_company_size: Optional[int] = None
    max_company_size: Optional[int] = None

    # Special interests
    interested_in_equity: bool = Field(default=True, description="Open to equity compensation")
    interested_in_founding: bool = Field(default=True, description="Open to co-founder roles")
    interested_in_grants: bool = Field(default=True, description="Research/creative grants")


class Opportunity(BaseModel):
    """A potential opportunity - job, grant, VC, etc."""

    id: str = Field(description="Unique identifier")
    title: str
    company: Optional[str] = None
    description: str
    opportunity_type: OpportunityType
    url: str

    # Compensation
    income_low: Optional[int] = None
    income_high: Optional[int] = None
    income_type: Optional[IncomeType] = None
    equity_offered: bool = False

    # Requirements
    effort_level: Optional[EffortLevel] = None
    hours_per_week: Optional[int] = None
    remote: bool = True
    location: Optional[str] = None

    # Metadata
    skills_required: list[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None
    posted_at: Optional[datetime] = None
    source: str = Field(description="Where this came from")

    # Scoring (filled by ranking engine)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    effort_score: float = Field(default=0.0, ge=0.0, le=1.0)  # Lower effort = higher score
    income_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchQuery(BaseModel):
    """What we're looking for."""

    query: str = Field(description="Natural language search query")
    opportunity_types: list[OpportunityType] = Field(default_factory=list)
    max_results: int = Field(default=20, ge=1, le=100)
    include_seen: bool = Field(default=False, description="Include previously seen opportunities")


class SearchResult(BaseModel):
    """Results from a search operation."""

    opportunities: list[Opportunity]
    query: SearchQuery
    sources_searched: list[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
