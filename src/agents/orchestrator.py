"""LangGraph orchestrator - the intelligent opportunity hunter."""

from typing import Annotated, TypedDict, Literal
from datetime import datetime
import json
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from src.models import (
    Opportunity,
    OpportunityType,
    UserProfile,
    SearchQuery,
    SearchResult,
)
from src.search.aggregator import SearchAggregator
from src.search.perplexity import PerplexitySearch
from src.memory.vector_store import OpportunityMemory
from src.ranking.scorer import OpportunityScorer
from config.settings import settings


class AgentState(TypedDict):
    """State passed through the agent graph."""
    messages: Annotated[list, operator.add]
    profile: UserProfile
    query: str
    opportunities: list[Opportunity]
    ranked_opportunities: list[Opportunity]
    search_complete: bool
    research_results: dict
    final_recommendations: list[dict]


class OpportunityFinder:
    """
    LangGraph-powered opportunity discovery agent.

    Workflow:
    1. Understand user query and intent
    2. Search across all sources
    3. Score and rank opportunities
    4. Deep research top candidates
    5. Present ranked recommendations
    """

    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.llm = ChatOpenAI(
            model=settings.reasoning_model,
            temperature=0.3,
            api_key=settings.openai_api_key
        )
        self.fast_llm = ChatOpenAI(
            model=settings.fast_model,
            temperature=0,
            api_key=settings.openai_api_key
        )
        self.search = SearchAggregator()
        self.memory = OpportunityMemory()
        self.scorer = OpportunityScorer()
        self.perplexity = PerplexitySearch()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        # Define the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("understand", self._understand_query)
        workflow.add_node("search", self._execute_search)
        workflow.add_node("rank", self._rank_opportunities)
        workflow.add_node("research", self._deep_research)
        workflow.add_node("recommend", self._generate_recommendations)

        # Define edges
        workflow.set_entry_point("understand")
        workflow.add_edge("understand", "search")
        workflow.add_edge("search", "rank")
        workflow.add_conditional_edges(
            "rank",
            self._should_research,
            {
                "research": "research",
                "recommend": "recommend"
            }
        )
        workflow.add_edge("research", "recommend")
        workflow.add_edge("recommend", END)

        return workflow.compile()

    async def _understand_query(self, state: AgentState) -> AgentState:
        """Understand what the user is looking for."""
        query = state["query"]
        profile = state["profile"]

        system_prompt = """You are an opportunity hunting assistant.
        Analyze the user's query and extract:
        1. What type of opportunity they want (job, freelance, grant, funding, etc.)
        2. Key skills or domains mentioned
        3. Any specific requirements or preferences

        Be direct and extract actionable search parameters."""

        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            User query: {query}

            User profile:
            - Skills: {', '.join(profile.skills)}
            - Industries: {', '.join(profile.industries)}
            - Target income: ${profile.min_income:,}
            - Max hours: {profile.max_hours_weekly}/week
            - Remote only: {profile.remote_only}

            What should we search for?
            """)
        ])

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response.content)]
        }

    async def _execute_search(self, state: AgentState) -> AgentState:
        """Execute search across all sources."""
        query = state["query"]
        profile = state["profile"]

        # Create search query
        search_query = SearchQuery(
            query=query,
            opportunity_types=profile.opportunity_types,
            max_results=30
        )

        # Execute search
        result = await self.search.search(search_query, profile)

        # Store in memory
        await self.memory.store_opportunities(result.opportunities)

        return {
            **state,
            "opportunities": result.opportunities,
            "search_complete": True,
            "messages": state["messages"] + [
                AIMessage(content=f"Found {len(result.opportunities)} opportunities from {', '.join(result.sources_searched)}")
            ]
        }

    async def _rank_opportunities(self, state: AgentState) -> AgentState:
        """Score and rank all opportunities."""
        opportunities = state["opportunities"]
        profile = state["profile"]

        # Score all opportunities
        ranked = self.scorer.score_opportunities(opportunities, profile)

        return {
            **state,
            "ranked_opportunities": ranked,
            "messages": state["messages"] + [
                AIMessage(content=f"Ranked {len(ranked)} opportunities by efficiency (min effort, max return)")
            ]
        }

    def _should_research(self, state: AgentState) -> Literal["research", "recommend"]:
        """Decide if we should do deep research on top opportunities."""
        ranked = state.get("ranked_opportunities", [])

        # If we have good candidates, research the top ones
        if ranked and len(ranked) >= 3:
            top_score = ranked[0].overall_score if ranked else 0
            if top_score > 0.6:
                return "research"

        return "recommend"

    async def _deep_research(self, state: AgentState) -> AgentState:
        """Do deep research on top opportunities."""
        ranked = state["ranked_opportunities"][:5]  # Top 5

        research_results = {}
        for opp in ranked:
            if opp.company:
                result = await self.perplexity.evaluate_opportunity(
                    opportunity_title=opp.title,
                    company=opp.company,
                    url=opp.url
                )
                research_results[opp.id] = result

        return {
            **state,
            "research_results": research_results,
            "messages": state["messages"] + [
                AIMessage(content=f"Completed deep research on top {len(research_results)} opportunities")
            ]
        }

    async def _generate_recommendations(self, state: AgentState) -> AgentState:
        """Generate final recommendations."""
        ranked = state["ranked_opportunities"][:10]
        research = state.get("research_results", {})
        profile = state["profile"]

        recommendations = []
        for opp in ranked:
            rec = {
                "opportunity": opp.model_dump(),
                "scores": {
                    "overall": opp.overall_score,
                    "income": opp.income_score,
                    "effort": opp.effort_score,
                    "fit": opp.relevance_score,
                },
                "efficiency": self.scorer.get_efficiency_metric(opp),
                "research": research.get(opp.id, {}),
            }
            recommendations.append(rec)

        # Generate summary with LLM
        summary_prompt = f"""
        Based on these top opportunities, create a brief executive summary.

        User wants: ${profile.min_income:,}+ with max {profile.max_hours_weekly} hrs/week

        Top 5 opportunities:
        {json.dumps([{
            'title': r['opportunity']['title'],
            'company': r['opportunity'].get('company'),
            'score': r['scores']['overall'],
            'income': r['opportunity'].get('income_high'),
            'hours': r['opportunity'].get('hours_per_week')
        } for r in recommendations[:5]], indent=2)}

        Provide:
        1. One sentence summary
        2. Top recommendation and why
        3. Any quick wins (easy to apply, high chance of success)
        """

        summary_response = await self.fast_llm.ainvoke([
            HumanMessage(content=summary_prompt)
        ])

        return {
            **state,
            "final_recommendations": recommendations,
            "messages": state["messages"] + [
                AIMessage(content=summary_response.content)
            ]
        }

    async def find(self, query: str) -> dict:
        """
        Main entry point - find opportunities matching the query.

        Returns:
            {
                "recommendations": [...],
                "summary": "...",
                "stats": {...}
            }
        """
        initial_state: AgentState = {
            "messages": [],
            "profile": self.profile,
            "query": query,
            "opportunities": [],
            "ranked_opportunities": [],
            "search_complete": False,
            "research_results": {},
            "final_recommendations": [],
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "recommendations": final_state["final_recommendations"],
            "summary": final_state["messages"][-1].content if final_state["messages"] else "",
            "total_found": len(final_state["opportunities"]),
            "sources_searched": ["brave", "perplexity", "web"],
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def quick_search(self, query: str) -> list[Opportunity]:
        """Quick search without full workflow - just search and rank."""
        search_query = SearchQuery(query=query, max_results=20)
        result = await self.search.search(search_query, self.profile)
        return self.scorer.score_opportunities(result.opportunities, self.profile)

    async def close(self):
        """Cleanup resources."""
        await self.search.close()
        await self.perplexity.close()


def create_search_graph(profile: UserProfile) -> OpportunityFinder:
    """Factory function to create an OpportunityFinder instance."""
    return OpportunityFinder(profile)
