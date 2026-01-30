"""Opportunity scoring engine - minimum effort, maximum return."""

from typing import Optional
from openai import OpenAI

from src.models import Opportunity, UserProfile, EffortLevel
from config.settings import settings


class OpportunityScorer:
    """
    Scores opportunities based on the holy grail:
    Maximum income / Minimum effort / Best fit

    The algorithm:
    1. Income Score: How well does compensation meet/exceed targets?
    2. Effort Score: How little time investment required?
    3. Fit Score: How well does it match skills and preferences?
    4. Overall = weighted combination optimized for efficiency

    We're hunting for the best ROI on your time.
    """

    # Scoring weights - tune these for your priorities
    WEIGHT_INCOME = 0.35
    WEIGHT_EFFORT = 0.35
    WEIGHT_FIT = 0.30

    # Effort level hours mapping
    EFFORT_HOURS = {
        EffortLevel.MINIMAL: 10,
        EffortLevel.LIGHT: 20,
        EffortLevel.MODERATE: 30,
        EffortLevel.FULL: 45,
        EffortLevel.VARIABLE: 25,  # Assume moderate
    }

    def __init__(self):
        self.openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def score_opportunity(
        self,
        opp: Opportunity,
        profile: UserProfile
    ) -> Opportunity:
        """
        Score an opportunity against user profile.

        Updates the opportunity's score fields and returns it.
        """
        # Calculate component scores
        income_score = self._score_income(opp, profile)
        effort_score = self._score_effort(opp, profile)
        fit_score = self._score_fit(opp, profile)

        # Calculate overall score
        overall = (
            self.WEIGHT_INCOME * income_score +
            self.WEIGHT_EFFORT * effort_score +
            self.WEIGHT_FIT * fit_score
        )

        # Update opportunity
        opp.income_score = income_score
        opp.effort_score = effort_score
        opp.relevance_score = fit_score
        opp.overall_score = overall

        return opp

    def score_opportunities(
        self,
        opportunities: list[Opportunity],
        profile: UserProfile
    ) -> list[Opportunity]:
        """Score and sort opportunities by overall score (descending)."""
        scored = [self.score_opportunity(opp, profile) for opp in opportunities]
        return sorted(scored, key=lambda x: x.overall_score, reverse=True)

    def _score_income(self, opp: Opportunity, profile: UserProfile) -> float:
        """
        Score based on income potential.

        Higher income relative to target = higher score.
        """
        if not opp.income_high and not opp.income_low:
            return 0.5  # Unknown, neutral score

        # Use the higher end of range, or low if that's all we have
        income = opp.income_high or opp.income_low or 0

        # Calculate ratio to target
        ratio = income / profile.min_income if profile.min_income > 0 else 1.0

        # Score: 0.5 at target, scales up to 1.0 at 2x target, down to 0 at 0
        if ratio >= 2.0:
            return 1.0
        elif ratio >= 1.0:
            return 0.5 + (ratio - 1.0) * 0.5  # 0.5-1.0
        else:
            return ratio * 0.5  # 0-0.5

    def _score_effort(self, opp: Opportunity, profile: UserProfile) -> float:
        """
        Score based on time investment required.

        Less time = higher score (we want efficiency).
        """
        # Get hours from opportunity
        hours = opp.hours_per_week
        if not hours and opp.effort_level:
            hours = self.EFFORT_HOURS.get(opp.effort_level, 30)
        if not hours:
            hours = 40  # Assume full-time if unknown

        max_hours = profile.max_hours_weekly

        # Score inversely proportional to hours
        # At or below max_hours: score based on how much below
        # Above max_hours: penalize heavily

        if hours <= max_hours:
            # Sweet spot: less hours = higher score
            # At max_hours = 0.5, at 0 hours = 1.0
            return 0.5 + 0.5 * (1 - hours / max_hours) if max_hours > 0 else 1.0
        else:
            # Over max hours: penalty
            overage_ratio = hours / max_hours if max_hours > 0 else 2
            # At 2x max_hours = 0, at max_hours = 0.5
            return max(0, 0.5 * (2 - overage_ratio))

    def _score_fit(self, opp: Opportunity, profile: UserProfile) -> float:
        """
        Score based on skill/preference match.
        """
        score = 0.5  # Base score

        # Skill match
        if opp.skills_required and profile.skills:
            opp_skills = set(s.lower() for s in opp.skills_required)
            user_skills = set(s.lower() for s in profile.skills)
            overlap = len(opp_skills & user_skills)
            if opp_skills:
                skill_ratio = overlap / len(opp_skills)
                score += 0.2 * skill_ratio

        # Remote preference
        if profile.remote_only:
            if opp.remote:
                score += 0.1
            else:
                score -= 0.2  # Penalty for non-remote when remote required

        # Opportunity type preference
        if opp.opportunity_type in profile.opportunity_types:
            score += 0.1

        # Equity interest
        if opp.equity_offered and profile.interested_in_equity:
            score += 0.1

        return min(1.0, max(0.0, score))

    async def llm_evaluate(
        self,
        opp: Opportunity,
        profile: UserProfile
    ) -> dict:
        """
        Use LLM for nuanced evaluation beyond simple scoring.

        Returns qualitative assessment and recommendations.
        """
        if not self.openai:
            return {"evaluation": "LLM not available", "recommendation": "unknown"}

        prompt = f"""
        Evaluate this opportunity for fit with the user profile.

        OPPORTUNITY:
        - Title: {opp.title}
        - Company: {opp.company or 'Unknown'}
        - Type: {opp.opportunity_type.value}
        - Income: ${opp.income_low or '?'} - ${opp.income_high or '?'}
        - Hours: {opp.hours_per_week or 'Unknown'}/week
        - Remote: {opp.remote}
        - Skills needed: {', '.join(opp.skills_required) or 'Unknown'}
        - Description: {opp.description[:500]}

        USER PROFILE:
        - Target income: ${profile.min_income:,}+
        - Max hours: {profile.max_hours_weekly}/week
        - Skills: {', '.join(profile.skills)}
        - Remote only: {profile.remote_only}
        - Experience: {profile.experience_years} years

        Provide a brief evaluation:
        1. Fit assessment (1-2 sentences)
        2. Key pros (bullet points)
        3. Key cons (bullet points)
        4. Recommendation: "apply", "skip", or "research more"
        5. Estimated effort-to-reward ratio (1-10, where 10 is best ROI)

        Be direct and actionable.
        """

        try:
            response = self.openai.chat.completions.create(
                model=settings.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return {
                "evaluation": response.choices[0].message.content,
                "model": settings.fast_model
            }
        except Exception as e:
            return {"evaluation": f"Error: {e}", "recommendation": "unknown"}

    def get_efficiency_metric(self, opp: Opportunity) -> Optional[float]:
        """
        Calculate dollars per hour worked.

        The ultimate efficiency metric.
        """
        if not opp.income_high or not opp.hours_per_week:
            return None

        annual_hours = opp.hours_per_week * 50  # Assume 50 weeks
        return opp.income_high / annual_hours

    def rank_by_efficiency(
        self,
        opportunities: list[Opportunity]
    ) -> list[tuple[Opportunity, float]]:
        """
        Rank opportunities by pure $/hour efficiency.

        Returns list of (opportunity, dollars_per_hour) tuples.
        """
        with_efficiency = []
        for opp in opportunities:
            efficiency = self.get_efficiency_metric(opp)
            if efficiency:
                with_efficiency.append((opp, efficiency))

        return sorted(with_efficiency, key=lambda x: x[1], reverse=True)
