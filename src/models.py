"""Core models - lean and mean."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class Opportunity(BaseModel):
    """An opportunity. That's it."""

    title: str
    company: Optional[str] = None
    url: str
    description: str = ""

    # The only things that matter
    pay_low: Optional[int] = None
    pay_high: Optional[int] = None
    hours_per_week: Optional[int] = None
    remote: bool = True

    # Computed
    efficiency: Optional[float] = None  # $/hour - the only metric

    # Metadata
    source: str = ""
    posted: Optional[datetime] = None

    @property
    def pay(self) -> Optional[int]:
        """Best estimate of pay."""
        return self.pay_high or self.pay_low

    @property
    def dollars_per_hour(self) -> Optional[float]:
        """The only metric that matters."""
        if not self.pay or not self.hours_per_week:
            return None
        annual_hours = self.hours_per_week * 50
        return self.pay / annual_hours

    def score(self) -> float:
        """
        Score = pay / hours. Higher is better.
        Unknown hours assumed 40 (penalized).
        Unknown pay assumed $0 (penalized).
        Non-remote penalized.
        """
        pay = self.pay or 0
        hours = self.hours_per_week or 40  # Assume worst case

        if hours == 0:
            return 0

        base_score = pay / (hours * 50)  # $/hour

        # Remote bonus
        if not self.remote:
            base_score *= 0.7  # 30% penalty for office

        return base_score
