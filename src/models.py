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
    pay_source: Optional[str] = None  # "posted" | "schema" | "ats" | None
    hours_source: Optional[str] = None

    @property
    def pay(self) -> Optional[int]:
        """Best estimate of pay: midpoint when both bounds are posted."""
        if self.pay_low is not None and self.pay_high is not None:
            return (self.pay_low + self.pay_high) // 2
        if self.pay_high is not None:
            return self.pay_high
        return self.pay_low

    @property
    def dollars_per_hour(self) -> Optional[float]:
        """Strict $/hour. None unless both pay and hours are known."""
        if self.pay is None or not self.hours_per_week:
            return None
        return self.pay / (self.hours_per_week * 50)

    @property
    def refined_rate(self) -> Optional[float]:
        """Displayable $/hour. Missing hours impute 40/wk. None if no pay."""
        if self.pay is None:
            return None
        hours = self.hours_per_week or 40
        if hours == 0:
            return 0.0
        return self.pay / (hours * 50)

    @property
    def rate_is_imputed(self) -> bool:
        return self.pay is not None and self.hours_per_week is None

    def score(self) -> float:
        """Rank key: refined $/hour, then 30% penalty for office roles."""
        rate = self.refined_rate or 0.0
        if not self.remote:
            rate *= 0.7
        return rate
