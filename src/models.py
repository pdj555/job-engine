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
        """Strict $/hour. None unless both pay and hours are known."""
        if not self.pay or not self.hours_per_week:
            return None
        return self.pay / (self.hours_per_week * 50)

    @property
    def refined_rate(self) -> Optional[float]:
        """Displayable $/hour. Missing hours impute 40/wk. None if no pay."""
        if not self.pay:
            return None
        hours = self.hours_per_week or 40
        if hours == 0:
            return 0.0
        return self.pay / (hours * 50)

    @property
    def rate_is_imputed(self) -> bool:
        return bool(self.pay) and self.hours_per_week is None

    def score(self) -> float:
        """Rank key: refined $/hour, then 30% penalty for office roles."""
        rate = self.refined_rate or 0.0
        if not self.remote:
            rate *= 0.7
        return rate
