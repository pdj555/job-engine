from io import StringIO

from rich.console import Console

import src.cli as cli
from src.models import Opportunity


def test_cli_displays_posted_range_and_midpoint_rate():
    buf = StringIO()
    cli.console = Console(file=buf, width=140, force_terminal=True, color_system=None)
    cli.display(
        [
            Opportunity(
                title="Staff Software Engineer",
                company="Engine",
                url="https://job-boards.greenhouse.io/engine/jobs/7994750003",
                pay_low=200_000,
                pay_high=245_000,
                hours_per_week=40,
                pay_source="ats",
            )
        ]
    )
    text = buf.getvalue()
    assert "$200,000–$245,000" in text
    assert "$111" in text
    assert "$222,500" not in text
