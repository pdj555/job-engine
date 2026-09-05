from io import StringIO

from rich.console import Console

from src.cli import display
from src.models import Opportunity


def test_display_marks_imputed_refined_rate():
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    import src.cli as cli

    original = cli.console
    cli.console = console
    try:
        display(
            [
                Opportunity(title="Thin", url="https://u", pay_high=100_000),
                Opportunity(title="Known", url="https://v", pay_high=100_000, hours_per_week=20),
            ]
        )
    finally:
        cli.console = original
    text = buf.getvalue()
    assert "~$50" in text
    assert "$100" in text
    assert "~40" in text
