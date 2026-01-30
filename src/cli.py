"""CLI - one command, instant results."""

import asyncio
import sys

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.engine import Engine
from src.models import Opportunity

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    query: str = typer.Argument(..., help="What you want"),
    limit: int = typer.Option(15, "-n", help="Results"),
):
    """
    Find opportunities. Ranked by $/hour.

    Examples:
        job-engine "AI engineer"
        job-engine "python freelance"
        job-engine "startup equity"
    """
    console.print(f"\n[bold]Hunting:[/bold] {query}\n")

    async def run():
        engine = Engine()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Searching everything...", total=None)
            results = await engine.find(query, limit)

        if not results:
            console.print("[yellow]Nothing found. Try different keywords.[/yellow]")
            return

        display(results)

    asyncio.run(run())


def display(opportunities: list[Opportunity]):
    """Show results."""
    table = Table(show_header=True, header_style="bold", box=None)

    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", max_width=45)
    table.add_column("Company", style="dim", max_width=20)
    table.add_column("Pay", style="green", justify="right")
    table.add_column("Hrs", style="yellow", justify="right")
    table.add_column("$/hr", style="bold magenta", justify="right")

    for i, opp in enumerate(opportunities, 1):
        pay = f"${opp.pay:,}" if opp.pay else "?"
        hours = str(opp.hours_per_week) if opp.hours_per_week else "?"
        efficiency = f"${opp.dollars_per_hour:.0f}" if opp.dollars_per_hour else "?"
        remote_tag = "" if opp.remote else " [red](office)[/red]"

        table.add_row(
            str(i),
            (opp.title[:42] + "..." if len(opp.title) > 45 else opp.title) + remote_tag,
            (opp.company or "")[:20],
            pay,
            hours,
            efficiency,
        )

    console.print(table)
    console.print()

    # Show top 3 URLs
    console.print("[bold]Top picks:[/bold]")
    for i, opp in enumerate(opportunities[:3], 1):
        eff = f"${opp.dollars_per_hour:.0f}/hr" if opp.dollars_per_hour else ""
        console.print(f"  {i}. {opp.title[:50]}")
        console.print(f"     [dim]{opp.url}[/dim]")
        if eff:
            console.print(f"     [green]{eff}[/green]")
        console.print()


@app.command()
def research(url: str = typer.Argument(..., help="URL to research")):
    """Deep dive on a specific opportunity."""
    console.print(f"\n[bold]Researching:[/bold] {url}\n")

    async def run():
        engine = Engine()
        opp = Opportunity(title="Research", url=url)
        result = await engine.research(opp)
        console.print(Panel(result, title="Research", border_style="green"))

    asyncio.run(run())


@app.command()
def serve(port: int = typer.Option(8000, "-p")):
    """Start API server."""
    import uvicorn
    console.print(f"[green]Starting on :{port}[/green]")
    uvicorn.run("src.api.routes:app", host="0.0.0.0", port=port)


def cli():
    """Entry point."""
    app()


if __name__ == "__main__":
    cli()
