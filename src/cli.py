"""CLI interface for the opportunity engine."""

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.models import UserProfile, OpportunityType
from src.agents.orchestrator import OpportunityFinder
from src.memory.vector_store import OpportunityMemory

app = typer.Typer(
    name="job-engine",
    help="AI-powered opportunity finder: minimum effort, maximum return"
)
console = Console()


def load_profile(profile_path: Optional[str] = None) -> UserProfile:
    """Load user profile from file or use defaults."""
    if profile_path:
        try:
            with open(profile_path) as f:
                data = json.load(f)
            return UserProfile(**data)
        except Exception as e:
            console.print(f"[yellow]Could not load profile: {e}. Using defaults.[/yellow]")

    return UserProfile(
        min_income=100000,
        max_hours_weekly=20,
        remote_only=True,
        skills=["python", "ai", "machine learning"],
        industries=["technology", "ai"],
        opportunity_types=[
            OpportunityType.JOB,
            OpportunityType.FREELANCE,
            OpportunityType.CONTRACT,
            OpportunityType.GRANT
        ]
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="What are you looking for?"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Path to profile JSON"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick search without deep analysis"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max results to show"),
):
    """Search for opportunities matching your criteria."""

    user_profile = load_profile(profile)

    console.print(Panel(
        f"[bold]Searching for:[/bold] {query}\n"
        f"[dim]Target: ${user_profile.min_income:,}+ | Max {user_profile.max_hours_weekly}hrs/week | Remote: {user_profile.remote_only}[/dim]",
        title="Job Engine",
        border_style="blue"
    ))

    async def run_search():
        finder = OpportunityFinder(user_profile)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if quick:
                task = progress.add_task("Quick searching...", total=None)
                results = await finder.quick_search(query)
                progress.update(task, completed=True)

                # Display results
                display_opportunities(results[:limit])
            else:
                task = progress.add_task("Searching & analyzing...", total=None)
                results = await finder.find(query)
                progress.update(task, completed=True)

                # Display summary
                console.print("\n")
                console.print(Panel(results["summary"], title="Summary", border_style="green"))

                # Display top recommendations
                display_recommendations(results["recommendations"][:limit])

        await finder.close()

    asyncio.run(run_search())


def display_opportunities(opportunities: list):
    """Display opportunities in a table."""
    table = Table(title="Opportunities Found", show_header=True, header_style="bold magenta")

    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Company", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Income", style="blue")
    table.add_column("Score", style="red")

    for opp in opportunities:
        income = f"${opp.income_high:,}" if opp.income_high else "—"
        score = f"{opp.overall_score:.2f}" if opp.overall_score else "—"

        table.add_row(
            opp.title[:40],
            (opp.company or "—")[:20],
            opp.opportunity_type.value,
            income,
            score
        )

    console.print(table)


def display_recommendations(recommendations: list):
    """Display detailed recommendations."""
    console.print("\n[bold]Top Recommendations:[/bold]\n")

    for i, rec in enumerate(recommendations, 1):
        opp = rec["opportunity"]
        scores = rec["scores"]
        efficiency = rec.get("efficiency")

        income_str = f"${opp.get('income_high', 0):,}" if opp.get('income_high') else "Unknown"
        eff_str = f"${efficiency:.0f}/hr" if efficiency else "—"

        console.print(Panel(
            f"[bold cyan]{opp['title']}[/bold cyan]\n"
            f"[dim]{opp.get('company', 'Unknown')}[/dim]\n\n"
            f"Type: {opp['opportunity_type']}\n"
            f"Income: {income_str}\n"
            f"Efficiency: {eff_str}\n\n"
            f"[dim]Scores - Overall: {scores['overall']:.2f} | Income: {scores['income']:.2f} | "
            f"Effort: {scores['effort']:.2f} | Fit: {scores['fit']:.2f}[/dim]\n\n"
            f"[link={opp['url']}]{opp['url']}[/link]",
            title=f"#{i}",
            border_style="green" if i == 1 else "blue"
        ))


@app.command()
def profile(
    output: str = typer.Option("profile.json", "--output", "-o", help="Output file path"),
    income: int = typer.Option(100000, "--income", "-i", help="Minimum income target"),
    hours: int = typer.Option(20, "--hours", "-h", help="Max hours per week"),
    remote: bool = typer.Option(True, "--remote/--no-remote", help="Remote only"),
    skills: str = typer.Option("", "--skills", "-s", help="Comma-separated skills"),
):
    """Create a profile configuration file."""

    skill_list = [s.strip() for s in skills.split(",") if s.strip()]

    profile = UserProfile(
        min_income=income,
        max_hours_weekly=hours,
        remote_only=remote,
        skills=skill_list or ["python", "software engineering"],
        industries=["technology"],
        opportunity_types=[
            OpportunityType.JOB,
            OpportunityType.FREELANCE,
            OpportunityType.CONTRACT,
        ]
    )

    with open(output, "w") as f:
        json.dump(profile.model_dump(), f, indent=2, default=str)

    console.print(f"[green]Profile saved to {output}[/green]")
    console.print(Panel(json.dumps(profile.model_dump(), indent=2, default=str), title="Profile"))


@app.command()
def stats():
    """Show system statistics."""
    memory = OpportunityMemory()
    stats = memory.get_stats()

    console.print(Panel(
        f"Opportunities stored: {stats['opportunities_stored']}\n"
        f"Interactions recorded: {stats['interactions_recorded']}\n"
        f"Preferences learned: {stats['preferences_learned']}",
        title="Memory Stats",
        border_style="blue"
    ))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the API server."""
    import uvicorn

    console.print(f"[bold green]Starting Job Engine API on {host}:{port}[/bold green]")
    uvicorn.run(
        "src.api.routes:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    app()
