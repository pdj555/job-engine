"""Installable hunt, OAuth connection, scheduler and application tracker commands."""

import asyncio
import json
import time
from pathlib import Path

import typer

from jobhunter.auth import authorize
from jobhunter.hunt import DryEngine, hunt
from jobhunter.profile import Profile, state_home
from jobhunter.registry import Registry
from jobhunter.sheets import Sheets

cli = typer.Typer(help="Jobhunter: verified roles, ranked by dollars per hour.", no_args_is_help=True)


def sheet_connection(home, profile, *, dry_run=False):
    if not profile.sheet_id:
        raise ValueError("Missing sheet_id. Run jobhunter connect first.")
    return Sheets(authorize(home), profile.sheet_id, tab="Jobhunter Dry Run" if dry_run else "Jobhunter")


def run_schedule(run, every_minutes=0, sleep=time.sleep):
    try:
        while True:
            run()
            if not every_minutes:
                return
            sleep(every_minutes * 60)
    except KeyboardInterrupt:
        return


@cli.command()
def connect(client_secrets: Path = typer.Option(..., exists=True, dir_okay=False),
            sheet_id: str = typer.Option("", help="Existing sheet. Opts into the spreadsheets OAuth scope."),
            home: Path = typer.Option(None)):
    """Connect your Google account and create a sheet, or select your own sheet."""
    home = home or state_home()
    try:
        profile = Profile.load(home)
        session = authorize(home, client_secrets=client_secrets, existing_sheet=bool(sheet_id))
        profile.sheet_id = sheet_id or Sheets.create(session)
        Sheets(session, profile.sheet_id).prepare()
        profile.save(home)
        typer.echo(f"https://docs.google.com/spreadsheets/d/{profile.sheet_id}")
    except Exception as exc:
        typer.echo(f"Connection failed: {type(exc).__name__}. Check your OAuth setup and profile.", err=True)
        raise typer.Exit(1) from exc


@cli.command("hunt")
def hunt_command(home: Path = typer.Option(None),
                 every_minutes: int = typer.Option(0, min=0, help="Repeat after this many minutes; 0 runs once."),
                 dry_run: bool = typer.Option(False, help="Synthetic listings; separate tab and seen registry.")):
    """Run once or on an interval. Stdout contains only new roles that clear all gates."""
    if dry_run and every_minutes:
        raise typer.BadParameter("A dry run cannot be scheduled.")
    home = home or state_home()
    try:
        registry = Registry(home / "dry-run" if dry_run else home)

        def once():
            profile = Profile.load(home)
            sheets = sheet_connection(home, profile, dry_run=dry_run)
            if dry_run:
                profile = profile.model_copy(update={"employer_domains": ["careers.example.com"]})
            matches = asyncio.run(hunt(profile, registry, sheets, DryEngine() if dry_run else None,
                                       dry_run=dry_run))
            if matches:
                typer.echo(json.dumps(matches, indent=2))

        run_schedule(once, every_minutes)
    except Exception as exc:
        # No credentials, request headers, or provider responses in diagnostics.
        typer.echo(f"Hunt failed: {type(exc).__name__}. Check profile, Google connection and sheet access.", err=True)
        raise typer.Exit(1) from exc


@cli.command()
def track(key: str, state: str, notes: str = typer.Option(None), home: Path = typer.Option(None)):
    """Update a posting's pipeline state. Does not submit applications or send messages."""
    home = home or state_home()
    try:
        sheet_connection(home, Profile.load(home)).track(key, state, notes)
    except Exception as exc:
        typer.echo(f"Tracker update failed: {type(exc).__name__}. Check the key, pipeline state and sheet access.", err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    cli()
