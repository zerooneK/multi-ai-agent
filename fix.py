#!/usr/bin/env python3
"""
fix.py — Interactive Runtime Bug Fixer
=======================================

Usage
-----
    python fix.py                          # prompts for project directory
    python fix.py output/my_todo_app       # skip directory prompt

Workflow
--------
1. You specify which generated project has the bug
2. You paste the runtime error (uvicorn traceback, browser console error, etc.)
3. FixerAgent diagnoses the error, reads relevant files, and writes fixes
4. Repeat for up to 3 rounds if more errors appear

This tool is meant to be run AFTER the main pipeline has finished.
It does NOT re-run QA — it trusts your runtime error as ground truth.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── Logging setup (minimal — fixer is interactive) ───────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fix")

# ── Try rich for prettier output ──────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.rule    import Rule
    _console = Console()
    _RICH    = True
except ImportError:
    _RICH    = False

def _print(msg: str, style: str = "") -> None:
    if _RICH:
        _console.print(msg, style=style or None)
    else:
        print(msg)

def _rule(title: str = "") -> None:
    if _RICH:
        _console.print(Rule(title, style="dim"))
    else:
        print(f"\n{'─' * 60} {title}")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from agents.fixer_agent import FixerAgent, MAX_FIX_ROUNDS
    from models.messages    import AgentMessage, TaskType
    import config as cfg_module
    cfg = cfg_module.cfg
except ImportError as e:
    print(f"[error] Cannot import framework modules: {e}")
    print("Make sure you run fix.py from the multi_agent_team/ directory.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner() -> None:
    if _RICH:
        _console.print(Panel.fit(
            "[bold cyan]🔧 Runtime Bug Fixer[/bold cyan]\n"
            "[dim]Paste your error → AI diagnoses → files fixed[/dim]",
            border_style="cyan",
        ))
    else:
        print("=" * 60)
        print("  Runtime Bug Fixer")
        print("  Paste your error → AI diagnoses → files fixed")
        print("=" * 60)


def _get_output_dir(argv_dir: str | None) -> str:
    if argv_dir and Path(argv_dir).exists():
        return str(Path(argv_dir).resolve())

    # Try to find the most recently modified project directory
    output_base = Path(getattr(cfg, "OUTPUT_DIR", "output"))
    candidates  = sorted(
        [d for d in output_base.glob("*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        _print("\nRecent generated projects:", "dim")
        for i, c in enumerate(candidates[:5], 1):
            _print(f"  {i}. {c}", "dim")
        _print("")

    _print("Project directory to fix (leave blank for most recent): ", "bold")
    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        sys.exit(0)

    if not raw and candidates:
        chosen = str(candidates[0].resolve())
        _print(f"Using: {chosen}", "dim")
        return chosen

    if raw and Path(raw).exists():
        return str(Path(raw).resolve())

    _print(f"[red]Directory not found: {raw}[/red]" if _RICH else f"Directory not found: {raw}")
    sys.exit(1)


def _read_multiline_error() -> str:
    """Read a multiline error from stdin until the user enters a blank line."""
    _print(
        "\n[bold]Paste the error message[/bold] (blank line to finish):" if _RICH
        else "\nPaste the error message (blank line to finish):"
    )
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line == "" and lines:
                # End on first blank line after some content
                break
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        pass
    return "\n".join(lines).strip()


def _make_fixer() -> FixerAgent:
    """Instantiate FixerAgent using QA provider/model settings (same class of model)."""
    provider = getattr(cfg, "QA_PROVIDER", getattr(cfg, "AGENT_PROVIDER", "ollama"))
    model    = getattr(cfg, "QA_MODEL",    getattr(cfg, "AGENT_MODEL",    "qwen2.5-coder:14b"))

    return FixerAgent(
        provider_name = provider,
        model         = model,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main fix loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner()

    # ── Resolve project directory ─────────────────────────────────────
    argv_dir   = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = _get_output_dir(argv_dir)
    _print(f"\nFixing project at: [bold]{output_dir}[/bold]" if _RICH
           else f"\nFixing project at: {output_dir}")

    fixer       = _make_fixer()
    round_count = 0

    while round_count < MAX_FIX_ROUNDS:
        round_count += 1
        _rule(f"Fix Round {round_count} / {MAX_FIX_ROUNDS}")

        # ── Get error from user ───────────────────────────────────────
        error_message = _read_multiline_error()
        if not error_message:
            _print("No error provided. Exiting.", "yellow")
            break

        # ── Extra files (optional) ────────────────────────────────────
        _print(
            "\n[dim]Any file to provide manually? "
            "(path:content or blank to skip)[/dim]" if _RICH
            else "\nAny file to provide manually? (path:content or blank to skip)"
        )
        extra_files: list[dict] = []
        try:
            extra_raw = input().strip()
            if extra_raw and ":" in extra_raw:
                path, _, content = extra_raw.partition(":")
                extra_files.append({"path": path.strip(), "content": content.strip()})
        except (EOFError, KeyboardInterrupt):
            pass

        # ── Run fixer ─────────────────────────────────────────────────
        _print("\n[cyan]⚙  Diagnosing and fixing…[/cyan]" if _RICH
               else "\nDiagnosing and fixing…")

        message = AgentMessage(
            sender   = "user",
            receiver = "fixer",
            task_type= TaskType.QA,   # reuse QA type — no dedicated type needed
            payload  = {
                "error_message": error_message,
                "output_dir":    output_dir,
                "extra_files":   extra_files,
            },
        )

        result = fixer.run(message)

        if result.status.value == "done":
            _print(
                f"\n[green]✓ Fixed![/green]\n{result.content}" if _RICH
                else f"\n✓ Fixed!\n{result.content}"
            )
        else:
            _print(
                f"\n[red]✗ Fixer failed:[/red] {result.content}" if _RICH
                else f"\n✗ Fixer failed: {result.content}"
            )

        # ── Ask if there's another error ──────────────────────────────
        if round_count < MAX_FIX_ROUNDS:
            _print(
                "\n[dim]Another error to fix? (y/N):[/dim]" if _RICH
                else "\nAnother error to fix? (y/N):"
            )
            try:
                again = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                again = "n"
            if again not in ("y", "yes"):
                break
        else:
            _print(
                f"\n[yellow]Max fix rounds ({MAX_FIX_ROUNDS}) reached.[/yellow]" if _RICH
                else f"\nMax fix rounds ({MAX_FIX_ROUNDS}) reached."
            )

    _rule()
    _print(
        "\n[bold]Done.[/bold] Re-run your server to verify the fixes." if _RICH
        else "\nDone. Re-run your server to verify the fixes."
    )
    _print("  Backend:  uvicorn backend.main:app --reload", "dim")
    _print("  Frontend: cd frontend && npm run dev", "dim")
    print()


if __name__ == "__main__":
    main()