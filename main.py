"""
main.py
-------
CLI entry point for the Multi-Agent Team.

Usage
-----
    # Interactive mode (prompted for requirement)
    python main.py

    # Direct mode (pass requirement as argument)
    python main.py "create a website for rental books"

    # With provider override
    AGENT_PROVIDER=openai python main.py "create a blog platform"
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# ── Configure logging before any local imports ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Silence noisy HTTP libraries
for lib in ("httpx", "httpcore", "openai", "anthropic"):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("main")

from config import cfg
from orchestrator import Orchestrator, PipelineResult


# ---------------------------------------------------------------------------
# Rich-aware display helpers
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None


def _print(msg: str, style: str = "") -> None:
    if _RICH:
        _console.print(msg if not style else f"[{style}]{msg}[/{style}]")
    else:
        print(msg)


def _banner() -> None:
    if _RICH:
        _console.print(Panel(
            "[bold cyan]🤖  Multi-Agent Team Framework[/bold cyan]\n"
            "[dim]Planner → Backend → Frontend → QA[/dim]\n"
            f"[dim]Provider: [bold]{cfg.PROVIDER}[/bold][/dim]",
            border_style="cyan", padding=(0, 2),
        ))
    else:
        print("=" * 60)
        print("  Multi-Agent Team Framework")
        print(f"  Provider: {cfg.PROVIDER}")
        print("=" * 60)


def _print_result(result: PipelineResult) -> None:
    """Pretty-print the pipeline result."""
    if _RICH:
        # Status panel
        status_color = "green" if result.success else "yellow"
        status_text  = "✅ SUCCESS" if result.success else "⚠️  COMPLETED WITH ISSUES"
        _console.print(Panel(
            f"[bold {status_color}]{status_text}[/bold {status_color}]\n"
            f"[dim]Project:[/dim] [bold]{result.project_name}[/bold]\n"
            f"[dim]Output :[/dim] {result.output_dir}\n"
            f"[dim]Files  :[/dim] {len(result.files_created)}\n"
            f"[dim]Time   :[/dim] {result.duration_s}s",
            border_style=status_color, title="Pipeline Result",
        ))

        # Files table
        if result.files_created:
            table = Table(title="Generated Files", box=box.SIMPLE)
            table.add_column("#",    style="dim", width=4)
            table.add_column("File", style="green")
            for i, f in enumerate(result.files_created, 1):
                table.add_row(str(i), f)
            _console.print(table)

        # Issues table
        issues = result.qa_report.get("issues", [])
        if issues:
            t = Table(title="QA Issues", box=box.SIMPLE)
            t.add_column("Severity", width=10)
            t.add_column("File")
            t.add_column("Issue")
            for issue in issues:
                sev   = issue.get("severity", "warning")
                color = "red" if sev == "error" else "yellow"
                t.add_row(
                    f"[{color}]{sev}[/{color}]",
                    issue.get("file", ""),
                    issue.get("description", ""),
                )
            _console.print(t)

    else:
        # Plain text output
        status = "SUCCESS" if result.success else "COMPLETED WITH ISSUES"
        print(f"\n{'='*60}")
        print(f"  {status}")
        print(f"  Project : {result.project_name}")
        print(f"  Output  : {result.output_dir}")
        print(f"  Files   : {len(result.files_created)}")
        print(f"  Time    : {result.duration_s}s")
        print(f"{'='*60}")

        if result.files_created:
            print("\nGenerated Files:")
            for f in result.files_created:
                print(f"  - {f}")

        issues = result.qa_report.get("issues", [])
        if issues:
            print(f"\nQA Issues ({len(issues)}):")
            for issue in issues:
                print(f"  [{issue.get('severity','?').upper()}] "
                      f"{issue.get('file','')} — {issue.get('description','')}")

    print()


def _progress_callback(step: str, detail: str) -> None:
    """Shown in real-time as each pipeline step runs."""
    if _RICH:
        if detail:
            _console.print(f"  [dim]{step}[/dim]  [italic]{detail}[/italic]")
        else:
            _console.print(f"  [bold]{step}[/bold]")
    else:
        if detail:
            print(f"  {step}  {detail}")
        else:
            print(f"  {step}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _banner()

    # Get requirement
    if len(sys.argv) > 1:
        requirement = " ".join(sys.argv[1:]).strip()
    else:
        _print("\nWhat would you like to build?", "dim")
        _print("Examples:", "dim")
        _print("  • create a website for rental books", "dim")
        _print("  • build a task management app with user auth", "dim")
        _print("  • create a blog platform with comments\n", "dim")
        try:
            requirement = input("Requirement: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

    if not requirement:
        _print("No requirement provided. Exiting.", "red")
        return

    _print(f"\n[bold]Requirement:[/bold] {requirement}\n" if _RICH
           else f"\nRequirement: {requirement}\n")

    # Run the pipeline
    orchestrator = Orchestrator(
        output_base_dir = cfg.OUTPUT_DIR,
        progress_cb     = _progress_callback,
    )

    try:
        result = orchestrator.run(requirement)
    except KeyboardInterrupt:
        _print("\nInterrupted.", "yellow")
        return
    except Exception as exc:
        _print(f"\nFatal error: {exc}", "red")
        logger.exception("Pipeline error")
        return

    _print_result(result)

    # Save result summary
    if result.output_dir:
        summary_path = Path(result.output_dir) / "pipeline_summary.json"
        summary_path.write_text(
            json.dumps({
                "success":       result.success,
                "project_name":  result.project_name,
                "output_dir":    result.output_dir,
                "files_created": result.files_created,
                "duration_s":    result.duration_s,
                "qa_report":     result.qa_report,
                "message_log":   result.message_log,
            }, indent=2),
            encoding="utf-8",
        )
        _print(f"Summary saved → {summary_path}", "dim")

    # Next steps hint
    if result.success and result.output_dir:
        _print("\n[bold]Next steps:[/bold]" if _RICH else "\nNext steps:")
        _print(f"  cd {result.output_dir}/backend")
        _print("  pip install -r requirements.txt")
        _print("  uvicorn main:app --reload")
        _print("  # Then open frontend/index.html in your browser")
    print()


if __name__ == "__main__":
    main()
