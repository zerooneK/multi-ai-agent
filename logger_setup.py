"""
logger_setup.py
---------------
Centralised logging configuration for the Multi-Agent Team framework.

Each pipeline run gets its own log file:
  logs/
  └── 2025-01-15_14-30-22_todo_list_app/
      ├── pipeline.log      ← all levels, structured format
      └── errors.log        ← ERROR and above only

Usage (call once at startup, before any imports that use logging):
  from logger_setup import setup_logging
  run_log_dir = setup_logging(project_name="todo_list_app")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    project_name: str = "unknown",
    log_base_dir: str = "logs",
    log_level: str    = "INFO",
) -> Path | None:
    """
    Configure root logger with:
      - StreamHandler  → stdout  (INFO+, human-friendly)
      - FileHandler    → pipeline.log (all levels, full format)
      - FileHandler    → errors.log   (ERROR+ only)

    Parameters
    ----------
    project_name : used as part of the log directory name
    log_base_dir : parent directory for all run logs ("" disables file logging)
    log_level    : root log level string (DEBUG/INFO/WARNING/ERROR)

    Returns
    -------
    Path to the run-specific log directory, or None if file logging disabled.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # ── Formatters ──────────────────────────────────────────────────────────
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Root logger ─────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # ── Console handler ─────────────────────────────────────────────────────
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(level)
    console_h.setFormatter(console_fmt)
    root.addHandler(console_h)

    # Silence noisy HTTP libraries regardless of log level
    for lib in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # ── File handlers (optional) ─────────────────────────────────────────
    if not log_base_dir:
        return None

    # Build run directory: logs/2025-01-15_14-30-22_todo_list_app/
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name   = _safe(project_name)
    run_dir     = Path(log_base_dir) / f"{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # pipeline.log — everything
    pipeline_h = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
    pipeline_h.setLevel(level)
    pipeline_h.setFormatter(file_fmt)
    root.addHandler(pipeline_h)

    # errors.log — ERROR and above only
    error_h = logging.FileHandler(run_dir / "errors.log", encoding="utf-8")
    error_h.setLevel(logging.ERROR)
    error_h.setFormatter(file_fmt)
    root.addHandler(error_h)

    logging.getLogger("logger_setup").info(
        "Logging to: %s", run_dir
    )
    return run_dir


def _safe(name: str) -> str:
    """Convert a project name to a filesystem-safe string."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]