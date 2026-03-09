"""
tools/shell_tools.py
--------------------
Shell execution tools used by the QA Agent to run syntax checks,
linters, and test suites on the generated code.

Supports both:
  - Python backend  : py_compile syntax check
  - Next.js frontend: npm install + tsc --noEmit TypeScript check
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: list[str],
    cwd: str | None = None,
    timeout: int = 60,
) -> dict:
    """
    Run a shell command and return a result dict.
    Returns {"success", "returncode", "stdout", "stderr", "cmd"}
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success":    proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout":     proc.stdout.strip(),
            "stderr":     proc.stderr.strip(),
            "cmd":        " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False, "returncode": -1,
            "stdout": "", "stderr": f"Command timed out after {timeout}s",
            "cmd": " ".join(cmd),
        }
    except FileNotFoundError as exc:
        return {
            "success": False, "returncode": -1,
            "stdout": "", "stderr": str(exc),
            "cmd": " ".join(cmd),
        }


# ---------------------------------------------------------------------------
# Python checks
# ---------------------------------------------------------------------------

def syntax_check_python(file_path: str) -> dict:
    """Check Python syntax using py_compile."""
    result = run_command([sys.executable, "-m", "py_compile", file_path])
    return {
        "file":    file_path,
        "success": result["success"],
        "errors":  result["stderr"] or result["stdout"] or "No errors",
    }


def syntax_check_all_python(directory: str) -> list[dict]:
    """Run syntax check on every .py file under directory."""
    root = Path(directory)
    if not root.exists():
        return []
    return [syntax_check_python(str(p)) for p in sorted(root.rglob("*.py"))]


def run_pytest(directory: str, timeout: int = 120) -> dict:
    """Run pytest in the given directory."""
    return run_command(
        [sys.executable, "-m", "pytest", "--tb=short", "-q"],
        cwd=directory,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Node / Next.js checks
# ---------------------------------------------------------------------------

def _npm_cmd() -> str:
    return "npm.cmd" if shutil.which("npm.cmd") else "npm"


def _npx_cmd() -> str:
    return "npx.cmd" if shutil.which("npx.cmd") else "npx"


def node_available() -> bool:
    return shutil.which("node") is not None


def npm_available() -> bool:
    return shutil.which("npm") is not None or shutil.which("npm.cmd") is not None


def run_npm_install(directory: str, timeout: int = 180) -> dict:
    """
    Run `npm install` in the given directory.
    Must be run before tsc so all @types packages are present.
    """
    if not npm_available():
        return {
            "success": False, "returncode": -1,
            "stdout": "", "stderr": "npm not found — Node.js is not installed",
            "cmd": "npm install",
        }
    return run_command(
        [_npm_cmd(), "install", "--prefer-offline", "--no-audit", "--no-fund"],
        cwd=directory,
        timeout=timeout,
    )


def run_tsc(directory: str, timeout: int = 120) -> dict:
    """
    Run TypeScript type-check via `npx tsc --noEmit`.

    Returns
    -------
    {
        "success"    : bool,
        "error_count": int,
        "errors"     : [{"file", "line", "code", "message"}, ...],
        "raw_output" : str,
        "cmd"        : str,
    }
    """
    if not npm_available():
        return {
            "success": False, "error_count": 0,
            "errors": [{"file": "", "line": 0, "code": "",
                        "message": "Node.js not installed — skipping TypeScript check"}],
            "raw_output": "Node.js not installed",
            "cmd": "npx tsc --noEmit",
        }

    result = run_command(
        [_npx_cmd(), "tsc", "--noEmit", "--pretty", "false"],
        cwd=directory,
        timeout=timeout,
    )
    raw    = (result["stdout"] + "\n" + result["stderr"]).strip()
    errors = _parse_tsc_output(raw)
    return {
        "success":     result["success"],
        "error_count": len(errors),
        "errors":      errors,
        "raw_output":  raw[:4000],
        "cmd":         result["cmd"],
    }


def run_next_build(directory: str, timeout: int = 300) -> dict:
    """
    Run `npm run build` (next build) — most thorough check.
    Catches all compile + lint errors. Takes 30-120s.
    """
    if not npm_available():
        return {
            "success": False, "returncode": -1,
            "stdout": "", "stderr": "npm not found",
            "cmd": "npm run build",
        }
    return run_command([_npm_cmd(), "run", "build"], cwd=directory, timeout=timeout)


def _parse_tsc_output(raw: str) -> list[dict]:
    """Parse `tsc --noEmit` output into structured error dicts."""
    pattern = re.compile(r"^(.+?)\((\d+),\d+\):\s+error\s+(TS\d+):\s+(.+)$")
    errors = []
    for line in raw.splitlines():
        m = pattern.match(line.strip())
        if m:
            errors.append({
                "file":    m.group(1).strip(),
                "line":    int(m.group(2)),
                "code":    m.group(3),
                "message": m.group(4).strip(),
            })
    return errors


def format_tsc_result(tsc_result: dict) -> str:
    """Format tsc result into a readable string for the QA prompt."""
    if tsc_result.get("success"):
        return "TypeScript check: PASS — no type errors"

    errors = tsc_result.get("errors", [])
    if any("not installed" in e.get("message", "") for e in errors):
        return "TypeScript check: SKIPPED — Node.js not installed"

    lines = [f"TypeScript check: FAIL — {len(errors)} error(s)"]
    lines.append("-" * 50)
    for e in errors[:20]:
        lines.append(f"  {e.get('file','')} line {e.get('line','')} [{e.get('code','')}]")
        lines.append(f"      {e.get('message','')}")
    if len(errors) > 20:
        lines.append(f"  ... and {len(errors) - 20} more errors")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared formatter
# ---------------------------------------------------------------------------

def format_check_result(results: list[dict]) -> str:
    """Format Python syntax check results into a readable string."""
    if not results:
        return "Syntax Check: no Python files found"
    passed = sum(1 for r in results if r.get("success"))
    failed = len(results) - passed
    lines  = [f"Python Syntax Check: {passed} passed, {failed} failed", "-" * 50]
    for r in results:
        icon = "OK" if r.get("success") else "FAIL"
        lines.append(f"  [{icon}]  {r['file']}")
        if not r.get("success"):
            lines.append(f"         {r.get('errors', '')}")
    return "\n".join(lines)
