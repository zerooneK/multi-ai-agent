"""
tools/file_tools.py
-------------------
Filesystem tools used by all agents to write generated code to disk.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

# Directories that are never useful for agents to read
_SKIP_DIRS = {
    "node_modules", ".next", ".git", "__pycache__",
    ".pytest_cache", "dist", "build", ".turbo",
}

# Binary file extensions that cannot be decoded as UTF-8 text
_BINARY_EXTENSIONS = {
    ".node", ".wasm", ".bin", ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".jpg", ".jpeg", ".png", ".gif", ".ico",
    ".pdf", ".zip", ".tar", ".gz",
}

# Files that are auto-generated and not useful for QA review
_SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "tsconfig.tsbuildinfo"}


def create_file(path: str, content: str) -> str:
    """
    Create a file and write content. Parent dirs are auto-created.

    Normalises forward-slash paths to the OS separator so that
    paths like 'frontend/app/(dashboard)/todos/[id]/page.tsx'
    are created correctly on both Windows and Unix.
    """
    # Normalise: replace forward slashes with OS separator
    normalised = os.path.normpath(path)
    target = Path(normalised)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    lines = content.count("\n") + 1
    return f"[OK] Created: {target}  ({lines} lines)"


def read_file(path: str) -> str:
    """
    Read and return the content of a file as UTF-8 text.

    Returns an error string (instead of raising) when:
      - File does not exist
      - File is binary (non-UTF-8 bytes)
    """
    normalised = os.path.normpath(path)
    target = Path(normalised)
    if not target.exists():
        return f"[Error] File not found: {path}"
    if target.suffix.lower() in _BINARY_EXTENSIONS:
        return f"[Skipped] Binary file: {path}"
    try:
        return target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return f"[Skipped] Cannot read as UTF-8: {path}"


def list_files(directory: str) -> list[str]:
    """
    Return a sorted list of all readable text files under a directory.

    Skips:
      - node_modules/, .next/, .git/, __pycache__/, dist/, build/
      - Binary file extensions (.node, .wasm, .bin, .pyc, ...)
      - Auto-generated lock files (package-lock.json, tsconfig.tsbuildinfo)
    """
    root = Path(os.path.normpath(directory))
    if not root.exists():
        return []

    results = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts[:-1])
        if parts & _SKIP_DIRS:
            continue
        if p.suffix.lower() in _BINARY_EXTENSIONS:
            continue
        if p.name in _SKIP_FILES:
            continue
        results.append(str(p.relative_to(root)))

    return sorted(results)


def create_directory(path: str) -> str:
    """Create a directory and all missing parents."""
    Path(os.path.normpath(path)).mkdir(parents=True, exist_ok=True)
    return f"[OK] Directory ready: {path}"


def file_exists(path: str) -> bool:
    return Path(os.path.normpath(path)).exists()


def patch_file(path: str, old_code: str, new_code: str) -> dict:
    """
    Apply a targeted patch to an existing file by replacing old_code with new_code.

    Returns a dict:
      {"success": True,  "method": "patch",   "path": path}
      {"success": False, "method": "patch",   "path": path, "error": reason}

    Does NOT fallback to full rewrite — caller decides what to do on failure.
    """
    normalised = os.path.normpath(path)
    target     = Path(normalised)

    if not target.exists():
        return {"success": False, "method": "patch", "path": path,
                "error": f"File not found: {path}"}

    try:
        current = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError) as exc:
        return {"success": False, "method": "patch", "path": path,
                "error": str(exc)}

    if old_code not in current:
        return {"success": False, "method": "patch", "path": path,
                "error": "old_code not found in file"}

    occurrences = current.count(old_code)
    if occurrences > 1:
        return {"success": False, "method": "patch", "path": path,
                "error": f"old_code found {occurrences} times — ambiguous patch"}

    patched = current.replace(old_code, new_code, 1)
    target.write_text(patched, encoding="utf-8")
    return {"success": True, "method": "patch", "path": path}


def actual_files_set(directory: str) -> set[str]:
    """
    Return a set of all existing file paths (relative, forward-slash normalised)
    under the given directory. Used by QA to cross-check 'MISSING' reports.
    """
    root = Path(os.path.normpath(directory))
    if not root.exists():
        return set()
    result = set()
    for p in root.rglob("*"):
        if p.is_file():
            # Use forward slashes for cross-platform consistency
            rel = p.relative_to(root).as_posix()
            result.add(rel)
    return result