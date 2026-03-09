"""
tools/file_tools.py
-------------------
Filesystem tools used by all agents to write generated code to disk.
"""

from __future__ import annotations

import os
from pathlib import Path


def create_file(path: str, content: str) -> str:
    """Create a file and write content. Parent dirs are auto-created."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    lines = content.count("\n") + 1
    return f"[OK] Created: {target}  ({lines} lines)"


def read_file(path: str) -> str:
    """Read and return the content of a file."""
    target = Path(path)
    if not target.exists():
        return f"[Error] File not found: {path}"
    return target.read_text(encoding="utf-8")


def list_files(directory: str) -> list[str]:
    """Return a sorted list of all files under a directory (recursive)."""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
    )


def create_directory(path: str) -> str:
    """Create a directory and all missing parents."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return f"[OK] Directory ready: {path}"


def file_exists(path: str) -> bool:
    return Path(path).exists()
