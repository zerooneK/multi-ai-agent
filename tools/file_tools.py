"""
tools/file_tools.py
-------------------
Filesystem tools used by all agents to write generated code to disk.
"""

from __future__ import annotations

from pathlib import Path

# Directories that are never useful for agents to read
_SKIP_DIRS = {
    "node_modules", ".next", ".git", "__pycache__",
    ".pytest_cache", "dist", "build", ".turbo",
}

# Binary file extensions that cannot be decoded as UTF-8 text
_BINARY_EXTENSIONS = {
    ".node", ".wasm", ".bin", ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".jpg", ".jpeg", ".png", ".gif", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".lock",
}


def create_file(path: str, content: str) -> str:
    """Create a file and write content. Parent dirs are auto-created."""
    target = Path(path)
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
    target = Path(path)
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
      - Any path component that starts with '.' (hidden dirs)
    """
    root = Path(directory)
    if not root.exists():
        return []

    results = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # Skip if any path part is a blocked directory
        parts = set(p.relative_to(root).parts[:-1])  # dirs only, not filename
        if parts & _SKIP_DIRS:
            continue
        # Skip binary extensions
        if p.suffix.lower() in _BINARY_EXTENSIONS:
            continue
        results.append(str(p.relative_to(root)))

    return sorted(results)


def create_directory(path: str) -> str:
    """Create a directory and all missing parents."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return f"[OK] Directory ready: {path}"


def file_exists(path: str) -> bool:
    return Path(path).exists()