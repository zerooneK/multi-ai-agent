# =============================================================================
# tools/__init__.py
# -----------------------------------------------------------------------------
# Re-exports all tool functions used by agents to interact with the filesystem
# and run shell commands.
#
# Two source files:
#   tools/file_tools.py   — read/write files and directories
#   tools/shell_tools.py  — run shell commands (Python checks, Node/tsc checks)
#
# Import anywhere with:
#   from tools import create_file, run_tsc, syntax_check_all_python, ...
# =============================================================================

# ── tools/file_tools.py ───────────────────────────────────────────────────────
from tools.file_tools import (
    create_file,       # write content to a file (auto-creates parent dirs)
    patch_file,        # apply a targeted patch to an existing file
    read_file,         # read and return file content as string
    list_files,        # recursively list all files under a directory
    create_directory,  # create a directory and all missing parents
    file_exists,       # check if a file exists → bool
)

# ── tools/shell_tools.py — core ───────────────────────────────────────────────
from tools.shell_tools import (
    run_command,       # generic subprocess runner → {success, returncode, stdout, stderr}
)

# ── tools/shell_tools.py — Python backend checks ─────────────────────────────
from tools.shell_tools import (
    syntax_check_python,      # py_compile one .py file → {file, success, errors}
    syntax_check_all_python,  # py_compile every .py under a directory → list[dict]
    run_pytest,               # run pytest in a directory → result dict
)

# ── tools/shell_tools.py — Node.js / Next.js frontend checks ─────────────────
from tools.shell_tools import (
    node_available,    # True if node is installed on this machine
    npm_available,     # True if npm is installed on this machine
    run_npm_install,   # npm install (must run before tsc)
    run_tsc,           # npx tsc --noEmit → {success, error_count, errors, raw_output}
    run_next_build,    # npm run build (most thorough, slowest ~30-120s)
)

# ── tools/shell_tools.py — formatters ────────────────────────────────────────
from tools.shell_tools import (
    format_check_result,  # format Python syntax results → readable string for QA prompt
    format_tsc_result,    # format tsc results → readable string for QA prompt
)

__all__ = [
    # file_tools
    "create_file", "patch_file", "read_file", "list_files", "create_directory", "file_exists",
    # shell — core
    "run_command",
    # shell — Python
    "syntax_check_python", "syntax_check_all_python", "run_pytest",
    # shell — Node / Next.js
    "node_available", "npm_available",
    "run_npm_install", "run_tsc", "run_next_build",
    # shell — formatters
    "format_check_result", "format_tsc_result",
]