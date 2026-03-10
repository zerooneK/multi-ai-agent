"""
tools/spec_validator.py
-----------------------
Phase 2 — Static validation of ModuleSpec dependency graph.

Runs BEFORE any LLM build call, so we catch plan errors cheaply.

Checks
------
1. Missing imports     — module imports a path not defined in the plan
2. Circular deps       — dependency graph has a cycle (would cause import errors)
3. Empty exports       — module has no exports (nothing to import from it)
4. Orphan imports      — a path is imported but never defined (same as #1, explicit)
5. Layer violations    — backend module importing a frontend module or vice versa

All checks return human-readable error strings.
An empty list means the spec is valid.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.project_plan import ModuleSpec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_module_specs(modules: list["ModuleSpec"]) -> list[str]:
    """
    Validate a list of ModuleSpec objects for logical consistency.

    Returns a list of error strings.  Empty list = valid.
    """
    if not modules:
        return []

    errors: list[str] = []
    defined = {m.path for m in modules}

    for m in modules:
        # ── Check 1: every import must exist in the plan ──────────────
        for imp in m.imports:
            if imp not in defined:
                errors.append(
                    f"[MISSING_IMPORT] {m.path} imports '{imp}' "
                    f"which is not defined in modules list"
                )

        # ── Check 2: exports must not be empty ────────────────────────
        # Exception: __init__.py files are allowed to have no exports
        if not m.exports and not m.path.endswith("__init__.py"):
            errors.append(
                f"[EMPTY_EXPORTS] {m.path} has no exports defined — "
                f"other modules cannot import from it"
            )

        # ── Check 3: layer violations ─────────────────────────────────
        for imp in m.imports:
            # Find the imported module's layer
            imp_module = next((x for x in modules if x.path == imp), None)
            if imp_module and imp_module.layer != m.layer:
                errors.append(
                    f"[LAYER_VIOLATION] {m.path} (layer={m.layer}) "
                    f"imports {imp} (layer={imp_module.layer}) — "
                    f"cross-layer imports are not allowed"
                )

    # ── Check 4: circular dependency detection (DFS) ──────────────────
    cycle_errors = _find_cycles(modules)
    errors.extend(cycle_errors)

    return errors


def topological_sort(modules: list["ModuleSpec"]) -> list["ModuleSpec"]:
    """
    Return modules in safe build order (dependencies before dependents).

    Uses Kahn's algorithm (BFS-based topological sort).
    If a cycle exists, raises ValueError — call validate_module_specs first.

    Parameters
    ----------
    modules : list of ModuleSpec (may be backend-only or frontend-only)

    Returns
    -------
    list[ModuleSpec] in dependency order (build this order)
    """
    if not modules:
        return []

    path_to_module = {m.path: m for m in modules}
    # Only consider imports that are within this modules list
    defined_paths  = set(path_to_module.keys())

    # Build in-degree map and adjacency list
    in_degree: dict[str, int]        = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)  # path → list of paths that depend on it

    for m in modules:
        if m.path not in in_degree:
            in_degree[m.path] = 0
        for imp in m.imports:
            if imp not in defined_paths:
                continue  # external import — ignore for sort
            in_degree[m.path] += 1
            dependents[imp].append(m.path)

    # Start with modules that have no dependencies
    queue:  deque[str]      = deque(p for p in in_degree if in_degree[p] == 0)
    result: list["ModuleSpec"] = []

    while queue:
        path = queue.popleft()
        result.append(path_to_module[path])
        for dep_path in dependents[path]:
            in_degree[dep_path] -= 1
            if in_degree[dep_path] == 0:
                queue.append(dep_path)

    if len(result) != len(modules):
        remaining = [p for p in in_degree if in_degree[p] > 0]
        raise ValueError(
            f"Circular dependency detected among: {remaining}. "
            f"Run validate_module_specs() first."
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_cycles(modules: list["ModuleSpec"]) -> list[str]:
    """DFS-based cycle detection. Returns error strings for each cycle found."""
    defined_paths = {m.path for m in modules}
    path_to_module = {m.path: m for m in modules}
    errors: list[str] = []

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {m.path: WHITE for m in modules}
    parent: dict[str, str | None] = {m.path: None for m in modules}

    def dfs(path: str) -> bool:
        """Returns True if a cycle is detected."""
        color[path] = GRAY
        module = path_to_module[path]
        for imp in module.imports:
            if imp not in defined_paths:
                continue
            if color[imp] == GRAY:
                # Cycle found — reconstruct path
                cycle = [imp, path]
                cur = path
                while parent[cur] and parent[cur] != imp:
                    cur = parent[cur]  # type: ignore[assignment]
                    cycle.append(cur)
                cycle.append(imp)
                errors.append(
                    f"[CIRCULAR_DEP] Cycle detected: "
                    + " → ".join(reversed(cycle))
                )
                return True
            if color[imp] == WHITE:
                parent[imp] = path
                if dfs(imp):
                    return True
        color[path] = BLACK
        return False

    for module in modules:
        if color[module.path] == WHITE:
            dfs(module.path)

    return errors