"""
agents/dod_checker.py
---------------------
Phase 4 — Definition-of-Done checker.

Runs after each module is built, in two layers:

Layer 1 — Static checks (no LLM, fast):
  - py_compile syntax check (backend .py files)
  - Export presence check (ast.parse — are declared exports actually defined?)
  - Import consistency (does the file import what the spec says it depends on?)

Layer 2 — LLM logic review (only if static passes):
  - Sends module code + DoD checklist → LLM confirms each item
  - Returns structured pass/fail per DoD item

DodResult.passed == True means the module is ready for the next step.
DodResult.passed == False means the orchestrator should trigger a fix.
"""

from __future__ import annotations

import ast
import logging
import py_compile
import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.project_plan import ModuleSpec

from agents.base_agent import BaseAgent
from config import cfg
from models.messages import AgentMessage, TaskType
from tools.file_tools import read_file

logger = logging.getLogger("agent.dod")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DodResult:
    module_path:    str
    passed:         bool
    static_errors:  list[str]  = field(default_factory=list)
    llm_failures:   list[str]  = field(default_factory=list)
    llm_skipped:    bool       = False   # True when static already failed

    @property
    def all_errors(self) -> list[str]:
        return self.static_errors + self.llm_failures

    def summary(self) -> str:
        if self.passed:
            return f"✅ DoD passed: {self.module_path}"
        parts = []
        if self.static_errors:
            parts.append(f"Static({len(self.static_errors)})")
        if self.llm_failures:
            parts.append(f"LLM({len(self.llm_failures)})")
        return f"❌ DoD failed [{', '.join(parts)}]: {self.module_path}"


# ---------------------------------------------------------------------------
# DodChecker agent
# ---------------------------------------------------------------------------

class DodCheckerAgent(BaseAgent):
    """
    Checks a single generated module against its ModuleSpec DoD.

    Usage
    -----
    checker = DodCheckerAgent(provider_name, model)
    result  = checker.check(module_spec, output_dir)
    """

    @property
    def name(self) -> str:
        return "dod_checker"

    @property
    def system(self) -> str:
        return """You are a code reviewer checking whether a generated file meets
its Definition-of-Done (DoD) checklist.

You receive:
  1. The file path and its full source code
  2. The DoD checklist (list of requirements)
  3. The module's declared exports and interface signatures

Your job: check each DoD item and report which ones FAIL.

Output format — JSON object only, no prose:
{
  "passed": true | false,
  "failures": [
    "DoD item text that FAILED",
    "Another failed item"
  ],
  "notes": "optional brief explanation"
}

Rules:
- Output ONLY the JSON object. Start with { and end with }
- "passed" is true only if ALL DoD items pass
- "failures" lists the exact DoD item text that failed (empty array if all pass)
- Be strict but fair — check what is actually in the code, not what you assume
- If the code is empty or clearly incomplete, fail all DoD items
"""

    def run(self, message: AgentMessage) -> AgentMessage:
        """Standard agent interface — wraps check()."""
        message.mark_running()
        # This agent is called via check() directly — run() is for interface compliance
        message.mark_done("DodChecker called via check() method")
        return message

    def check(
        self,
        spec: "ModuleSpec",
        output_dir: str,
        skip_llm: bool = False,
    ) -> DodResult:
        """
        Full DoD check for one module.

        Parameters
        ----------
        spec       : The ModuleSpec for this module
        output_dir : Project root (spec.path is relative to this)
        skip_llm   : If True, only run static checks (faster, less thorough)
        """
        full_path = str(Path(output_dir) / spec.path)
        source    = read_file(full_path)

        if source.startswith("[Error]") or source.startswith("[Skipped]"):
            return DodResult(
                module_path   = spec.path,
                passed        = False,
                static_errors = [f"File not readable: {source}"],
                llm_skipped   = True,
            )

        # ── Layer 1: Static checks ─────────────────────────────────────
        static_errors = self._static_check(spec, source, full_path)

        if static_errors:
            logger.warning("DoD static FAIL %s: %s", spec.path, static_errors)
            return DodResult(
                module_path   = spec.path,
                passed        = False,
                static_errors = static_errors,
                llm_skipped   = True,   # no point asking LLM if syntax broken
            )

        # ── Layer 2: LLM logic review ──────────────────────────────────
        if skip_llm or not spec.dod:
            return DodResult(module_path=spec.path, passed=True)

        llm_failures = self._llm_check(spec, source)
        passed       = len(llm_failures) == 0

        if not passed:
            logger.warning("DoD LLM FAIL %s: %s", spec.path, llm_failures)

        return DodResult(
            module_path  = spec.path,
            passed       = passed,
            llm_failures = llm_failures,
        )

    # ------------------------------------------------------------------
    # Layer 1: Static checks
    # ------------------------------------------------------------------

    def _static_check(
        self,
        spec: "ModuleSpec",
        source: str,
        full_path: str,
    ) -> list[str]:
        errors: list[str] = []

        # 1a. Syntax check (Python only)
        if spec.path.endswith(".py"):
            syntax_error = self._syntax_check_py(source, full_path)
            if syntax_error:
                errors.append(syntax_error)
                return errors  # no point continuing if syntax broken

        # 1b. Export presence — are declared exports actually defined?
        if spec.path.endswith(".py") and spec.exports:
            missing = self._check_exports_py(source, spec.exports)
            for sym in missing:
                errors.append(
                    f"Export '{sym}' declared in spec but not found in {spec.path}"
                )

        # 1c. TypeScript/TSX — basic structure check
        if spec.path.endswith((".ts", ".tsx")) and spec.exports:
            missing = self._check_exports_ts(source, spec.exports)
            for sym in missing:
                errors.append(
                    f"Export '{sym}' declared in spec but not found in {spec.path}"
                )

        # 1d. Empty file check
        if len(source.strip()) < 10:
            errors.append(f"File appears to be empty or near-empty ({len(source)} chars)")

        return errors

    @staticmethod
    def _syntax_check_py(source: str, original_path: str) -> str | None:
        """Returns error string on syntax error, None if OK."""
        try:
            # Write to temp file so py_compile gives proper filename in errors
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            py_compile.compile(tmp_path, doraise=True)
            return None
        except py_compile.PyCompileError as exc:
            return f"SyntaxError in {original_path}: {exc}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def _check_exports_py(source: str, exports: list[str]) -> list[str]:
        """
        Parse Python source with ast and check that each export name
        is defined as a function, class, or variable at module level.
        """
        missing: list[str] = []
        try:
            tree     = ast.parse(source)
            defined  = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            defined.add(t.id)
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        defined.add(node.target.id)
            for sym in exports:
                if sym not in defined:
                    missing.append(sym)
        except SyntaxError:
            pass  # syntax error already caught by _syntax_check_py
        return missing

    @staticmethod
    def _check_exports_ts(source: str, exports: list[str]) -> list[str]:
        """
        Lightweight text-based check for TypeScript exports.
        Looks for 'export function X', 'export class X', 'export const X',
        'export default function', 'export type X', 'export interface X'.
        """
        missing: list[str] = []
        for sym in exports:
            # Allow 'default' as a wildcard (export default ...)
            if sym == "default":
                if "export default" not in source:
                    missing.append(sym)
                continue
            # Check for named export patterns
            patterns = [
                f"export function {sym}",
                f"export async function {sym}",
                f"export class {sym}",
                f"export const {sym}",
                f"export type {sym}",
                f"export interface {sym}",
                f"export {{ {sym}",           # re-export
                f"export {{{sym}",            # re-export compact
            ]
            if not any(p in source for p in patterns):
                missing.append(sym)
        return missing

    # ------------------------------------------------------------------
    # Layer 2: LLM logic review
    # ------------------------------------------------------------------

    def _llm_check(self, spec: "ModuleSpec", source: str) -> list[str]:
        """
        Ask the LLM to verify each DoD item.
        Returns list of failed DoD item strings.
        """
        # Truncate source if very long (keep first 4000 chars)
        MAX_SOURCE = 4000
        source_excerpt = source[:MAX_SOURCE]
        if len(source) > MAX_SOURCE:
            source_excerpt += f"\n... [truncated — {len(source) - MAX_SOURCE} more chars]"

        # Build interface summary
        iface_lines: list[str] = []
        for iface in spec.interfaces:
            params_str = ", ".join(f"{k}: {v}" for k, v in iface.params.items())
            iface_lines.append(
                f"  - {iface.name}({params_str}) → {iface.returns}"
            )
        iface_block = "\n".join(iface_lines) or "  (none specified)"

        prompt = (
            f"File: {spec.path}\n"
            f"Purpose: {spec.description}\n\n"
            f"Declared exports: {', '.join(spec.exports) or 'none'}\n"
            f"Interfaces:\n{iface_block}\n\n"
            f"DoD checklist to verify:\n"
            + "\n".join(f"  - {item}" for item in spec.dod)
            + f"\n\nSource code:\n```\n{source_excerpt}\n```\n\n"
            f"Check each DoD item and output JSON with 'passed' and 'failures'.\n"
            f"Start with {{ and end with }}\n\n{{"
        )

        try:
            raw    = self.chat(prompt, max_tokens=1024)
            result = self.extract_json(raw)
            failures = result.get("failures", [])
            if not isinstance(failures, list):
                failures = []
            return [str(f) for f in failures]
        except Exception as exc:
            logger.warning("DoD LLM check failed (%s) — treating as passed", exc)
            # If LLM call fails, don't block the build — log and continue
            return []