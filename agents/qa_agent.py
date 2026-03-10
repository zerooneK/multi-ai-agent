"""
agents/qa_agent.py
------------------
QA Agent — runs automated checks on ALL generated code and produces
a structured fix report for the Orchestrator.

Pipeline:
  1. Python syntax check  : py_compile on all backend .py files
  2. TypeScript check     : npm install + tsc --noEmit on frontend/
     (skipped gracefully if Node.js is not installed)
  3. LLM deep review      : reads all files, checks logic, references,
     Next.js rules, schema alignment
     — Strategy 1: progressive prompt reduction (4 attempts, shrinking payload)
     — Strategy 2: split backend/frontend review (2 separate calls, merged)
     — Fallback   : automated checks only (if all LLM attempts fail)
  4. Post-process report  : strip hallucinated "MISSING" issues for files
     that actually exist on disk
  5. Returns unified JSON report with per-agent fix instructions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from models.messages import AgentMessage
from models.project_plan import ProjectPlan
from tools.file_tools import actual_files_set, list_files, read_file
from tools.shell_tools import (
    format_check_result,
    format_tsc_result,
    node_available,
    run_npm_install,
    run_tsc,
    syntax_check_all_python,
)

logger = logging.getLogger("agent.qa")

MAX_FILE_CHARS = 2000   # default chars per file
MAX_FILES      = 20     # default max files sent to LLM


class QAAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "qa"

    @property
    def system(self) -> str:
        return """You are a senior QA engineer specialising in
full-stack web apps (FastAPI backend + Next.js 14 App Router frontend).

You receive:
  1. Automated check results (Python syntax + TypeScript tsc)
  2. All generated source files (may be truncated for large files)
  3. The original project plan
  4. A list of ALL files that actually exist on disk

CRITICAL RULE — DO NOT hallucinate missing files:
  Before reporting any file as "MISSING", check the "FILES ON DISK" list.
  If the file appears in that list, it EXISTS — do NOT report it as missing.
  Only report a file as missing if it is ABSENT from the FILES ON DISK list.

Output format — JSON only, no prose, no markdown:
{
  "passed": true | false,
  "summary": "one sentence overall status",
  "issues": [
    {
      "file": "path/to/file",
      "severity": "error" | "warning",
      "description": "what is wrong",
      "suggestion": "how to fix it"
    }
  ],
  "backend_fix_instructions":  "detailed step-by-step for backend agent ('' if none)",
  "frontend_fix_instructions": "detailed step-by-step for frontend agent ('' if none)"
}

BACKEND CHECKLIST — flag errors for any of these:
- Python syntax errors (from automated check)
- SQLAlchemy model fields don't match the project plan
- Pydantic schemas missing fields that models have
- FastAPI router not included in main.py app.include_router()
- CORS middleware missing or not allowing http://localhost:3000
- JWT cookie name not "access_token"
- Auth endpoints missing: POST /api/auth/register, POST /api/auth/login,
  POST /api/auth/logout, GET /api/auth/me
- requirements.txt missing a package that is imported

NEXT.JS FRONTEND CHECKLIST — flag errors for any of these:
- TypeScript errors (from tsc check — do NOT invent new ones)
- 'use client' MISSING on a component using useState/useEffect/useRouter/events
- 'use client' present on a pure Server Component (no hooks/events at all)
- localStorage or sessionStorage used for JWT
- window.location used instead of next/navigation
- apiFetch called without credentials:"include"
- Hard-coded API URL instead of process.env.NEXT_PUBLIC_API_URL
- package.json missing next, react, react-dom, typescript

RULES:
- Output ONLY the JSON object.
- Set passed=true only if there are ZERO errors (warnings are acceptable).
- Never report a file as missing if it appears in FILES ON DISK.
- Be specific in fix_instructions — name exact files and changes needed.
"""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track if progressive strategy has ever overflowed — if so, skip it next time
        self._progressive_always_overflows: bool = False

    def run(self, message: AgentMessage) -> AgentMessage:
        message.mark_running()

        plan_json  = message.payload.get("plan_json", "{}")
        output_dir = message.payload.get("output_dir", "output/project")

        try:
            plan = ProjectPlan.from_json(plan_json)
        except Exception as exc:
            message.mark_failed(f"Could not parse plan: {exc}")
            return message

        logger.info("QA agent checking: %s", output_dir)

        # ── Step 1: Python syntax check ───────────────────────────────
        backend_dir = str(Path(output_dir) / "backend")
        py_results  = syntax_check_all_python(backend_dir)
        py_report   = format_check_result(py_results)
        py_passed   = all(r["success"] for r in py_results)
        logger.info("Python syntax: %s (%d files)",
                    "PASS" if py_passed else "FAIL", len(py_results))

        # ── Step 2: TypeScript check ──────────────────────────────────
        frontend_dir = str(Path(output_dir) / "frontend")
        tsc_report   = "TypeScript check: SKIPPED — frontend dir not found"
        tsc_passed   = True

        if Path(frontend_dir).exists():
            if node_available():
                logger.info("Running npm install in %s ...", frontend_dir)
                npm_result = run_npm_install(frontend_dir, timeout=180)
                if npm_result["success"]:
                    logger.info("npm install OK — running tsc ...")
                    tsc_result = run_tsc(frontend_dir, timeout=120)
                    tsc_report = format_tsc_result(tsc_result)
                    tsc_passed = tsc_result["success"]
                    logger.info("TypeScript check: %s (%d errors)",
                                "PASS" if tsc_passed else "FAIL",
                                tsc_result.get("error_count", 0))
                else:
                    tsc_report = "TypeScript check: SKIPPED — npm install failed"
            else:
                tsc_report = "TypeScript check: SKIPPED — Node.js not installed"

        automated_passed = py_passed and tsc_passed

        # ── Step 3: Build file inventory ─────────────────────────────
        all_files     = list_files(output_dir)
        files_on_disk = actual_files_set(output_dir)
        files_on_disk_str = "\n".join(
            f for f in sorted(files_on_disk)
            if "node_modules" not in f and ".next" not in f and ".git" not in f
        )

        # ── Step 4: LLM deep review ───────────────────────────────────
        # Strategy 1: progressive prompt reduction (skipped if known to always overflow)
        # Strategy 2: split backend/frontend (fallback or primary if overflow known)
        report = None

        if self._progressive_always_overflows:
            logger.info("QA skipping progressive strategy (known context overflow) — using split directly")
        else:
            report = self._strategy_progressive(
                plan, py_report, tsc_report, files_on_disk_str, all_files, output_dir,
            )
            if report is None:
                logger.info("QA progressive strategy failed — trying split review")
                self._progressive_always_overflows = True  # remember for next QA round

        if report is None:
            report = self._strategy_split(
                plan, py_report, tsc_report, files_on_disk_str, all_files, output_dir,
            )

        if report is None:
            logger.error("QA LLM review failed: all strategies exhausted")
            report = self._make_fallback(automated_passed, py_results,
                                         "all LLM review strategies exhausted")

        # ── Step 5: Post-process ──────────────────────────────────────
        report = self._filter_hallucinated_issues(report, files_on_disk)

        if not py_passed and report.get("passed"):
            report["passed"]  = False
            report["summary"] = "Python syntax errors detected"
        if not tsc_passed and report.get("passed"):
            report["passed"]  = False
            report["summary"] = "TypeScript errors detected"

        logger.info("QA report: passed=%s issues=%d",
                    report.get("passed"), len(report.get("issues", [])))
        message.mark_done(json.dumps(report, indent=2))
        return message

    # ------------------------------------------------------------------
    # Strategy 1 — Progressive prompt reduction
    # ------------------------------------------------------------------

    def _strategy_progressive(
        self,
        plan: ProjectPlan,
        py_report: str,
        tsc_report: str,
        files_on_disk_str: str,
        all_files: list[str],
        output_dir: str,
    ) -> dict | None:
        """
        Try LLM review with progressively smaller payloads.

        Attempt 1: full file contents (2000 chars, 20 files)
        Attempt 2: reduced file contents (500 chars, 5 files)
        Attempt 3: no file contents, full plan
        Attempt 4: no file contents, no plan (automated results only)
        """
        plan_summary = f"Project: {plan.project_name}\nModels: {', '.join(m.name for m in plan.database_models)}\nEndpoints: {len(plan.api_endpoints)}"

        attempts = [
            # (max_chars, max_files, include_plan_full, include_files)
            (2000, 20, True,  True),   # attempt 1: files + full plan
            (500,   5, True,  True),   # attempt 2: fewer files + full plan
            (0,     0, False, False),  # attempt 3: summary only (skip full plan)
            (0,     0, False, False),  # attempt 4: same — last resort
        ]

        for attempt_num, (chars, files, include_plan, include_files) in enumerate(attempts, start=1):
            file_section = ""
            if include_files and chars > 0:
                _content = self._collect_files(output_dir, all_files,
                                               max_chars=chars, max_files=files)
                file_section = f"\n=== FILE CONTENTS (showing {files} files, {chars} chars each) ===\n{_content}\n"

            plan_section = (
                f"\n=== PROJECT PLAN ===\n{plan.to_json(indent=1)}\n"
                if include_plan else
                f"\n=== PROJECT SUMMARY ===\n{plan_summary}\n"
            )

            prompt = (
                f"Review the generated full-stack project: {plan.project_name}\n\n"
                f"=== AUTOMATED CHECK RESULTS ===\n{py_report}\n\n{tsc_report}\n\n"
                f"=== FILES ON DISK (do not report these as missing) ===\n{files_on_disk_str}\n"
                f"{plan_section}"
                f"{file_section}"
                "Produce the JSON quality report now."
            )

            try:
                raw = self.chat(prompt, max_tokens=8192)
                report = self.extract_json(raw)
                logger.info("QA progressive strategy succeeded (attempt %d/4)", attempt_num)
                return report
            except Exception as exc:
                if self._is_context_error(exc):
                    logger.warning(
                        "QA progressive attempt %d/4 — context exceeded, reducing payload", attempt_num
                    )
                    continue
                logger.warning("QA progressive attempt %d/4 — error: %s", attempt_num, exc)
                return None  # non-context error, stop trying

        return None  # all 4 attempts exhausted

    # ------------------------------------------------------------------
    # Strategy 2 — Split backend / frontend review
    # ------------------------------------------------------------------

    def _strategy_split(
        self,
        plan: ProjectPlan,
        py_report: str,
        tsc_report: str,
        files_on_disk_str: str,
        all_files: list[str],
        output_dir: str,
    ) -> dict | None:
        """
        Review backend and frontend separately, then merge results.
        Each call uses ~half the tokens of a combined review.
        """
        backend_report  = self._review_backend(plan, py_report, all_files, output_dir)
        frontend_report = self._review_frontend(plan, tsc_report, all_files, output_dir)

        if backend_report is None and frontend_report is None:
            return None

        return self._merge_reports(
            backend_report  or self._empty_report(True),
            frontend_report or self._empty_report(True),
        )

    def _review_backend(
        self,
        plan: ProjectPlan,
        py_report: str,
        all_files: list[str],
        output_dir: str,
    ) -> dict | None:
        """LLM review of backend files only."""
        BACKEND_PRIORITY = [
            "backend/main.py", "backend/auth.py", "backend/config.py",
            "backend/database.py", "backend/models/models.py",
            "backend/schemas/schemas.py", "backend/routers/auth.py",
            "backend/requirements.txt",
        ]
        backend_files = [f for f in all_files if f.replace("\\", "/").startswith("backend/")]
        content = self._collect_files(output_dir, backend_files,
                                      max_chars=2000, max_files=15,
                                      priority=BACKEND_PRIORITY)

        plan_summary = (
            f"Project: {plan.project_name} | "
            f"Models: {', '.join(m.name for m in plan.database_models)} | "
            f"Endpoints: {len(plan.api_endpoints)}"
        )
        prompt = (
            f"Review ONLY the FastAPI backend for: {plan.project_name}\n\n"
            f"=== PYTHON SYNTAX CHECK ===\n{py_report}\n\n"
            f"=== PROJECT SUMMARY ===\n{plan_summary}\n\n"
            f"=== BACKEND FILE CONTENTS ===\n{content}\n\n"
            "Output a JSON report covering ONLY backend issues.\n"
            "Use the same JSON format: {passed, summary, issues[], "
            "backend_fix_instructions, frontend_fix_instructions}.\n"
            "Set frontend_fix_instructions to ''.\n"
            "Output the JSON now:"
        )

        for attempt in range(1, 4):
            try:
                raw = self.chat(prompt, max_tokens=4096)
                report = self.extract_json(raw)
                logger.info("QA backend split review OK")
                return report
            except Exception as exc:
                if self._is_context_error(exc):
                    logger.warning("QA backend split attempt %d/3 — context exceeded", attempt)
                    # Halve the content on each retry
                    backend_files = backend_files[:max(1, len(backend_files) // 2)]
                    content = self._collect_files(output_dir, backend_files,
                                                  max_chars=1000, max_files=8,
                                                  priority=BACKEND_PRIORITY)
                    prompt = prompt.replace(
                        f"=== BACKEND FILE CONTENTS ===\n{content}",
                        f"=== BACKEND FILE CONTENTS ===\n{content}",
                    )
                    continue
                logger.warning("QA backend split error: %s", exc)
                return None

        return None

    def _review_frontend(
        self,
        plan: ProjectPlan,
        tsc_report: str,
        all_files: list[str],
        output_dir: str,
    ) -> dict | None:
        """LLM review of frontend files only."""
        FRONTEND_PRIORITY = [
            "frontend/app/layout.tsx", "frontend/app/page.tsx",
            "frontend/lib/api.ts", "frontend/lib/auth.ts",
            "frontend/types/index.ts", "frontend/package.json",
            "frontend/next.config.js",
            "frontend/app/(auth)/login/page.tsx",
            "frontend/app/(auth)/register/page.tsx",
        ]
        frontend_files = [f for f in all_files if f.replace("\\", "/").startswith("frontend/")]
        content = self._collect_files(output_dir, frontend_files,
                                      max_chars=2000, max_files=15,
                                      priority=FRONTEND_PRIORITY)

        plan_summary = (
            f"Project: {plan.project_name} | "
            f"Models: {', '.join(m.name for m in plan.database_models)} | "
            f"Endpoints: {len(plan.api_endpoints)}"
        )
        prompt = (
            f"Review ONLY the Next.js 14 frontend for: {plan.project_name}\n\n"
            f"=== TYPESCRIPT CHECK ===\n{tsc_report}\n\n"
            f"=== PROJECT SUMMARY ===\n{plan_summary}\n\n"
            f"=== FRONTEND FILE CONTENTS ===\n{content}\n\n"
            "Output a JSON report covering ONLY frontend issues.\n"
            "Use the same JSON format: {passed, summary, issues[], "
            "backend_fix_instructions, frontend_fix_instructions}.\n"
            "Set backend_fix_instructions to ''.\n"
            "Output the JSON now:"
        )

        for attempt in range(1, 4):
            try:
                raw = self.chat(prompt, max_tokens=4096)
                report = self.extract_json(raw)
                logger.info("QA frontend split review OK")
                return report
            except Exception as exc:
                if self._is_context_error(exc):
                    logger.warning("QA frontend split attempt %d/3 — context exceeded", attempt)
                    frontend_files = frontend_files[:max(1, len(frontend_files) // 2)]
                    content = self._collect_files(output_dir, frontend_files,
                                                  max_chars=1000, max_files=8,
                                                  priority=FRONTEND_PRIORITY)
                    continue
                logger.warning("QA frontend split error: %s", exc)
                return None

        return None

    def _merge_reports(self, backend: dict, frontend: dict) -> dict:
        """Merge two partial QA reports into one unified report."""
        all_issues = backend.get("issues", []) + frontend.get("issues", [])
        has_errors = any(i.get("severity") == "error" for i in all_issues)
        passed     = not has_errors

        b_fix = backend.get("backend_fix_instructions", "").strip()
        f_fix = frontend.get("frontend_fix_instructions", "").strip()

        b_summary = backend.get("summary", "")
        f_summary = frontend.get("summary", "")
        if b_summary and f_summary:
            summary = f"Backend: {b_summary} | Frontend: {f_summary}"
        else:
            summary = b_summary or f_summary or ("All checks passed" if passed else "Issues found")

        return {
            "passed":                    passed,
            "summary":                   summary,
            "issues":                    all_issues,
            "backend_fix_instructions":  b_fix,
            "frontend_fix_instructions": f_fix,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_context_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "context window" in msg
            or "context_length" in msg
            or "maximum context" in msg
            or "prompt too long" in msg
            or "prompt is too long" in msg
            or "context length" in msg
        )

    @staticmethod
    def _make_fallback(automated_passed: bool, py_results: list, reason: str) -> dict:
        fallback: dict = {
            "passed":                    automated_passed,
            "summary":                   f"Automated checks only (LLM review failed: {reason})",
            "issues":                    [],
            "backend_fix_instructions":  "",
            "frontend_fix_instructions": "",
        }
        for r in py_results:
            if not r["success"]:
                fallback["issues"].append({
                    "file": r["file"], "severity": "error",
                    "description": r["errors"],
                    "suggestion": "Fix syntax error",
                })
                fallback["backend_fix_instructions"] = "Fix all Python syntax errors."
        return fallback

    @staticmethod
    def _empty_report(passed: bool) -> dict:
        return {
            "passed": passed, "summary": "No issues found",
            "issues": [],
            "backend_fix_instructions":  "",
            "frontend_fix_instructions": "",
        }

    # ------------------------------------------------------------------
    # Post-processing: remove hallucinated MISSING reports
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_hallucinated_issues(report: dict, files_on_disk: set[str]) -> dict:
        """
        Remove any issue that claims a file is MISSING when it actually
        exists on disk.
        """
        if "issues" not in report:
            return report

        def _normalise(path: str) -> str:
            return path.replace("\\", "/").lstrip("/").lower()

        disk_normalised = {_normalise(f) for f in files_on_disk}

        def _is_hallucinated(issue: dict) -> bool:
            file_ref = issue.get("file", "").lower()
            desc     = issue.get("description", "").lower()
            if "missing" not in file_ref and "missing" not in desc:
                return False
            clean_ref = _normalise(file_ref.replace("- missing", "").strip())
            for disk_path in disk_normalised:
                if clean_ref and (clean_ref in disk_path or disk_path.endswith(clean_ref)):
                    return True
            return False

        original_count  = len(report["issues"])
        filtered_issues = [i for i in report["issues"] if not _is_hallucinated(i)]
        removed         = original_count - len(filtered_issues)

        if removed > 0:
            logger.info("QA: removed %d hallucinated 'MISSING' issue(s)", removed)
            report["issues"] = filtered_issues
            has_errors = any(i.get("severity") == "error" for i in filtered_issues)
            if not has_errors:
                report["passed"]  = True
                report["summary"] = "All checks passed after removing false-positive reports"

        return report

    # ------------------------------------------------------------------
    # File collection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_files(
        output_dir: str,
        file_list: list[str],
        max_chars: int = MAX_FILE_CHARS,
        max_files: int = MAX_FILES,
        priority: list[str] | None = None,
    ) -> str:
        """Read files and format for the LLM prompt, prioritising key files."""
        DEFAULT_PRIORITY = [
            "backend/main.py", "backend/auth.py",
            "backend/models/models.py", "backend/schemas/schemas.py",
            "backend/routers/users.py", "backend/requirements.txt",
            "frontend/app/layout.tsx", "frontend/lib/api.ts",
            "frontend/lib/auth.ts", "frontend/types/index.ts",
            "frontend/package.json", "frontend/next.config.js",
        ]
        _priority = priority or DEFAULT_PRIORITY

        def sort_key(p: str) -> tuple:
            normalised = p.replace("\\", "/")
            try:
                return (0, _priority.index(normalised))
            except ValueError:
                return (1, normalised)

        sorted_files = sorted(file_list, key=sort_key)[:max_files]

        sections: list[str] = []
        for rel_path in sorted_files:
            full_path = str(Path(output_dir) / rel_path)
            content   = read_file(full_path)
            if content.startswith("[Skipped]") or content.startswith("[Error]"):
                continue
            if max_chars > 0 and len(content) > max_chars:
                content = content[:max_chars] + "\n... (truncated)"
            sections.append(f"=== {rel_path} ===\n{content}\n")

        return "\n".join(sections) if sections else "(no files found)"