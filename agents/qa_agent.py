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

MAX_FILE_CHARS = 4000   # raised from 2500 so LLM sees more complete files
MAX_FILES      = 30     # cap to keep prompt size sane


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
  "backend_fix_instructions":  "detailed step-by-step for backend agent  ('' if none)",
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

        # ── Step 3: Build file inventory for LLM ─────────────────────
        all_files     = list_files(output_dir)
        files_on_disk = actual_files_set(output_dir)
        files_content = self._collect_files(output_dir, all_files)
        files_on_disk_str = "\n".join(sorted(files_on_disk))

        # ── Step 4: LLM deep review ───────────────────────────────────
        prompt = f"""Review the generated full-stack project: {plan.project_name}

=== AUTOMATED CHECK RESULTS ===
{py_report}

{tsc_report}

=== FILES ON DISK (these files EXIST — do not report them as missing) ===
{files_on_disk_str}

=== PROJECT PLAN (reference) ===
{plan.to_json(indent=1)}

=== FILE CONTENTS ===
{files_content}

Produce the JSON quality report now. Remember: never report a file as missing if it is in FILES ON DISK."""

        try:
            raw    = self.chat(prompt, max_tokens=8192)
            report = self.extract_json(raw)

            # ── Step 5: Strip hallucinated MISSING issues ─────────────
            report = self._filter_hallucinated_issues(report, files_on_disk)

            # Sync automated failures into LLM report
            if not py_passed and report.get("passed"):
                report["passed"]  = False
                report["summary"] = "Python syntax errors detected"
            if not tsc_passed and report.get("passed"):
                report["passed"]  = False
                report["summary"] = "TypeScript errors detected"

            logger.info("QA report: passed=%s issues=%d",
                        report.get("passed"), len(report.get("issues", [])))
            message.mark_done(json.dumps(report, indent=2))

        except Exception as exc:
            logger.error("QA LLM review failed: %s", exc)
            fallback = {
                "passed":  automated_passed,
                "summary": f"Automated checks only (LLM review failed: {exc})",
                "issues":  [],
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
            message.mark_done(json.dumps(fallback, indent=2))

        return message

    # ------------------------------------------------------------------
    # Post-processing: remove hallucinated MISSING reports
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_hallucinated_issues(report: dict, files_on_disk: set[str]) -> dict:
        """
        Remove any issue that claims a file is MISSING when it actually
        exists on disk. Also normalise file paths for comparison.
        """
        if "issues" not in report:
            return report

        def _normalise(path: str) -> str:
            """Strip leading separators and normalise to forward slashes."""
            return path.replace("\\", "/").lstrip("/").lower()

        disk_normalised = {_normalise(f) for f in files_on_disk}

        def _is_hallucinated(issue: dict) -> bool:
            file_ref  = issue.get("file", "").lower()
            desc      = issue.get("description", "").lower()
            is_missing_claim = "missing" in file_ref or "missing" in desc
            if not is_missing_claim:
                return False
            # Extract the actual filename from the issue reference
            # e.g. "backend/routers/users.py - MISSING" → "backend/routers/users.py"
            clean_ref = _normalise(file_ref.replace("- missing", "").strip())
            # Check if the file or any similar path exists on disk
            for disk_path in disk_normalised:
                if clean_ref and (clean_ref in disk_path or disk_path.endswith(clean_ref)):
                    return True
            return False

        original_count   = len(report["issues"])
        filtered_issues  = [i for i in report["issues"] if not _is_hallucinated(i)]
        removed          = original_count - len(filtered_issues)

        if removed > 0:
            logger.info("QA: removed %d hallucinated 'MISSING' issue(s)", removed)
            report["issues"] = filtered_issues
            # Re-evaluate passed based on remaining errors
            has_errors = any(i.get("severity") == "error" for i in filtered_issues)
            if not has_errors:
                report["passed"]  = True
                report["summary"] = "All checks passed after removing false-positive reports"

        return report

    # ------------------------------------------------------------------
    # File collection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_files(output_dir: str, file_list: list[str]) -> str:
        """Read files and format for the LLM prompt, prioritising key files."""
        PRIORITY = [
            "backend/main.py", "backend/auth.py",
            "backend/models/models.py", "backend/schemas/schemas.py",
            "backend/routers/users.py", "backend/requirements.txt",
            "frontend/app/layout.tsx", "frontend/lib/api.ts",
            "frontend/lib/auth.ts", "frontend/types/index.ts",
            "frontend/package.json", "frontend/next.config.ts",
        ]

        def sort_key(p: str) -> tuple:
            normalised = p.replace("\\", "/")
            try:
                return (0, PRIORITY.index(normalised))
            except ValueError:
                return (1, normalised)

        sorted_files = sorted(file_list, key=sort_key)[:MAX_FILES]

        sections: list[str] = []
        for rel_path in sorted_files:
            full_path = str(Path(output_dir) / rel_path)
            content   = read_file(full_path)
            if content.startswith("[Skipped]") or content.startswith("[Error]"):
                continue
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n... (truncated)"
            sections.append(f"=== {rel_path} ===\n{content}\n")

        return "\n".join(sections) if sections else "(no files found)"