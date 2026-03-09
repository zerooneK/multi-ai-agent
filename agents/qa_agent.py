"""
agents/qa_agent.py
------------------
QA Agent — runs automated checks on ALL generated code and produces
a structured fix report for the Orchestrator.

Pipeline (Phase 3 upgrade):
  1. Python syntax check  : py_compile on all backend .py files
  2. TypeScript check     : npm install + tsc --noEmit on frontend/
     (skipped gracefully if Node.js is not installed)
  3. LLM deep review      : reads all files, checks logic, references,
     Next.js rules, schema alignment, missing files
  4. Returns unified JSON report with per-agent fix instructions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from models.messages import AgentMessage
from models.project_plan import ProjectPlan
from tools.file_tools import list_files, read_file
from tools.shell_tools import (
    format_check_result,
    format_tsc_result,
    node_available,
    run_npm_install,
    run_tsc,
    syntax_check_all_python,
)

logger = logging.getLogger("agent.qa")

# Max chars per file sent to LLM (keep prompt manageable)
MAX_FILE_CHARS = 2500
# Max files sent to LLM review
MAX_FILES      = 35


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
  2. All generated source files
  3. The original project plan

Your job is to find ALL real issues and produce a JSON report.

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
- Missing __init__.py in any package directory
- SQLAlchemy model fields don't match the project plan
- Pydantic schemas missing fields that models have
- FastAPI router not registered in main.py
- CORS middleware missing or not allowing http://localhost:3000
- JWT cookie name not "access_token"
- Auth endpoints missing: POST /api/auth/register, POST /api/auth/login,
  POST /api/auth/logout, GET /api/auth/me
- requirements.txt missing a package that is imported

NEXT.JS FRONTEND CHECKLIST — flag errors for any of these:
- TypeScript errors (from tsc check)
- 'use client' present on a Server Component (no hooks/events used)
- 'use client' MISSING on a component that uses useState/useEffect/
  useRouter/event handlers
- localStorage or sessionStorage used for JWT (must use httpOnly cookie)
- window.location used instead of next/navigation useRouter / redirect
- <a href> used for internal links instead of next/link <Link>
- <img> used instead of next/image <Image>
- Hard-coded API URL instead of process.env.NEXT_PUBLIC_API_URL
- Client-side env var missing NEXT_PUBLIC_ prefix
- apiFetch called without credentials:"include"
- TypeScript interface missing fields that exist in the DB model
- package.json missing next, react, react-dom, typescript
- tsconfig.json missing paths alias @/*
- next.config.ts missing API rewrite to backend

RULES:
- Output ONLY the JSON object.
- Set passed=true only if there are NO errors (warnings are OK).
- Be specific in fix_instructions — name exact files and what to change.
- Do NOT report issues that are clearly intentional design choices.
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
        backend_dir  = str(Path(output_dir) / "backend")
        py_results   = syntax_check_all_python(backend_dir)
        py_report    = format_check_result(py_results)
        py_passed    = all(r["success"] for r in py_results)
        logger.info("Python syntax: %s (%d files)",
                    "PASS" if py_passed else "FAIL", len(py_results))

        # ── Step 2: TypeScript check (npm install + tsc) ──────────────
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
                    tsc_report = (
                        "TypeScript check: SKIPPED — npm install failed:\n"
                        + npm_result["stderr"][:500]
                    )
                    logger.warning("npm install failed: %s", npm_result["stderr"][:200])
            else:
                tsc_report = "TypeScript check: SKIPPED — Node.js not installed on this machine"
                logger.info("Node.js not found — skipping tsc")

        automated_passed = py_passed and tsc_passed

        # ── Step 3: LLM deep review ───────────────────────────────────
        all_files     = list_files(output_dir)
        files_content = self._collect_files(output_dir, all_files)

        prompt = f"""Review the generated full-stack project: {plan.project_name}

=== AUTOMATED CHECK RESULTS ===

{py_report}

{tsc_report}

=== PROJECT PLAN (reference) ===
{plan.to_json(indent=1)}

=== GENERATED FILES ===
{files_content}

Produce the JSON quality report now:"""

        try:
            raw    = self.chat(prompt, max_tokens=4096)
            report = self.extract_json(raw)

            # Merge automated failures into LLM report
            if not py_passed and report.get("passed"):
                report["passed"]  = False
                report["summary"] = "Python syntax errors detected"

            if not tsc_passed and report.get("passed"):
                report["passed"]  = False
                report["summary"] = "TypeScript errors detected"

            logger.info("QA report: passed=%s issues=%d",
                        report.get("passed"), len(report.get("issues", [])))

            message.mark_done(json.dumps(report, indent=2))

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("QA LLM review failed: %s", exc)
            # Fallback: use automated results only
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
                        "description": r["errors"], "suggestion": "Fix syntax error",
                    })
                    fallback["backend_fix_instructions"] = "Fix all Python syntax errors."
            message.mark_done(json.dumps(fallback, indent=2))

        return message

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_files(output_dir: str, file_list: list[str]) -> str:
        """
        Read each file and format for the LLM prompt.
        Prioritises important files, caps total files and per-file size.
        """
        # Priority order: core config files first, then source
        PRIORITY = [
            "backend/main.py", "backend/models/models.py",
            "backend/schemas/schemas.py", "backend/auth.py",
            "frontend/app/layout.tsx", "frontend/app/page.tsx",
            "frontend/lib/api.ts", "frontend/lib/auth.ts",
            "frontend/types/index.ts",
            "frontend/package.json", "frontend/tsconfig.json",
            "frontend/next.config.ts",
        ]

        # Sort: priority files first, then alphabetical
        def sort_key(p: str) -> tuple:
            try:
                return (0, PRIORITY.index(p))
            except ValueError:
                return (1, p)

        sorted_files = sorted(file_list, key=sort_key)[:MAX_FILES]

        sections: list[str] = []
        for rel_path in sorted_files:
            full_path = str(Path(output_dir) / rel_path)
            content   = read_file(full_path)
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n... (truncated)"
            sections.append(f"=== {rel_path} ===\n{content}\n")

        return "\n".join(sections) if sections else "(no files found)"
