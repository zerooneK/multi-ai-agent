"""
orchestrator.py
---------------
The Orchestrator is the master coordinator of the multi-agent pipeline.

Responsibilities
----------------
1. Receive the user's raw requirement.
2. Invoke PlannerAgent  → get ProjectPlan.
3. Invoke BackendAgent  → generate backend code.
4. Invoke FrontendAgent → generate frontend code.
5. Invoke QAAgent       → check all generated code.
6. If QA fails:
     a. Send fix instructions to BackendAgent and/or FrontendAgent.
     b. Re-run QA.
     c. Repeat up to MAX_FIX_ATTEMPTS times.
7. Report final status to the caller.

The Orchestrator itself does NOT call any LLM — it only routes messages
between agents and manages pipeline state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from agents import BackendAgent, FrontendAgent, PlannerAgent, QAAgent
from config import cfg
from models.messages import AgentMessage, TaskStatus, TaskType
from models.project_plan import ProjectPlan
from tools.file_tools import create_directory, list_files, patch_file, create_file

logger = logging.getLogger("orchestrator")

MAX_FIX_ATTEMPTS = 5  # QA → fix loop limit


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Final result returned to the caller after the full pipeline completes."""
    success:       bool
    project_name:  str
    output_dir:    str
    files_created: list[str]  = field(default_factory=list)
    plan_json:     str        = ""
    qa_report:     dict       = field(default_factory=dict)
    errors:        list[str]  = field(default_factory=list)
    duration_s:    float      = 0.0
    message_log:   list[str]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Drives the full multi-agent pipeline from requirement → working codebase.

    Parameters
    ----------
    output_base_dir : Root directory where generated projects are saved.
    progress_cb     : Optional callback(step: str, detail: str) for UI updates.
    provider_name   : LLM provider for all agents (default: cfg.PROVIDER).
    model           : LLM model override for all agents.
    """

    def __init__(
        self,
        output_base_dir: str = "output",
        progress_cb: Callable[[str, str], None] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> None:
        self.output_base_dir = output_base_dir
        self.progress_cb     = progress_cb or self._default_progress

        # Resolve per-agent provider + model from cfg.
        # Explicit constructor args (provider_name / model) override .env settings.
        def _resolve(agent: str):
            ac = cfg.for_agent(agent)
            return (
                provider_name or ac.provider,
                model         or ac.model,
            )

        p_prov, p_model = _resolve("planner")
        b_prov, b_model = _resolve("backend")
        f_prov, f_model = _resolve("frontend")
        q_prov, q_model = _resolve("qa")

        # Initialise each agent with its own provider + model
        self.planner  = PlannerAgent(p_prov,  p_model)
        self.backend  = BackendAgent(b_prov,  b_model)
        self.frontend = FrontendAgent(f_prov, f_model)
        self.qa       = QAAgent(q_prov,       q_model)

        logger.info("Agent models:")
        logger.info("  Planner  -> %s / %s", p_prov, p_model)
        logger.info("  Backend  -> %s / %s", b_prov, b_model)
        logger.info("  Frontend -> %s / %s", f_prov, f_model)
        logger.info("  QA       -> %s / %s", q_prov, q_model)

        self._log: list[str] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, requirement: str) -> PipelineResult:
        """
        Execute the full pipeline for a user requirement.

        Parameters
        ----------
        requirement : Free-text description of the project to build,
                      e.g. "create a website for rental books".

        Returns
        -------
        PipelineResult with success flag, output directory, and file list.
        """
        t_start = time.perf_counter()
        self._log.clear()
        self._emit("🚀 Starting pipeline", requirement[:120])

        # ── Step 1: Plan ─────────────────────────────────────────────
        plan_message = self._run_agent(
            agent=self.planner,
            sender="orchestrator",
            receiver="planner",
            task_type=TaskType.PLAN,
            payload={"requirement": requirement},
            step_label="📋 Planning project",
        )
        if plan_message.status == TaskStatus.FAILED:
            return self._failed(plan_message.error or "Planner failed", t_start)

        plan_json = plan_message.result
        try:
            plan = ProjectPlan.from_json(plan_json)
        except Exception as exc:
            return self._failed(f"Could not parse plan: {exc}", t_start)

        # Create output directory
        output_dir = str(Path(self.output_base_dir) / plan.project_name)
        create_directory(output_dir)
        self._emit("📁 Output directory", output_dir)

        # Save the plan to disk
        plan_path = str(Path(output_dir) / "project_plan.json")
        Path(plan_path).write_text(plan_json, encoding="utf-8")
        self._emit("💾 Plan saved", plan_path)

        # ── Step 2+3: Backend & Frontend — parallel if supported ────
        self._emit("⚙️  Generating backend + frontend (parallel)", "")
        backend_message, frontend_message = self._run_parallel(
            plan_json=plan_json,
            output_dir=output_dir,
        )
        if backend_message.status == TaskStatus.FAILED:
            return self._failed(backend_message.error or "Backend failed", t_start)
        if frontend_message.status == TaskStatus.FAILED:
            return self._failed(frontend_message.error or "Frontend failed", t_start)

        # ── Step 4: QA + fix loop ─────────────────────────────────────
        qa_report:    dict       = {}
        prev_issues:  frozenset  = frozenset()  # track stale/looping issues
        stale_rounds: int        = 0
        MAX_STALE                = 2            # stop if same issues repeat N times

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            label = f"🔍 QA check (attempt {attempt}/{MAX_FIX_ATTEMPTS})"
            qa_message = self._run_agent(
                agent=self.qa,
                sender="orchestrator",
                receiver="qa",
                task_type=TaskType.QA,
                payload={"plan_json": plan_json, "output_dir": output_dir},
                step_label=label,
            )

            if qa_message.status == TaskStatus.FAILED:
                self._emit("⚠️  QA agent error", qa_message.error or "")
                break

            try:
                qa_report = json.loads(qa_message.result or "{}")
            except json.JSONDecodeError:
                qa_report = {"passed": False, "summary": qa_message.result}

            qa_passed = qa_report.get("passed", False)
            self._emit(
                "✅ QA passed" if qa_passed else "❌ QA failed",
                qa_report.get("summary", ""),
            )

            if qa_passed:
                break

            # ── Stale issue detection ─────────────────────────────────
            # If QA keeps reporting the exact same issues after fixes,
            # the model is hallucinating — stop the loop early.
            current_issues = frozenset(
                i.get("description", "") for i in qa_report.get("issues", [])
            )
            if current_issues and current_issues == prev_issues:
                stale_rounds += 1
                if stale_rounds >= MAX_STALE:
                    self._emit(
                        "⚠️  QA loop detected",
                        f"Same issues reported {stale_rounds} times — likely hallucination. Stopping.",
                    )
                    break
            else:
                stale_rounds = 0
            prev_issues = current_issues

            # ── Fix cycle ─────────────────────────────────────────────
            if attempt < MAX_FIX_ATTEMPTS:
                patches      = qa_report.get("patches", [])
                backend_fix  = qa_report.get("backend_fix_instructions", "")
                frontend_fix = qa_report.get("frontend_fix_instructions", "")

                # Strategy 1: apply patches (targeted, fast)
                if patches:
                    patch_results = self._apply_patches(patches, output_dir)
                    failed_patches = [r for r in patch_results if not r["success"]]
                    if failed_patches:
                        self._emit(
                            f"⚠️  {len(failed_patches)}/{len(patches)} patches failed",
                            "falling back to full rewrite for affected files",
                        )
                        # Collect files that need full rewrite
                        failed_files = {r["path"] for r in failed_patches}
                        backend_fix  = backend_fix or (
                            f"Rewrite these files (patch failed): {', '.join(failed_files)}"
                            if any("backend" in p for p in failed_files) else ""
                        )
                        frontend_fix = frontend_fix or (
                            f"Rewrite these files (patch failed): {', '.join(failed_files)}"
                            if any("frontend" in p for p in failed_files) else ""
                        )
                    else:
                        self._emit(
                            f"✅ Applied {len(patches)} patch(es) successfully", ""
                        )
                        # Patches applied — skip full rewrite unless fix_instructions also set
                        backend_fix  = "" if not backend_fix else backend_fix
                        frontend_fix = "" if not frontend_fix else frontend_fix

                # Strategy 2: full rewrite (fallback)
                if backend_fix:
                    self._emit("🔧 Sending fixes to Backend Agent", "")
                    self._run_agent(
                        agent=self.backend,
                        sender="orchestrator",
                        receiver="backend",
                        task_type=TaskType.FIX,
                        payload={
                            "plan_json":   plan_json,
                            "output_dir":  output_dir,
                            "fix_context": backend_fix,
                        },
                        step_label="🔧 Backend fixing errors",
                    )

                if frontend_fix:
                    self._emit("🔧 Sending fixes to Frontend Agent", "")
                    self._run_agent(
                        agent=self.frontend,
                        sender="orchestrator",
                        receiver="frontend",
                        task_type=TaskType.FIX,
                        payload={
                            "plan_json":   plan_json,
                            "output_dir":  output_dir,
                            "fix_context": frontend_fix,
                        },
                        step_label="🔧 Frontend fixing errors",
                    )

                if not patches and not backend_fix and not frontend_fix:
                    self._emit(
                        "⚠️  QA reported failure but gave no fix instructions",
                        "Stopping fix loop.",
                    )
                    break
            else:
                self._emit(
                    "⚠️  Max fix attempts reached",
                    "Delivering with known issues — review QA report.",
                )

        # ── Step 5: Collect results ───────────────────────────────────
        files_created = list_files(output_dir)
        duration      = round(time.perf_counter() - t_start, 1)

        self._emit(
            "🎉 Pipeline complete",
            f"{len(files_created)} files in {duration}s → {output_dir}",
        )

        return PipelineResult(
            success       = qa_report.get("passed", True),
            project_name  = plan.project_name,
            output_dir    = output_dir,
            files_created = files_created,
            plan_json     = plan_json,
            qa_report     = qa_report,
            errors        = [
                i.get("description", "") for i in qa_report.get("issues", [])
                if i.get("severity") == "error"
            ],
            duration_s    = duration,
            message_log   = list(self._log),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_agent(
        self,
        agent,
        sender: str,
        receiver: str,
        task_type: TaskType,
        payload: dict,
        step_label: str,
    ) -> AgentMessage:
        """Create a message, run the agent, log result, return message."""
        self._emit(step_label, "")
        message = AgentMessage(
            sender=sender, receiver=receiver,
            task_type=task_type, payload=payload,
        )
        t = time.perf_counter()
        message = agent.run(message)
        elapsed = round(time.perf_counter() - t, 1)

        self._log.append(message.summary() + f"  [{elapsed}s]")
        self.progress_cb(message.summary(), f"Took {elapsed}s")

        if message.status == TaskStatus.FAILED:
            logger.error("Agent %s failed: %s", receiver, message.error)
        else:
            logger.info("Agent %s done in %.1fs", receiver, elapsed)

        return message

    def _apply_patches(
        self,
        patches: list[dict],
        output_dir: str,
        max_attempts: int = 3,
    ) -> list[dict]:
        """
        Apply a list of patch objects to files on disk.

        Each patch: {"file": str, "old_code": str, "new_code": str}

        Retries up to max_attempts times per patch.
        Returns list of results: {"success": bool, "path": str, "error"?: str}
        """
        results = []
        for patch in patches:
            rel_path = patch.get("file", "")
            old_code = patch.get("old_code", "")
            new_code = patch.get("new_code", "")

            if not rel_path or not old_code:
                results.append({
                    "success": False, "path": rel_path,
                    "error": "missing file or old_code in patch",
                })
                continue

            full_path = str(Path(output_dir) / rel_path)
            result    = None

            for attempt in range(1, max_attempts + 1):
                result = patch_file(full_path, old_code, new_code)
                if result["success"]:
                    logger.info("Patch applied (%d/%d): %s", attempt, max_attempts, rel_path)
                    break
                logger.warning(
                    "Patch attempt %d/%d failed for %s: %s",
                    attempt, max_attempts, rel_path, result.get("error"),
                )

            results.append(result or {
                "success": False, "path": full_path,
                "error": f"patch failed after {max_attempts} attempts",
            })

        return results

    def _run_parallel(
        self,
        plan_json: str,
        output_dir: str,
    ) -> tuple:
        """Run backend and frontend agents in parallel using threads.

        Safe because backend writes to output_dir/backend/ and frontend
        writes to output_dir/frontend/ — no shared files at root level.
        Falls back to sequential if PARALLEL_GENERATION=false in .env.
        """
        use_parallel = getattr(cfg, "parallel_generation", True)

        def _run_backend() -> AgentMessage:
            return self._run_agent(
                agent=self.backend,
                sender="orchestrator", receiver="backend",
                task_type=TaskType.BACKEND,
                payload={"plan_json": plan_json, "output_dir": output_dir},
                step_label="⚙️  Generating backend code",
            )

        def _run_frontend() -> AgentMessage:
            return self._run_agent(
                agent=self.frontend,
                sender="orchestrator", receiver="frontend",
                task_type=TaskType.FRONTEND,
                payload={"plan_json": plan_json, "output_dir": output_dir},
                step_label="🎨 Generating frontend code",
            )

        if not use_parallel:
            logger.info("Parallel generation disabled — running sequential")
            return _run_backend(), _run_frontend()

        logger.info("Running backend + frontend in parallel")
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_backend  = executor.submit(_run_backend)
            f_frontend = executor.submit(_run_frontend)
            return f_backend.result(), f_frontend.result()

    def _failed(self, reason: str, t_start: float) -> PipelineResult:
        logger.error("Pipeline failed: %s", reason)
        return PipelineResult(
            success=False,
            project_name="unknown",
            output_dir="",
            errors=[reason],
            duration_s=round(time.perf_counter() - t_start, 1),
            message_log=list(self._log),
        )

    def _emit(self, step: str, detail: str) -> None:
        entry = f"{step}: {detail}" if detail else step
        self._log.append(entry)
        logger.info(entry)
        self.progress_cb(step, detail)

    @staticmethod
    def _default_progress(step: str, detail: str) -> None:
        """Default progress printer used when no callback is provided."""
        if detail:
            print(f"  {step}  {detail}")
        else:
            print(f"  {step}")