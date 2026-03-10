"""
orchestrator.py
---------------
v2 — Sequential module-by-module build with DoD checking.

Pipeline
--------
1. PlannerAgent       → ProjectPlan with modules[] spec
2. Spec Validator     → validate dependency graph (no LLM, fast)
3. Sequential Build   → ModuleBuilderAgent builds each module in topo order
                        DoD check after each module — fix immediately if fail
4. Config Files       → write requirements.txt, .env.example, package.json, etc.
5. QA Agent           → final syntax + tsc + LLM review (same as before)
6. QA fix loop        → up to MAX_FIX_ATTEMPTS (same as before)

Fallback
--------
If the plan has no modules[] (LLM returned old-style plan), the orchestrator
falls back to the legacy BackendAgent + FrontendAgent bulk generation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from agents import BackendAgent, FrontendAgent, PlannerAgent, QAAgent
from agents.module_builder_agent import ModuleBuilderAgent
from agents.dod_checker import DodCheckerAgent
from config import cfg
from models.messages import AgentMessage, TaskStatus, TaskType
from models.project_plan import ProjectPlan, ModuleSpec
from tools.file_tools import create_directory, list_files, read_file
from tools.spec_validator import validate_module_specs, topological_sort

logger = logging.getLogger("orchestrator")

MAX_FIX_ATTEMPTS   = 5   # QA → fix loop limit
MAX_DOD_FIX_ROUNDS = 2   # per-module DoD fix attempts before skipping


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    success:       bool
    project_name:  str
    output_dir:    str
    files_created: list[str] = field(default_factory=list)
    plan_json:     str       = ""
    qa_report:     dict      = field(default_factory=dict)
    errors:        list[str] = field(default_factory=list)
    duration_s:    float     = 0.0
    message_log:   list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(
        self,
        output_base_dir: str = "output",
        progress_cb: Callable[[str, str], None] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
    ) -> None:
        self.output_base_dir = output_base_dir
        self.progress_cb     = progress_cb or self._default_progress

        def _resolve(agent: str):
            ac = cfg.for_agent(agent)
            return (provider_name or ac.provider, model or ac.model)

        p_prov, p_model = _resolve("planner")
        b_prov, b_model = _resolve("backend")
        f_prov, f_model = _resolve("frontend")
        q_prov, q_model = _resolve("qa")

        self.planner        = PlannerAgent(p_prov, p_model)
        self.backend        = BackendAgent(b_prov, b_model)      # fallback
        self.frontend       = FrontendAgent(f_prov, f_model)     # fallback
        self.qa             = QAAgent(q_prov, q_model)
        self.module_builder = ModuleBuilderAgent(b_prov, b_model)
        self.dod_checker    = DodCheckerAgent(q_prov, q_model)   # use QA model

        logger.info("Agents: planner=%s/%s builder=%s/%s qa=%s/%s",
                    p_prov, p_model, b_prov, b_model, q_prov, q_model)
        self._log: list[str] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, requirement: str) -> PipelineResult:
        t_start = time.perf_counter()
        self._log.clear()
        self._emit("🚀 Starting pipeline", requirement[:120])

        # ── Step 1: Plan ──────────────────────────────────────────────
        plan_message = self._run_agent(
            agent=self.planner,
            sender="orchestrator", receiver="planner",
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

        # Save plan to disk
        Path(output_dir, "project_plan.json").write_text(plan_json, encoding="utf-8")

        # ── Step 2: Spec Validation ───────────────────────────────────
        if plan.modules:
            self._emit("🔎 Validating module specs", f"{len(plan.modules)} modules")
            spec_errors = validate_module_specs(plan.modules)
            if spec_errors:
                for err in spec_errors:
                    logger.warning("Spec error: %s", err)
                    self._emit("⚠️  Spec issue", err)
                # Non-fatal — we log and continue; build will expose real errors
                self._emit("⚠️  Spec validation had issues",
                           "Continuing — some modules may fail DoD")

        # ── Step 3: Build ─────────────────────────────────────────────
        if plan.modules:
            self._emit("🏗️  Sequential module build", f"{len(plan.modules)} modules")
            build_ok = self._sequential_build(plan, output_dir, plan_json)
            if not build_ok:
                self._emit("⚠️  Some modules failed DoD",
                           "Continuing to QA with best-effort files")
        else:
            # Fallback: legacy bulk generation
            self._emit("⚙️  Legacy bulk generation (no modules[] in plan)", "")
            back_msg, front_msg = self._run_parallel(plan_json, output_dir)
            if back_msg.status == TaskStatus.FAILED:
                return self._failed(back_msg.error or "Backend failed", t_start)
            if front_msg.status == TaskStatus.FAILED:
                return self._failed(front_msg.error or "Frontend failed", t_start)

        # Write config files (requirements.txt, .env.example, package.json, etc.)
        self._write_config_files(plan, output_dir)

        # ── Step 4: QA + fix loop ─────────────────────────────────────
        qa_report: dict       = {}
        prev_issues: frozenset = frozenset()
        prev_tsc_errors: frozenset = frozenset()
        stale_rounds: int      = 0
        tsc_stale_rounds: int  = 0
        MAX_STALE              = 2
        MAX_TSC_STALE          = 2

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            qa_message = self._run_agent(
                agent=self.qa,
                sender="orchestrator", receiver="qa",
                task_type=TaskType.QA,
                payload={"plan_json": plan_json, "output_dir": output_dir},
                step_label=f"🔍 QA check (attempt {attempt}/{MAX_FIX_ATTEMPTS})",
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

            # Stale detection
            current_issues = frozenset(
                i.get("description", "") for i in qa_report.get("issues", [])
            )
            if current_issues and current_issues == prev_issues:
                stale_rounds += 1
                if stale_rounds >= MAX_STALE:
                    self._emit("⚠️  QA loop detected",
                               "Same issues repeated — stopping")
                    break
            else:
                stale_rounds = 0
            prev_issues = current_issues

            # TSC stale detection
            tsc_errors = frozenset(
                i.get("description", "") for i in qa_report.get("issues", [])
                if "typescript" in i.get("description", "").lower()
                or i.get("file", "").endswith((".tsx", ".ts"))
            )
            if tsc_errors and tsc_errors == prev_tsc_errors:
                tsc_stale_rounds += 1
                if tsc_stale_rounds >= MAX_TSC_STALE:
                    affected = [
                        i.get("file", "") for i in qa_report.get("issues", [])
                        if i.get("file", "").endswith((".tsx", ".ts"))
                    ]
                    qa_report["frontend_fix_instructions"] = (
                        f"FORCE FULL REWRITE: {', '.join(set(affected))}"
                    )
                    tsc_stale_rounds = 0
            else:
                tsc_stale_rounds = 0
            prev_tsc_errors = tsc_errors

            if attempt < MAX_FIX_ATTEMPTS:
                self._run_qa_fixes(qa_report, plan_json, output_dir)
            else:
                self._emit("⚠️  Max fix attempts reached",
                           "Delivering with known issues")

        # ── Step 5: Collect results ───────────────────────────────────
        files_created = list_files(output_dir)
        duration      = round(time.perf_counter() - t_start, 1)
        self._emit("🎉 Pipeline complete",
                   f"{len(files_created)} files in {duration}s → {output_dir}")

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
    # Sequential build (Phase 3 + 4)
    # ------------------------------------------------------------------

    def _sequential_build(
        self,
        plan: ProjectPlan,
        output_dir: str,
        plan_json: str,
    ) -> bool:
        """
        Build every module in topological order.
        After each module: run DoD check. If fail → fix up to MAX_DOD_FIX_ROUNDS.

        Returns True if all modules passed DoD, False if any were skipped/failed.
        """
        # Sort by dependency order — no circular deps (validated in Step 2)
        try:
            ordered = topological_sort(plan.modules)
        except ValueError as exc:
            logger.error("Topological sort failed: %s — using original order", exc)
            ordered = plan.modules

        # built_contents: path → source (fed as context to later modules)
        built_contents: dict[str, str] = {}
        all_passed = True

        for spec in ordered:
            self._emit(f"  🔨 Building", spec.path)

            for dod_round in range(1, MAX_DOD_FIX_ROUNDS + 2):
                fix_ctx = ""
                if dod_round > 1:
                    # Retrieve DoD errors from previous round for fix prompt
                    fix_ctx = self._last_dod_errors.get(spec.path, "")

                # Gather dependency contents for this module
                dep_contents = {
                    imp: built_contents[imp]
                    for imp in spec.imports
                    if imp in built_contents
                }

                # Build the module
                msg = self._run_agent(
                    agent=self.module_builder,
                    sender="orchestrator", receiver="module_builder",
                    task_type=TaskType.BACKEND
                             if spec.layer == "backend" else TaskType.FRONTEND,
                    payload={
                        "module_spec":  spec.to_dict(),
                        "plan_json":    plan_json,
                        "output_dir":   output_dir,
                        "dep_contents": dep_contents,
                        "fix_context":  fix_ctx,
                    },
                    step_label=f"    Building {spec.path} (round {dod_round})",
                )

                if msg.status == TaskStatus.FAILED:
                    logger.error("ModuleBuilder failed for %s", spec.path)
                    all_passed = False
                    break

                # DoD check
                skip_llm = (dod_round == MAX_DOD_FIX_ROUNDS + 1)  # last chance: skip LLM
                dod_result = self.dod_checker.check(spec, output_dir, skip_llm=skip_llm)
                self._emit(f"    {'✅' if dod_result.passed else '❌'} DoD",
                           dod_result.summary())

                if dod_result.passed:
                    # Read the built file into context for later modules
                    full_path = str(Path(output_dir) / spec.path)
                    source    = read_file(full_path)
                    if not source.startswith("[Error]"):
                        built_contents[spec.path] = source
                    break

                # DoD failed
                if dod_round <= MAX_DOD_FIX_ROUNDS:
                    errors_text = "\n".join(dod_result.all_errors)
                    logger.warning("DoD failed %s (round %d): %s",
                                   spec.path, dod_round, errors_text)
                    # Store errors for next round's fix_context
                    if not hasattr(self, "_last_dod_errors"):
                        self._last_dod_errors: dict[str, str] = {}
                    self._last_dod_errors[spec.path] = (
                        f"Previous attempt failed these DoD checks:\n{errors_text}\n"
                        f"Fix all of them in the new version."
                    )
                else:
                    # Exhausted rounds — include best-effort file in context
                    logger.error("DoD exhausted for %s — using best-effort output", spec.path)
                    all_passed = False
                    full_path  = str(Path(output_dir) / spec.path)
                    source     = read_file(full_path)
                    if not source.startswith("[Error]"):
                        built_contents[spec.path] = source

        return all_passed

    # ------------------------------------------------------------------
    # Config file writer
    # ------------------------------------------------------------------

    def _write_config_files(self, plan: ProjectPlan, output_dir: str) -> None:
        """
        Write non-module config files: requirements.txt, .env.example,
        package.json, tsconfig.json, next.config.js, tailwind.config.ts.

        These are NOT in modules[] but are needed to run the project.
        Only writes files that don't already exist (module builder may have
        created some of them).
        """
        from tools.file_tools import file_exists

        # ── backend/requirements.txt ──────────────────────────────────
        req_path = str(Path(output_dir) / "backend" / "requirements.txt")
        if not file_exists(req_path):
            create_directory(str(Path(output_dir) / "backend"))
            from tools.file_tools import create_file
            create_file(req_path, (
                "fastapi\nuvicorn[standard]\nsqlalchemy\n"
                "python-jose[cryptography]\npasslib[bcrypt]\n"
                "python-multipart\npydantic-settings\n"
                "pydantic[email]\nemail-validator\npython-dotenv\n"
            ))
            self._emit("  📄 Written", "backend/requirements.txt")

        # ── backend/.env.example ──────────────────────────────────────
        env_path = str(Path(output_dir) / "backend" / ".env.example")
        if not file_exists(env_path):
            from tools.file_tools import create_file
            create_file(env_path, (
                "DATABASE_URL=sqlite:///./app.db\n"
                "SECRET_KEY=change-me-in-production-use-long-random-string\n"
                "ALGORITHM=HS256\n"
                "ACCESS_TOKEN_EXPIRE_MINUTES=60\n"
            ))
            self._emit("  📄 Written", "backend/.env.example")

        # ── backend/__init__.py ───────────────────────────────────────
        for pkg in ["backend", "backend/models", "backend/routers", "backend/schemas"]:
            init_path = str(Path(output_dir) / pkg / "__init__.py")
            if not file_exists(init_path):
                from tools.file_tools import create_file
                create_file(init_path, "")

        # ── frontend/package.json ─────────────────────────────────────
        pkg_path = str(Path(output_dir) / "frontend" / "package.json")
        if not file_exists(pkg_path):
            create_directory(str(Path(output_dir) / "frontend"))
            from tools.file_tools import create_file
            create_file(pkg_path, json.dumps({
                "name": plan.project_name.replace("_", "-"),
                "version": "0.1.0",
                "private": True,
                "scripts": {
                    "dev": "next dev",
                    "build": "next build",
                    "start": "next start",
                    "lint": "next lint"
                },
                "dependencies": {
                    "next": "14.2.3",
                    "react": "^18",
                    "react-dom": "^18",
                    "@radix-ui/react-slot": "^1.0.2",
                    "class-variance-authority": "^0.7.0",
                    "clsx": "^2.1.1",
                    "lucide-react": "^0.383.0",
                    "tailwind-merge": "^2.3.0"
                },
                "devDependencies": {
                    "@types/node": "^20",
                    "@types/react": "^18",
                    "@types/react-dom": "^18",
                    "autoprefixer": "^10.0.1",
                    "postcss": "^8",
                    "tailwindcss": "^3.4.1",
                    "typescript": "^5"
                }
            }, indent=2))
            self._emit("  📄 Written", "frontend/package.json")

        # ── frontend/next.config.js ───────────────────────────────────
        next_cfg = str(Path(output_dir) / "frontend" / "next.config.js")
        if not file_exists(next_cfg):
            from tools.file_tools import create_file
            create_file(next_cfg, (
                "/** @type {import('next').NextConfig} */\n"
                "const nextConfig = {};\n"
                "module.exports = nextConfig;\n"
            ))

        # ── frontend/tsconfig.json ────────────────────────────────────
        ts_cfg = str(Path(output_dir) / "frontend" / "tsconfig.json")
        if not file_exists(ts_cfg):
            from tools.file_tools import create_file
            create_file(ts_cfg, json.dumps({
                "compilerOptions": {
                    "target": "es5",
                    "lib": ["dom", "dom.iterable", "esnext"],
                    "allowJs": True,
                    "skipLibCheck": True,
                    "strict": True,
                    "noEmit": True,
                    "esModuleInterop": True,
                    "module": "esnext",
                    "moduleResolution": "bundler",
                    "resolveJsonModule": True,
                    "isolatedModules": True,
                    "jsx": "preserve",
                    "incremental": True,
                    "plugins": [{"name": "next"}],
                    "paths": {"@/*": ["./*"]}
                },
                "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
                "exclude": ["node_modules"]
            }, indent=2))

        # ── frontend/tailwind.config.ts ───────────────────────────────
        tw_cfg = str(Path(output_dir) / "frontend" / "tailwind.config.ts")
        if not file_exists(tw_cfg):
            from tools.file_tools import create_file
            create_file(tw_cfg, (
                'import type { Config } from "tailwindcss";\n\n'
                'const config: Config = {\n'
                '  content: [\n'
                '    "./pages/**/*.{js,ts,jsx,tsx,mdx}",\n'
                '    "./components/**/*.{js,ts,jsx,tsx,mdx}",\n'
                '    "./app/**/*.{js,ts,jsx,tsx,mdx}",\n'
                '  ],\n'
                '  theme: { extend: {} },\n'
                '  plugins: [],\n'
                '};\n'
                'export default config;\n'
            ))

        # ── frontend/.env.local ───────────────────────────────────────
        env_local = str(Path(output_dir) / "frontend" / ".env.local")
        if not file_exists(env_local):
            from tools.file_tools import create_file
            create_file(env_local, "NEXT_PUBLIC_API_URL=http://localhost:8000\n")

        self._emit("  📄 Config files ready", "")

    # ------------------------------------------------------------------
    # QA fix helpers (unchanged from v1)
    # ------------------------------------------------------------------

    def _run_qa_fixes(
        self,
        qa_report: dict,
        plan_json: str,
        output_dir: str,
    ) -> None:
        backend_fix  = qa_report.get("backend_fix_instructions", "")
        frontend_fix = qa_report.get("frontend_fix_instructions", "")

        if backend_fix:
            self._emit("🔧 Sending fixes to Backend Agent", "")
            self._run_agent(
                agent=self.backend,
                sender="orchestrator", receiver="backend",
                task_type=TaskType.FIX,
                payload={
                    "plan_json":   plan_json,
                    "output_dir":  output_dir,
                    "fix_context": self._build_fix_context(
                        instructions=backend_fix,
                        output_dir=output_dir,
                        prefix="backend",
                        issues=qa_report.get("issues", []),
                    ),
                },
                step_label="🔧 Backend fixing errors",
            )

        if frontend_fix:
            self._emit("🔧 Sending fixes to Frontend Agent", "")
            self._run_agent(
                agent=self.frontend,
                sender="orchestrator", receiver="frontend",
                task_type=TaskType.FIX,
                payload={
                    "plan_json":   plan_json,
                    "output_dir":  output_dir,
                    "fix_context": self._build_fix_context(
                        instructions=frontend_fix,
                        output_dir=output_dir,
                        prefix="frontend",
                        issues=qa_report.get("issues", []),
                    ),
                },
                step_label="🔧 Frontend fixing errors",
            )

        if not backend_fix and not frontend_fix:
            self._emit("⚠️  QA gave no fix instructions", "Stopping fix loop")

    # ------------------------------------------------------------------
    # Unchanged helpers from v1
    # ------------------------------------------------------------------

    def _run_agent(self, agent, sender, receiver, task_type, payload, step_label) -> AgentMessage:
        self._emit(step_label, "")
        message = AgentMessage(sender=sender, receiver=receiver,
                               task_type=task_type, payload=payload)
        t       = time.perf_counter()
        message = agent.run(message)
        elapsed = round(time.perf_counter() - t, 1)
        self._log.append(message.summary() + f"  [{elapsed}s]")
        self.progress_cb(message.summary(), f"Took {elapsed}s")
        if message.status == TaskStatus.FAILED:
            logger.error("Agent %s failed: %s", receiver, message.error)
        else:
            logger.info("Agent %s done in %.1fs", receiver, elapsed)
        return message

    def _build_fix_context(self, instructions, output_dir, prefix, issues,
                           max_file_chars=3000, max_files=8) -> str:
        all_files  = list_files(output_dir)
        agent_files = [f for f in all_files if f.replace("\\", "/").startswith(f"{prefix}/")]
        issue_files: list[str] = []
        for issue in issues:
            f = issue.get("file", "").replace("\\", "/")
            if f.startswith(f"{prefix}/") and f not in issue_files:
                issue_files.append(f)
        if prefix == "backend":
            priority = ["backend/main.py", "backend/auth.py", "backend/routers/auth.py"]
        else:
            priority = ["frontend/app/layout.tsx", "frontend/lib/api.ts", "frontend/types/index.ts"]
            for issue_file in issue_files:
                component_name = Path(issue_file).stem
                for candidate in all_files:
                    if not candidate.replace("\\", "/").startswith("frontend/"):
                        continue
                    if candidate in priority or candidate in issue_files:
                        continue
                    try:
                        src = read_file(str(Path(output_dir) / candidate))
                        if component_name in src:
                            priority.append(candidate)
                    except Exception:
                        pass
        ordered: list[str] = []
        for f in priority + issue_files + agent_files:
            if f not in ordered:
                ordered.append(f)
        sections: list[str] = []
        for rel_path in ordered[:max_files]:
            raw = read_file(str(Path(output_dir) / rel_path))
            if raw.startswith("[Error]") or raw.startswith("[Skipped]"):
                continue
            truncated = raw[:max_file_chars]
            if len(raw) > max_file_chars:
                truncated += f"\n... [truncated]"
            sections.append(f"=== {rel_path} ===\n{truncated}")
        return (
            f"INSTRUCTIONS:\n{instructions}\n\n"
            f"CURRENT FILE CONTENTS:\n" + "\n\n".join(sections)
        )

    def _run_parallel(self, plan_json, output_dir):
        use_parallel = getattr(cfg, "parallel_generation", True)
        def _back(): return self._run_agent(self.backend, "orchestrator", "backend",
            TaskType.BACKEND, {"plan_json": plan_json, "output_dir": output_dir},
            "⚙️  Generating backend code")
        def _front(): return self._run_agent(self.frontend, "orchestrator", "frontend",
            TaskType.FRONTEND, {"plan_json": plan_json, "output_dir": output_dir},
            "🎨 Generating frontend code")
        if not use_parallel:
            return _back(), _front()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fb, ff = ex.submit(_back), ex.submit(_front)
            return fb.result(), ff.result()

    def _failed(self, reason, t_start):
        logger.error("Pipeline failed: %s", reason)
        return PipelineResult(
            success=False, project_name="unknown", output_dir="",
            errors=[reason], duration_s=round(time.perf_counter() - t_start, 1),
            message_log=list(self._log),
        )

    def _emit(self, step, detail):
        entry = f"{step}: {detail}" if detail else step
        self._log.append(entry)
        logger.info(entry)
        self.progress_cb(step, detail)

    @staticmethod
    def _default_progress(step, detail):
        print(f"  {step}  {detail}" if detail else f"  {step}")