"""
agents/frontend_agent.py
------------------------
The Frontend Agent reads the ProjectPlan and generates a complete
Next.js 14 (App Router) + TypeScript + Tailwind CSS + shadcn/ui frontend.

Key rules enforced in prompts:
- 'use client' only on components that use hooks / browser APIs
- JWT stored in httpOnly cookie (set by backend), never localStorage
- next/navigation for routing (useRouter, redirect)
- NEXT_PUBLIC_ prefix for client-side env vars
- Server Components by default, Client Components only when needed
- Fetch wrapper in lib/api.ts handles auth via cookie (credentials: include)
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents.base_agent import BaseAgent
from models.messages import AgentMessage
from models.project_plan import ProjectPlan
from tools.file_tools import create_file

logger = logging.getLogger("agent.frontend")

# ---------------------------------------------------------------------------
# Few-shot examples embedded in the system prompt.
# Concrete patterns dramatically reduce LLM mistakes.
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLES = r"""
## EXAMPLE 1 — lib/api.ts (fetch wrapper — always follow this pattern)
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: "include",
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json() as Promise<T>;
}

## EXAMPLE 2 — Server Component page (NO 'use client')
// app/(dashboard)/books/page.tsx
import { apiFetch } from "@/lib/api";
import { Book } from "@/types";

export default async function BooksPage() {
  const books = await apiFetch<Book[]>("/api/books");
  return (
    <main className="p-6">
      <h1 className="text-2xl font-bold mb-4">Books</h1>
      <ul className="space-y-2">
        {books.map((b) => <li key={b.id} className="border rounded p-3">{b.title}</li>)}
      </ul>
    </main>
  );
}

## EXAMPLE 3 — Client Component login form (MUST have 'use client')
// app/(auth)/login/page.tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState("");
  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    try {
      await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email: form.get("email"), password: form.get("password") }),
      });
      router.push("/");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed");
    }
  }
  return (
    <main className="flex min-h-screen items-center justify-center">
      <form onSubmit={handleSubmit} className="space-y-4 w-80">
        <h1 className="text-2xl font-bold">Login</h1>
        {error && <p className="text-red-500 text-sm">{error}</p>}
        <input name="email" type="email" placeholder="Email"
          className="w-full border rounded px-3 py-2" required />
        <input name="password" type="password" placeholder="Password"
          className="w-full border rounded px-3 py-2" required />
        <button type="submit"
          className="w-full bg-blue-600 text-white rounded py-2 hover:bg-blue-700">
          Login
        </button>
      </form>
    </main>
  );
}

## EXAMPLE 4 — next.config.js (API rewrite to FastAPI backend)
/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://localhost:8000/api/:path*" }];
  },
};
module.exports = nextConfig;

## EXAMPLE 5 — package.json (exact versions — do not change)
{
  "name": "frontend", "version": "0.1.0", "private": true,
  "scripts": {
    "dev": "next dev", "build": "next build",
    "start": "next start", "lint": "next lint",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "next": "14.2.29", "react": "^18", "react-dom": "^18",
    "class-variance-authority": "^0.7.0", "clsx": "^2.1.1",
    "lucide-react": "^0.383.0", "tailwind-merge": "^2.3.0"
  },
  "devDependencies": {
    "@types/node": "^20", "@types/react": "^18", "@types/react-dom": "^18",
    "autoprefixer": "^10.0.1", "eslint": "^8", "eslint-config-next": "14.2.29",
    "postcss": "^8", "tailwindcss": "^3.4.1", "typescript": "^5"
  }
}
"""


class FrontendAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "frontend"

    @property
    def system(self) -> str:
        return f"""You are a senior Next.js engineer specialising in
Next.js 14 App Router, TypeScript, Tailwind CSS, and shadcn/ui.
You generate COMPLETE, PRODUCTION-READY frontend code.

Output a JSON array of file objects:
[
  {{"path": "relative/path/to/file.tsx", "content": "full file content"}},
  ...
]

ABSOLUTE RULES — never break these:

1. OUTPUT: ONLY the JSON array. No prose, no markdown fences. Every file 100% complete.

2. use client RULE (most common mistake):
   - Default = Server Component = NO 'use client'.
   - Add 'use client' ONLY when using: useState, useEffect, useCallback,
     useRef, useContext, useRouter, event handlers, or browser APIs.
   - async/await data fetch in a page = Server Component = NO 'use client'.

3. AUTH — httpOnly COOKIE ONLY:
   - NEVER localStorage or sessionStorage for tokens.
   - Backend sets JWT as httpOnly cookie named "access_token".
   - Always pass credentials:"include" in fetch — cookie sent automatically.
   - Logout = POST /api/auth/logout which clears the cookie.

4. ROUTING:
   - next/navigation: useRouter(), redirect(), notFound().
   - NEVER window.location.href.
   - next/link <Link> for all internal navigation.

5. ENV VARS:
   - Client-side vars MUST start with NEXT_PUBLIC_.
   - API base via lib/api.ts only — never hard-code URLs in components.

6. IMAGES: next/image <Image> instead of <img>.

7. PACKAGE.JSON: Always Next.js 14.2.29 exactly. Include "type-check": "tsc --noEmit".
   next.config MUST be next.config.js (NOT .ts) — Next.js 14 does not support next.config.ts.

8. STYLING: Tailwind utility classes only. Mobile-first responsive design.

CODE PATTERNS TO FOLLOW EXACTLY:
{_FEW_SHOT_EXAMPLES}
"""

    def run(self, message: AgentMessage) -> AgentMessage:
        message.mark_running()

        plan_json   = message.payload.get("plan_json", "{}")
        output_dir  = message.payload.get("output_dir", "output/project")
        fix_context = message.payload.get("fix_context", "")

        try:
            plan = ProjectPlan.from_json(plan_json)
        except Exception as exc:
            message.mark_failed(f"Could not parse plan: {exc}")
            return message

        logger.info("Frontend agent generating Next.js app for: %s", plan.project_name)

        prompt = self._fix_prompt(plan, fix_context) if fix_context \
            else self._generate_prompt(plan)

        # ── Attempt 1: normal ────────────────────────────────────────
        written: list[str] = []
        try:
            raw   = self.chat(prompt, max_tokens=16384)
            files = self.extract_json(raw)
            written = self._write_files(files, output_dir)
            message.mark_done(self._summary(written))
            return message
        except Exception as exc:
            logger.warning("Frontend attempt 1 failed (%s) — retrying", exc)

        # ── Attempt 2: force JSON-only prompt ────────────────────────
        try:
            retry_prompt = self._json_only_prompt(plan, fix_context)
            raw   = self.chat(retry_prompt, max_tokens=16384)
            files = self.extract_json(raw)
            written = self._write_files(files, output_dir)
            message.mark_done(self._summary(written))
        except Exception as exc2:
            logger.error("Frontend agent failed: %s", exc2)
            if written:
                message.mark_done(self._summary(written) + f"\n[partial — error: {exc2}]")
            else:
                message.mark_failed(str(exc2))

        return message

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _generate_prompt(self, plan: ProjectPlan) -> str:
        endpoints_summary = "\n".join(
            f"  {e.method} {e.path} — {e.description} (auth_required={e.auth_required})"
            for e in plan.api_endpoints
        )
        return f"""Generate the COMPLETE Next.js 14 frontend for this project.

PROJECT PLAN:
{plan.to_json()}

BACKEND API ENDPOINTS:
{endpoints_summary}

Generate ALL of these files:

── Config & Setup ──────────────────────────────────────────────────
  frontend/package.json              (Next.js 14.2.29 — use example from system prompt)
  frontend/tsconfig.json             (strict: true, paths: {{"@/*": ["./*"]}})
  frontend/next.config.js            (API rewrite to http://localhost:8000 — must be .js not .ts)
  frontend/tailwind.config.ts        (content: ["./app/**/*.tsx","./components/**/*.tsx"])
  frontend/postcss.config.js         (tailwind + autoprefixer plugins)
  frontend/.env.local                (NEXT_PUBLIC_API_URL=http://localhost:8000)

── Types ───────────────────────────────────────────────────────────
  frontend/types/index.ts            (ALL TypeScript interfaces matching every DB model)

── Lib ─────────────────────────────────────────────────────────────
  frontend/lib/api.ts                (apiFetch with credentials:include — see example)
  frontend/lib/auth.ts               (getCurrentUser(), logout() helpers)

── App Router ──────────────────────────────────────────────────────
  frontend/app/globals.css           (@tailwind base; @tailwind components; @tailwind utilities;)
  frontend/app/layout.tsx            (root layout + NavBar + Inter font — NO use client)
  frontend/app/page.tsx              (home/landing — Server Component)
  frontend/app/(auth)/login/page.tsx        (login form — use client)
  frontend/app/(auth)/register/page.tsx     (register form — use client)
{self._dashboard_pages(plan)}

── Components ──────────────────────────────────────────────────────
  frontend/components/NavBar.tsx     (nav links — use client for mobile menu toggle)
{self._model_components(plan)}

Output the JSON array of file objects now:"""

    def _fix_prompt(self, plan: ProjectPlan, fix_context: str) -> str:
        return f"""The QA agent found issues in the Next.js frontend. Fix them.

PROJECT PLAN:
{plan.to_json()}

ISSUES TO FIX:
{fix_context}

CHECKLIST before fixing:
- 'use client' only when using hooks or event handlers
- JWT in httpOnly cookie — NEVER localStorage
- useRouter from next/navigation (not next/router)
- NEXT_PUBLIC_ prefix for all client-side env vars
- next/image instead of <img>
- next/link instead of <a> for internal links

Generate ONLY the corrected files as a JSON array.
Output the JSON array now:"""


    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _write_files(self, files, output_dir: str) -> list:
        if not isinstance(files, list):
            files = [files]
        written = []
        for f in files:
            if not isinstance(f, dict):
                continue
            rel_path = f.get("path", "")
            content  = f.get("content", "")
            if not rel_path or content is None:
                continue
            from pathlib import Path
            from tools.file_tools import create_file
            full_path = str(Path(output_dir) / rel_path)
            create_file(full_path, content)
            written.append(full_path)
            logger.info("Frontend wrote: %s", full_path)
        return written

    @staticmethod
    def _summary(written: list) -> str:
        return (
            "Next.js frontend generation complete.\n"
            f"Files written ({len(written)}):\n"
            + "\n".join(f"  - {p}" for p in written)
        )

    def _json_only_prompt(self, plan, fix_context: str) -> str:
        models_summary = ", ".join(m.name for m in plan.database_models) or "Item"
        context = f"Fix these errors:\n{fix_context}\n\n" if fix_context else ""
        return (
            f"{context}Output JSON array only. Start with [ end with ]. No text outside.\n\n"
            f"Generate a complete Next.js 14 frontend for {plan.project_name} ({models_summary}).\n"
            "ALL paths MUST start with 'frontend/' prefix.\n"
            "Include: frontend/package.json, frontend/tsconfig.json, frontend/next.config.js, "
            "frontend/tailwind.config.ts, frontend/postcss.config.js, frontend/.env.local, "
            "frontend/types/index.ts, frontend/lib/api.ts, frontend/lib/auth.ts, "
            "frontend/app/globals.css, frontend/app/layout.tsx, frontend/app/page.tsx, "
            "frontend/app/(auth)/login/page.tsx, frontend/app/(auth)/register/page.tsx, "
            "frontend/components/NavBar.tsx, frontend/app/(dashboard)/items/page.tsx.\n"
            "Rules: use client only on hook/event components; JWT in httpOnly cookie; "
            "apiFetch with credentials:include; NEXT_PUBLIC_API_URL env var.\n"
            "Output the complete JSON array now:"
        )

    @staticmethod
    def _dashboard_pages(plan: ProjectPlan) -> str:
        lines = []
        for model in plan.database_models:
            n  = model.name.lower()
            ns = n + "s"
            lines.append(f"  frontend/app/(dashboard)/{ns}/page.tsx          (list {ns} — Server Component)")
            lines.append(f"  frontend/app/(dashboard)/{ns}/[id]/page.tsx      (detail/edit — Server Component)")
            lines.append(f"  frontend/app/(dashboard)/{ns}/new/page.tsx       (create form — use client)")
        return "\n".join(lines) if lines else \
            "  frontend/app/(dashboard)/items/page.tsx  (main list — Server Component)"

    @staticmethod
    def _model_components(plan: ProjectPlan) -> str:
        lines = []
        for model in plan.database_models:
            n = model.name
            lines.append(f"  frontend/components/{n}Card.tsx       ({n} summary card — NO use client)")
            lines.append(f"  frontend/components/{n}Form.tsx       (create/edit form — use client)")
        return "\n".join(lines) if lines else \
            "  frontend/components/ItemCard.tsx   (item card component)"