# Multi-Agent Team Framework

Send a single requirement — a full working codebase comes out the other side.

```
"create a website for rental books"
              │
              ▼
    ┌─────────────────────┐
    │    ORCHESTRATOR     │  coordinates everything
    └──┬──────┬──────┬────┘
       │      │      │
       ▼      ▼      ▼
   PLANNER BACKEND FRONTEND   each is a separate LLM call
       │      │      │        with a specialist prompt
       └──────┴──────┘
                │
                ▼
             QA AGENT         syntax check + LLM review
                │
         pass? │ fail?
               ▼
          fix loop (max 3x)
               │
               ▼
        output/rental_books/
          ├── backend/
          │   ├── main.py
          │   ├── models/
          │   ├── routers/
          │   └── ...
          └── frontend/
              ├── index.html
              ├── js/
              └── css/
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your .env
cp .env.example .env
# Edit .env — set AGENT_PROVIDER and your API key

# 3. Run
python main.py "create a website for rental books"
```

---

## Project Structure

```
multi_agent_team/
├── main.py                    ← CLI entry point
├── orchestrator.py            ← Pipeline coordinator
├── config.py                  ← Config + .env loading
│
├── agents/
│   ├── base_agent.py          ← Abstract base (LLM chat, JSON extract)
│   ├── planner_agent.py       ← Requirement → ProjectPlan JSON
│   ├── backend_agent.py       ← ProjectPlan → FastAPI backend code
│   ├── frontend_agent.py      ← ProjectPlan → HTML/JS frontend code
│   └── qa_agent.py            ← Syntax check + LLM code review
│
├── models/
│   ├── messages.py            ← AgentMessage, TaskStatus dataclasses
│   └── project_plan.py        ← ProjectPlan, ApiEndpoint, etc.
│
├── tools/
│   ├── file_tools.py          ← create_file, read_file, list_files
│   └── shell_tools.py         ← run_command, syntax_check_python
│
├── providers/
│   └── __init__.py            ← get_provider() factory (all 5 providers)
│
├── output/                    ← Generated projects land here
├── .env                       ← Your secrets (never commit)
├── .env.example               ← Template to share
└── requirements.txt
```

---

## Pipeline Flow

```
User requirement
    │
    ▼
[1] PlannerAgent
    • Sends requirement to LLM
    • Gets back ProjectPlan JSON:
      - tech stack, folder structure
      - database models, API endpoints
      - ordered task list
    │
    ▼
[2] BackendAgent
    • Reads ProjectPlan
    • Generates: main.py, models, schemas,
                 routers, auth, requirements.txt
    • Writes files to output/<project>/backend/
    │
    ▼
[3] FrontendAgent
    • Reads ProjectPlan + API contract
    • Generates: HTML pages, JS (fetch API),
                 CSS (TailwindCSS via CDN)
    • Writes files to output/<project>/frontend/
    │
    ▼
[4] QAAgent
    • Runs py_compile on all .py files
    • LLM reviews all files for logic errors,
      broken references, missing imports
    • Returns pass/fail + fix instructions
    │
    ├── PASS ──► Done ✅
    │
    └── FAIL ──► BackendAgent + FrontendAgent fix
                 Re-run QA (max 3 attempts)
```

---

## Provider Setup

Set `AGENT_PROVIDER` in `.env`:

| Provider | ENV key needed | Example model |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o` |
| `openrouter` | `OPENROUTER_API_KEY` | `mistralai/mistral-large` |
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| `ollama` | *(none)* | `llama3.3` |

---

## Running the Generated Project

After the pipeline completes:

```bash
cd output/<project_name>/backend

# Install backend dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload
# API available at http://localhost:8000
# Docs at       http://localhost:8000/docs

# Open the frontend
# Just open output/<project_name>/frontend/index.html in your browser
```

---

## Example

```bash
$ python main.py "create a website for rental books"

  🚀 Starting pipeline  create a website for rental books
  📋 Planning project
  ✅ [PLAN] orchestrator → planner | status=done  [8.3s]
  📁 Output directory  output/rental_book_website
  ⚙️  Generating backend code
  ✅ [BACKEND] orchestrator → backend | status=done  [22.1s]
  🎨 Generating frontend code
  ✅ [FRONTEND] orchestrator → frontend | status=done  [18.4s]
  🔍 QA check (attempt 1/3)
  ✅ QA passed  All files pass syntax check and code review
  🎉 Pipeline complete  14 files in 51.2s → output/rental_book_website

  ┌─────────────────────────────┐
  │  ✅ SUCCESS                 │
  │  Project: rental_book_website │
  │  Files:   14                │
  │  Time:    51.2s             │
  └─────────────────────────────┘

Next steps:
  cd output/rental_book_website/backend
  pip install -r requirements.txt
  uvicorn main:app --reload
```
