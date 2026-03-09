# =============================================================================
# agents/__init__.py
# -----------------------------------------------------------------------------
# Re-exports all specialist agent classes so the Orchestrator and other
# modules can import them with a single clean line:
#
#   from agents import PlannerAgent, BackendAgent, FrontendAgent, QAAgent
#
# Each agent inherits from BaseAgent (agents/base_agent.py) and implements:
#   - name   : str     — agent identifier
#   - system : str     — LLM system prompt
#   - run()  : method  — receives AgentMessage, returns AgentMessage
# =============================================================================

from agents.planner_agent  import PlannerAgent   # requirement → ProjectPlan JSON
from agents.backend_agent  import BackendAgent   # ProjectPlan → FastAPI code
from agents.frontend_agent import FrontendAgent  # ProjectPlan → Next.js code
from agents.qa_agent       import QAAgent        # syntax check + tsc + LLM review

__all__ = ["PlannerAgent", "BackendAgent", "FrontendAgent", "QAAgent"]