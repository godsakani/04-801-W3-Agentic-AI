"""
Role Output Models — Structured I/O for role-separated nodes.

Each node in the Planner → Executor → Critic pipeline produces
a typed output that the next node consumes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlanOutput:
    """Structured output from the PlannerNode."""
    action: str                         # "tool_call", "final_answer", "retrieve"
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    reasoning: str = ""


@dataclass
class ExecutionResult:
    """Structured output from the ExecutorNode."""
    success: bool
    output: str = ""
    error: Optional[str] = None
    tool_used: Optional[str] = None
    params_used: Optional[dict] = None


@dataclass
class CriticOutput:
    """Structured output from the CriticNode."""
    should_continue: bool
    confidence: str = "medium"          # "low", "medium", "high"
    groundedness_score: float = 0.0
    feedback: str = ""
    recommendation: str = "proceed"     # "proceed", "re_plan", "re_retrieve", "escalate", "clarify"
