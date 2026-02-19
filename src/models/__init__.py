"""
Models Package — Data structures for the Alumni RAG Agent.

Provides AgentState for session tracking and structured I/O
dataclasses for role-separated nodes.
"""

from src.models.agent_state_model import AgentState
from src.models.role_output_model import PlanOutput, ExecutionResult, CriticOutput

__all__ = [
    "AgentState",
    "PlanOutput",
    "ExecutionResult",
    "CriticOutput",
]
