"""
Agent State Model — Tracks agent state across loop iterations.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AgentState:
    """Tracks agent state across loop iterations."""
    query: str
    context: str = ""
    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    iteration: int = 0
    current_alumni_id: Optional[str] = None
    session_id: str = ""
    memory_context: str = ""
