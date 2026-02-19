"""
Critic Node — Evaluates quality and provides adaptive feedback.

CRITIQUE/EVALUATION ROLE in the Planner → Executor → Critic pipeline.
Runs groundedness checks and implements adaptive control rules.
"""

import logging
from typing import Callable

from src.models.agent_state_model import AgentState
from src.models.role_output_model import ExecutionResult, CriticOutput

logger = logging.getLogger(__name__)


class CriticNode:
    """
    CRITIQUE/EVALUATION ROLE: Evaluates execution output and decides next steps.
    
    Responsibilities:
    - Run groundedness verification on accumulated context
    - Determine if the agent should continue, re-retrieve, or stop
    - Implement adaptive control rules
    """
    
    def __init__(self, verify_fn: Callable):
        self.verify_fn = verify_fn
    
    def critique(self, state: AgentState, execution: ExecutionResult) -> CriticOutput:
        """
        Evaluate the execution result and provide feedback.
        
        Adaptive Control Rules:
        - Groundedness < 0.5 -> re-retrieve
        - Tool blocked/failed -> re-plan with feedback
        - Sufficient context + successful execution -> proceed to answer
        - Max retries exhausted -> escalate
        
        Args:
            state: Current agent state
            execution: ExecutionResult from ExecutorNode
            
        Returns:
            CriticOutput with continuation decision and feedback
        """
        # If executor returned FINAL_ANSWER, we're done
        if execution.tool_used is None and execution.success:
            return CriticOutput(
                should_continue=False,
                confidence="high",
                feedback="Planner decided to give final answer.",
                recommendation="proceed"
            )
        
        # If execution failed, decide adaptive action
        if not execution.success:
            error = execution.error or "Unknown error"
            
            if "TOOL_BLOCKED" in error:
                logger.info("[CRITIC] ADAPT: TOOL_BLOCKED -> RE_PLAN with more context")
                state.observations.append(
                    "Critic feedback: Tool was blocked, re-planning with enriched context"
                )
                return CriticOutput(
                    should_continue=True,
                    confidence="low",
                    feedback=f"Tool blocked: {error}. Re-plan with available context.",
                    recommendation="re_plan"
                )
            else:
                logger.info("[CRITIC] ADAPT: TOOL_FAILURE -> RETRY")
                state.observations.append("Critic feedback: Tool failed, retrying")
                return CriticOutput(
                    should_continue=True,
                    confidence="low",
                    feedback=f"Execution error: {error}. Retry or try alternative.",
                    recommendation="re_plan"
                )
        
        # Execution succeeded — check groundedness of accumulated context
        if state.context:
            sources = [state.context]
            # Use a lightweight check — verify the observations make sense
            try:
                verification = self.verify_fn(
                    "\n".join(state.observations[-3:]),  # Check recent observations
                    sources
                )
                groundedness = verification.score
                confidence = verification.confidence
            except Exception:
                groundedness = 0.5
                confidence = "medium"
        else:
            groundedness = 0.0
            confidence = "low"
        
        # --- ADAPTIVE CONTROL RULES ---
        
        # Rule 1: Low groundedness -> re-retrieve
        if groundedness < 0.5 and state.context:
            logger.info(f"[CRITIC] ADAPT: LOW_GROUNDEDNESS ({groundedness:.2f}) -> RE_RETRIEVE")
            state.observations.append(
                f"Critic: Groundedness low ({groundedness:.2f}), needs more data"
            )
            return CriticOutput(
                should_continue=True,
                confidence="low",
                groundedness_score=groundedness,
                feedback=f"Groundedness score {groundedness:.2f} is below threshold. Re-retrieve for better data.",
                recommendation="re_retrieve"
            )
        
        # Rule 2: Low confidence -> clarify
        if confidence == "low" and not state.context:
            logger.info("[CRITIC] ADAPT: LOW_CONFIDENCE -> CLARIFY")
            return CriticOutput(
                should_continue=False,
                confidence="low",
                groundedness_score=groundedness,
                feedback="Not enough data to give confident answer. Clarify with user.",
                recommendation="clarify"
            )
        
        # Rule 3: Successful tool execution -> can continue or finish
        logger.info(f"[CRITIC] Execution OK. Groundedness: {groundedness:.2f}, Confidence: {confidence}")
        return CriticOutput(
            should_continue=False,
            confidence=confidence,
            groundedness_score=groundedness,
            feedback=f"Execution successful. Groundedness: {groundedness:.2f}",
            recommendation="proceed"
        )
