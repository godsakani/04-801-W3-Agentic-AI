"""
Orchestrator — ReAct-style reasoning loop.

Coordinates the Planner → Executor → Critic pipeline with
adaptive control and persistent memory integration.
"""

import os
import uuid
import logging
from typing import Callable, List

from langchain_openai import ChatOpenAI

from src.models.agent_state_model import AgentState
from src.models.role_output_model import PlanOutput, ExecutionResult, CriticOutput
from src.nodes.planner import PlannerNode
from src.nodes.executor import ExecutorNode
from src.nodes.critic import CriticNode
from src.memory.agent_memory import PersistentMemory
from src.verification.groundedness import GroundednessResult, handle_verification

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct-style orchestrator for alumni tracking.
    
    Uses structured role separation:
    - PlannerNode: Decides what action to take (REASON + DECIDE)
    - ExecutorNode: Validates and executes actions (VALIDATE + ACT)
    - CriticNode: Evaluates results and provides feedback (EVALUATE + UPDATE)
    
    Loop: Planner → Executor → Critic → (repeat or finish)
    
    Adaptive Control:
    - LOW_GROUNDEDNESS → re-retrieve
    - TOOL_BLOCKED → re-plan with feedback
    - TOOL_FAILURE → retry with fallback
    - MAX_ITERATIONS → escalate
    - LOW_CONFIDENCE → clarify
    """
    
    def __init__(
        self,
        retrieval_fn: Callable,
        tools: dict,
        verify_fn: Callable,
        memory: PersistentMemory = None,
        max_iterations: int = 5
    ):
        self.retrieval_fn = retrieval_fn
        self.tools = tools
        self.verify_fn = verify_fn
        self.memory = memory
        self.max_iterations = max_iterations
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-2024-08-06",
            temperature=0,
            base_url='https://ai-gateway.andrew.cmu.edu/',
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Bind tools to LLM for native function calling
        tool_list = [t for t in self.tools.values() if hasattr(t, 'name')]
        self.llm_with_tools = self.llm.bind_tools(tool_list)
        
        # Initialize role-separated nodes
        self.planner = PlannerNode(self.llm_with_tools)
        self.executor = ExecutorNode(self.tools, self.retrieval_fn)
        self.critic = CriticNode(self.verify_fn)
        
        logger.info("ReActAgent initialized with role separation: Planner → Executor → Critic")
    
    def run(self, query: str, initial_observation: str = None) -> dict:
        """
        Execute the orchestrator loop: Planner → Executor → Critic → repeat.
        
        Args:
            query: User query or trigger
            initial_observation: Optional trigger (e.g., "LinkedIn change detected")
            
        Returns:
            {
                "response": final response,
                "verification": groundedness result,
                "trace": list of steps with role labels
            }
        """
        state = AgentState(query=query, session_id=str(uuid.uuid4())[:8])
        
        if initial_observation:
            state.observations.append(initial_observation)
        
        trace = []
        
        # ========== MEMORY READ: Load cross-session context ==========
        if self.memory:
            try:
                recent_sessions = self.memory.get_recent_sessions(limit=5)
                task_history = self.memory.get_task_history(query)
                memory_context = self.memory.format_memory_context(recent_sessions)
                state.memory_context = memory_context
                if memory_context:
                    logger.info(f"[ORCHESTRATOR] Loaded memory: {len(recent_sessions)} sessions")
                    trace.append({
                        "phase": "MEMORY_READ", "role": "ORCHESTRATOR",
                        "sessions_loaded": len(recent_sessions),
                        "task_history_matches": len(task_history),
                        "iteration": 0
                    })
                # Prune old sessions periodically
                self.memory.prune_old_sessions(max_age_days=30)
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Memory read failed: {e}")
        
        # ========== PHASE 0: AUTO-RETRIEVE ==========
        logger.info("[ORCHESTRATOR] Phase 0: Initial retrieval")
        docs = self.retrieval_fn(query)
        if docs:
            new_context = "\n\n".join([doc.page_content for doc in docs])
            state.context = new_context
            state.observations.append(f"Initial retrieval: {len(docs)} documents")
            state.actions.append("RETRIEVAL_MODULE")
            trace.append({
                "phase": "RETRIEVE", "role": "EXECUTOR",
                "result": f"Retrieved {len(docs)} documents",
                "iteration": 0
            })
        else:
            state.observations.append("Initial retrieval returned no documents")
        
        # ========== MAIN ORCHESTRATOR LOOP ==========
        while state.iteration < self.max_iterations:
            state.iteration += 1
            logger.info(f"[ORCHESTRATOR] === Iteration {state.iteration}/{self.max_iterations} ===")
            
            # ---- STEP 1: PLANNER (Reason + Decide) ----
            plan = self.planner.plan(state)
            trace.append({
                "phase": "PLAN", "role": "PLANNER",
                "action": plan.action,
                "tool": plan.tool_name,
                "reasoning": plan.reasoning[:200],
                "iteration": state.iteration
            })
            logger.info(f"[ORCHESTRATOR] Planner decided: {plan.action}" + 
                       (f" → {plan.tool_name}" if plan.tool_name else ""))
            
            # ---- STEP 2: EXECUTOR (Validate + Act) ----
            execution = self.executor.execute(plan, state)
            trace.append({
                "phase": "EXECUTE", "role": "EXECUTOR",
                "success": execution.success,
                "tool_used": execution.tool_used,
                "output": execution.output[:200] if execution.output else None,
                "error": execution.error,
                "iteration": state.iteration
            })
            
            if execution.success:
                state.observations.append(f"Executed {execution.tool_used}: {execution.output[:100]}")
                if execution.tool_used and execution.tool_used != "RETRIEVAL_MODULE":
                    state.actions.append(f"TOOL_MODULE:{execution.tool_used}")
            else:
                state.observations.append(f"Execution failed: {execution.error}")
            
            # ---- STEP 3: CRITIC (Evaluate + Feedback) ----
            critique = self.critic.critique(state, execution)
            trace.append({
                "phase": "CRITIQUE", "role": "CRITIC",
                "should_continue": critique.should_continue,
                "confidence": critique.confidence,
                "groundedness": critique.groundedness_score,
                "recommendation": critique.recommendation,
                "feedback": critique.feedback[:200],
                "iteration": state.iteration
            })
            logger.info(f"[ORCHESTRATOR] Critic: continue={critique.should_continue}, "
                       f"confidence={critique.confidence}, rec={critique.recommendation}")
            
            # ---- STEP 4: ADAPTIVE CONTROL ----
            if not critique.should_continue:
                if critique.recommendation == "clarify":
                    logger.info("[ORCHESTRATOR] ADAPT: LOW_CONFIDENCE → CLARIFY")
                    trace.append({
                        "phase": "ADAPT", "role": "ORCHESTRATOR",
                        "action": "CLARIFY",
                        "iteration": state.iteration
                    })
                break
            
            # Adaptive: re-retrieve if critic says so
            if critique.recommendation == "re_retrieve":
                logger.info("[ORCHESTRATOR] ADAPT: RE_RETRIEVE triggered by Critic")
                docs = self.retrieval_fn(state.query)
                if docs:
                    new_context = "\n\n".join([doc.page_content for doc in docs])
                    state.context += "\n\n" + new_context
                    state.observations.append(f"Adaptive re-retrieval: {len(docs)} docs")
                    state.actions.append("ADAPTIVE_RETRIEVAL")
                trace.append({
                    "phase": "ADAPT", "role": "ORCHESTRATOR",
                    "action": "RE_RETRIEVE",
                    "result": f"Retrieved {len(docs)} additional documents",
                    "iteration": state.iteration
                })
        
        # ========== ESCALATION CHECK ==========
        if state.iteration >= self.max_iterations:
            logger.warning("[ORCHESTRATOR] ADAPT: MAX_ITERATIONS → ESCALATE")
            trace.append({
                "phase": "ADAPT", "role": "ORCHESTRATOR",
                "action": "ESCALATE",
                "reason": f"Reached max iterations ({self.max_iterations})",
                "iteration": state.iteration
            })
        
        # ========== GENERATE FINAL RESPONSE ==========
        response_text = self._generate_response(state)
        logger.info(f"[ORCHESTRATOR] Response generated ({len(response_text)} chars)")
        
        # ========== FINAL VERIFICATION ==========
        logger.info("[ORCHESTRATOR] Running final groundedness verification")
        sources = [state.context] if state.context else []
        verification = self.verify_fn(response_text, sources)
        
        trace.append({
            "phase": "VERIFY", "role": "CRITIC",
            "score": verification.score,
            "confidence": verification.confidence,
            "recommendation": verification.recommendation
        })
        
        final_response = handle_verification(verification, response_text)
        
        # ========== MEMORY WRITE: Save session to persistent memory ==========
        if self.memory:
            try:
                self.memory.save_session(
                    session_id=state.session_id,
                    query=query,
                    response=final_response,
                    tools_used=state.actions,
                    metrics={
                        "groundedness_score": verification.score,
                        "confidence": verification.confidence,
                        "iterations": state.iteration,
                        "recommendation": verification.recommendation
                    },
                    trace_summary=f"{len(trace)} steps, final confidence: {verification.confidence}"
                )
                trace.append({
                    "phase": "MEMORY_WRITE", "role": "ORCHESTRATOR",
                    "session_id": state.session_id,
                    "data_saved": "session record with metrics"
                })
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Memory write failed: {e}")
        
        return {
            "response": final_response,
            "verification": verification,
            "trace": trace,
            "session_id": state.session_id,
            "iterations": state.iteration,
            "actions": state.actions
        }
    
    def _generate_response(self, state: AgentState) -> str:
        """Generate final response from accumulated context."""
        prompt = f"""Based on the following context and actions, generate a helpful response to the user's query.

Query: {state.query}

Retrieved Context:
{state.context[:2000] if state.context else 'No specific context available'}

Actions Taken: {state.actions}
Observations: {state.observations}

Generate a clear, accurate, and helpful response. Only include information that is supported by the context."""

        result = self.llm.invoke(prompt)
        return result.content
