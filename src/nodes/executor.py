"""
Executor Node — Validates prerequisites and executes actions.

EXECUTION ROLE in the Planner → Executor → Critic pipeline.
Handles tool parameter validation, tool execution, and fallback retrieval.
"""

import logging
from typing import Callable

from src.models.agent_state_model import AgentState
from src.models.role_output_model import PlanOutput, ExecutionResult
from src.utils.tool_validators import validate_tool_params

logger = logging.getLogger(__name__)


class ExecutorNode:
    """
    EXECUTION ROLE: Validates prerequisites and executes the planner's decision.
    
    Responsibilities:
    - Validate tool parameters against TOOL_PREREQUISITES
    - Execute tool calls via LangChain tool.invoke()
    - Handle execution errors gracefully
    - Perform fallback retrieval when validation fails
    """
    
    def __init__(self, tools: dict, retrieval_fn: Callable):
        self.tools = tools
        self.retrieval_fn = retrieval_fn
    
    def execute(self, plan: PlanOutput, state: AgentState) -> ExecutionResult:
        """
        Execute the planner's decision after validation.
        
        Args:
            plan: PlanOutput from PlannerNode
            state: Current agent state
            
        Returns:
            ExecutionResult with success status, output, and any errors
        """
        if plan.action == "FINAL_ANSWER":
            return ExecutionResult(
                success=True,
                output=plan.reasoning,
                tool_used=None
            )
        
        if plan.action == "RETRIEVE":
            return self._execute_retrieval(state)
        
        if plan.action == "TOOL_CALL" and plan.tool_name:
            return self._execute_tool(plan, state)
        
        return ExecutionResult(
            success=False,
            error=f"Unknown action: {plan.action}"
        )
    
    def _execute_tool(self, plan: PlanOutput, state: AgentState) -> ExecutionResult:
        """Execute a tool call with prerequisite validation."""
        tool_name = plan.tool_name
        tool_args = plan.tool_args or {}
        
        # Check if tool exists
        if tool_name not in self.tools:
            return ExecutionResult(
                success=False,
                error=f"Unknown tool: {tool_name}",
                tool_used=tool_name
            )
        
        # --- PREREQUISITE VALIDATION ---
        has_context = bool(state.context.strip())
        is_valid, error_msg = validate_tool_params(tool_name, tool_args, has_context)
        
        if not is_valid:
            logger.warning(f"[EXECUTOR] Prerequisite check FAILED for {tool_name}: {error_msg}")
            
            # Attempt fallback retrieval
            fallback_result = self._execute_retrieval(state)
            
            return ExecutionResult(
                success=False,
                error=f"TOOL_BLOCKED: {error_msg}. Fallback retrieval: {fallback_result.output}",
                tool_used=tool_name,
                params_used=tool_args
            )
        
        # --- EXECUTE TOOL ---
        logger.info(f"[EXECUTOR] Params validated for {tool_name}: {list(tool_args.keys())}")
        
        try:
            tool_fn = self.tools[tool_name]
            result = tool_fn.invoke(tool_args)
            logger.info(f"[EXECUTOR] Tool {tool_name} executed successfully")
            
            return ExecutionResult(
                success=True,
                output=str(result),
                tool_used=tool_name,
                params_used=tool_args
            )
        except Exception as e:
            logger.error(f"[EXECUTOR] Tool {tool_name} failed: {e}")
            
            # Fallback on execution error
            fallback_result = self._execute_retrieval(state)
            
            return ExecutionResult(
                success=False,
                error=f"Tool {tool_name} execution failed: {str(e)}. Fallback: {fallback_result.output}",
                tool_used=tool_name,
                params_used=tool_args
            )
    
    def _execute_retrieval(self, state: AgentState) -> ExecutionResult:
        """Execute retrieval to gather more context."""
        try:
            docs = self.retrieval_fn(state.query)
            new_context = "\n\n".join([doc.page_content for doc in docs])
            state.context += "\n\n" + new_context
            state.observations.append(f"Retrieved {len(docs)} documents")
            state.actions.append("RETRIEVAL_MODULE")
            
            return ExecutionResult(
                success=True,
                output=f"Retrieved {len(docs)} documents",
                tool_used="RETRIEVAL_MODULE"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Retrieval failed: {str(e)}",
                tool_used="RETRIEVAL_MODULE"
            )
