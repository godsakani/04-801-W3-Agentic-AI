"""
Planner Node — Decides what action to take.

PLANNING ROLE in the Planner → Executor → Critic pipeline.
Uses LLM with bound tools to reason about the next step.
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.agent_state_model import AgentState
from src.models.role_output_model import PlanOutput

logger = logging.getLogger(__name__)


class PlannerNode:
    """
    PLANNING ROLE: Analyzes the current state and decides what action to take.
    
    Uses LLM with bound tools to decide:
    - Call a specific tool (with parameters)
    - Give a final answer (no tool call)
    """
    
    def __init__(self, llm_with_tools):
        self.llm_with_tools = llm_with_tools
    
    def plan(self, state: AgentState) -> PlanOutput:
        """
        Decide the next action based on current state.
        
        Args:
            state: Current agent state with query, context, observations
            
        Returns:
            PlanOutput with action type, tool name/args, and reasoning
        """
        messages = self._build_messages(state)
        response = self.llm_with_tools.invoke(messages)
        
        thought = response.content or ""
        tool_calls = response.tool_calls or []
        
        logger.info(f"[PLANNER] Reasoning: {thought[:100]}...")
        
        if not tool_calls:
            return PlanOutput(
                action="FINAL_ANSWER",
                reasoning=thought
            )
        
        # Use the first tool call
        tc = tool_calls[0]
        return PlanOutput(
            action="TOOL_CALL",
            tool_name=tc["name"],
            tool_args=tc["args"],
            reasoning=thought
        )
    
    def _build_messages(self, state: AgentState) -> list:
        """Build messages for the planner LLM."""
        system_prompt = """You are the PLANNER for a CMU Africa alumni tracking system.
Your job is to analyze the query and context, then decide what action to take.

You have access to tools for: sending emails, scraping LinkedIn profiles, and creating surveys.

IMPORTANT RULES:
1. If you don't have alumni data yet, do NOT call email_sender or survey_tool. 
   Instead, respond with text explaining what data is needed.
2. When calling email_sender, you MUST use the exact email address from the retrieved context.
3. When calling survey_tool, you MUST use a valid alumni_id from the retrieved context.
4. If you have enough information to answer, respond with text only (no tool call).

Think step by step:
1. What does the user need?
2. Do I have enough context from retrieved data?
3. What specific action should I take?"""

        messages = [SystemMessage(content=system_prompt)]
        
        user_content = f"Query: {state.query}\n"
        
        if state.memory_context:
            user_content += f"\nPrior Session Context:\n{state.memory_context[:500]}\n"
        
        if state.context:
            user_content += f"\nRetrieved Alumni Context:\n{state.context[:2000]}\n"
        else:
            user_content += "\nNo alumni context retrieved yet.\n"
        
        if state.observations:
            user_content += f"\nPrevious observations: {state.observations}\n"
        
        if state.actions:
            user_content += f"\nPrevious actions: {state.actions}\n"
        
        user_content += "\nDecide what to do next. Either call a tool with the correct parameters, or respond with your final answer."
        
        messages.append(HumanMessage(content=user_content))
        return messages
