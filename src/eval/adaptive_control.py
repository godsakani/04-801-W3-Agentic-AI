"""
Adaptive Control Module (HW3 Task 4)

Implements closed-loop behavioral adaptation based on feedback.

Adaptive Behaviors:
1. If groundedness < threshold → re-retrieve with refined query
2. If tool fails → retry or select alternative tool
3. If iteration limit exceeded → escalate to human
4. If plan quality low → re-plan with feedback
5. If confidence low → request clarification

Pattern: Observe → Reason → Decide → Act → Evaluate → Update → Repeat
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveAction:
    """Represents an adaptive action taken by the system."""
    trigger: str              # What caused this adaptation
    condition: str            # The condition that was met
    action_type: str          # Type of adaptation (re-retrieve, retry, escalate, etc.)
    action_details: str       # Specific action taken
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AdaptiveController:
    """
    Implements closed-loop adaptive control for the agent.

    Monitors execution quality and adapts behavior in real-time.

    Usage:
        controller = AdaptiveController(
            groundedness_threshold=0.7,
            max_retrieval_attempts=2
        )

        # During execution
        if controller.should_re_retrieve(groundedness_score):
            new_query = controller.refine_query(original_query, feedback)
            docs = retrieval_fn(new_query)
    """

    def __init__(
        self,
        groundedness_threshold: float = 0.7,
        tool_retry_limit: int = 2,
        max_replanning_attempts: int = 2,
        confidence_threshold: float = 0.6,
        max_retrieval_attempts: int = 2
    ):
        self.groundedness_threshold = groundedness_threshold
        self.tool_retry_limit = tool_retry_limit
        self.max_replanning_attempts = max_replanning_attempts
        self.confidence_threshold = confidence_threshold
        self.max_retrieval_attempts = max_retrieval_attempts

        # Track adaptations
        self.adaptations: List[AdaptiveAction] = []
        self.retrieval_attempts = 0
        self.tool_retries: Dict[str, int] = {}

        logger.info("[ADAPTIVE CONTROL] Initialized")
        logger.info(f"[ADAPTIVE CONTROL]   Groundedness threshold: {groundedness_threshold}")
        logger.info(f"[ADAPTIVE CONTROL]   Max retrieval attempts: {max_retrieval_attempts}")
        logger.info(f"[ADAPTIVE CONTROL]   Tool retry limit: {tool_retry_limit}")

    def reset_for_new_query(self):
        """Reset state for a new query."""
        self.adaptations = []
        self.retrieval_attempts = 0
        self.tool_retries = {}

    # ========== ADAPTIVE BEHAVIOR 1: Low Groundedness → Re-Retrieve ==========

    def should_re_retrieve(self, groundedness_score: float) -> bool:
        """
        Decide if we should re-retrieve based on groundedness.

        Condition: groundedness < threshold AND attempts < max
        """
        should_adapt = (
            groundedness_score < self.groundedness_threshold and
            self.retrieval_attempts < self.max_retrieval_attempts
        )

        if should_adapt:
            logger.info(f"[ADAPTIVE CONTROL] 🔄 ADAPTATION TRIGGERED")
            logger.info(f"[ADAPTIVE CONTROL]   Condition: Groundedness {groundedness_score:.2f} < {self.groundedness_threshold}")
            logger.info(f"[ADAPTIVE CONTROL]   Action: Re-retrieve with refined query (attempt {self.retrieval_attempts + 1}/{self.max_retrieval_attempts})")

            action = AdaptiveAction(
                trigger="low_groundedness",
                condition=f"groundedness={groundedness_score:.2f} < threshold={self.groundedness_threshold}",
                action_type="re_retrieve",
                action_details=f"Refining query and retrieving again (attempt {self.retrieval_attempts + 1})"
            )
            self.adaptations.append(action)
            self.retrieval_attempts += 1

        return should_adapt

    def refine_query(self, original_query: str, feedback: str) -> str:
        """
        Refine the retrieval query based on feedback.

        Strategy: Make query more specific based on what was missing.
        """
        refined = f"{original_query} (focus on: {feedback})"

        logger.info(f"[ADAPTIVE CONTROL]   Original query: '{original_query}'")
        logger.info(f"[ADAPTIVE CONTROL]   Refined query: '{refined}'")

        return refined

    # ========== ADAPTIVE BEHAVIOR 2: Tool Failure → Retry or Alternative ==========

    def should_retry_tool(self, tool_name: str, error: str) -> bool:
        """
        Decide if we should retry a failed tool.

        Condition: retries < limit
        """
        current_retries = self.tool_retries.get(tool_name, 0)
        should_retry = current_retries < self.tool_retry_limit

        if should_retry:
            logger.info(f"[ADAPTIVE CONTROL] 🔄 ADAPTATION TRIGGERED")
            logger.info(f"[ADAPTIVE CONTROL]   Condition: Tool '{tool_name}' failed: {error}")
            logger.info(f"[ADAPTIVE CONTROL]   Action: Retry tool (attempt {current_retries + 1}/{self.tool_retry_limit})")

            action = AdaptiveAction(
                trigger="tool_failure",
                condition=f"tool={tool_name} failed with error={error[:50]}",
                action_type="retry_tool",
                action_details=f"Retrying {tool_name} (attempt {current_retries + 1})"
            )
            self.adaptations.append(action)
            self.tool_retries[tool_name] = current_retries + 1
        else:
            logger.warning(f"[ADAPTIVE CONTROL] ⚠ Tool '{tool_name}' retry limit reached")

        return should_retry

    def select_alternative_tool(self, failed_tool: str, available_tools: List[str]) -> Optional[str]:
        """
        Select an alternative tool when one fails repeatedly.

        Strategy: Choose from available tools that serve similar purpose.
        """
        # Tool alternatives mapping
        alternatives = {
            "linkedin_scraper": ["linkedin_discovery"],  # If scraper fails, try discovery
            "email_sender": [],  # No alternative for email
            "survey_tool": ["email_sender"],  # Can send survey link via email
        }

        options = [t for t in alternatives.get(failed_tool, []) if t in available_tools]

        if options:
            alternative = options[0]
            logger.info(f"[ADAPTIVE CONTROL] 🔄 ADAPTATION TRIGGERED")
            logger.info(f"[ADAPTIVE CONTROL]   Condition: Tool '{failed_tool}' exhausted retries")
            logger.info(f"[ADAPTIVE CONTROL]   Action: Switch to alternative tool '{alternative}'")

            action = AdaptiveAction(
                trigger="tool_exhausted",
                condition=f"tool={failed_tool} retry_limit_reached",
                action_type="alternative_tool",
                action_details=f"Switching from {failed_tool} to {alternative}"
            )
            self.adaptations.append(action)

            return alternative

        logger.warning(f"[ADAPTIVE CONTROL] ⚠ No alternative tool available for '{failed_tool}'")
        return None

    # ========== ADAPTIVE BEHAVIOR 3: Iteration Limit → Escalate ==========

    def should_escalate(self, iterations: int, max_iterations: int) -> bool:
        """
        Decide if we should escalate to human based on iterations.

        Condition: iterations >= max_iterations
        """
        should_escalate = iterations >= max_iterations

        if should_escalate:
            logger.warning(f"[ADAPTIVE CONTROL] 🚨 ADAPTATION TRIGGERED")
            logger.warning(f"[ADAPTIVE CONTROL]   Condition: Iterations {iterations} >= max {max_iterations}")
            logger.warning(f"[ADAPTIVE CONTROL]   Action: ESCALATE to human review")

            action = AdaptiveAction(
                trigger="iteration_limit",
                condition=f"iterations={iterations} >= max={max_iterations}",
                action_type="escalate",
                action_details="Escalating to human due to iteration limit"
            )
            self.adaptations.append(action)

        return should_escalate

    # ========== ADAPTIVE BEHAVIOR 4: Low Plan Quality → Re-plan ==========

    def should_replan(self, plan_quality: float, current_attempt: int) -> bool:
        """
        Decide if we should re-plan based on plan quality.

        Condition: quality < threshold AND attempts < max
        """
        should_adapt = (
            plan_quality < self.confidence_threshold and
            current_attempt < self.max_replanning_attempts
        )

        if should_adapt:
            logger.info(f"[ADAPTIVE CONTROL] 🔄 ADAPTATION TRIGGERED")
            logger.info(f"[ADAPTIVE CONTROL]   Condition: Plan quality {plan_quality:.2f} < {self.confidence_threshold}")
            logger.info(f"[ADAPTIVE CONTROL]   Action: Re-plan with critique feedback (attempt {current_attempt + 1}/{self.max_replanning_attempts})")

            action = AdaptiveAction(
                trigger="low_plan_quality",
                condition=f"plan_quality={plan_quality:.2f} < threshold={self.confidence_threshold}",
                action_type="replan",
                action_details=f"Creating new plan with feedback (attempt {current_attempt + 1})"
            )
            self.adaptations.append(action)

        return should_adapt

    # ========== ADAPTIVE BEHAVIOR 5: Low Confidence → Request Clarification ==========

    def should_request_clarification(self, confidence_score: float, has_ambiguity: bool = False) -> bool:
        """
        Decide if we should request user clarification.

        Condition: confidence < threshold OR ambiguity detected
        """
        should_clarify = confidence_score < self.confidence_threshold or has_ambiguity

        if should_clarify:
            logger.info(f"[ADAPTIVE CONTROL] 🔄 ADAPTATION TRIGGERED")
            logger.info(f"[ADAPTIVE CONTROL]   Condition: Confidence {confidence_score:.2f} < {self.confidence_threshold} OR ambiguity={has_ambiguity}")
            logger.info(f"[ADAPTIVE CONTROL]   Action: Request clarification from user")

            action = AdaptiveAction(
                trigger="low_confidence",
                condition=f"confidence={confidence_score:.2f} < threshold={self.confidence_threshold}",
                action_type="request_clarification",
                action_details="Asking user for more specific information"
            )
            self.adaptations.append(action)

        return should_clarify

    def generate_clarification_request(self, query: str, ambiguity_reason: str) -> str:
        """Generate a clarification request for the user."""
        return (
            f"I need clarification on your query: '{query}'\n\n"
            f"Issue: {ambiguity_reason}\n\n"
            f"Could you please provide more specific details?"
        )

    # ========== Logging & Reporting ==========

    def get_adaptations_summary(self) -> str:
        """Get summary of all adaptations made."""
        if not self.adaptations:
            return "No adaptations were needed."

        summary = f"\n{'='*80}\n"
        summary += f"ADAPTIVE CONTROL SUMMARY: {len(self.adaptations)} adaptation(s)\n"
        summary += f"{'='*80}\n"

        for i, adaptation in enumerate(self.adaptations, 1):
            summary += f"\n{i}. [{adaptation.action_type.upper()}]\n"
            summary += f"   Trigger: {adaptation.trigger}\n"
            summary += f"   Condition: {adaptation.condition}\n"
            summary += f"   Action: {adaptation.action_details}\n"
            summary += f"   Time: {adaptation.timestamp.strftime('%H:%M:%S')}\n"

        summary += f"\n{'='*80}\n"

        return summary

    def log_decision_cycle(
        self,
        observation: str,
        reasoning: str,
        decision: str,
        action: str,
        evaluation: str,
        update: str
    ):
        """
        Log a complete decision cycle for transparency.

        Pattern: Observe → Reason → Decide → Act → Evaluate → Update
        """
        logger.info(f"\n{'='*80}")
        logger.info("[ADAPTIVE CONTROL] DECISION CYCLE")
        logger.info(f"{'='*80}")
        logger.info(f"OBSERVE:   {observation}")
        logger.info(f"REASON:    {reasoning}")
        logger.info(f"DECIDE:    {decision}")
        logger.info(f"ACT:       {action}")
        logger.info(f"EVALUATE:  {evaluation}")
        logger.info(f"UPDATE:    {update}")
        logger.info(f"{'='*80}\n")


def demonstrate_adaptive_cycle():
    """
    Demonstrate a complete adaptive cycle.

    Shows: Observe → Reason → Decide → Act → Evaluate → Update → Repeat
    """
    controller = AdaptiveController()

    print("\n" + "="*80)
    print("DEMONSTRATING ADAPTIVE CONTROL CYCLE")
    print("="*80)

    # Cycle 1: Low groundedness triggers re-retrieval
    print("\n--- CYCLE 1: Initial Attempt ---")
    controller.log_decision_cycle(
        observation="Groundedness score is 0.45 (below threshold 0.7)",
        reasoning="Response may contain unverified claims. Need more context.",
        decision="RE-RETRIEVE: Fetch more relevant documents",
        action="Refining query from 'AI alumni' to 'AI alumni (focus on: current positions)'",
        evaluation="New documents retrieved. Groundedness improved to 0.85",
        update="Retrieval attempts: 1. Continue with enhanced context."
    )

    # Cycle 2: Tool failure triggers retry
    print("\n--- CYCLE 2: Tool Failure ---")
    controller.log_decision_cycle(
        observation="LinkedIn scraper tool failed with timeout error",
        reasoning="Tool may have hit rate limit. Should retry with backoff.",
        decision="RETRY TOOL: Attempt linkedin_scraper again",
        action="Retrying linkedin_scraper with 5-second delay",
        evaluation="Tool succeeded on retry. Profile data retrieved.",
        update="Tool retries: {linkedin_scraper: 1}. Continue execution."
    )

    # Cycle 3: High quality, proceed
    print("\n--- CYCLE 3: Success ---")
    controller.log_decision_cycle(
        observation="All metrics above thresholds. Quality score: 0.88",
        reasoning="Response is well-grounded, all tools executed successfully",
        decision="PROCEED: Return response to user",
        action="Generating final response with high confidence",
        evaluation="Task completed successfully",
        update="No further adaptation needed. DONE."
    )

    print("\n" + controller.get_adaptations_summary())


if __name__ == "__main__":
    demonstrate_adaptive_cycle()
