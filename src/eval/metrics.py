"""
Automated Evaluation Metrics for Alumni Agent (HW3 Task 3)

Implements quantitative metrics:
1. Groundedness Score - Factuality of responses
2. Tool Selection Accuracy - Correct tool usage
3. Task Completion Rate - Success rate
4. Iterations Before Convergence - Efficiency
5. Plan Adherence Score - Following the plan
6. Hallucination Frequency - False information rate
7. Response Quality Score - Overall quality
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""

    # Core metrics
    groundedness_score: float = 0.0          # 0-1: Factuality
    tool_selection_accuracy: float = 0.0     # 0-1: Correct tools used
    task_completion_rate: float = 0.0        # 0-1: Did it complete successfully
    iterations_before_convergence: int = 0    # Number of iterations
    plan_adherence_score: float = 0.0        # 0-1: Followed the plan
    hallucination_frequency: float = 0.0     # 0-1: Rate of false claims
    response_quality_score: float = 0.0      # 0-1: Overall quality

    # Supporting data
    execution_time_seconds: float = 0.0
    num_tools_used: int = 0
    num_errors: int = 0
    confidence_level: str = "unknown"        # low/medium/high

    # Detailed breakdown
    metrics_breakdown: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage/display."""
        return {
            "groundedness_score": self.groundedness_score,
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "task_completion_rate": self.task_completion_rate,
            "iterations_before_convergence": self.iterations_before_convergence,
            "plan_adherence_score": self.plan_adherence_score,
            "hallucination_frequency": self.hallucination_frequency,
            "response_quality_score": self.response_quality_score,
            "execution_time_seconds": self.execution_time_seconds,
            "num_tools_used": self.num_tools_used,
            "num_errors": self.num_errors,
            "confidence_level": self.confidence_level,
        }


class MetricsCalculator:
    """
    Calculates automated evaluation metrics for agent performance.

    Usage:
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(result, expected_output)
    """

    def __init__(self):
        logger.info("[METRICS] MetricsCalculator initialized")

    def calculate_all_metrics(
        self,
        result: dict,
        expected_tools: Optional[List[str]] = None,
        expected_answer: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> EvaluationMetrics:
        """
        Calculate all metrics for a query result.

        Args:
            result: Agent execution result dictionary
            expected_tools: Expected tools for this query (for accuracy)
            expected_answer: Expected answer (for validation)
            start_time: Query start time (for execution time)

        Returns:
            EvaluationMetrics object with all computed metrics
        """
        logger.info("[METRICS] Calculating all evaluation metrics...")

        metrics = EvaluationMetrics()

        # 1. Groundedness Score (from verification module)
        metrics.groundedness_score = self._calculate_groundedness(result)

        # 2. Tool Selection Accuracy
        metrics.tool_selection_accuracy = self._calculate_tool_accuracy(result, expected_tools)

        # 3. Task Completion Rate
        metrics.task_completion_rate = self._calculate_completion_rate(result)

        # 4. Iterations Before Convergence
        metrics.iterations_before_convergence = self._calculate_iterations(result)

        # 5. Plan Adherence Score
        metrics.plan_adherence_score = self._calculate_plan_adherence(result)

        # 6. Hallucination Frequency
        metrics.hallucination_frequency = self._calculate_hallucination_rate(result)

        # 7. Response Quality Score (aggregate)
        metrics.response_quality_score = self._calculate_overall_quality(metrics)

        # Supporting metrics
        if start_time:
            metrics.execution_time_seconds = (datetime.now() - start_time).total_seconds()

        metrics.num_tools_used = len(result.get('tools_used', []))

        execution = result.get('execution')
        if execution:
            metrics.num_errors = len(execution.errors) if hasattr(execution, 'errors') else 0

        # Confidence level
        metrics.confidence_level = self._determine_confidence(metrics)

        # Detailed breakdown
        metrics.metrics_breakdown = {
            "groundedness_details": self._get_groundedness_details(result),
            "tool_details": self._get_tool_details(result, expected_tools),
            "plan_details": self._get_plan_details(result)
        }

        logger.info(f"[METRICS] ✓ All metrics calculated:")
        logger.info(f"[METRICS]   Groundedness: {metrics.groundedness_score:.2f}")
        logger.info(f"[METRICS]   Tool Accuracy: {metrics.tool_selection_accuracy:.2f}")
        logger.info(f"[METRICS]   Completion: {metrics.task_completion_rate:.2f}")
        logger.info(f"[METRICS]   Iterations: {metrics.iterations_before_convergence}")
        logger.info(f"[METRICS]   Plan Adherence: {metrics.plan_adherence_score:.2f}")
        logger.info(f"[METRICS]   Overall Quality: {metrics.response_quality_score:.2f}")

        return metrics

    def _calculate_groundedness(self, result: dict) -> float:
        """
        Metric 1: Groundedness Score

        Measures how well the response is grounded in retrieved facts.
        Source: Critique node's groundedness_score
        """
        critique = result.get('critique')
        if critique and hasattr(critique, 'groundedness_score'):
            return critique.groundedness_score

        # Fallback: check verification
        verification = result.get('verification')
        if verification:
            if hasattr(verification, 'score'):
                return verification.score
            elif isinstance(verification, dict):
                return verification.get('score', 0.0)

        return 0.0

    def _calculate_tool_accuracy(self, result: dict, expected_tools: Optional[List[str]]) -> float:
        """
        Metric 2: Tool Selection Accuracy

        Measures if the correct tools were selected.
        Formula: (correct_tools ∩ used_tools) / expected_tools
        """
        if not expected_tools:
            # No expected tools defined, can't measure accuracy
            return 1.0  # Assume correct if not specified

        used_tools = set(result.get('tools_used', []))
        expected_set = set(expected_tools)

        if not expected_set:
            return 1.0

        # Calculate intersection
        correct_tools = used_tools & expected_set

        # Accuracy = correct / expected
        accuracy = len(correct_tools) / len(expected_set)

        logger.info(f"[METRICS] Tool Accuracy: {correct_tools} / {expected_set} = {accuracy:.2f}")

        return accuracy

    def _calculate_completion_rate(self, result: dict) -> float:
        """
        Metric 3: Task Completion Rate

        Did the task complete successfully?
        Based on critique recommendation and execution success.
        """
        critique = result.get('critique')

        if critique:
            recommendation = critique.recommendation if hasattr(critique, 'recommendation') else None

            if recommendation == "proceed":
                return 1.0  # Full success
            elif recommendation == "proceed_with_caveats":
                return 0.8  # Partial success
            elif recommendation == "replan":
                return 0.5  # Needed replanning
            elif recommendation == "escalate":
                return 0.0  # Failed

        # Fallback: check execution success
        execution = result.get('execution')
        if execution and hasattr(execution, 'success'):
            return 1.0 if execution.success else 0.0

        # Last fallback: check if response exists
        return 1.0 if result.get('response') else 0.0

    def _calculate_iterations(self, result: dict) -> int:
        """
        Metric 4: Iterations Before Convergence

        How many replanning cycles were needed?
        Lower is better (more efficient).
        """
        return result.get('replans', 0) + 1  # +1 for initial attempt

    def _calculate_plan_adherence(self, result: dict) -> float:
        """
        Metric 5: Plan Adherence Score

        Did the execution follow the plan?
        Compares planned tools vs actual tools used.
        """
        plan = result.get('plan')
        execution = result.get('execution')

        if not plan or not execution:
            return 0.5  # Unknown

        expected_tools = set(plan.expected_tools) if hasattr(plan, 'expected_tools') else set()
        used_tools = set(execution.tools_used) if hasattr(execution, 'tools_used') else set()

        if not expected_tools:
            return 1.0  # No plan constraints

        # Calculate overlap
        matched_tools = expected_tools & used_tools

        # Adherence = matched / expected
        adherence = len(matched_tools) / len(expected_tools)

        logger.info(f"[METRICS] Plan Adherence: {matched_tools} / {expected_tools} = {adherence:.2f}")

        return adherence

    def _calculate_hallucination_rate(self, result: dict) -> float:
        """
        Metric 6: Hallucination Frequency

        Rate of unverified claims in the response.
        Inverse of groundedness: hallucination = 1 - groundedness
        """
        groundedness = self._calculate_groundedness(result)

        # Hallucination is inverse of groundedness
        hallucination = 1.0 - groundedness

        return hallucination

    def _calculate_overall_quality(self, metrics: EvaluationMetrics) -> float:
        """
        Metric 7: Response Quality Score

        Weighted aggregate of all metrics.

        Formula:
        Quality = 0.3 * groundedness
                + 0.2 * tool_accuracy
                + 0.2 * completion
                + 0.1 * plan_adherence
                + 0.1 * (1 - iterations_penalty)
                + 0.1 * (1 - hallucination)
        """
        # Normalize iterations (penalty for >3 iterations)
        iterations_penalty = min(metrics.iterations_before_convergence / 5.0, 1.0)

        quality = (
            0.3 * metrics.groundedness_score +
            0.2 * metrics.tool_selection_accuracy +
            0.2 * metrics.task_completion_rate +
            0.1 * metrics.plan_adherence_score +
            0.1 * (1.0 - iterations_penalty) +
            0.1 * (1.0 - metrics.hallucination_frequency)
        )

        return min(quality, 1.0)  # Cap at 1.0

    def _determine_confidence(self, metrics: EvaluationMetrics) -> str:
        """Determine confidence level based on metrics."""
        quality = metrics.response_quality_score

        if quality >= 0.8:
            return "high"
        elif quality >= 0.6:
            return "medium"
        else:
            return "low"

    def _get_groundedness_details(self, result: dict) -> dict:
        """Get detailed groundedness breakdown."""
        critique = result.get('critique')
        if critique:
            return {
                "score": critique.groundedness_score if hasattr(critique, 'groundedness_score') else 0.0,
                "strengths": critique.strengths if hasattr(critique, 'strengths') else [],
                "weaknesses": critique.weaknesses if hasattr(critique, 'weaknesses') else []
            }
        return {}

    def _get_tool_details(self, result: dict, expected_tools: Optional[List[str]]) -> dict:
        """Get detailed tool usage breakdown."""
        used = result.get('tools_used', [])
        expected = expected_tools or []

        return {
            "used_tools": used,
            "expected_tools": expected,
            "correct": list(set(used) & set(expected)),
            "missing": list(set(expected) - set(used)),
            "extra": list(set(used) - set(expected))
        }

    def _get_plan_details(self, result: dict) -> dict:
        """Get detailed plan execution breakdown."""
        plan = result.get('plan')
        if plan:
            return {
                "steps": plan.steps if hasattr(plan, 'steps') else [],
                "expected_tools": plan.expected_tools if hasattr(plan, 'expected_tools') else [],
                "complexity": plan.estimated_complexity if hasattr(plan, 'estimated_complexity') else "unknown"
            }
        return {}


def format_metrics_table(metrics_list: List[EvaluationMetrics]) -> str:
    """
    Format metrics as a table for display/reporting.

    Args:
        metrics_list: List of EvaluationMetrics objects

    Returns:
        Formatted table string
    """
    if not metrics_list:
        return "No metrics to display"

    # Header
    table = "=" * 120 + "\n"
    table += f"{'Test':<6} | {'Ground.':<7} | {'Tool Acc.':<9} | {'Complet.':<9} | {'Iters':<5} | {'Plan Adh.':<9} | {'Halluc.':<8} | {'Quality':<7} | {'Confid.':<7}\n"
    table += "=" * 120 + "\n"

    # Rows
    for i, m in enumerate(metrics_list, 1):
        table += (
            f"{i:<6} | "
            f"{m.groundedness_score:<7.2f} | "
            f"{m.tool_selection_accuracy:<9.2f} | "
            f"{m.task_completion_rate:<9.2f} | "
            f"{m.iterations_before_convergence:<5} | "
            f"{m.plan_adherence_score:<9.2f} | "
            f"{m.hallucination_frequency:<8.2f} | "
            f"{m.response_quality_score:<7.2f} | "
            f"{m.confidence_level:<7}\n"
        )

    table += "=" * 120 + "\n"

    # Averages
    avg_ground = sum(m.groundedness_score for m in metrics_list) / len(metrics_list)
    avg_tool = sum(m.tool_selection_accuracy for m in metrics_list) / len(metrics_list)
    avg_comp = sum(m.task_completion_rate for m in metrics_list) / len(metrics_list)
    avg_iters = sum(m.iterations_before_convergence for m in metrics_list) / len(metrics_list)
    avg_plan = sum(m.plan_adherence_score for m in metrics_list) / len(metrics_list)
    avg_hall = sum(m.hallucination_frequency for m in metrics_list) / len(metrics_list)
    avg_qual = sum(m.response_quality_score for m in metrics_list) / len(metrics_list)

    table += (
        f"{'AVG':<6} | "
        f"{avg_ground:<7.2f} | "
        f"{avg_tool:<9.2f} | "
        f"{avg_comp:<9.2f} | "
        f"{avg_iters:<5.1f} | "
        f"{avg_plan:<9.2f} | "
        f"{avg_hall:<8.2f} | "
        f"{avg_qual:<7.2f} |\n"
    )
    table += "=" * 120 + "\n"

    return table
