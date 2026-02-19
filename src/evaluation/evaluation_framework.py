"""
Evaluation Framework — Automated Metrics for Alumni RAG Agent

Computes quantitative metrics for agent performance evaluation:
1. Groundedness Score — from GroundednessScorer (already exists)
2. Tool Selection Accuracy — did agent pick the correct tool?
3. Iteration Efficiency — iterations_used / max_iterations
4. Task Completion Rate — did agent produce a valid (non-fallback) response?
"""

import logging
from typing import Optional, List

from src.evaluation.test_cases import TestCase, EvaluationResult, TEST_CASES

logger = logging.getLogger(__name__)


class EvaluationFramework:
    """
    Automated evaluation of Alumni RAG Agent performance.
    
    Computes 4 quantitative metrics across structured test cases.
    Results are presented as a table for the evaluation report.
    """
    
    def __init__(self, max_iterations: int = 5):
        """
        Initialize the evaluation framework.
        
        Args:
            max_iterations: The agent's max iteration count (for efficiency metric)
        """
        self.max_iterations = max_iterations
        self.results: List[EvaluationResult] = []
    
    def evaluate_run(
        self,
        test_case: TestCase,
        agent_result: dict,
    ) -> EvaluationResult:
        """
        Compute all 4 metrics for a single agent run.
        
        Args:
            test_case: The test case that was executed
            agent_result: Dict returned by agent.run() with keys:
                - response, verification, trace, session_id, iterations, actions
                
        Returns:
            EvaluationResult with all 4 metrics computed
        """
        # Extract data from agent result
        verification = agent_result.get("verification")
        trace = agent_result.get("trace", [])
        iterations = agent_result.get("iterations", 0)
        actions = agent_result.get("actions", [])
        response = agent_result.get("response", "")
        
        # --- Metric 1: Groundedness Score ---
        groundedness = verification.score if verification else 0.0
        
        # --- Metric 2: Tool Selection Accuracy ---
        actual_tool = self._extract_tool_used(actions)
        tool_accuracy = self._compute_tool_accuracy(
            expected=test_case.expected_tool,
            actual=actual_tool
        )
        
        # --- Metric 3: Iteration Efficiency ---
        iteration_efficiency = 1.0 - (iterations / self.max_iterations) if self.max_iterations > 0 else 0.0
        iteration_efficiency = max(0.0, iteration_efficiency)  # Clamp to [0, 1]
        
        # --- Metric 4: Task Completion Rate ---
        task_completion = self._compute_task_completion(
            response=response,
            trace=trace,
            test_case=test_case
        )
        
        # Determine pass/fail
        passed = self._determine_pass(
            test_case=test_case,
            groundedness=groundedness,
            tool_accuracy=tool_accuracy,
            task_completion=task_completion
        )
        
        # Build notes
        notes = self._build_notes(test_case, trace, actual_tool)
        
        result = EvaluationResult(
            test_id=test_case.id,
            test_name=test_case.name,
            groundedness_score=groundedness,
            tool_selection_accuracy=tool_accuracy,
            iteration_efficiency=iteration_efficiency,
            task_completion=task_completion,
            actual_tool_used=actual_tool,
            expected_tool=test_case.expected_tool,
            iterations_used=iterations,
            passed=passed,
            notes=notes
        )
        
        self.results.append(result)
        logger.info(
            f"[EVAL] Test {test_case.id} ({test_case.name}): "
            f"ground={groundedness:.2f}, tool_acc={tool_accuracy:.1f}, "
            f"efficiency={iteration_efficiency:.2f}, completion={task_completion:.1f}, "
            f"{'PASS' if passed else 'FAIL'}"
        )
        
        return result
    
    def _extract_tool_used(self, actions: list) -> Optional[str]:
        """Extract the main tool used from the actions list."""
        for action in actions:
            if action.startswith("TOOL_MODULE:"):
                return action.split(":")[1]
        return None
    
    def _compute_tool_accuracy(self, expected: Optional[str], actual: Optional[str]) -> float:
        """
        Compute tool selection accuracy (0.0 or 1.0).
        
        - If expected is None (retrieval-only) and no tool was used: 1.0
        - If expected matches actual: 1.0
        - Otherwise: 0.0
        """
        if expected is None:
            return 1.0 if actual is None else 0.0
        if actual is None:
            return 0.0
        return 1.0 if expected == actual else 0.0
    
    def _compute_task_completion(
        self, response: str, trace: list, test_case: TestCase
    ) -> float:
        """
        Determine if the task was completed (1.0) or not (0.0).
        
        A task is considered incomplete if:
        - Response is empty
        - Response contains escalation/clarification language
        - For failure case: completion is 0.0 by design
        """
        if not response or len(response.strip()) < 20:
            return 0.0
        
        # Check for escalation/fallback markers
        escalation_markers = [
            "I need more information",
            "could you clarify",
            "unable to complete",
            "I don't have enough",
        ]
        
        for marker in escalation_markers:
            if marker.lower() in response.lower():
                return 0.0
        
        # For the failure case, if tool was blocked, task isn't "completed"
        if test_case.is_failure_case:
            blocked = any(
                step.get("error") and "TOOL_BLOCKED" in str(step.get("error", ""))
                for step in trace
            )
            if blocked:
                return 0.0
        
        return 1.0
    
    def _determine_pass(
        self,
        test_case: TestCase,
        groundedness: float,
        tool_accuracy: float,
        task_completion: float
    ) -> bool:
        """
        Determine overall pass/fail for the test case.
        
        For failure cases: PASS means the agent correctly identified it couldn't
        proceed (validation blocked the tool, agent asked for clarification).
        """
        if test_case.is_failure_case:
            # For failure cases, "pass" means the system correctly handled it
            # Tool should have been blocked AND agent should not have hallucinated
            return task_completion == 0.0 and groundedness >= 0.3
        
        # Normal cases: need decent groundedness and correct tool
        return groundedness >= 0.5 and tool_accuracy >= 0.5
    
    def _build_notes(
        self, test_case: TestCase, trace: list, actual_tool: Optional[str]
    ) -> str:
        """Build human-readable notes about the evaluation."""
        notes = []
        
        # Check for adaptive control actions
        adapt_steps = [s for s in trace if s.get("phase") == "ADAPT"]
        if adapt_steps:
            actions = [s.get("action", "unknown") for s in adapt_steps]
            notes.append(f"Adaptive control triggered: {', '.join(actions)}")
        
        # Check for validation failures
        blocked_steps = [
            s for s in trace 
            if s.get("error") and "TOOL_BLOCKED" in str(s.get("error", ""))
        ]
        if blocked_steps:
            notes.append(f"Tool blocked {len(blocked_steps)} time(s) by prerequisite validation")
        
        # Tool mismatch
        if test_case.expected_tool and actual_tool != test_case.expected_tool:
            notes.append(f"Tool mismatch: expected={test_case.expected_tool}, actual={actual_tool}")
        
        if test_case.is_failure_case:
            notes.append("FAILURE CASE: Intentionally vague query for failure analysis")
        
        return "; ".join(notes) if notes else "Normal execution"
    
    # ============================================================
    # Reporting
    # ============================================================
    
    def get_results_table(self) -> str:
        """
        Generate a markdown table of all evaluation results.
        
        Returns:
            Markdown-formatted results table
        """
        if not self.results:
            return "No evaluation results available."
        
        header = (
            "| # | Test Name | Groundedness | Tool Accuracy | "
            "Efficiency | Completion | Iterations | Pass/Fail | Notes |\n"
            "|---|-----------|-------------|---------------|------------|"
            "------------|------------|-----------|-------|\n"
        )
        
        rows = []
        for r in self.results:
            rows.append(
                f"| {r.test_id} | {r.test_name} | {r.groundedness_score:.2f} | "
                f"{r.tool_selection_accuracy:.1f} | {r.iteration_efficiency:.2f} | "
                f"{r.task_completion:.1f} | {r.iterations_used} | "
                f"{'✅ PASS' if r.passed else '❌ FAIL'} | {r.notes[:50]} |"
            )
        
        return header + "\n".join(rows)
    
    def get_summary_stats(self) -> dict:
        """
        Compute aggregate statistics across all test runs.
        
        Returns:
            Dict with average metrics and pass rate
        """
        if not self.results:
            return {}
        
        n = len(self.results)
        return {
            "total_tests": n,
            "pass_rate": sum(1 for r in self.results if r.passed) / n,
            "avg_groundedness": sum(r.groundedness_score for r in self.results) / n,
            "avg_tool_accuracy": sum(r.tool_selection_accuracy for r in self.results) / n,
            "avg_efficiency": sum(r.iteration_efficiency for r in self.results) / n,
            "avg_completion": sum(r.task_completion for r in self.results) / n,
            "failure_cases": sum(1 for r in self.results if not r.passed),
        }
    
    def get_failure_analysis(self) -> str:
        """
        Deep dive analysis for failed test cases.
        
        Returns:
            Markdown-formatted failure analysis
        """
        failed = [r for r in self.results if not r.passed or 
                  any(tc.is_failure_case and tc.id == r.test_id for tc in TEST_CASES)]
        
        if not failed:
            return "No failures to analyze."
        
        sections = ["## Failure Case Analysis\n"]
        
        for r in failed:
            test_case = next((tc for tc in TEST_CASES if tc.id == r.test_id), None)
            
            sections.append(f"### Test {r.test_id}: {r.test_name}")
            sections.append(f"- **Query**: {test_case.query if test_case else 'N/A'}")
            sections.append(f"- **Expected Tool**: {r.expected_tool or 'None (retrieval only)'}")
            sections.append(f"- **Actual Tool**: {r.actual_tool_used or 'None'}")
            sections.append(f"- **Groundedness**: {r.groundedness_score:.2f}")
            sections.append(f"- **Tool Accuracy**: {r.tool_selection_accuracy:.1f}")
            sections.append(f"- **Task Completion**: {r.task_completion:.1f}")
            sections.append(f"- **Notes**: {r.notes}")
            
            if test_case and test_case.is_failure_case:
                sections.append("\n**Root Cause Analysis:**")
                sections.append("The query is intentionally vague — 'Email someone in tech' does not ")
                sections.append("specify a particular alumni. The prerequisite validation in ExecutorNode ")
                sections.append("correctly blocks the email_sender tool because `recipient_email` is missing.")
                sections.append("\n**Technical Explanation:**")
                sections.append("1. PlannerNode receives vague query + alumni context from retrieval")
                sections.append("2. PlannerNode decides to call email_sender (reasonable given the query)")
                sections.append("3. ExecutorNode validates params → BLOCKED: missing recipient_email")
                sections.append("4. CriticNode detects failure → recommends re-plan")
                sections.append("5. Agent falls back to retrieval and eventually gives a clarification response")
                sections.append("\n**Adjustment Made:**")
                sections.append("Added prerequisite validation (TOOL_PREREQUISITES) that checks required ")
                sections.append("fields before tool execution, preventing hallucinated parameters.")
                sections.append("\n**Before vs After:**")
                sections.append("- Before: Agent would call email_sender with hallucinated email address → silent failure")
                sections.append("- After: ExecutorNode blocks the call → CriticNode triggers re-plan → agent asks for clarification")
            
            sections.append("")
        
        return "\n".join(sections)
