"""
Test Cases — Predefined test suite for the evaluation framework.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestCase:
    """A structured test case for the evaluation framework."""
    id: int
    name: str
    query: str
    expected_tool: Optional[str]        # None = retrieval-only
    expected_behavior: str              # Description of expected outcome
    is_failure_case: bool = False       # True for intentional failure test


@dataclass
class EvaluationResult:
    """Metrics computed for a single test case."""
    test_id: int
    test_name: str
    groundedness_score: float
    tool_selection_accuracy: float      # 1.0 = correct, 0.0 = wrong
    iteration_efficiency: float         # iterations_used / max_iterations
    task_completion: float              # 1.0 = completed, 0.0 = fallback/failure
    actual_tool_used: Optional[str]
    expected_tool: Optional[str]
    iterations_used: int
    passed: bool
    notes: str = ""


# ============================================================
# Predefined Test Suite
# ============================================================
TEST_CASES = [
    TestCase(
        id=1,
        name="Alumni Info Retrieval",
        query="Tell me about alumni working in fintech",
        expected_tool=None,  # Retrieval-only, no action tool
        expected_behavior="Returns grounded answer from vector store about fintech alumni"
    ),
    TestCase(
        id=2,
        name="Email Outreach",
        query="Send a check-in email to our alumni in the database",
        expected_tool="email_sender",
        expected_behavior="Agent uses email_sender tool with correct params from context"
    ),
    TestCase(
        id=3,
        name="LinkedIn Profile Check",
        query="Check the LinkedIn profile of an alumni for career updates",
        expected_tool="linkedin_scraper",
        expected_behavior="Agent scrapes LinkedIn profile for changes"
    ),
    TestCase(
        id=4,
        name="Survey Distribution",
        query="Send a career update survey to our alumni",
        expected_tool="survey_tool",
        expected_behavior="Agent uses survey_tool to create and send survey"
    ),
    TestCase(
        id=5,
        name="Vague Request (Failure Case)",
        query="Email someone in tech",
        expected_tool="email_sender",
        expected_behavior="Validation blocks tool — no specific alumni identified. Fallback to retrieval.",
        is_failure_case=True
    ),
]
