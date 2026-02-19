"""
Test Cases for Evaluation Framework (HW3 Task 3)

Defines 5+ structured test cases with expected outputs.
Includes at least one failure case for analysis.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestCase:
    """A structured test case for evaluation."""
    id: str
    query: str
    expected_tools: List[str]
    expected_answer_keywords: List[str]
    description: str
    should_succeed: bool = True
    failure_reason: Optional[str] = None


# ========== TEST CASES ==========

TEST_CASES = [
    TestCase(
        id="TC001",
        query="Find alumni who work in artificial intelligence or machine learning",
        expected_tools=["RETRIEVE"],
        expected_answer_keywords=["AI", "machine learning", "alumni"],
        description="Simple retrieval query - should succeed with high groundedness",
        should_succeed=True
    ),

    TestCase(
        id="TC002",
        query="Find alumni working in fintech and prepare to send them a networking email",
        expected_tools=["RETRIEVE", "email_sender"],
        expected_answer_keywords=["fintech", "email", "prepared"],
        description="Multi-tool query - retrieval + email preparation",
        should_succeed=True
    ),

    TestCase(
        id="TC003",
        query="Discover new MSIT alumni from 2023 graduating class",
        expected_tools=["linkedin_discovery", "RETRIEVE"],
        expected_answer_keywords=["MSIT", "2023", "alumni"],
        description="Discovery query using web search",
        should_succeed=True
    ),

    TestCase(
        id="TC004",
        query="Who is the CEO of Tesla and what is their educational background?",
        expected_tools=["RETRIEVE"],
        expected_answer_keywords=[],
        description="OFF-TOPIC query - FAILURE CASE (not about CMU alumni)",
        should_succeed=False,
        failure_reason="Query is off-topic (not about CMU Africa alumni). System should have low groundedness."
    ),

    TestCase(
        id="TC005",
        query="Find alumni with Python and MongoDB skills who graduated in 2022-2023",
        expected_tools=["RETRIEVE"],
        expected_answer_keywords=["Python", "MongoDB", "2022", "2023"],
        description="Specific skills query with time range",
        should_succeed=True
    ),

    TestCase(
        id="TC006",
        query="Send a congratulations email to all alumni who got promoted this year",
        expected_tools=["RETRIEVE", "linkedin_scraper", "email_sender"],
        expected_answer_keywords=["promoted", "email", "congratulations"],
        description="Complex multi-step query requiring multiple tools",
        should_succeed=True
    ),
]


def get_test_case(test_id: str) -> Optional[TestCase]:
    """Get a test case by ID."""
    for tc in TEST_CASES:
        if tc.id == test_id:
            return tc
    return None


def get_failure_cases() -> List[TestCase]:
    """Get all test cases that are expected to fail."""
    return [tc for tc in TEST_CASES if not tc.should_succeed]


def get_success_cases() -> List[TestCase]:
    """Get all test cases that are expected to succeed."""
    return [tc for tc in TEST_CASES if tc.should_succeed]
