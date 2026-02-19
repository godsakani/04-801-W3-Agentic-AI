"""
Groundedness Scorer for Verification Module

Calculates groundedness scores to detect hallucinations and verify responses.

Groundedness Score = (verified claims) / (total claims)

Per HW2 requirements: "The Groundedness Score (0 to 1) corresponds to the fraction
of generated claims that are supported by retrieved context."
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, List
from langchain_openai import ChatOpenAI


@dataclass
class VerifiedClaim:
    """A claim with verification status."""
    claim: str
    verified: bool
    evidence: Optional[str] = None
    source_doc: Optional[str] = None


@dataclass
class GroundednessResult:
    """Complete groundedness evaluation result."""
    score: float
    claims: List[VerifiedClaim]
    confidence: str
    recommendation: str


class GroundednessScorer:
    """
    Calculates groundedness scores for agent responses.
    
    Score Interpretation:
    - 0.9-1.0: High confidence - Proceed with response
    - 0.7-0.9: Medium confidence - Add caveats
    - 0.5-0.7: Low confidence - Request clarification
    - < 0.5: Very low - Reject and retry
    """
    
    def __init__(self, openai_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4o-2024-08-06",
            temperature=0,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            base_url="https://ai-gateway.andrew.cmu.edu/"
        )
    
    def extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from agent response.
        
        A claim is a statement that can be verified as true or false.
        """
        prompt = f"""Extract all factual claims from this text. 
A claim is a specific statement that can be verified as true or false.
Return as a JSON array of strings.

Text: {response}

Return ONLY a valid JSON array like: ["claim 1", "claim 2"]
Do not include opinions or general statements."""

        result = self.llm.invoke(prompt)
        
        try:
            claims = json.loads(result.content)
            return claims if isinstance(claims, list) else [response]
        except json.JSONDecodeError:
            # If parsing fails, treat whole response as one claim
            return [response]
    
    def find_evidence(self, claim: str, sources: list) -> tuple:
        """
        Check if claim is supported by source documents.
        
        Returns:
            (is_verified, evidence_quote, source_doc_id)
        """
        # Combine source texts
        source_texts = []
        for i, source in enumerate(sources):
            if hasattr(source, 'page_content'):
                source_texts.append(f"[Source {i+1}]: {source.page_content}")
            else:
                source_texts.append(f"[Source {i+1}]: {str(source)}")
        
        combined_sources = "\n\n".join(source_texts)
        
        prompt = f"""Determine if this claim is supported by the sources.

Claim: {claim}

Sources:
{combined_sources[:3000]}

Answer in this exact format:
VERDICT: YES or NO
EVIDENCE: Quote the specific text that supports the claim (if YES) or explain why not supported (if NO)
SOURCE: Which source number (if YES)"""

        result = self.llm.invoke(prompt)
        content = result.content
        
        # Parse response
        is_verified = "VERDICT: YES" in content.upper()
        
        evidence = None
        if "EVIDENCE:" in content:
            evidence = content.split("EVIDENCE:")[1].split("\n")[0].strip()
        
        source_id = None
        if "SOURCE:" in content and is_verified:
            source_id = content.split("SOURCE:")[1].split("\n")[0].strip()
        
        return is_verified, evidence, source_id
    
    def calculate_groundedness(self, response: str, sources: list) -> GroundednessResult:
        """
        Calculate groundedness score for a response.
        
        Args:
            response: Agent's generated response
            sources: List of retrieved documents (LangChain Document objects or strings)
            
        Returns:
            GroundednessResult with score, claims, confidence, and recommendation
        """
        # Extract claims
        claims = self.extract_claims(response)
        
        if not claims:
            return GroundednessResult(
                score=1.0,
                claims=[],
                confidence="high",
                recommendation="proceed"
            )
        
        # Verify each claim
        verified_claims = []
        for claim in claims:
            is_verified, evidence, source_id = self.find_evidence(claim, sources)
            verified_claims.append(VerifiedClaim(
                claim=claim,
                verified=is_verified,
                evidence=evidence,
                source_doc=source_id
            ))
        
        # Calculate score
        verified_count = sum(1 for c in verified_claims if c.verified)
        score = verified_count / len(claims)
        
        # Determine confidence and recommendation
        if score >= 0.9:
            confidence = "high"
            recommendation = "proceed"
        elif score >= 0.7:
            confidence = "medium"
            recommendation = "add_caveats"
        elif score >= 0.5:
            confidence = "low"
            recommendation = "clarify"
        else:
            confidence = "very_low"
            recommendation = "reject"
        
        return GroundednessResult(
            score=score,
            claims=verified_claims,
            confidence=confidence,
            recommendation=recommendation
        )
    
    def format_for_log(self, result: GroundednessResult) -> str:
        """Format result for implementation trace logging."""
        lines = [
            "=" * 50,
            "VERIFICATION REPORT",
            "=" * 50,
            f"Groundedness Score: {result.score:.2f}",
            f"Confidence: {result.confidence}",
            f"Recommendation: {result.recommendation}",
            "",
            "Claims Analysis:",
        ]
        
        for i, claim in enumerate(result.claims, 1):
            status = "✓" if claim.verified else "✗"
            lines.append(f"  {i}. [{status}] {claim.claim[:60]}...")
            if claim.evidence:
                lines.append(f"      Evidence: {claim.evidence[:50]}...")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def handle_verification(result: GroundednessResult, response: str) -> str:
    """
    Handle verification result by modifying response if needed.
    
    Args:
        result: GroundednessResult from scorer
        response: Original agent response
        
    Returns:
        Modified response based on verification outcome
    """
    if result.recommendation == "proceed":
        return response
    
    elif result.recommendation == "add_caveats":
        unverified = [c.claim for c in result.claims if not c.verified]
        caveat = f"\n\nNote: The following statements could not be fully verified: {', '.join(unverified[:2])}"
        return response + caveat
    
    elif result.recommendation == "clarify":
        return f"I found some relevant information, but I'm not fully confident in the details. {response}\n\nWould you like me to search for more specific information?"
    
    else:  # reject
        return "I apologize, but I couldn't find reliable information to answer your question. Could you please rephrase or provide more context?"
