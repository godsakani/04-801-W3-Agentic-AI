"""
LinkedIn Scraper Tool

Monitors alumni LinkedIn profiles for career changes.
WARNING: LinkedIn scraping may violate ToS. Use for educational purposes only.
"""

import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field


@dataclass
class AlumniProfile:
    """Structured alumni profile data from LinkedIn."""
    name: str
    headline: str
    current_job: str
    company: str
    location: str
    skills: list
    scraped_at: datetime


class LinkedInScraper:
    """
    Scrapes public LinkedIn profiles for alumni monitoring.
    Rate limited to avoid detection.
    """
    
    def __init__(self, rate_limit_seconds: int = 30):
        self.rate_limit = rate_limit_seconds
        self.last_request_time = None
    
    def scrape_profile(self, profile_url: str) -> dict:
        """
        Scrape a single LinkedIn profile (mock implementation for demo).
        
        In production, this would use Selenium to scrape public profiles.
        """
        # Mock response for demo - avoids actual scraping
        return {
            "success": True,
            "profile_data": {
                "name": "John Doe",
                "headline": "Senior Data Engineer at TechCorp",
                "current_job": "Senior Data Engineer",
                "company": "TechCorp",
                "location": "Nairobi, Kenya",
                "skills": ["Python", "MongoDB", "Machine Learning"],
                "scraped_at": datetime.now().isoformat()
            },
            "error": None
        }
    
    def detect_changes(self, new_data: dict, old_data: dict) -> list:
        """Compare profiles to detect career changes."""
        changes = []
        
        if new_data.get("current_job") != old_data.get("current_job"):
            if self._is_promotion(old_data.get("current_job", ""), new_data.get("current_job", "")):
                changes.append("promotion")
            else:
                changes.append("job_change")
        
        if new_data.get("company") != old_data.get("company"):
            changes.append("company_change")
        
        if new_data.get("location") != old_data.get("location"):
            changes.append("location_change")
        
        return changes
    
    def _is_promotion(self, old_title: str, new_title: str) -> bool:
        """Check if job change is a promotion."""
        senior_keywords = ["senior", "lead", "principal", "director", "head", "manager"]
        old_lower = old_title.lower()
        new_lower = new_title.lower()
        
        for keyword in senior_keywords:
            if keyword in new_lower and keyword not in old_lower:
                return True
        return False


# LangChain Tool
class LinkedInInput(BaseModel):
    profile_url: str = Field(description="LinkedIn profile URL to scrape")


@tool(args_schema=LinkedInInput)
def linkedin_scraper(profile_url: str) -> dict:
    """
    Scrape a LinkedIn profile for job changes and updates.
    Returns profile data and detected changes.
    """
    scraper = LinkedInScraper()
    return scraper.scrape_profile(profile_url)


def create_linkedin_tool():
    """Create configured LinkedIn tool instance."""
    return linkedin_scraper
