"""
Survey Tool for Alumni Feedback Collection

Creates and sends surveys to alumni via Google Forms.
"""

from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field


@dataclass
class SurveyResult:
    """Result of survey creation/send."""
    success: bool
    survey_url: Optional[str]
    alumni_id: str
    survey_type: str
    error: Optional[str] = None


# Survey types
SURVEY_TYPES = {
    "career_update": {
        "title": "CMU Africa Alumni Career Update Survey",
        "description": "Help us stay updated on your career journey",
        "questions": [
            "What is your current job title?",
            "What company do you work for?",
            "What skills have been most valuable in your career?",
            "Would you be interested in mentoring current students?"
        ]
    },
    "feedback": {
        "title": "CMU Africa Program Feedback Survey",
        "description": "Share your experience to help improve our programs",
        "questions": [
            "How well did CMU Africa prepare you for your career?",
            "What courses were most valuable?",
            "What would you change about the program?",
            "Would you recommend CMU Africa to others?"
        ]
    },
    "networking": {
        "title": "CMU Africa Alumni Networking Preferences",
        "description": "Help us connect you with fellow alumni",
        "questions": [
            "What industries are you interested in?",
            "What type of networking events do you prefer?",
            "Would you like to be matched with a mentor or mentee?",
            "What skills would you like to learn from others?"
        ]
    },
    "support_needs": {
        "title": "CMU Africa Alumni Support Assessment",
        "description": "Let us know how we can help",
        "questions": [
            "Are you currently seeking new career opportunities?",
            "Would you benefit from career counseling?",
            "Are you interested in further education?",
            "How can the alumni network best support you?"
        ]
    }
}


class SurveyTool:
    """Create and send surveys to alumni."""
    
    def create_survey(self, survey_type: str, alumni_id: str) -> SurveyResult:
        """
        Create a survey for an alumni (mock implementation).
        
        In production, this would integrate with Google Forms API.
        """
        if survey_type not in SURVEY_TYPES:
            return SurveyResult(
                success=False, survey_url=None, alumni_id=alumni_id,
                survey_type=survey_type, error=f"Unknown survey type: {survey_type}"
            )
        
        survey_config = SURVEY_TYPES[survey_type]
        
        # Mock URL generation
        survey_url = f"https://forms.google.com/d/e/mock_{survey_type}_{alumni_id}_{datetime.now().timestamp()}/viewform"
        
        print(f"[MOCK] Created survey: {survey_config['title']}")
        print(f"  URL: {survey_url}")
        print(f"  Alumni: {alumni_id}")
        
        return SurveyResult(
            success=True, survey_url=survey_url,
            alumni_id=alumni_id, survey_type=survey_type
        )
    
    def get_available_types(self) -> list:
        return list(SURVEY_TYPES.keys())


# LangChain Tool
class SurveyInput(BaseModel):
    survey_type: str = Field(description="Survey type: career_update, feedback, networking, support_needs")
    alumni_id: str = Field(description="Alumni ID to send survey to")


@tool(args_schema=SurveyInput)
def survey_tool(survey_type: str, alumni_id: str) -> dict:
    """
    Create and send a survey to an alumni.
    Available types: career_update, feedback, networking, support_needs
    """
    tool = SurveyTool()
    result = tool.create_survey(survey_type, alumni_id)
    return {
        "success": result.success,
        "survey_url": result.survey_url,
        "alumni_id": result.alumni_id,
        "survey_type": result.survey_type,
        "error": result.error
    }


def create_survey_tool():
    """Create configured survey tool instance."""
    return survey_tool
