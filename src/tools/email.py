"""
Email Tool for Alumni Outreach

Sends personalized emails to alumni based on templates.
"""

import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field


@dataclass
class EmailResult:
    """Result of email send attempt."""
    success: bool
    message_id: Optional[str]
    recipient: str
    sent_at: Optional[datetime]
    error: Optional[str] = None


# Email templates
TEMPLATES = {
    "congratulations_promotion": {
        "subject": "Congratulations on Your Promotion, {name}!",
        "body": """Dear {name},

We were thrilled to learn about your promotion to {new_role} at {company}!

This is a wonderful achievement that reflects your hard work and dedication since graduating from CMU Africa in {graduation_year}.

We'd love to hear more about your journey. Would you be interested in sharing your experience with current students or fellow alumni?

Warm regards,
CMU Africa Alumni Relations Team"""
    },
    
    "congratulations_new_job": {
        "subject": "Congratulations on Your New Position, {name}!",
        "body": """Dear {name},

Congratulations on your new role as {new_role} at {company}!

We're proud to see CMU Africa alumni making strides in their careers. Your success story is an inspiration to our community.

Best wishes in your new position!

Warm regards,
CMU Africa Alumni Relations Team"""
    },
    
    "offer_support": {
        "subject": "Checking In - CMU Africa Alumni Support",
        "body": """Dear {name},

We hope this message finds you well. As a valued member of the CMU Africa alumni community, we wanted to reach out and see how things are going.

We noticed you might be in a period of career transition. Please know that our alumni network is here to support you:

- Career counseling sessions
- Networking events with industry professionals  
- Job board access
- Mentorship matching

Would you like to schedule a call to discuss how we can support you?

With warm regards,
CMU Africa Alumni Relations Team"""
    },
    
    "general_check_in": {
        "subject": "Catching Up - CMU Africa Alumni",
        "body": """Dear {name},

Greetings from CMU Africa!

It's been a while since we connected, and we'd love to hear how you're doing.

As part of our alumni community, your journey matters to us. We'd be grateful if you could take a moment to share any updates about your career, accomplishments, or ways we might support you.

Looking forward to hearing from you!

Best regards,
CMU Africa Alumni Relations Team"""
    },
    
    "survey_request": {
        "subject": "Quick Survey - Help Shape CMU Africa's Future",
        "body": """Dear {name},

Your feedback matters! As a CMU Africa graduate, your insights help us improve our programs and better serve future students.

Please take 5 minutes to complete our annual alumni survey:
{survey_link}

Your responses will remain confidential and will be used to enhance our curriculum and student support services.

Thank you for contributing to CMU Africa's mission!

Best regards,
CMU Africa Alumni Relations Team"""
    }
}


class EmailTool:
    """Send personalized emails to alumni."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.from_email = os.environ.get("FROM_EMAIL", "dev.ngodwill@gmail.com")
    
    def send_email(self, recipient_email: str, template_name: str, personalization: dict) -> EmailResult:
        """Send a personalized email."""
        template = TEMPLATES.get(template_name)
        if not template:
            return EmailResult(
                success=False, message_id=None, recipient=recipient_email,
                sent_at=None, error=f"Unknown template: {template_name}"
            )
        
        try:
            subject = template["subject"].format(**personalization)
            body = template["body"].format(**personalization)
        except KeyError as e:
            return EmailResult(
                success=False, message_id=None, recipient=recipient_email,
                sent_at=None, error=f"Missing personalization key: {e}"
            )
        
        # Dry run mode - don't actually send
        if self.dry_run:
            message_id = f"dry_run_{datetime.now().timestamp()}"
            print(f"[DRY RUN] Would send email to {recipient_email}")
            print(f"  Subject: {subject}")
            print(f"  Template: {template_name}")
            return EmailResult(
                success=True, message_id=message_id,
                recipient=recipient_email, sent_at=datetime.now()
            )
        
        # Production would send via SMTP here
        try:
            import smtplib
            import ssl
            from email.message import EmailMessage
            
            smtp_host = os.environ.get("SMTP_HOST")
            smtp_port = int(os.environ.get("SMTP_PORT", 587))
            smtp_user = os.environ.get("SMTP_USER")
            smtp_password = os.environ.get("SMTP_PASSWORD")
            
            if not all([smtp_host, smtp_user, smtp_password]):
                return EmailResult(
                    success=False, message_id=None, recipient=recipient_email, 
                    sent_at=None, error="Missing SMTP configuration"
                )
                
            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = recipient_email
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
                
            return EmailResult(
                success=True, message_id=f"msg_{datetime.now().timestamp()}",
                recipient=recipient_email, sent_at=datetime.now()
            )
        except Exception as e:
            return EmailResult(
                success=False, message_id=None, recipient=recipient_email,
                sent_at=None, error=f"SMTP Error: {str(e)}"
            )
    
    def get_available_templates(self) -> list:
        return list(TEMPLATES.keys())


# LangChain Tool
class EmailInput(BaseModel):
    recipient_email: str = Field(description="Email address")
    template: str = Field(description="Template: congratulations_promotion, congratulations_new_job, offer_support, general_check_in, survey_request")
    personalization: dict = Field(description="Personalization data: name, new_role, company, graduation_year")


@tool(args_schema=EmailInput)
def email_sender(recipient_email: str, template: str, personalization: dict) -> dict:
    """
    Send a personalized email to an alumni.
    Available templates: congratulations_promotion, congratulations_new_job, offer_support, general_check_in, survey_request
    """
    tool = EmailTool(dry_run=False)
    result = tool.send_email(recipient_email, template, personalization)
    return {
        "success": result.success,
        "message_id": result.message_id,
        "recipient": result.recipient,
        "error": result.error
    }


def create_email_tool(dry_run: bool = True):
    """Create configured email tool instance."""
    return email_sender
