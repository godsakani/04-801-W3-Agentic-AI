"""
Tool Validators — Prerequisite validation for agent tools.

Defines schemas and validation logic to ensure tool parameters
meet required conditions before execution.
"""


# ============================================================
# Prerequisite Validation Schemas
# ============================================================
TOOL_PREREQUISITES = {
    "email_sender": {
        "required": ["recipient_email", "template", "personalization"],
        "requires_context": True,
        "validators": {
            "recipient_email": lambda v: isinstance(v, str) and "@" in v and len(v) > 5,
            "template": lambda v: v in [
                "general_check_in", "congratulations_promotion",
                "congratulations_new_job", "offer_support", "survey_request"
            ],
            "personalization": lambda v: isinstance(v, dict) and "name" in v,
        }
    },
    "linkedin_scraper": {
        "required": ["profile_url"],
        "requires_context": False,
        "validators": {
            "profile_url": lambda v: isinstance(v, str) and ("linkedin.com" in v or v.startswith("http")),
        }
    },
    "survey_tool": {
        "required": ["survey_type", "alumni_id"],
        "requires_context": True,
        "validators": {
            "survey_type": lambda v: v in ["career_update", "feedback", "networking", "support_needs"],
            "alumni_id": lambda v: isinstance(v, str) and len(v) > 0,
        }
    },
}


def validate_tool_params(tool_name: str, params: dict, has_context: bool) -> tuple:
    """
    Validate tool parameters against prerequisite schema.
    
    Returns:
        (is_valid: bool, error_message: str or None)
    """
    schema = TOOL_PREREQUISITES.get(tool_name)
    if not schema:
        return True, None  # No schema defined, allow execution
    
    # Check if context is required but missing
    if schema["requires_context"] and not has_context:
        return False, f"Tool '{tool_name}' requires retrieved context before execution. Must RETRIEVE first."
    
    # Check required fields are present and non-empty
    for field in schema["required"]:
        if field not in params or params[field] is None or params[field] == "":
            return False, f"Missing required parameter '{field}' for tool '{tool_name}'. Got params: {list(params.keys())}"
    
    # Run field-level validators
    for field, validator_fn in schema.get("validators", {}).items():
        if field in params:
            try:
                if not validator_fn(params[field]):
                    return False, f"Invalid value for '{field}' in tool '{tool_name}': {repr(params[field])}"
            except Exception as e:
                return False, f"Validation error for '{field}' in tool '{tool_name}': {e}"
    
    return True, None
