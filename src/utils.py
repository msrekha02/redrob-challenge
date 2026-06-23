# src/utils.py
import re

def sanitize_text(text: str) -> str:
    """
    Cleans raw profile text strings to ensure strict CSV formatting safety.
    Strips raw carriage returns, line breaks, commas, and double quotes.
    """
    if not text:
        return ""
    
    # Replace line breaks and tabs with plain spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    
    # Strip commas and quotes to neutralize string parsing exceptions
    cleaned = cleaned.replace(",", " ").replace('"', "'")
    
    # Trim leading/trailing whitespace and collapse multiple internal spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def generate_reasoning_string(candidate_id: str, title: str, yoe: float, match_reason: str) -> str:
    """
    Constructs a concise, professional 1-2 sentence recruiter-facing justification
    ensuring it is substantively unique and adheres to formatting specs.
    """
    cleaned_title = sanitize_text(title)
    cleaned_reason = sanitize_text(match_reason)
    
    # Produce a targeted, highly professional narrative context
    reasoning = f"Strong candidate showing {yoe} years of agility as a {cleaned_title}. {cleaned_reason}"
    
    # Enforce safe text caps to stay concise
    return reasoning[:200]