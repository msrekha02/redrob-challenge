# src/filters.py

def is_honeypot_or_invalid(candidate: dict) -> bool:
    """
    Evaluates profile schema properties to catch keyword-stuffer traps,
    unsupported location anomalies, and out-of-bounds career footprints.
    Returns True if the profile should be dropped immediately.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    # 1. Experience Envelope Constraint (Targeting 5-9 Years)
    yoe = profile.get("years_of_experience", 0)
    if yoe < 5.0 or yoe > 9.0:
        return True

    # 2. Non-Technical / Core Target Role Title Check (Trap Defusal)
    current_title = profile.get("current_title", "").lower()
    invalid_titles = ["marketing manager", "sales executive", "content writer", "graphic designer", "hr manager"]
    if any(title in current_title for title in invalid_titles):
        return True

    # 3. Location & Relocation Track Constraints
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    
    allowed_cities = ["noida", "pune", "bangalore", "bengaluru", "hyderabad", "chennai", "mumbai", "delhi", "gurgaon", "gurugram"]
    is_target_city = any(city in location for city in allowed_cities) or country == "india"
    
    if not is_target_city and not willing_to_relocate:
        return True

    # 4. Availability / Fresh Engagement Trap
    recruiter_response = signals.get("recruiter_response_rate", 0.0)
    if recruiter_response < 0.05:
        return True

    return False