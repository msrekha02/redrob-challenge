# src/scoring.py
from datetime import datetime

def calculate_active_multiplier(last_active_str: str) -> float:
    """
    Computes a continuous active decay curve based on the challenge baseline.
    M_Active = max(0.85, 1.0 - (days_since_active / 365.0))
    """
    try:
        # Fixed challenge context reference date: June 23, 2026
        baseline_date = datetime(2026, 6, 23)
        last_active = datetime.strptime(last_active_str.strip(), "%Y-%m-%d")
        days_diff = (baseline_date - last_active).days
        
        if days_diff <= 30:
            return 1.0
        return max(0.85, 1.0 - (days_diff / 365.0))
    except Exception:
        return 0.85

def calculate_engagement_multiplier(recruiter_response_rate: float) -> float:
    """
    Smoothly down-weights inactive candidate accounts without zeroing out their scores.
    M_Response = 0.4 + (0.6 * rate)
    """
    rate = max(0.0, min(1.0, recruiter_response_rate))
    return 0.4 + (0.6 * rate)

def calculate_notice_multiplier(notice_period_days: int) -> float:
    """
    Applies continuous prioritization weights for immediate availability.
    W_Notice = 0.8 + (0.2 * (1.0 if notice <= 30 else 0.5))
    """
    factor = 1.0 if notice_period_days <= 30 else 0.5
    return 0.8 + (0.2 * factor)