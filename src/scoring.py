# src/scoring.py
from datetime import datetime

def calculate_active_multiplier(last_active_str: str) -> float:
    """
    Computes a smooth decay curve tracking baseline activity over months.
    M_Active = max(0.85, 1.0 - (days_since_last_active / 365))
    """
    try:
        # Challenge baseline date context is 2026
        current_date = datetime(2026, 6, 23)
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
        days_diff = (current_date - last_active).days
        
        if days_diff <= 30:
            return 1.0
        
        decay = 1.0 - (days_diff / 365.0)
        return max(0.85, decay)
    except Exception:
        return 0.85  # Safe protective baseline floor

def calculate_engagement_multiplier(recruiter_response_rate: float) -> float:
    """
    Down-weights uncommunicative candidate profiles without resetting scores to zero.
    M_Response = 0.4 + (0.6 * recruiter_response_rate)
    """
    rate = max(0.0, min(1.0, recruiter_response_rate))
    return 0.4 + (0.6 * rate)

def calculate_notice_multiplier(notice_period_days: int) -> float:
    """
    Adjusts priority based on immediate availability to match Series-A startup agility.
    W_Notice = 0.8 + (0.2 * (1.0 if notice_period_days <= 30 else 0.5))
    """
    factor = 1.0 if notice_period_days <= 30 else 0.5
    return 0.8 + (0.2 * factor)