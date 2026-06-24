# src/filters.py
import re

def is_honeypot_or_invalid(candidate: dict) -> bool:
    """
    Senior AI Engineer Honeypot Shield. Identifies and defuses complex synthetic 
    traps, timeline anachronisms, and keyword stuffing behavior.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    
    # 1. Experience Bracket Enforcement (Strict Target: 5-9 Years)
    yoe = profile.get("years_of_experience", 0.0)
    if yoe < 5.0 or yoe > 9.0:
        return True

    # 2. Trap Defusal 1: Expert Proficiency with Zero Experience Claims
    expert_zero_traps = [
        s for s in skills 
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero_traps) >= 3:
        return True

    # 3. Trap Defusal 2: Startup Timeline Verification (Sarvam / Krutrim founded 2023)
    for job in history:
        comp = job.get("company", "")
        start = job.get("start_date", "")
        dur = job.get("duration_months", 0)
        
        if comp in ["Sarvam AI", "Krutrim"]:
            if dur > 36:  # Exceeds operational lifespan relative to mid-2026 baseline
                return True
            if start:
                try:
                    start_year = int(start.split("-")[0])
                    if start_year < 2023:  # Timeline violation
                        return True
                except (ValueError, IndexError):
                    pass

    # 4. Trap Defusal 3: Modern AI Framework Lifecycles (Post-2021 adoption markers)
    modern_frameworks = {"Pinecone", "LoRA", "Fine-tuning LLMs", "Weights & Biases", "RAG"}
    for s in skills:
        name = s.get("name", "")
        dur = s.get("duration_months", 0)
        if name in modern_frameworks and dur > 60:
            return True

    # 5. Trap Defusal 4: Single Professional Job Tenure Impossibility
    for job in history:
        dur = job.get("duration_months", 0)
        if dur > (yoe * 12 + 6):
            return True

    # 6. Corporate Track Penalty: Consulting-Only Exclusions
    consulting_firms = {
        "tcs", "tata consultancy services", "infosys", "wipro", 
        "accenture", "cognizant", "capgemini", "tech mahindra"
    }
    all_consulting = True if history else False
    for job in history:
        comp_lower = job.get("company", "").lower()
        if not any(firm in comp_lower for firm in consulting_firms):
            all_consulting = False
            break
            
    if all_consulting and len(history) > 0:
        return True  # Filter profiles with only consulting backgrounds per the JD spec

    # 7. Basic Platform Availability Floor
    if signals.get("recruiter_response_rate", 0.0) < 0.05:
        return True

    return False