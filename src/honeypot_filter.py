import json
import os
from datetime import datetime

# --- CONFIGURATION & PATHS ---
CANDIDATES_FILE = "../data/candidates.jsonl"
CLEANED_OUTPUT_FILE = "../data/cleaned_candidates.jsonl"

# --- HONEYPOT CONFIGURATION CONSTANTS ---
GLOBAL_BASELINE_YEAR = 2026

FICTIONAL_COMPANIES = {
    "INITECH", "PIED PIPER", "WAYNE ENTERPRISES", "ACME CORP", 
    "STARK INDUSTRIES", "HOOLI", "GLOBEX INC", "DUNDER MIFFLIN"
}

COMPANY_FOUNDING_YEARS = {
    "CRED": 2018,
    "GLANCE": 2019,
    "REPHRASE.AI": 2019,
    "SARVAM AI": 2023,       
    "KRUTRIM": 2023
}

# Academic stream establishment ceilings (Earliest valid initialization years)
ESTABLISHMENT_CEILINGS = {
    "B.TECH":   {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "B.SC.":    {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "M.TECH.":  {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "M.SC.":    {"ARTIFICIAL INTELLIGENCE": 2020, "DATA SCIENCE": 2018, "MACHINE LEARNING": 2020},
    "PH.D.":    {"ARTIFICIAL INTELLIGENCE": 2015, "DATA SCIENCE": 2016, "MACHINE LEARNING": 2015}
}

# Modern technology real-world market emergence restrictions
TECH_INVENTION_CEILINGS = {
    "QLORA": 34, # Maximum months active in June 2026 (Invented Mid-2023)
    "LORA": 45   # Maximum months active in June 2026 (Published Late-2021)
}


def is_honeypot_candidate(cand: dict) -> tuple:
    """
    Evaluates a candidate profile against all known synthetic dataset anomalies.
    Returns: (bool, str) -> (True and reason if trapped, False and empty string if safe)
    """
    profile = cand.get("profile", {})
    education = cand.get("education", [])
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # --------------------------------------------------------------------------
    # TRAP 1: FICTIONAL POP-CULTURE COMPANIES
    # --------------------------------------------------------------------------
    for job in career:
        comp_name = str(job.get("company", "")).strip().upper()
        if any(fictional in comp_name for fictional in FICTIONAL_COMPANIES):
            return True, f"Fictional corporate footprint: {comp_name}"

    # --------------------------------------------------------------------------
    # TRAP 2: COMPANY ESTABLISHMENT VS. TENURE CHRONOLOGY
    # --------------------------------------------------------------------------
    for job in career:
        comp_name = str(job.get("company", "")).strip().upper()
        start_date_str = job.get("start_date")
        if start_date_str and len(start_date_str) >= 4:
            try:
                job_start_year = int(start_date_str[:4])
                for target_comp, founding_year in COMPANY_FOUNDING_YEARS.items():
                    if target_comp in comp_name and job_start_year < founding_year:
                        return True, f"Worked at {comp_name} in {job_start_year} before incorporation ({founding_year})"
            except ValueError:
                continue

    # --------------------------------------------------------------------------
    # TRAP 3: ACADEMIC TIMELINE STRUCTURES (DURATIONS & COHORT DRIFTS)
    # --------------------------------------------------------------------------
    max_undergrad_grad_year = None
    earliest_undergrad_start_year = None
    postgrad_end_year = None
    bachelors_start_year = None

    for edu in education:
        degree = str(edu.get("degree", "")).strip().upper()
        field = str(edu.get("field_of_study", "")).strip().upper()
        start = edu.get("start_year")
        end = edu.get("end_year")
        
        if not start or not end:
            continue
            
        duration = end - start
        
        # Track edge indicators for the Time-Travel Education Trap
        if any(b_deg in degree for b_deg in ["B.TECH", "B.SC", "BACHELOR", "B.E"]):
            bachelors_start_year = start
            if max_undergrad_grad_year is None or end > max_undergrad_grad_year:
                max_undergrad_grad_year = end
            if earliest_undergrad_start_year is None or start < earliest_undergrad_start_year:
                earliest_undergrad_start_year = start
        
        if any(m_deg in degree for m_deg in ["M.TECH", "M.SC", "MBA",  "PH.D"]):
            postgrad_end_year = end

        # Extended Postgrad Duration Trap (Master's stretched to Undergraduate length)
        if any(m_deg in degree for m_deg in ["M.SC", "MBA", "M.TECH", "M.E"]) and duration >= 4:
            return True, f"Impossible Master degree duration: {degree} held for {duration} years"
            
        # Extended Undergrad Duration Trap (Standard courses stretched excessively)
        if any(b_deg in degree for b_deg in ["B.TECH", "B.SC", "BACHELOR", "B.E"]) and duration >= 5:
            return True, f"Impossible Undergraduate degree duration: {degree} held for {duration} years"

        # Stream Establishment Year Trap (Dedicated AI/ML courses did not exist prior to these lines)
        if degree in ESTABLISHMENT_CEILINGS:
            for specialized_stream, milestone_year in ESTABLISHMENT_CEILINGS[degree].items():
                if specialized_stream in field and start < milestone_year:
                    return True, f"Historical anomaly: {degree} in {field} initiated in {start} (Pre-dates {milestone_year} infrastructure)"

    # --------------------------------------------------------------------------
    # TRAP 4: CHRONOLOGICAL PARADOXES (TIME-TRAVEL IN EDUCATION & WORK)
    # --------------------------------------------------------------------------
    # Chronological Paradox A: Completed Master's program before starting Bachelor's
    if postgrad_end_year and bachelors_start_year and postgrad_end_year < bachelors_start_year:
        return True, "Chronological Paradox: Completed postgraduate track before entering undergraduate tier"

    # Chronological Paradox B: Corporate career started before college graduation (Applying Safe >1 Year Filter)
    if max_undergrad_grad_year:
        for job in career:
            start_date_str = job.get("start_date")
            if start_date_str and len(start_date_str) >= 4:
                try:
                    job_start_year = int(start_date_str[:4])
                    if (max_undergrad_grad_year - job_start_year) > 1:
                        return True, f"Full-time employment started in {job_start_year} before graduation tier in {max_undergrad_grad_year}"
                except ValueError:
                    continue

    # --------------------------------------------------------------------------
    # TRAP 5: CAREER INTEGRITY INVERSIONS & METRIC OVERLAPS
    # --------------------------------------------------------------------------
    for job in career:
        start_date_str = job.get("start_date")
        end_date_str = job.get("end_date")
        
        if start_date_str and end_date_str:
            try:
                # Direct string comparisons are safe and lightning-fast for ISO format 'YYYY-MM-DD'
                if end_date_str < start_date_str:
                    return True, f"Career Timeline Inversion: Job end date ({end_date_str}) predates start date ({start_date_str})"
            except Exception:
                continue

    # --------------------------------------------------------------------------
    # TRAP 6: PLATFORM BEHAVIORAL SIGNAL DISCONNECTS
    # --------------------------------------------------------------------------
    signup_date = signals.get("signup_date")
    last_active = signals.get("last_active_date")
    
    # Platform date logic anomaly
    if signup_date and last_active and signup_date > last_active:
        return True, f"Platform chronological mismatch: Signup ({signup_date}) occurs after Last Active ({last_active})"

    # Financial scale contradiction
    salary_obj = signals.get("expected_salary_range_inr_lpa", {})
    min_salary = salary_obj.get("min", 0.0)
    max_salary = salary_obj.get("max", 0.0)
    if min_salary > max_salary:
        return True, f"Broken scale range: Stated minimum salary ({min_salary} LPA) exceeds maximum ({max_salary} LPA)"

    # Ceiling evaluation stuffing (Impossibly high scores across unrelated domains)
    assessment_scores = signals.get("skill_assessment_scores") or {}
    suspicious_scores = sum(1 for score_val in assessment_scores.values() if float(score_val) >= 99.0)
    if suspicious_scores >= 5:
        return True, "Suspicious performance padding: Near-perfect assessment scores across >= 5 technical modules"

    # --------------------------------------------------------------------------
    # TRAP 7: SKILL DURATION INFLATION VS. PROFESSIONAL FOOTPRINT
    # --------------------------------------------------------------------------
    total_experience_years = profile.get("years_of_experience", 0.0)
    total_experience_months = total_experience_years * 12.0
    
    for s in skills:
        s_name = str(s.get("name", "")).strip().upper()
        s_dur = s.get("duration_months", 0)
        
        # Skill runtime duration exceeds overall professional experience footprint
        if s_dur > (total_experience_months + 2.0): # 2-month threshold to account for slight rounding variances
            return True, f"Skill runtime inflation: {s_name} held for {s_dur} months, exceeding total profile runtime ({round(total_experience_months, 1)} months)"
            
        # Tech emergence ceilings (Catches time-traveling modern frameworks)
        if s_name in TECH_INVENTION_CEILINGS:
            allowed_max = TECH_INVENTION_CEILINGS[s_name]
            if s_dur > allowed_max:
                return True, f"Time-traveling framework footprint: {s_name} runtime states {s_dur} months (Market availability ceiling is {allowed_max} months)"

    return False, ""


def execution_pipeline():
    print("=" * 65)
    print("REDROB TALENT RANKER — HONEYPOT PREPROCESSING ENGINE")
    print("=" * 65)
    
    if not os.path.exists(CANDIDATES_FILE):
        print(f"[!] SYSTEM ERROR: Source file '{CANDIDATES_FILE}' missing.")
        return

    # Ensure output destination directory exists securely
    os.makedirs(os.path.dirname(CLEANED_OUTPUT_FILE), exist_ok=True)
    
    processed_count = 0
    clean_count = 0
    trap_counts = collections.Counter()
    
    print("Processing candidate stream execution data... Please wait.")
    start_time = datetime.now()

    with open(CANDIDATES_FILE, "r", encoding="utf-8") as infile, \
         open(CLEANED_OUTPUT_FILE, "w", encoding="utf-8") as outfile:
         
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            processed_count += 1
            
            # Evaluate current candidate block for data traps
            is_trapped, reason_str = is_honeypot_candidate(candidate)
            
            if is_trapped:
                # Isolate core label categories for reporting metrics
                category_label = reason_str.split(":")[0]
                trap_counts[category_label] += 1
                
                # Append a hard null-score tracking token so your subsequent ranking layer 
                # can permanently skip evaluating this specific candidate ID.
                candidate["suitability_rank_score"] = 0.0
                candidate["is_honeypot_trap"] = True
                candidate["filter_pruning_reason"] = reason_str
            else:
                clean_count += 1
                candidate["suitability_rank_score"] = -1.0 # Initialize safe state for ranker
                candidate["is_honeypot_trap"] = False
                candidate["filter_pruning_reason"] = ""
                outfile.write(json.dumps(candidate, ensure_ascii=False) + "\n")

                
            # Write all candidate tracking records back to store to maintain streaming line index alignment
            # outfile.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 65)
    print("### DATASET SANITIZATION EXECUTION REPORT ###")
    print("=" * 65)
    print(f"Total Stream Processing Time   : {elapsed_time:.2f} seconds")
    print(f"Total Candidate Files Scanned  : {processed_count}")
    print(f"Sanitized Profiles Passed (NO) : {clean_count}")
    print(f"Honeypots Pruned Out (YES)     : {processed_count - clean_count}")
    print(f"Dataset Global Purity Rate     : {(clean_count / processed_count) * 100:.2f}%")
    print("-" * 65)
    print("### ISOLATED ANOMALY TRACKING METRICS ###")
    for trap_type, count in trap_counts.items():
        print(f" - {trap_type:<35}: {count} candidates")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import collections
    execution_pipeline()