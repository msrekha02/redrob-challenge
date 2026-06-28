import json
import os
import collections
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
    "B.TECH.":  {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "B.SC":     {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "B.SC.":    {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "M.TECH":   {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "M.TECH.":  {"ARTIFICIAL INTELLIGENCE": 2019, "DATA SCIENCE": 2019, "MACHINE LEARNING": 2019},
    "M.SC":     {"ARTIFICIAL INTELLIGENCE": 2020, "DATA SCIENCE": 2018, "MACHINE LEARNING": 2020},
    "M.SC.":    {"ARTIFICIAL INTELLIGENCE": 2020, "DATA SCIENCE": 2018, "MACHINE LEARNING": 2020},
    "PH.D":     {"ARTIFICIAL INTELLIGENCE": 2015, "DATA SCIENCE": 2016, "MACHINE LEARNING": 2015},
    "PH.D.":    {"ARTIFICIAL INTELLIGENCE": 2015, "DATA SCIENCE": 2016, "MACHINE LEARNING": 2015}
}

# Modern technology real-world market emergence restrictions
TECH_INVENTION_CEILINGS = {
    # PARAMETER-EFFICIENT FINE-TUNING (PEFT)
    "QLORA": 37,                # INVENTED MAY 2023 (TIM DETTMERS PAPER)
    "LORA": 60,                 # PUBLISHED JUNE 2021 (MICROSOFT PAPER)
    "PEFT": 40,                 # HUGGINGFACE PEFT LIBRARY RELEASED EARLY 2023
    "FINE-TUNING": 108,         # WIDESPREAD POST-TRANSFORMERS ERA
    "FINE TUNING": 108,
    "FINETUNING": 108,
    
    # MODERN FRAMEWORKS & ORCHESTRATORS
    "LANGCHAIN": 44,            # OPEN-SOURCED OCTOBER 2022 BY HARRISON CHASE
    "LLAMAINDEX": 43,           # OPEN-SOURCED NOVEMBER 2022 AS GPT_INDEX
    "HAYSTACK": 72,             # DEEPSET HAYSTACK FRAMEWORK EVOLVED INTO LLM SPACE ~2020
    "PROMPT ENGINEERING": 66,   # ROSE AS A DEFINED DISCIPLINE ALONGSIDE GPT-3 ACCESS
    
    # STATE-OF-THE-ART EMBEDDING ARCHITECTURES
    "BGE": 34,                  # BAAI GENERAL EMBEDDINGS PUBLISHED AUGUST 2023
    "E5": 34,                   # MICROSOFT E5 EMBEDDING FAMILY RELEASED MID-2023
    "OPENAI EMBEDDINGS": 54,     # TEXT-EMBEDDING-ADA-002 LAUNCHED DECEMBER 2020
    "SENTENCE TRANSFORMERS": 79, # SBERT PAPER PUBLISHED NOVEMBER 2019
    "SENTENCE-TRANSFORMERS": 79,
    
    # SPECIALIZED VECTOR SERVING & INFRASTRUCTURE
    "VLLM": 36,                 # PAGEDATTENTION ENGINE RELEASED JUNE 2023
    "PINECONE": 65,             # VECTOR DB LAUNCHED COMMERCIALLY IN EARLY 2021
    "QDRANT": 65,               # VECTOR DB ENGINE LAUNCHED EARLY 2021
    "WEAVIATE": 65,             # VECTOR DB PLATFORM STRUCTURED FOR EMBEDDINGS EARLY 2021
    "MILVUS": 79,               # PURPOSE-BUILT OPEN-SOURCE VECTOR CLUSTER LATE 2019
    "FAISS": 96,                # META OPEN-SOURCED FAISS LIBRARIES IN EARLY 2017
    "PGVECTOR": 60,             # POSTGRESQL EXTENSION GAINED VECTOR TRACK MID-2021
    
    # CORE LLM/RAG ARCHITECTURAL TERMS
    "RAG": 66,                  # RETRIEVAL-AUGMENTED GENERATION PAPER MID-2020
    "LLM": 72,                  # "LARGE LANGUAGE MODEL" PHRASE SCALED POST-GPT-3 (2020)
    "LARGE LANGUAGE MODEL": 72, 
    "TRANSFORMER": 108,         # ATTENTION IS ALL YOU NEED PAPER JUNE 2017
    
    # DEEP LEARNING FOUNDATIONS (POST-ATTENTION IS ALL YOU NEED)
    "PYTORCH": 116,             # RELEASED LATE 2016 (ABSOLUTE UPPER BOUND FOR DL)
    "TRANSFORMERS": 108,        # JUNE 2017
    "HUGGING FACE TRANSFORMERS": 96, # TRACTION SCALED LATE 2018
    "HUGGINGFACE": 96,
}


def is_honeypot_candidate(cand: dict) -> tuple:
    """
    Evaluates a candidate profile against known synthetic dataset anomalies.
    Returns: (bool, str) -> (True and reason if trapped, False and empty string if safe)
    """
    profile = cand.get("profile", {})
    education = cand.get("education", [])
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # --------------------------------------------------------------------------
    # TRAP 1: FICTIONAL POP-CULTURE COMPANIES (UNTOUCHED)
    # --------------------------------------------------------------------------
    for job in career:
        comp_name = str(job.get("company", "")).strip().upper()
        if any(fictional in comp_name for fictional in FICTIONAL_COMPANIES):
            return True, f"Fictional corporate footprint: {comp_name}"

    # --------------------------------------------------------------------------
    # TRAP 2: COMPANY ESTABLISHMENT VS. TENURE CHRONOLOGY (UNTOUCHED)
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
    # TRAP 3: ACADEMIC TIMELINE STRUCTURES (DURATIONS & COHORT DRIFTS) (UNTOUCHED)
    # --------------------------------------------------------------------------
    max_undergrad_grad_year = None
    earliest_undergrad_start_year = None
    postgrad_end_year = None
    bachelors_start_year = None
    earliest_undergrad_grad_year = None

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
            if earliest_undergrad_grad_year is None or start < earliest_undergrad_start_year:
                earliest_undergrad_start_year = start
            if earliest_undergrad_grad_year is None or end < earliest_undergrad_grad_year:
                earliest_undergrad_grad_year = end
        
        if any(m_deg in degree for m_deg in ["M.TECH", "M.SC", "MBA", "M.E", "PH.D", "PHD"]):
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

    # ==========================================================================
    # MODIFIED CODE SEGMENT: PRECISE NEW REQUESTED TRAP EXECUTION FLOW
    # ==========================================================================

    # --------------------------------------------------------------------------
    # NEW TRAP 1: COMPLETE MASTER'S PROGRAM BEFORE UNDERGRADUATION
    # --------------------------------------------------------------------------
    if postgrad_end_year and bachelors_start_year and postgrad_end_year < bachelors_start_year:
        return True, "Chronological Paradox: Completed postgraduate track before entering undergraduate tier"

    # --------------------------------------------------------------------------
    # NEW TRAP 2: CORPORATE CAREER STARTED BEFORE COLLEGE GRADUATION (>2 BUFFER)
    # --------------------------------------------------------------------------
    if max_undergrad_grad_year:
        for job in career:
            start_date_str = job.get("start_date")
            if start_date_str and len(start_date_str) >= 4:
                try:
                    job_start_year = int(start_date_str[:4])
                    if (max_undergrad_grad_year - job_start_year) > 2:
                        return True, f"Full-time employment started in {job_start_year} before graduation tier in {max_undergrad_grad_year}"
                except ValueError:
                    continue

    # --------------------------------------------------------------------------
    # NEW TRAP 3: END DATE BEFORE START DATE & SIGNUP AFTER LAST ACTIVE
    # --------------------------------------------------------------------------
    # Part A: Job block timeline sanity checking
    for job in career:
        start_date_str = job.get("start_date")
        end_date_str = job.get("end_date")
        if start_date_str and end_date_str:
            if end_date_str < start_date_str:
                return True, f"Timeline Inversion: Job end date ({end_date_str}) predates start date ({start_date_str})"

    # Part B: Platform behavioral timeline registration check
    signup_date = signals.get("signup_date")
    last_active = signals.get("last_active_date")
    if signup_date and last_active and signup_date > last_active:
        return True, f"Platform chronological mismatch: Signup ({signup_date}) occurs after Last Active ({last_active})"

    # --------------------------------------------------------------------------
    # NEW TRAP 4: YEARS OVERLAP BETWEEN COMPANIES (OVERLAP TRAP)
    # --------------------------------------------------------------------------
    job_intervals = []
    for job in career:
        s_str = job.get("start_date")
        e_str = job.get("end_date")
        if s_str and len(s_str) >= 4:
            try:
                s_yr = int(s_str[:4])
                # If currently working, anchor to 2026 challenge baseline context
                e_yr = int(e_str[:4]) if (e_str and len(e_str) >= 4) else GLOBAL_BASELINE_YEAR
                job_intervals.append((s_yr, e_yr))
            except ValueError:
                continue

    # Evaluate all pairs for structural time overlap anomalies
    for i in range(len(job_intervals)):
        for j in range(i + 1, len(job_intervals)):
            s1, e1 = job_intervals[i]
            s2, e2 = job_intervals[j]
            # Check overlap condition: Start of one is strictly within the active duration window of another
            if max(s1, s2) < min(e1, e2):
                return True, f"Overlap Trap: Detected overlapping operational years between distinct corporate histories"

    # --------------------------------------------------------------------------
    # NEW TRAP 5: MIN SALARY GREATER THAN MAX SALARY
    # --------------------------------------------------------------------------
    salary_obj = signals.get("expected_salary_range_inr_lpa", {})
    min_salary = salary_obj.get("min")
    max_salary = salary_obj.get("max")
    if min_salary is not None and max_salary is not None:
        if min_salary > max_salary:
            return True, f"Broken scale range: Stated minimum salary ({min_salary} LPA) exceeds maximum ({max_salary} LPA)"

    # --------------------------------------------------------------------------
    # NEW TRAP 6: SKILL DURATION EXCEEDS TOTAL EXPERIENCE
    # --------------------------------------------------------------------------
    total_experience_years = profile.get("years_of_experience", 0.0)
    total_experience_months = total_experience_years * 12.0
    for s in skills:
        s_name = str(s.get("name", "")).strip().upper()
        s_dur = s.get("duration_months", 0)
        # Apply a minor 2-month allowance window for variance rounding
        if s_dur > (total_experience_months + 2.0):
            return True, f"Skill runtime inflation: {s_name} held for {s_dur} months, exceeding total profile runtime ({round(total_experience_months, 1)} months)"

    # --------------------------------------------------------------------------
    # NEW TRAP 7: SKILL EMERGENCE YEARS EXCEEDED
    # --------------------------------------------------------------------------
    for s in skills:
        s_name = str(s.get("name", "")).strip().upper()
        s_dur = s.get("duration_months", 0)
        if s_name in TECH_INVENTION_CEILINGS:
            allowed_max = TECH_INVENTION_CEILINGS[s_name]
            if s_dur > allowed_max:
                return True, f"Time-traveling framework footprint: {s_name} runtime states {s_dur} months (Market availability ceiling is {allowed_max} months)"

    # --------------------------------------------------------------------------
    # NEW TRAP 8: EXPERIENCE EXCEEDS TIME ELAPSED SINCE EARLIEST GRADUATION (+1.5)
    # --------------------------------------------------------------------------
    if earliest_undergrad_grad_year:
        total_claimed_exp = profile.get("years_of_experience", 0.0)
        max_possible_exp = GLOBAL_BASELINE_YEAR - earliest_undergrad_grad_year
        if total_claimed_exp > (max_possible_exp + 1.5):
            return True, f"Experience Inflation Trap: Stated experience ({total_claimed_exp} yrs) exceeds time elapsed since undergrad graduation in {earliest_undergrad_grad_year} by more than 1.5 years"

    return False, ""

def normalize_reason(reason):
    """Strips dynamic variables (years/company names) from the reason string."""
    # Maps specific complex strings to clean category labels
    if "Fictional corporate footprint" in reason: return "Fictional Corporate Footprint"
    if "Impossible Master degree duration" in reason: return "Invalid PG Duration"
    if "Historical anomaly" in reason: return "Stream Establishment Year Trap"
    if "Skill runtime inflation" in reason: return "Skill Runtime Inflation"
    if "Chronological Paradox" in reason: return "Chronological Paradox"
    if "Impossible Undergraduate degree duration" in reason: return "Invalid UG Duration"
    if "Full-time employment started in" in reason: return "Pre-Graduation Employment"
    if "Platform chronological mismatch" in reason: return "Platform Timeline Anomaly"
    if "Broken scale range" in reason: return "Invalid Salary Range"
    if "Experience Inflation Trap" in reason: return "Experience vs. Graduation Inflation"
    if "Time-traveling framework" in reason: return "Time-Travel Framework Trap"
    if "Worked at" in reason and "before incorporation" in reason: return "Company Founding Year Anomaly"
    return reason

def execution_pipeline():
    print("=" * 65)
    print("REDROB TALENT RANKER — HONEYPOT PREPROCESSING ENGINE")
    print("=" * 65)
    
    if not os.path.exists(CANDIDATES_FILE):
        print(f"[!] SYSTEM ERROR: Source file '{CANDIDATES_FILE}' missing.")
        return

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
            is_trapped, reason_str = is_honeypot_candidate(candidate)
            
            if is_trapped:
                # category_label = reason_str.split(":")[0]
                # trap_counts[category_label] += 1
                # candidate["suitability_rank_score"] = 0.0
                # candidate["is_honeypot_trap"] = True
                # candidate["filter_pruning_reason"] = reason_str
                category_label = normalize_reason(reason_str)
                trap_counts[category_label] += 1
            else:
                clean_count += 1
                candidate["suitability_rank_score"] = -1.0 
                candidate["is_honeypot_trap"] = False
                candidate["filter_pruning_reason"] = ""
                
            outfile.write(json.dumps(candidate, ensure_ascii=False) + "\n")

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
    execution_pipeline()