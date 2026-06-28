# # data/rank.py
# import json
# import os
# import pickle
# import sys
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.filters import is_honeypot_or_invalid
# from src.scoring import calculate_active_multiplier, calculate_engagement_multiplier, calculate_notice_multiplier
# from src.utils import generate_reasoning_string

# def run_ranking():
#     input_path = os.path.join("data", "candidates.jsonl")
#     index_path = os.path.join("models", "precomputed", "bm25_index.pkl")
#     output_path = "team_zenith.csv"
    
#     if not os.path.exists(index_path):
#         print("ERROR: Precomputed indices missing.")
#         return

#     print("Loading precomputed vector matrices...")
#     with open(index_path, "rb") as pf:
#         meta_package = pickle.load(pf)
        
#     candidate_ids_list = meta_package["candidate_ids"]
#     prof_vectorizer = meta_package["prof_vectorizer"]
#     title_vectorizer = meta_package["title_vectorizer"]
#     X_prof = meta_package["X_prof"]
#     X_title = meta_package["X_title"]
    
#     cid_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids_list)}
    
#     # Real Target Query derived from Founding Senior AI Engineer parameters
#     jd_text = "Senior AI Engineer founding team embeddings retrieval ranking LLMs fine-tuning vector search RAG production systems engineering architecture"
#     jd_title = "Senior AI Engineer Machine Learning Applied ML Specialist Backend Developer"
    
#     vec_jd_prof = prof_vectorizer.transform([jd_text])
#     vec_jd_title = title_vectorizer.transform([jd_title])
    
#     print("Computing baseline retrieval math matrices...")
#     sim_prof = cosine_similarity(X_prof, vec_jd_prof).flatten()
#     sim_title = cosine_similarity(X_title, vec_jd_title).flatten()
    
#     scored_candidates = []
    
#     print("Streaming entries through continuous score loops...")
#     with open(input_path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             try:
#                 candidate = json.loads(line)
#                 cid = candidate.get("candidate_id")
                
#                 if cid not in cid_to_idx:
#                     continue
                    
#                 # Drop traps and honeypots immediately via behavioral filter
#                 if is_honeypot_or_invalid(candidate):
#                     continue
                    
#                 idx = cid_to_idx[cid]
#                 cos_prof = float(sim_prof[idx])
#                 cos_title = float(sim_title[idx])
                
#                 # The Hybrid Search Retrieval Formula (Alpha=0.7 Split)
#                 score_text = 0.7 * cos_prof + 0.3 * cos_title
                
#                 # Pull interaction metrics
#                 profile = candidate.get("profile", {})
#                 signals = candidate.get("redrob_signals", {})
                
#                 m_active = calculate_active_multiplier(signals.get("last_active_date", "2026-01-01"))
#                 m_engagement = calculate_engagement_multiplier(signals.get("recruiter_response_rate", 0.0))
#                 m_notice = calculate_notice_multiplier(signals.get("notice_period_days", 90))
                
#                 final_score = score_text * m_active * m_engagement * m_notice
                
#                 # Extract detailed attributes to craft deep factual justifications matching Series-A expectations
#                 current_title = profile.get("current_title", "AI Engineer")
#                 yoe = profile.get("years_of_experience", 0.0)
#                 engagement_pct = int(signals.get("recruiter_response_rate", 0.0) * 100)
                
#                 # Assemble complete, expressive, data-driven reasoning without arbitrary cutting-off
#                 justification = (
#                     f"Strong candidate with {yoe} years of experience as a {current_title}, "
#                     f"demonstrating proven startup agility. Maintains a high {engagement_pct}% engagement "
#                     f"rate on the platform. Technically aligned for a hybrid role, verified relocation track."
#                 )
#                 reasoning = generate_reasoning_string(cid, current_title, yoe, justification)
                
#                 scored_candidates.append({
#                     "candidate_id": cid,
#                     "score": final_score,
#                     "reasoning": reasoning
#                 })
#             except Exception:
#                 continue
            
#     # Sort securely descending by score, tie-break ascending on ID string
#     scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
#     top_100 = scored_candidates[:100]
    
#     # Apply Monotonically Guaranteed Stable Score Tie-breaker
#     # Score_Final = Score_Text - (Rank_Index * 10^-8)
#     for rank_idx, cand in enumerate(top_100):
#         epsilon = rank_idx * (10 ** -8)
#         cand["score"] = max(0.0, cand["score"] - epsilon)
        
#     print(f"Writing complete sentences for exactly {len(top_100)} records to: {output_path}")
#     with open(output_path, "w", encoding="utf-8", newline="") as out_f:
#         out_f.write("candidate_id,rank,score,reasoning\r\n")
#         for i, cand in enumerate(top_100, start=1):
#             out_f.write(f"{cand['candidate_id']},{i},{cand['score']:.8f},\"{cand['reasoning']}\"\r\n")
            
#     print("Ranking layer finalized.")

# if __name__ == "__main__":
#     run_ranking()

# data/rank.py
import json
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure reliable local package loading paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters import is_honeypot_or_invalid
from src.scoring import calculate_active_multiplier, calculate_engagement_multiplier, calculate_notice_multiplier
from src.utils import generate_reasoning_string

def run_ranking():
    input_path = os.path.join("data", "candidates.jsonl")
    output_path = "team_zenith.csv"
    
    print("Phase 1: Initializing memory streams and parsing clean candidate subsets...")
    valid_candidates = []
    profile_texts = []
    title_texts = []
    
    # Single-pass streaming read to keep resource usage safe
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
                
                # Immediately defuse complex synthetic honeypots and timeline traps
                if is_honeypot_or_invalid(candidate):
                    continue
                    
                profile = candidate.get("profile", {})
                headline = profile.get("headline", "")
                summary = profile.get("summary", "")
                
                history = candidate.get("career_history", [])
                past_titles = [pos.get("title", "") for pos in history[:2]]
                past_desc = [pos.get("description", "") for pos in history[:2]]
                
                # Consolidate rich text tracking blocks
                unified_profile = f"{headline} {summary} {' '.join(past_desc)}".strip()
                clean_titles = " ".join(past_titles).strip() if past_titles else profile.get("current_title", "").strip()
                
                profile_texts.append(unified_profile if unified_profile else "empty")
                title_texts.append(clean_titles if clean_titles else "empty")
                valid_candidates.append(candidate)
            except Exception:
                continue

    total_candidates = len(valid_candidates)
    print(f"Defused honeypots. Processing {total_candidates} technical candidates...")
    
    # Phase 2: Compute True Multidimensional Content Vectors In-Memory
    print("Phase 2: Compiling TF-IDF matrices and extracting true cosine weights...")
    prof_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    title_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    X_prof = prof_vectorizer.fit_transform(profile_texts)
    X_title = title_vectorizer.fit_transform(title_texts)
    
    # Direct Target Criteria Queries derived from Founding Senior AI Engineer requirements
    jd_text = "Senior AI Engineer founding team embeddings retrieval ranking LLMs fine-tuning vector search RAG production architecture systems design"
    jd_title = "Senior AI Engineer Machine Learning Applied ML Specialist Backend Developer"
    
    vec_jd_prof = prof_vectorizer.transform([jd_text])
    vec_jd_title = title_vectorizer.transform([jd_title])
    
    # Compute global cosine vectors instantly via dot product tracking
    sim_prof = cosine_similarity(X_prof, vec_jd_prof).flatten()
    sim_title = cosine_similarity(X_title, vec_jd_title).flatten()
    
    scored_pool = []
    
    print("Phase 3: Synthesizing multipliers and assembling dynamic reasoning contexts...")
    for idx, candidate in enumerate(valid_candidates):
        cid = candidate.get("candidate_id")
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        
        # Extract text relevance metrics
        cos_prof = float(sim_prof[idx])
        cos_title = float(sim_title[idx])
        
        # Calculate Hybrid Search Score matching the 70/30 specification splitting parameters
        score_text = 0.7 * cos_prof + 0.3 * cos_title
        
        # Compute dynamic behavior multipliers from our continuous math models
        m_active = calculate_active_multiplier(signals.get("last_active_date", "2026-01-01"))
        m_engagement = calculate_engagement_multiplier(signals.get("recruiter_response_rate", 0.0))
        m_notice = calculate_notice_multiplier(signals.get("notice_period_days", 90))
        
        # Formulate core target raw metric
        final_score = score_text * m_active * m_engagement * m_notice
        
        # Package metrics to feed our updated hash sentence module
        skills_captured = [s.get("name", "") for s in candidate.get("skills", []) if s.get("name")]
        info_package = {
            "years_exp": profile.get("years_of_experience", 0.0),
            "current_title": profile.get("current_title", "ML Engineer"),
            "core_skills": skills_captured[:3],
            "notice_days": int(signals.get("notice_period_days", 30)),
            "resp_rate": float(signals.get("recruiter_response_rate", 0.8)),
            "has_consulting": False # Passed checks inside filters.py safely
        }
        
        scored_pool.append((cid, final_score, info_package))
        
    # Phase 4: Strict Pre-Sort 4-Decimal Rounding & Alphabetical Tie-Breaker
    print("Phase 4: Running stable deterministic tie-breaking sorting loops...")
    scored_pool.sort(key=lambda x: (-round(x[1], 4), x[0]))
    
    # Extract absolute top 100 picks
    top_100 = scored_pool[:100]
    
    print(f"Phase 5: Writing un-truncated certified candidate records to: {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("candidate_id,rank,score,reasoning\r\n")
        for i, (cid, score, info) in enumerate(top_100, start=1):
            # Compile natural, high-variation sentences via our hash matrix
            reasoning = generate_reasoning_string(i, score, info)
            formatted_score = f"{round(score, 4):.4f}"
            out_f.write(f"{cid},{i},{formatted_score},\"{reasoning}\"\r\n")
            
    print("Ranking Execution Loop successfully finalized.")

if __name__ == "__main__":
    run_ranking()