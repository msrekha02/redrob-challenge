# data/rank.py
import json
import os
import pickle
import sys

# Append root directory to the module path to ensure reliable internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters import is_honeypot_or_invalid
from src.scoring import calculate_active_multiplier, calculate_engagement_multiplier, calculate_notice_multiplier
from src.utils import sanitize_text, generate_reasoning_string

def run_ranking():
    input_path = os.path.join("data", "candidates.jsonl")
    index_path = os.path.join("models", "precomputed", "bm25_index.pkl")
    output_path = "team_zenith.csv"  # Your production-ready submission file name
    
    if not os.path.exists(index_path):
        print("ERROR: Precomputed indices not found! Please run data/precompute.py first.")
        return

    print("Loading precomputed candidate metadata tracks...")
    with open(index_path, "rb") as pf:
        meta_package = pickle.load(pf)
        
    candidate_ids_list = meta_package["candidate_ids"]
    profile_blocks = meta_package["profile_blocks"]
    title_blocks = meta_package["title_blocks"]
    
    # Map for rapid direct lookup during stream reading
    candidate_indices = {cid: idx for idx, cid in enumerate(candidate_ids_list)}
    
    # Core Target Keyword Clusters derived from the Job Description requirements
    core_keywords = ["rag", "llm", "embeddings", "retrieval", "ranking", "fine-tuning", "vector", "search", "ml", "backend"]
    title_keywords = ["ai", "ml", "machine learning", "backend", "software", "data", "engineer"]

    scored_candidates = []

    print("Streaming candidates through scoring matrix and honeypot isolation screens...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            cid = candidate.get("candidate_id")
            
            if cid not in candidate_indices:
                continue
                
            # 1. Immediate Trap and Honeypot Disqualification Filter
            if is_honeypot_or_invalid(candidate):
                continue
                
            idx = candidate_indices[cid]
            profile_text = profile_blocks[idx].lower()
            title_text = title_blocks[idx].lower()
            
            # 2. Compute Hybrid Keyword Alignment Profile Match
            profile_hits = sum(1 for kw in core_keywords if kw in profile_text)
            title_hits = sum(1 for kw in title_keywords if kw in title_text)
            
            # Base content weight calculated safely
            content_score = (profile_hits * 0.1) + (title_hits * 0.2)
            
            # 3. Pull Dynamic Platform Interaction Variables
            signals = candidate.get("redrob_signals", {})
            last_active = signals.get("last_active_date", "2026-01-01")
            response_rate = signals.get("recruiter_response_rate", 0.0)
            notice_days = signals.get("notice_period_days", 90)
            
            # 4. Synthesize Continuous Multiplier Vector Matrix
            m_active = calculate_active_multiplier(last_active)
            m_engagement = calculate_engagement_multiplier(response_rate)
            m_notice = calculate_notice_multiplier(notice_days)
            
            # Combined final score calculation
            final_score = content_score * m_active * m_engagement * m_notice
            
            # Generate highly tailored unique contextual justification
            current_title = candidate.get("profile", {}).get("current_title", "Engineer")
            yoe = candidate.get("profile", {}).get("years_of_experience", 0.0)
            justification = f"Matches {profile_hits} core engineering signals with high engagement score."
            reasoning = generate_reasoning_string(cid, current_title, yoe, justification)
            
            scored_candidates.append({
                "candidate_id": cid,
                "score": round(final_score, 4),
                "reasoning": reasoning
            })

    # 5. Professional Rank Selection Sort (Sort by Score Descending, Tie-break on ID Ascending)
    scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Slice the absolute top 100 target records
    top_100 = scored_candidates[:100]
    
    # 6. Generate Validated Submission CSV Asset
    print(f"Writing sanitized submission file containing exactly {len(top_100)} records to: {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("candidate_id,rank,score,reasoning\r\n")
        for i, cand in enumerate(top_100, start=1):
            line = f"{cand['candidate_id']},{i},{cand['score']:.4f},\"{cand['reasoning']}\"\r\n"
            out_f.write(line)
            
    print("Core Ranking completed successfully.")

if __name__ == "__main__":
    run_ranking()