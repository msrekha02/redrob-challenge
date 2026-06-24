# data/precompute.py
import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def run_precomputation():
    input_path = os.path.join("data", "candidates.jsonl")
    output_dir = os.path.join("models", "precomputed")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Streaming and processing raw blocks from: {input_path}")
    profile_blocks = []
    title_blocks = []
    candidate_ids = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
                profile = candidate.get("profile", {})
                
                headline = profile.get("headline", "")
                summary = profile.get("summary", "")
                
                history = candidate.get("career_history", [])
                past_titles = [pos.get("title", "") for pos in history[:2]]
                past_desc = [pos.get("description", "") for pos in history[:2]]
                
                unified_profile = f"{headline} {summary} {' '.join(past_desc)}"
                clean_titles = " ".join(past_titles) if past_titles else profile.get("current_title", "")
                
                profile_blocks.append(unified_profile if unified_profile.strip() else "empty")
                title_blocks.append(clean_titles if clean_titles.strip() else "empty")
                candidate_ids.append(candidate.get("candidate_id"))
            except Exception:
                continue

    print(f"Fitting vector matrices for {len(candidate_ids)} candidates...")
    
    # Fit independent vectorizers to isolate profile context from job titles
    prof_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    title_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    X_prof = prof_vectorizer.fit_transform(profile_blocks)
    X_title = title_vectorizer.fit_transform(title_blocks)
    
    meta_package = {
        "candidate_ids": candidate_ids,
        "prof_vectorizer": prof_vectorizer,
        "title_vectorizer": title_vectorizer,
        "X_prof": X_prof,
        "X_title": X_title
    }
    
    output_meta = os.path.join(output_dir, "bm25_index.pkl")
    with open(output_meta, "wb") as pf:
        pickle.dump(meta_package, pf)
        
    print(f"Successfully generated offline serialized vectors at: {output_meta}")

if __name__ == "__main__":
    run_precomputation()