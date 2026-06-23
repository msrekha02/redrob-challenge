# data/precompute.py
import json
import os
import pickle
import numpy as np

def run_precomputation():
    input_path = os.path.join("data", "candidates.jsonl")
    
    print(f"Starting sequential streaming parsing from: {input_path}")
    profile_blocks = []
    title_blocks = []
    candidate_ids = []
    
    # Process sequentially using a loop to prevent RAM overflow on 100K items
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            
            # Extract attributes cleanly
            profile = candidate.get("profile", {})
            candidate_id = candidate.get("candidate_id")
            
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            
            # Safely grab top 2 job titles and descriptions from career history
            history = candidate.get("career_history", [])
            past_titles = []
            past_descriptions = []
            for position in history[:2]:
                past_titles.append(position.get("title", ""))
                past_descriptions.append(position.get("description", ""))
                
            # Build isolation block 1: Unified Profile Text Block
            unified_profile = f"{headline} {summary} {' '.join(past_descriptions)}"
            # Build isolation block 2: Clean Title Track Block
            clean_titles = " ".join(past_titles) if past_titles else profile.get("current_title", "")
            
            profile_blocks.append(unified_profile)
            title_blocks.append(clean_titles)
            candidate_ids.append(candidate_id)
            
    print(f"Processed {len(candidate_ids)} records. Saving layout dictionary mapping...")
    
    # Save a lightweight metadata map file for direct execution retrieval
    meta_package = {
        "candidate_ids": candidate_ids,
        "profile_blocks": profile_blocks,
        "title_blocks": title_blocks
    }
    
    # Serialize our local tracking artifact safely
    output_meta = os.path.join("models", "precomputed", "bm25_index.pkl")
    with open(output_meta, "wb") as pf:
        pickle.dump(meta_package, pf)
        
    print(f"Successfully generated offline serialized indices at: {output_meta}")

if __name__ == "__main__":
    run_precomputation()