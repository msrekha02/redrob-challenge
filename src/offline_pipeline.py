"""
offline_pipeline.py — Phase A: Artifact Builder
================================================
"""

from __future__ import annotations
import argparse
import json
import logging
import pickle
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from honeypot import filter_honeypots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OFFLINE] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REFERENCE_DATE = date(2026, 6, 29)

def build_role_text(cand: dict[str, Any]) -> str:
    profile = cand.get("profile", {})
    parts = []
    if t := profile.get("current_title"): parts.append(t)
    if c := profile.get("current_company"): parts.append(f"at {c}")
    if i := profile.get("current_industry"): parts.append(i)
    if h := profile.get("headline"): parts.append(h)
    for job in cand.get("career_history", []):
        if t := job.get("title"): parts.append(t)
        if c := job.get("company"): parts.append(c)
        if i := job.get("industry"): parts.append(i)
    for edu in cand.get("education", []):
        if d := edu.get("degree"): parts.append(d)
        if inst := edu.get("institution"): parts.append(inst)
    return " ".join(filter(None, parts))

def build_cap_text(cand: dict[str, Any]) -> str:
    parts = []
    profile = cand.get("profile", {})
    if s := profile.get("summary"): parts.append(s)
    for skill in cand.get("skills", []):
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        if name: parts.append(f"{name} {prof}".strip())
    for job in cand.get("career_history", []):
        if d := job.get("description"): parts.append(d)
    for cert in cand.get("certifications", []):
        if n := cert.get("name"): parts.append(n)
    for edu in cand.get("education", []):
        if f := edu.get("field_of_study"): parts.append(f)
    return " ".join(filter(None, parts))

def compute_behavioral_score(cand: dict[str, Any]) -> float:
    signals = cand.get("redrob_signals", {})
    career  = cand.get("career_history", [])
    try:
        last_active = date.fromisoformat(str(signals.get("last_active_date")))
        days_inactive = (REFERENCE_DATE - last_active).days
    except (TypeError, ValueError): days_inactive = 999

    recency_norm = 1.00 if days_inactive <= 30 else (0.85 if days_inactive <= 60 else (0.65 if days_inactive <= 90 else (0.40 if days_inactive <= 180 else 0.10)))
    try: response_norm = float(signals.get("recruiter_response_rate", 0.5))
    except (TypeError, ValueError): response_norm = 0.5
    try: notice_days = int(signals.get("notice_period_days", 90))
    except (TypeError, ValueError): notice_days = 90
    notice_norm = 1.00 if notice_days <= 30 else (0.85 if notice_days <= 60 else (0.65 if notice_days <= 90 else 0.40))
    completed = [j.get("duration_months") or 0 for j in career if not j.get("is_current") and j.get("duration_months")]
    avg_tenure_months = sum(completed) / len(completed) if completed else 18.0
    tenure_norm = 1.00 if avg_tenure_months >= 24 else (0.75 if avg_tenure_months >= 18 else (0.50 if avg_tenure_months >= 12 else 0.25))

    return float(np.clip(recency_norm * 0.35 + response_norm * 0.30 + notice_norm * 0.20 + tenure_norm * 0.15, 0.0, 1.0))

import re as _re
def tokenize(text: str) -> list[str]:
    text = _re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    tokens = [t.strip("-") for t in text.split() if len(t) > 1]
    return [t for t in tokens if t not in cfg.STOPWORDS and len(t) > 1]

class EmbeddingEngine:
    def __init__(self) -> None:
        self._model = None
    def _load(self) -> None:
        if self._model is not None: return
        from model_engine import ONNXEmbedder
        # Alignment with Vocabulary: ONNXEmbedder safely loads native HuggingFace tokenizer
        self._model = ONNXEmbedder(cfg.BGE_EMBED_MODEL_DIR, fallback_model_id=cfg.BGE_EMBED_MODEL_ID)
    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        self._load()
        return self._model.encode(texts, batch_size=cfg.EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=show_progress, convert_to_numpy=True).astype(np.float32)

def load_candidates(path: Path) -> list[dict[str, Any]]:
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: candidates.append(json.loads(line))
            except json.JSONDecodeError: pass
    return candidates

def run_offline_pipeline(candidates_path: Path, skip_embed: bool = False) -> None:
    # Ensure this offline pipeline does not try to invoke the online jd_extractor_int8.onnx model
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    all_candidates = load_candidates(candidates_path)
    clean_candidates, honeypot_results = filter_honeypots(all_candidates, verbose=False)
    N = len(clean_candidates)
    if N == 0: return

    role_texts, cap_texts = [], []
    for cand in clean_candidates:
        role_texts.append(build_role_text(cand))
        cap_texts.append(build_cap_text(cand))

    if not skip_embed:
        engine = EmbeddingEngine()
        role_vecs = engine.encode(role_texts)
        role_mm = np.memmap(cfg.ROLE_VECTORS_PATH, dtype="float32", mode="w+", shape=(N, cfg.EMBED_DIM))
        role_mm[:] = role_vecs
        role_mm.flush()
        cap_vecs = engine.encode(cap_texts)
        cap_mm = np.memmap(cfg.CAP_VECTORS_PATH, dtype="float32", mode="w+", shape=(N, cfg.EMBED_DIM))
        cap_mm[:] = cap_vecs
        cap_mm.flush()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise
    role_corpus = [tokenize(t) for t in role_texts]
    cap_corpus  = [tokenize(t) for t in cap_texts]
    role_bm25 = BM25Okapi(role_corpus)
    cap_bm25  = BM25Okapi(cap_corpus)
    with open(cfg.ROLE_BM25_PATH, "wb") as f: pickle.dump(role_bm25, f)
    with open(cfg.CAP_BM25_PATH, "wb") as f: pickle.dump(cap_bm25, f)

    behaviors = np.zeros(N, dtype=np.float32)
    meta = {"n_candidates": N, "built_at": datetime.utcnow().isoformat(), "index_to_id": [], "id_to_index": {}, "snapshots": {}}
    for i, cand in enumerate(clean_candidates):
        behaviors[i] = compute_behavioral_score(cand)
        cid = cand.get("candidate_id", f"CAND_{i:07d}")
        profile  = cand.get("profile", {})
        signals  = cand.get("redrob_signals", {})
        meta["index_to_id"].append(cid)
        meta["id_to_index"][cid] = i
        meta["snapshots"][cid] = {
            "anonymized_name": profile.get("anonymized_name", ""),
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "years_of_experience": profile.get("years_of_experience", 0),
            "location": profile.get("location", ""),
            "country": profile.get("country", "IN"),
            "open_to_work_flag": signals.get("open_to_work_flag", False),
            "recruiter_response_rate": signals.get("recruiter_response_rate", 0.5),
            "interview_completion_rate": signals.get("interview_completion_rate", 1.0),
            "notice_period_days": signals.get("notice_period_days", 90),
            "willing_to_relocate": signals.get("willing_to_relocate", False),
            "github_activity_score": signals.get("github_activity_score", -1),
            "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
            "profile_completeness_score": signals.get("profile_completeness_score", 0),
            "verified_email": signals.get("verified_email", False),
            "linkedin_connected": signals.get("linkedin_connected", False),
            "skill_assessment_scores": signals.get("skill_assessment_scores", {}) or {},
            "expected_salary_min": (signals.get("expected_salary_range_inr_lpa") or {}).get("min"),
            "expected_salary_max": (signals.get("expected_salary_range_inr_lpa") or {}).get("max"),
            "career_history": [{"company": j.get("company", ""), "title": j.get("title", ""), "duration_months": j.get("duration_months", 0), "is_current": j.get("is_current", False), "description": j.get("description", "")[:300]} for j in cand.get("career_history", [])],
            "skills": [{"name": s.get("name", ""), "proficiency": s.get("proficiency", ""), "endorsements": s.get("endorsements", 0)} for s in cand.get("skills", [])],
            "education": [{"degree": e.get("degree", ""), "field_of_study": e.get("field_of_study", ""), "institution": e.get("institution", ""), "tier": e.get("tier", "")} for e in cand.get("education", [])],
            # For 14-trap checks
            "signup_date": signals.get("signup_date", ""), "last_active_date": signals.get("last_active_date", ""), "avg_response_time_hours": signals.get("avg_response_time_hours", 0), "profile_views_received_30d": signals.get("profile_views_received_30d", 0), "applications_submitted_30d": signals.get("applications_submitted_30d", 0), "connection_count": signals.get("connection_count", 0), "endorsements_received": signals.get("endorsements_received", 0), "search_appearance_30d": signals.get("search_appearance_30d", 0), "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
        }
    np.save(cfg.BEHAVIORS_PATH, behaviors)
    with open(cfg.CANDIDATE_META_PATH, "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", "-c", default=str(cfg.CANDIDATES_PATH))
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()
    run_offline_pipeline(Path(args.candidates), skip_embed=args.skip_embed)