"""
offline_pipeline.py — Phase A: Artifact Builder
================================================
Run this script ONCE whenever candidates.jsonl changes.
It builds all artifacts needed by the online ranking phase (rank.py).

Usage:
    python offline_pipeline.py --candidates ../data/candidates.jsonl
    python offline_pipeline.py --candidates ../data/candidates.jsonl --skip-embed

What it builds:
    artifacts/role_vectors.npy      (N × 384 memmapped float32)
    artifacts/cap_vectors.npy       (N × 384 memmapped float32)
    artifacts/role_bm25.pkl         (BM25Okapi over role text)
    artifacts/cap_bm25.pkl          (BM25Okapi over capability text)
    artifacts/behaviors.npy         (N × 1 float32 behavioral tensor)
    artifacts/candidate_meta.json   (candidate_id → array index + snapshot fields)

Design principle: All 100K candidates are loaded first.
Honeypot filter runs in-process. Only clean candidates are embedded.
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

# Ensure src/ is importable from anywhere
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from honeypot import filter_honeypots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OFFLINE] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REFERENCE_DATE = date(2026, 6, 29)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text builders — Role Track & Capability Track
# ─────────────────────────────────────────────────────────────────────────────

def build_role_text(cand: dict[str, Any]) -> str:
    """
    Role Track: Encodes WHO the candidate is (seniority, job family, industry).
    Adapts to any candidate schema — gracefully skips missing fields.
    """
    profile = cand.get("profile", {})
    parts: list[str] = []

    # Current position
    if t := profile.get("current_title"):
        parts.append(t)
    if c := profile.get("current_company"):
        parts.append(f"at {c}")
    if i := profile.get("current_industry"):
        parts.append(i)

    # Headline is the candidate's self-description of their role identity
    if h := profile.get("headline"):
        parts.append(h)

    # All past job titles give the role trajectory
    for job in cand.get("career_history", []):
        if t := job.get("title"):
            parts.append(t)
        if c := job.get("company"):
            parts.append(c)
        if i := job.get("industry"):
            parts.append(i)

    # Education (degree + institution signals academic seniority)
    for edu in cand.get("education", []):
        if d := edu.get("degree"):
            parts.append(d)
        if inst := edu.get("institution"):
            parts.append(inst)

    return " ".join(filter(None, parts))


def build_cap_text(cand: dict[str, Any]) -> str:
    """
    Capability Track: Encodes WHAT the candidate can do (tools, skills, projects).
    Adapts to any candidate schema.
    """
    parts: list[str] = []
    profile = cand.get("profile", {})

    # Summary often contains the most concise technical narrative
    if s := profile.get("summary"):
        parts.append(s)

    # All skills — name + proficiency level
    for skill in cand.get("skills", []):
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        if name:
            parts.append(f"{name} {prof}".strip())

    # Career descriptions contain real project details
    for job in cand.get("career_history", []):
        if d := job.get("description"):
            parts.append(d)

    # Certifications signal specific tool competency
    for cert in cand.get("certifications", []):
        if n := cert.get("name"):
            parts.append(n)
        if org := cert.get("issuing_organization"):
            parts.append(org)

    # Degree field is a capability signal (e.g. "Computer Science", "AI/ML")
    for edu in cand.get("education", []):
        if f := edu.get("field_of_study"):
            parts.append(f)

    return " ".join(filter(None, parts))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Behavioral tensor
# ─────────────────────────────────────────────────────────────────────────────

def compute_behavioral_score(cand: dict[str, Any]) -> float:
    """
    Precompute a JD-INDEPENDENT behavioral score [0, 1], matching the
    architecture's exact four named components: recency, response rate,
    notice period, tenure-consistency.

    This is NOT the full penalty system — guardrails.py's Layer 1 (m_avail)
    and Layer 3 (m_notice) score recency/response/notice far more precisely
    and JD-conditionally (e.g. an urgent JD weighs notice period harder than
    a flexible one). This tensor is consequently used in guardrails.py as a
    free, precomputed TIE-BREAKER between candidates with equal penalized
    scores — not as a second multiplicative factor on top of the Layer 1/3
    penalties, since that would score the same underlying facts twice.

    v2 fix: earlier versions of this function substituted open_to_work_flag
    and interview_completion_rate for two of the four named components.
    Those two signals are real and important, but they were never part of
    this architecture's stated tensor definition and are already scored
    precisely in Layer 1 (P-2, P-4). This version uses exactly the four
    components the architecture specifies.

    Formula: recency(0.35) + response_rate(0.30) + notice_period(0.20) + tenure_consistency(0.15)
    """
    signals = cand.get("redrob_signals", {})
    career  = cand.get("career_history", [])

    # ── Recency ────────────────────────────────────────────────────────────
    last_active_raw = signals.get("last_active_date")
    try:
        last_active = date.fromisoformat(str(last_active_raw))
        days_inactive = (REFERENCE_DATE - last_active).days
    except (TypeError, ValueError):
        days_inactive = 999

    if days_inactive <= 30:
        recency_norm = 1.00
    elif days_inactive <= 60:
        recency_norm = 0.85
    elif days_inactive <= 90:
        recency_norm = 0.65
    elif days_inactive <= 180:
        recency_norm = 0.40
    else:
        recency_norm = 0.10

    # ── Response rate ──────────────────────────────────────────────────────
    rr = signals.get("recruiter_response_rate", 0.5)
    try:
        response_norm = float(rr) if 0.0 <= float(rr) <= 1.0 else 0.5
    except (TypeError, ValueError):
        response_norm = 0.5

    # ── Notice period (shorter = higher score; JD-independent raw signal —
    #    the JD-conditional URGENCY weighting of this same fact happens
    #    separately in guardrails.py's m_notice) ──────────────────────────
    notice_raw = signals.get("notice_period_days", 90)
    try:
        notice_days = int(notice_raw)
    except (TypeError, ValueError):
        notice_days = 90

    if notice_days <= 30:
        notice_norm = 1.00
    elif notice_days <= 60:
        notice_norm = 0.85
    elif notice_days <= 90:
        notice_norm = 0.65
    else:
        notice_norm = 0.40

    # ── Tenure consistency (average completed-role duration; longer/more
    #    consistent tenure = higher score, a JD-independent positive trait
    #    distinct from m_jd's JD-conditional job-hopping THRESHOLD penalty) ──
    completed = [
        j.get("duration_months") or 0 for j in career
        if not j.get("is_current") and j.get("duration_months")
    ]
    if completed:
        avg_tenure_months = sum(completed) / len(completed)
    else:
        avg_tenure_months = 18.0   # neutral default for candidates with no completed roles yet

    if avg_tenure_months >= 24:
        tenure_norm = 1.00
    elif avg_tenure_months >= 18:
        tenure_norm = 0.75
    elif avg_tenure_months >= 12:
        tenure_norm = 0.50
    else:
        tenure_norm = 0.25

    score = (
        recency_norm * 0.35
        + response_norm * 0.30
        + notice_norm * 0.20
        + tenure_norm * 0.15
    )
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. BM25 tokenizer (shared with JDProcessor)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, return token list."""
    text = _re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    tokens = [t.strip("-") for t in text.split() if len(t) > 1]
    return [t for t in tokens if t not in cfg.STOPWORDS and len(t) > 1]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Embedding engine (lazy-loaded, one model instance per run)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps model_engine.ONNXEmbedder — runs as ONNX INT8 when an exported
    artifact exists (see setup_models.py), automatically falls back to
    PyTorch sentence-transformers otherwise. All embeddings are
    L2-normalised (cosine sim = dot product for normalised vecs).
    """

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from model_engine import ONNXEmbedder
        model_path = (
            str(cfg.BGE_EMBED_MODEL_DIR)
            if cfg.BGE_EMBED_MODEL_DIR.exists()
            else cfg.BGE_EMBED_MODEL_ID
        )
        log.info("Loading embedding model from: %s", model_path)
        self._model = ONNXEmbedder(cfg.BGE_EMBED_MODEL_DIR, fallback_model_id=cfg.BGE_EMBED_MODEL_ID)
        log.info(
            "Embedding model loaded (dim=%d, backend=%s)",
            cfg.EMBED_DIM, "ONNX INT8" if self._model.using_onnx else "PyTorch (fallback)",
        )

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Return (N, EMBED_DIM) float32 array of L2-normalised embeddings."""
        self._load()
        return self._model.encode(
            texts,
            batch_size=cfg.EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load all candidates from a JSONL file. Skips malformed lines."""
    candidates: list[dict] = []
    errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    log.warning("Line %d: JSON error — %s", i, e)
    log.info("Loaded %d candidates (%d parse errors)", len(candidates), errors)
    return candidates


def run_offline_pipeline(
    candidates_path: Path,
    skip_embed: bool = False,
) -> None:
    t0 = time.perf_counter()
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load & honeypot filter ───────────────────────────────────────
    log.info("Step 1/5 — Loading candidates from %s", candidates_path)
    all_candidates = load_candidates(candidates_path)

    log.info("Step 1/5 — Running honeypot filter on %d candidates …", len(all_candidates))
    clean_candidates, honeypot_results = filter_honeypots(all_candidates, verbose=False)
    log.info(
        "Honeypot filter complete: %d clean / %d flagged (%.1f%%)",
        len(clean_candidates),
        len(honeypot_results),
        len(honeypot_results) / max(1, len(all_candidates)) * 100,
    )
    N = len(clean_candidates)
    if N == 0:
        log.error("No clean candidates after filter. Aborting.")
        return

    # ── Step 2: Build text representations ────────────────────────────────────
    log.info("Step 2/5 — Building role & capability text for %d candidates …", N)
    role_texts: list[str] = []
    cap_texts:  list[str] = []
    for cand in clean_candidates:
        role_texts.append(build_role_text(cand))
        cap_texts.append(build_cap_text(cand))

    # ── Step 3: Embed (skip if only indexes need rebuilding) ──────────────────
    if skip_embed:
        log.info("Step 3/5 — Skipping embedding (--skip-embed flag set)")
    else:
        log.info("Step 3/5 — Embedding %d × 2 text tracks …", N)
        engine = EmbeddingEngine()

        # Role vectors
        log.info("  Encoding role track …")
        role_vecs = engine.encode(role_texts)
        role_mm = np.memmap(
            cfg.ROLE_VECTORS_PATH, dtype="float32", mode="w+", shape=(N, cfg.EMBED_DIM)
        )
        role_mm[:] = role_vecs
        role_mm.flush()
        log.info("  role_vectors.npy → %s (%.1f MB)", cfg.ROLE_VECTORS_PATH,
                 role_mm.nbytes / 1e6)

        # Capability vectors
        log.info("  Encoding capability track …")
        cap_vecs = engine.encode(cap_texts)
        cap_mm = np.memmap(
            cfg.CAP_VECTORS_PATH, dtype="float32", mode="w+", shape=(N, cfg.EMBED_DIM)
        )
        cap_mm[:] = cap_vecs
        cap_mm.flush()
        log.info("  cap_vectors.npy → %s (%.1f MB)", cfg.CAP_VECTORS_PATH,
                 cap_mm.nbytes / 1e6)

    # ── Step 4: Build dual BM25 indexes ───────────────────────────────────────
    log.info("Step 4/5 — Building BM25 indexes …")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.error("rank_bm25 not installed. Run: pip install rank-bm25")
        raise

    role_corpus = [tokenize(t) for t in role_texts]
    cap_corpus  = [tokenize(t) for t in cap_texts]

    role_bm25 = BM25Okapi(role_corpus)
    cap_bm25  = BM25Okapi(cap_corpus)

    with open(cfg.ROLE_BM25_PATH, "wb") as f:
        pickle.dump(role_bm25, f)
    with open(cfg.CAP_BM25_PATH, "wb") as f:
        pickle.dump(cap_bm25, f)
    log.info("  BM25 indexes saved.")

    # ── Step 5: Behavioral tensor + candidate metadata ─────────────────────────
    log.info("Step 5/5 — Computing behavioral tensor & metadata …")
    behaviors = np.zeros(N, dtype=np.float32)
    meta: dict[str, Any] = {
        "n_candidates": N,
        "built_at": datetime.utcnow().isoformat(),
        "candidates_path": str(candidates_path),
        "index_to_id": [],
        "id_to_index": {},
        "snapshots": {},
    }

    for i, cand in enumerate(clean_candidates):
        behaviors[i] = compute_behavioral_score(cand)

        cid = cand.get("candidate_id", f"CAND_{i:07d}")
        profile  = cand.get("profile", {})
        signals  = cand.get("redrob_signals", {})
        meta["index_to_id"].append(cid)
        meta["id_to_index"][cid] = i

        # Snapshot: all fields needed by guardrails + reasoner at runtime
        # Store here so rank.py doesn't re-parse candidates.jsonl
        meta["snapshots"][cid] = {
            # Profile fields
            "anonymized_name":      profile.get("anonymized_name", ""),
            "headline":             profile.get("headline", ""),
            "summary":              profile.get("summary", ""),
            "current_title":        profile.get("current_title", ""),
            "current_company":      profile.get("current_company", ""),
            "current_industry":     profile.get("current_industry", ""),
            "years_of_experience":  profile.get("years_of_experience", 0),
            "location":             profile.get("location", ""),
            "country":              profile.get("country", "IN"),

            # Signals
            "last_active_date":          signals.get("last_active_date", ""),
            "open_to_work_flag":         signals.get("open_to_work_flag", False),
            "recruiter_response_rate":   signals.get("recruiter_response_rate", 0.5),
            "interview_completion_rate": signals.get("interview_completion_rate", 1.0),
            "notice_period_days":        signals.get("notice_period_days", 90),
            "willing_to_relocate":       signals.get("willing_to_relocate", False),
            "preferred_work_mode":       signals.get("preferred_work_mode", ""),
            "github_activity_score":     signals.get("github_activity_score", -1),
            "offer_acceptance_rate":     signals.get("offer_acceptance_rate", -1),
            "profile_completeness_score":signals.get("profile_completeness_score", 0),
            "verified_email":            signals.get("verified_email", False),
            "verified_phone":            signals.get("verified_phone", False),
            "linkedin_connected":        signals.get("linkedin_connected", False),
            "skill_assessment_scores":   signals.get("skill_assessment_scores", {}) or {},
            "expected_salary_min":       (signals.get("expected_salary_range_inr_lpa") or {}).get("min"),
            "expected_salary_max":       (signals.get("expected_salary_range_inr_lpa") or {}).get("max"),

            # NOTE: the fields below (signup_date + 7 count fields) are not
            # used by any penalty/scoring logic — they exist solely so the
            # final-stage honeypot safety net (guardrails.apply_honeypot_safety_net)
            # can re-run the FULL 14-trap check on the final top-100 without
            # needing to re-read candidates.jsonl. Trivial storage cost
            # (small scalars × ~8K candidates), but makes the safety net a
            # complete re-check rather than a partial approximation.
            "signup_date":                signals.get("signup_date", ""),
            "avg_response_time_hours":    signals.get("avg_response_time_hours", 0),
            "profile_views_received_30d": signals.get("profile_views_received_30d", 0),
            "applications_submitted_30d": signals.get("applications_submitted_30d", 0),
            "connection_count":           signals.get("connection_count", 0),
            "endorsements_received":      signals.get("endorsements_received", 0),
            "search_appearance_30d":      signals.get("search_appearance_30d", 0),
            "saved_by_recruiters_30d":    signals.get("saved_by_recruiters_30d", 0),

            # Career history (summarised to save space)
            "career_history": [
                {
                    "company":        j.get("company", ""),
                    "title":          j.get("title", ""),
                    "start_date":     j.get("start_date", ""),
                    "end_date":       j.get("end_date"),
                    "duration_months":j.get("duration_months", 0),
                    "is_current":     j.get("is_current", False),
                    "industry":       j.get("industry", ""),
                    "description":    j.get("description", "")[:300],  # cap at 300 chars
                }
                for j in cand.get("career_history", [])
            ],

            # Skills
            "skills": [
                {
                    "name":        s.get("name", ""),
                    "proficiency": s.get("proficiency", ""),
                    "endorsements":s.get("endorsements", 0),
                    "duration_months": s.get("duration_months", 0),
                }
                for s in cand.get("skills", [])
            ],

            # Education
            "education": [
                {
                    "degree":         e.get("degree", ""),
                    "field_of_study": e.get("field_of_study", ""),
                    "institution":    e.get("institution", ""),
                    "start_year":     e.get("start_year"),
                    "end_year":       e.get("end_year"),
                    "tier":           e.get("tier", ""),
                }
                for e in cand.get("education", [])
            ],
        }

    np.save(cfg.BEHAVIORS_PATH, behaviors)
    with open(cfg.CANDIDATE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    log.info("─" * 60)
    log.info(
        "Offline pipeline complete in %.1fs — %d clean candidates embedded",
        elapsed, N,
    )
    log.info("Artifacts in: %s", cfg.ARTIFACTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Redrob — Phase A offline artifact builder"
    )
    parser.add_argument(
        "--candidates", "-c",
        default=str(cfg.CANDIDATES_PATH),
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding step (useful when only rebuilding BM25/meta)",
    )
    args = parser.parse_args()
    run_offline_pipeline(Path(args.candidates), skip_embed=args.skip_embed)