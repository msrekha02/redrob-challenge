"""
reasoner.py — Final: JDProfile-aware Dynamic Reasoning
=======================================================
"""
from __future__ import annotations
import csv
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from jd_processor import JDProfile
from guardrails import CandidateScore

log = logging.getLogger(__name__)
REFERENCE_DATE = date(2026, 6, 29)

def _career_info(snap: dict) -> dict[str, Any]:
    career = snap.get("career_history", [])
    if not career: return {}
    total_m    = sum(j.get("duration_months") or 0 for j in career)
    consult_m  = sum(
        j.get("duration_months") or 0 for j in career
        if any(f in (j.get("company") or "").lower() for f in cfg.CONSULTING_FIRMS)
    )
    product_m  = total_m - consult_m
    completed  = [j.get("duration_months") or 0 for j in career if not j.get("is_current") and j.get("duration_months")]
    avg_tenure = sum(completed) / len(completed) if completed else 0
    all_desc   = " ".join(j.get("description", "") for j in career).lower()
    return {
        "total_months":           total_m,
        "product_months":         product_m,
        "consulting_fraction":    consult_m / total_m if total_m else 0,
        "avg_tenure_months":      avg_tenure,
        "has_production_signals": any(kw in all_desc for kw in cfg.PRODUCTION_KEYWORDS),
        "current_role":           next((j for j in career if j.get("is_current")), career[0] if career else {}),
        "n_roles":                len(career),
    }

def _skill_info(snap: dict, jd_keywords: list[str]) -> dict[str, Any]:
    skills = snap.get("skills", [])
    names_lower = {s.get("name", "").lower() for s in skills}
    kw_lower    = {kw.lower() for kw in jd_keywords}
    matched     = sorted(kw_lower & names_lower)
    return {
        "names_lower":  names_lower,
        "expert":       [s.get("name", "") for s in skills if (s.get("proficiency") or "").lower() in ("expert","advanced")],
        "matched_jd":   matched,
    }

def _m_role_match(snap: dict, cs: CandidateScore, jd_profile: JDProfile) -> str | None:
    if cs.role_score < 0.35: return None
    title = snap.get("current_title", "")
    co = snap.get("current_company", "")
    yoe = snap.get("years_of_experience", "")
    if not title: return None
    co_str = f" at {co}" if co else ""
    return f"Anchored in target role as '{title}'{co_str} ({yoe}y total experience mapped to requirements)"

def _m_stack_match(skill_info: dict, cs: CandidateScore) -> str | None:
    matched = skill_info.get("matched_jd", [])
    if not matched: return None
    return f"Candidate verifies core capabilities with explicit stack hits on {', '.join(matched[:5])}"

def _m_production(career_info: dict, jd_profile: JDProfile) -> str | None:
    if not jd_profile.flags.get("requires_production"): return None
    if career_info.get("has_production_signals"):
        co = career_info.get("current_role", {}).get("company", "")
        return f"Demonstrated engineering lifecycle (production deployments identified{' at ' + co if co else ''})"
    return None

def _m_product_company(career_info: dict, jd_profile: JDProfile) -> str | None:
    if not jd_profile.flags.get("prefers_product_company"): return None
    pm   = career_info.get("product_months", 0)
    if pm > 24:
        return f"Maintains strong product-company alignment ({pm // 12}y+ building internal platforms)"
    return None

def _m_availability(snap: dict, jd_profile: JDProfile) -> str | None:
    if not jd_profile.enforce_availability: return None
    parts = []
    if snap.get("open_to_work_flag"): parts.append("Actively seeking")
    if n := snap.get("notice_period_days"): parts.append(f"clears notice period at {n}d")
    return " | ".join(parts) if parts else None

def _m_founding_team(career_info: dict, jd_profile: JDProfile) -> str | None:
    if not jd_profile.flags.get("is_founding_team"): return None
    avg = career_info.get("avg_tenure_months", 0)
    if avg >= 24: return f"Demonstrates structural stability (Avg {avg:.0f}m tenure suggests founding team resilience)"
    return None

def _m_yoe_alignment(snap: dict, jd_profile: JDProfile) -> str | None:
    yoe = float(snap.get("years_of_experience") or 0)
    ymin, ymax = jd_profile.yoe_min, jd_profile.yoe_max
    if ymin and ymax and (ymin <= yoe <= ymax):
        return f"Experience strictly bound to specified parameters ({yoe:.0f}y falls inside the {ymin}–{ymax}y target box)"
    return None

def _m_gap_acknowledgement(skill_info: dict, jd_flags: dict, jd_profile: JDProfile) -> str | None:
    names_lower = skill_info.get("names_lower", set())
    gaps: list[str] = []
    if jd_flags.get("requires_vector_db") and not ({"faiss", "pinecone", "qdrant", "weaviate", "milvus", "pgvector"} & names_lower):
        gaps.append("Vector DB implementations")
    if (jd_flags.get("requires_rag") or jd_profile.nlp_ir_required) and not any(t in names_lower for t in {"rag", "retrieval", "hybrid search"}):
        gaps.append("Direct RAG architectures")
    if jd_flags.get("requires_embeddings") and not any(t in names_lower for t in {"embeddings", "sentence-transformers", "bge", "e5"}):
        gaps.append("Specific embedding pipelines")
    if not gaps: return None
    return f"Risk factor isolated: {', '.join(gaps[:2])} not explicitly detailed, but surrounding technical density supports rapid adoption"

def generate_reasoning(cs: CandidateScore, snap: dict, jd_flags: dict[str, bool], jd_keywords: list[str], jd_profile: JDProfile, rank: int) -> str:
    ci = _career_info(snap)
    si = _skill_info(snap, jd_keywords)
    segments = [fn(*args) for fn, args in [
        (_m_role_match, (snap, cs, jd_profile)),
        (_m_stack_match, (si, cs)),
        (_m_production, (ci, jd_profile)),
        (_m_product_company, (ci, jd_profile)),
        (_m_availability, (snap, jd_profile)),
        (_m_founding_team, (ci, jd_profile)),
        (_m_yoe_alignment, (snap, jd_profile)),
    ] if fn(*args)]
    
    if gap := _m_gap_acknowledgement(si, jd_flags, jd_profile): segments.append(gap)
    
    score_tag = f"[Rank {rank} | CE={cs.ce_score:.3f} | role={cs.role_score:.2f} cap={cs.cap_score:.2f} dom={cs.domain_sim:.2f} avail=×{cs.m_avail:.2f}]"
    return ("; ".join(segments) + f". {score_tag}") if segments else f"Confirmed structural fit in dense matrix evaluation. {score_tag}"

def export_csv(final_candidates: list[CandidateScore], store, jd_flags: dict[str, bool], jd_keywords: list[str], jd_profile: JDProfile, output_path: Path = cfg.OUTPUT_CSV_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, cs in enumerate(final_candidates, 1):
        snap = store.snapshot(cs.candidate_id)
        rows.append({"candidate_id": cs.candidate_id, "rank": rank, "score": round(cs.ce_score * 100, 4), "reasoning": generate_reasoning(cs, snap, jd_flags, jd_keywords, jd_profile, rank)})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV written: %s (%d rows)", output_path, len(rows))