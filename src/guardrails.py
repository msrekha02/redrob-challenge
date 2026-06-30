"""
guardrails.py — Production: All Penalties + Cross-Encoder
=================================================================
"""

from __future__ import annotations
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
import honeypot
from jd_processor import JDProfile
from retriever import ArtifactStore, RetrievalResult

log = logging.getLogger(__name__)
REFERENCE_DATE = date(2026, 6, 29)

@dataclass
class CandidateScore:
    candidate_id:     str
    array_idx:        int
    rrf_score:        float
    role_score:       float
    cap_score:        float
    bm25_role:        float
    bm25_cap:         float
    behavioral:       float
    domain_sim:       float   = 1.0
    role_score_norm:  float   = 0.0
    cap_score_norm:   float   = 0.0
    combined_dense_raw:  float = 0.0
    combined_dense_norm: float = 0.0
    is_disqualified:  bool    = False
    dq_reason:        str     = ""
    m_avail:          float   = 1.0
    m_loc:            float   = 1.0
    m_notice:         float   = 1.0
    m_integrity:      float   = 1.0
    m_jd:             float   = 1.0
    penalized_score:  float   = 0.0
    ce_score:         float   = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    penalty_notes:    list[str] = field(default_factory=list)


def _norm_co(name: str) -> str:
    return name.lower().strip().replace(".", "").replace(",", "").replace("-", " ")


# ─────────────────────────────────────────────────────────────────────────────
# Restored Layer 0 — Hard Disqualifiers 
# ─────────────────────────────────────────────────────────────────────────────
def apply_hard_disqualifiers(snap: dict, jd_profile: JDProfile) -> tuple[bool, str]:
    career = snap.get("career_history", [])
    if not career:
        return False, ""
    
    # 100% Consulting-Only Career Hard Disqualifier (Restored Challenge DQ-1)
    if jd_profile.consulting_penalty_active:
        if all(_norm_co(j.get("company", "")) in cfg.CONSULTING_FIRMS for j in career):
            return True, "100% consulting career (DQ-1)"
            
    # Pure Research / Zero Production Hard Disqualifier (Restored Challenge DQ-2)
    if jd_profile.research_penalty_active:
        if all(any(kw in (j.get("title") or "").lower() for kw in cfg.RESEARCH_TITLE_KEYWORDS) for j in career):
            all_desc = " ".join(j.get("description", "").lower() for j in career)
            if not any(kw in all_desc for kw in cfg.PRODUCTION_KEYWORDS):
                return True, "Pure research / zero production (DQ-2)"
                
    return False, ""


def compute_domain_similarity(cand_role_vec: np.ndarray, jd_profile: JDProfile) -> float:
    anchor = jd_profile.category_anchor_vec
    if not np.any(anchor): return 1.0
    return float(np.dot(cand_role_vec, anchor))

# Semantic Domain Anchor Mismatch Penalty (Fixed Rule)
def domain_mismatch_multiplier(domain_sim: float, threshold: float) -> float:
    if domain_sim >= threshold: return 1.0
    return max(0.15, domain_sim / threshold)


# Framework-Only AI Tourist Penalty (Challenge Fixed P-17)
def _check_framework_only_ai(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    if not jd_profile.flags.get("is_ml_ai"): return 0.0, ""
    skills = snap.get("skills", [])
    if not skills: return 0.0, ""
    all_skill_text = " ".join(s.get("name", "").lower() for s in skills)
    has_wrapper = any(fw in all_skill_text for fw in cfg.FRAMEWORK_WRAPPER_SKILLS)
    has_foundational = any(fd in all_skill_text for fd in cfg.FOUNDATIONAL_AI_SKILLS)
    if has_wrapper and not has_foundational:
        return cfg.FRAMEWORK_ONLY_HARD_PENALTY, "framework-only AI (no foundational ML)"
    return 0.0, ""


# Bypassed Management Track Drift Guard (Dynamic Exclusion of P-18)
def _check_non_coding_senior(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    if not jd_profile.flags.get("is_engineering"): return 0.0, ""
    if jd_profile.flags.get("is_founding_team"): return 0.0, "" # Bypass for founding teams
    career = snap.get("career_history", [])
    current_jobs = [j for j in career if j.get("is_current")]
    for job in current_jobs:
        title, duration = (job.get("title") or "").lower(), job.get("duration_months") or 0
        if any(et in title for et in cfg.EXECUTIVE_NON_CODER_TITLES) and duration > cfg.NON_CODING_SENIOR_DURATION_THRESHOLD:
            return cfg.NON_CODING_SENIOR_DEDUCTION, f"non-coding senior role: '{job.get('title')}' for {duration}m"
    return 0.0, ""


# Domain Specialization Isolation Penalty (Challenge Fixed P-19)
def _check_cv_speech_no_nlp(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    if not jd_profile.nlp_ir_required: return 0.0, ""
    career, skills = snap.get("career_history", []), snap.get("skills", [])
    combined = " ".join((j.get("title") or "") + " " + (j.get("description") or "") for j in career).lower() + " " + " ".join(s.get("name", "") for s in skills).lower()
    has_cv_speech = any(kw in combined for kw in cfg.CV_SPEECH_DOMAIN_TITLES)
    has_nlp_ir = any(sig in combined for sig in cfg.NLP_IR_SIGNALS)
    if has_cv_speech and not has_nlp_ir:
        return cfg.CV_SPEECH_WITHOUT_NLP_DEDUCTION, "CV/Speech/Robotics specialist with no NLP/IR background"
    return 0.0, ""


# Recency, Open to Work, Interview & Response Dynamics (Dynamic)
def _compute_m_avail(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    notes: list[str] = []
    m_active, m_open, m_response, m_interview = 1.0, 1.0, 1.0, 1.0
    
    # Only enforce if JD specifies availability enforcement or has high importance
    if jd_profile.enforce_availability or jd_profile.avail_importance > 0.5:
        try:
            last = date.fromisoformat(str(snap.get("last_active_date", "")))
            days = (REFERENCE_DATE - last).days
        except (TypeError, ValueError): days = 999
        m_active = cfg.ACTIVITY_FLOOR_MULT
        for max_d, mult in cfg.ACTIVITY_TIERS:
            if days <= max_d: m_active = mult; break

        m_open = cfg.OPEN_TO_WORK_MULT if snap.get("open_to_work_flag") else cfg.NOT_OPEN_TO_WORK_MULT
        
        rr = max(0.0, min(1.0, float(snap.get("recruiter_response_rate") or 0.5)))
        m_response = float(np.clip(cfg.RESPONSE_RATE_BASE + rr * cfg.RESPONSE_RATE_SCALE, cfg.RESPONSE_RATE_MIN, cfg.RESPONSE_RATE_MAX))
        
        ic = max(0.0, min(1.0, float(snap.get("interview_completion_rate") or 1.0)))
        m_interview = cfg.INTERVIEW_POOR_MULT if ic < cfg.INTERVIEW_COMPLETION_THRESHOLD else cfg.INTERVIEW_OK_MULT

    m_avail_raw = m_active * m_open * m_response * m_interview
    importance = float(jd_profile.avail_importance)
    m_avail = m_avail_raw * importance + (1.0 - importance)
    return max(0.05, min(1.0, m_avail)), notes


# Strict Target Location vs. Sourcing Hub Multiplier (Fixed Rule)
def _compute_m_loc(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    if not jd_profile.flags.get("prefers_local"): return 1.0, []
    pref_cities = jd_profile.preferred_cities or cfg.PREFERRED_CITIES
    location, country = (snap.get("location") or "").lower(), (snap.get("country") or "in").lower()
    relocate = bool(snap.get("willing_to_relocate", False))
    in_pref = any(c in location for c in pref_cities)
    in_india = country in ("in", "india") or any(c in location for c in ("india", "bengaluru", "bangalore", "mumbai", "delhi", "pune", "hyderabad", "noida", "gurgaon"))

    m_raw = cfg.LOC_PREFERRED if in_pref else (cfg.LOC_INDIA_RELOCATE if in_india and relocate else (cfg.LOC_INDIA_NO_RELOCATE if in_india else (cfg.LOC_INTL_RELOCATE if relocate else cfg.LOC_INTL_NO_RELOCATE)))
    importance = float(jd_profile.loc_importance)
    return max(0.20, min(1.0, m_raw * importance + (1.0 - importance))), []


# Urgent JD Notice Period Penalty Modifier (Fixed logic)
def _compute_m_notice(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    if not jd_profile.flags.get("prefers_short_notice") and jd_profile.urgency < 0.60: return 1.0, []
    notice = int(snap.get("notice_period_days") or 90)
    stated = jd_profile.notice_period_days_stated
    tiers = [(stated, 1.00), (stated * 2, 0.90), (stated * 3, 0.80)] if stated else cfg.NOTICE_TIERS
    m_notice = cfg.NOTICE_FLOOR_MULT
    for max_d, mult in tiers:
        if notice <= max_d: m_notice = mult; break
    threshold = stated if stated is not None else 30
    if jd_profile.urgency > 0.60 and notice > threshold:
        m_notice = max(0.50, m_notice * (1.0 - (jd_profile.urgency - 0.60) * 0.30))
    return float(m_notice), []


# Integrity & Hollow Claims (Dynamic based on JD toggles)
def _compute_m_integrity(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    if not jd_profile.requires_high_integrity: return 1.0, []
    deductions = 0.0
    s_min, s_max = snap.get("expected_salary_min"), snap.get("expected_salary_max")
    if s_min and s_max:
        try:
            if float(s_min) > float(s_max): deductions += cfg.SALARY_INVERSION_PENALTY
        except (TypeError, ValueError): pass

    anchors = sum([bool(snap.get("verified_email")), bool(snap.get("verified_phone")), bool(snap.get("linkedin_connected"))])
    if anchors == 0: deductions += cfg.TRUST_ZERO_ANCHOR_PENALTY
    elif anchors == 1: deductions += cfg.TRUST_ONE_ANCHOR_PENALTY

    comp = float(snap.get("profile_completeness_score") or 0)
    if comp < cfg.COMPLETENESS_LOW_THRESHOLD: deductions += cfg.COMPLETENESS_LOW_PENALTY
    elif comp < cfg.COMPLETENESS_MED_THRESHOLD: deductions += cfg.COMPLETENESS_MED_PENALTY

    assessments = snap.get("skill_assessment_scores") or {}
    n_hollow = sum(1 for s in snap.get("skills", []) if (s.get("proficiency") or "").lower() == "expert" and int(s.get("endorsements") or 0) == 0 and (s.get("name") or "") not in assessments)
    if n_hollow: deductions += min(n_hollow * cfg.HOLLOW_EXPERT_PER_SKILL, cfg.HOLLOW_EXPERT_DEDUCTION_CAP)

    return float(np.clip(1.0 - deductions, cfg.INTEGRITY_FLOOR, 1.0)), []


# JD Fit Modifier
def _compute_m_jd(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    delta = 0.0
    career = snap.get("career_history", [])

    # Dynamic Job Hopping Penalty
    if jd_profile.penalize_job_hoppers:
        completed = [j.get("duration_months") or 0 for j in career if not j.get("is_current") and j.get("duration_months")]
        if completed:
            avg = sum(completed) / len(completed)
            if avg < cfg.JOB_HOP_VERY_SHORT_MONTHS: delta -= cfg.JOB_HOP_HARD_PENALTY
            elif avg < cfg.JOB_HOP_SHORT_MONTHS: delta -= cfg.JOB_HOP_SOFT_PENALTY
            
        offer_rate = float(snap.get("offer_acceptance_rate") or -1)
        if offer_rate != -1.0 and offer_rate < cfg.OFFER_ACCEPTANCE_POOR_THRESHOLD:
            delta -= cfg.OFFER_ACCEPTANCE_PENALTY

    if jd_profile.flags.get("prefers_open_source"):
        gh = float(snap.get("github_activity_score") or -1)
        if gh >= cfg.GITHUB_HIGH_THRESHOLD: delta += cfg.GITHUB_HIGH_BONUS
        elif gh >= cfg.GITHUB_MED_THRESHOLD: delta += cfg.GITHUB_MED_BONUS

    fw_deduction, _ = _check_framework_only_ai(snap, jd_profile)
    delta -= fw_deduction
    nc_deduction, _ = _check_non_coding_senior(snap, jd_profile)
    delta -= nc_deduction
    cs_deduction, _ = _check_cv_speech_no_nlp(snap, jd_profile)
    delta -= cs_deduction

    # Dynamic Experience (YoE) Out-of-Bounds Penalty (Fixed Rule)
    yoe_min, yoe_max = jd_profile.yoe_min, jd_profile.yoe_max
    if yoe_min is not None or yoe_max is not None:
        cand_yoe = float(snap.get("years_of_experience") or 0)
        if cand_yoe > 0:
            if yoe_min is not None and cand_yoe < yoe_min - cfg.YOE_TOLERANCE: delta -= cfg.YOE_BELOW_RANGE_DEDUCTION
            elif yoe_max is not None and cand_yoe > yoe_max + cfg.YOE_TOLERANCE: delta -= cfg.YOE_ABOVE_RANGE_DEDUCTION

    return float(np.clip(1.0 + delta, cfg.JD_FIT_MIN, cfg.JD_FIT_MAX)), []


def _find_matched_keywords(snap: dict, jd_keywords: list[str]) -> list[str]:
    blob = " ".join([snap.get("summary", ""), " ".join(s.get("name", "") for s in snap.get("skills", [])), " ".join(j.get("description", "") for j in snap.get("career_history", []))]).lower()
    return [kw for kw in jd_keywords if kw in blob][:20]

def _robust_minmax_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0: return scores
    lo, hi = np.percentile(scores, cfg.SEMANTIC_NORM_LOW_PERCENTILE), np.percentile(scores, cfg.SEMANTIC_NORM_HIGH_PERCENTILE)
    if hi - lo < cfg.SEMANTIC_NORM_MIN_SPREAD: return np.full_like(scores, 0.5, dtype=np.float32)
    return np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

def apply_penalties(retrieval: RetrievalResult, store: ArtifactStore, jd_profile: JDProfile, jd_keywords: list[str], top_k: int = cfg.TOP_K_PENALTY) -> list[CandidateScore]:
    role_norm = _robust_minmax_normalize(retrieval.role_scores.astype(np.float64))
    cap_norm  = _robust_minmax_normalize(retrieval.cap_scores.astype(np.float64))
    combined_raw = jd_profile.role_weight * retrieval.role_scores + jd_profile.cap_weight * retrieval.cap_scores
    combined_norm = _robust_minmax_normalize(combined_raw.astype(np.float64))

    scored: list[CandidateScore] = []
    for i, (arr_idx, cid) in enumerate(zip(retrieval.indices, retrieval.candidate_ids)):
        arr_idx = int(arr_idx)
        snap = store.snapshot(cid)
        behav = float(store.behaviors[arr_idx])

        cs = CandidateScore(
            candidate_id=cid, array_idx=arr_idx,
            rrf_score=float(retrieval.rrf_scores[i]),
            role_score=float(retrieval.role_scores[i]), cap_score=float(retrieval.cap_scores[i]),
            role_score_norm=float(role_norm[i]), cap_score_norm=float(cap_norm[i]),
            combined_dense_raw=float(combined_raw[i]), combined_dense_norm=float(combined_norm[i]),
            bm25_role=float(retrieval.bm25_role_scores[i]), bm25_cap=float(retrieval.bm25_cap_scores[i]),
            behavioral=behav,
        )

        is_dq, dq_reason = apply_hard_disqualifiers(snap, jd_profile)
        if is_dq:
            cs.is_disqualified, cs.dq_reason, cs.penalized_score = True, dq_reason, cfg.SCORE_DISQUALIFIED
            scored.append(cs); continue

        dom_sim = compute_domain_similarity(store.role_vecs[arr_idx], jd_profile)
        cs.domain_sim = dom_sim
        domain_mult = domain_mismatch_multiplier(dom_sim, jd_profile.domain_mismatch_threshold)
        stuffer_mult = cfg.FAKER_PENALTY_MULTIPLIER if cs.cap_score_norm >= cfg.CAP_SCORE_FAKER_THRESHOLD and cs.role_score_norm < cfg.ROLE_SCORE_FAKER_THRESHOLD else 1.0

        cs.m_avail, _ = _compute_m_avail(snap, jd_profile)
        cs.m_loc, _ = _compute_m_loc(snap, jd_profile)
        cs.m_notice, _ = _compute_m_notice(snap, jd_profile)
        cs.m_integrity, _ = _compute_m_integrity(snap, jd_profile)
        cs.m_jd, _ = _compute_m_jd(snap, jd_profile)

        cs.penalized_score = cs.combined_dense_norm * domain_mult * stuffer_mult * cs.m_avail * cs.m_loc * cs.m_notice * cs.m_integrity * cs.m_jd
        cs.matched_keywords = _find_matched_keywords(snap, jd_keywords)
        scored.append(cs)

    non_dq_scored = [s for s in scored if not s.is_disqualified]
    if len(non_dq_scored) <= top_k:
        non_dq = sorted(non_dq_scored, key=lambda x: (x.penalized_score, x.behavioral), reverse=True)
    else:
        scores_arr = np.array([s.penalized_score for s in non_dq_scored], dtype=np.float64)
        top_pos = np.argpartition(scores_arr, -top_k)[-top_k:]
        top_pos_sorted = sorted(top_pos, key=lambda i: (scores_arr[i], non_dq_scored[i].behavioral), reverse=True)
        non_dq = [non_dq_scored[i] for i in top_pos_sorted]

    return non_dq

class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None
    def _load(self) -> None:
        if self._model is not None: return
        from model_engine import ONNXCrossEncoder
        self._model = ONNXCrossEncoder(cfg.BGE_RERANKER_MODEL_DIR, fallback_model_id=cfg.BGE_RERANKER_MODEL_ID)
    def rerank(self, scored: list[CandidateScore], store: ArtifactStore, raw_jd_text: str, batch_size: int = 32) -> list[CandidateScore]:
        self._load()
        pairs = [(raw_jd_text[:1500], self._build_ce_text(store.snapshot(cs.candidate_id))) for cs in scored]
        raw = np.array(self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo + 1e-9)
        for cs, n in zip(scored, norm):
            cs.ce_score = max(0.0, 0.70 * float(n) + 0.30 * cs.penalized_score)
        scored.sort(key=lambda x: x.ce_score, reverse=True)
        return scored
    def _build_ce_text(self, snap: dict) -> str:
        parts = []
        title, company = snap.get("current_title", ""), snap.get("current_company", "")
        if title: parts.append(f"Current: {title}" + (f" at {company}" if company else ""))
        if yoe := snap.get("years_of_experience"): parts.append(f"Experience: {yoe}y")
        if s := snap.get("summary"): parts.append(s[:400])
        skills = [s.get("name", "") for s in snap.get("skills", []) if s.get("name")]
        if skills: parts.append("Skills: " + ", ".join(skills[:15]))
        for job in snap.get("career_history", [])[:3]:
            parts.append(f"{job.get('title','')} at {job.get('company','')}. {(job.get('description') or '')[:150]}".strip())
        for edu in snap.get("education", [])[:1]:
            parts.append(f"Education: {edu.get('degree','')} in {edu.get('field_of_study','')} from {edu.get('institution','')}".strip())
        return " | ".join(filter(None, parts))[:2000]

def apply_honeypot_safety_net(ranked: list[CandidateScore], store: ArtifactStore, final_k: int = cfg.TOP_K_FINAL) -> list[CandidateScore]:
    clean = []
    for cs in ranked:
        if len(clean) >= final_k: break
        snap = store.snapshot(cs.candidate_id)
        hp_dict = {
            "candidate_id": cs.candidate_id, "profile": {"years_of_experience": snap.get("years_of_experience", 0)},
            "career_history": snap.get("career_history", []), "education": snap.get("education", []), "skills": snap.get("skills", []),
            "redrob_signals": {
                "signup_date": snap.get("signup_date", ""), "last_active_date": snap.get("last_active_date", ""),
                "profile_completeness_score": snap.get("profile_completeness_score", 0), "recruiter_response_rate": snap.get("recruiter_response_rate", 0.5),
                "interview_completion_rate": snap.get("interview_completion_rate", 1.0), "offer_acceptance_rate": snap.get("offer_acceptance_rate", -1),
                "github_activity_score": snap.get("github_activity_score", -1), "notice_period_days": snap.get("notice_period_days", 90),
                "avg_response_time_hours": snap.get("avg_response_time_hours", 0), "profile_views_received_30d": snap.get("profile_views_received_30d", 0),
                "applications_submitted_30d": snap.get("applications_submitted_30d", 0), "connection_count": snap.get("connection_count", 0),
                "endorsements_received": snap.get("endorsements_received", 0), "search_appearance_30d": snap.get("search_appearance_30d", 0),
                "saved_by_recruiters_30d": snap.get("saved_by_recruiters_30d", 0),
            }
        }
        if not honeypot.is_honeypot(hp_dict)["is_honeypot"]: clean.append(cs)
    return clean