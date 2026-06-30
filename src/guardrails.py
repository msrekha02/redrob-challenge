"""
guardrails.py — Production: All Penalties + Cross-Encoder (v3)
=================================================================
v3 CHANGES (this revision):
  1. Removed the hard Layer-0 binary DQ for consulting/research career
     history (DQ-1, DQ-2). Replaced with continuous, recency-weighted,
     content-aware multipliers (see _compute_consulting_penalty_multiplier,
     _compute_research_penalty_multiplier) — a job's pull on the penalty
     decays the further in the past it ended, and a "consulting firm" job
     whose own description shows real production/infra signal counts at a
     fraction of full weight. No candidate is ever zeroed out purely by a
     company-name match against a fixed list; a genuinely current, recent,
     pure consulting/research career with zero production signal still
     gets crushed toward a steep floor — functionally equivalent to
     disqualification for the case the JD actually targets, without an
     absolute, context-blind cutoff catching edge cases it shouldn't.
  2. apply_penalties() now does the semantic scoring in two passes: first
     a vectorized pass computes the raw combined dense score for every
     retrieved candidate and applies a robust (percentile-clipped) min-max
     normalization across the pool BEFORE any behavioral multiplier is
     applied. Raw BGE cosine similarities cluster tightly across a
     candidate pool; multiplying that narrow band directly by behavioral
     multipliers let behavioral signal dominate the ranking. Stretching to
     the full [0,1] range first restores the intended balance between
     semantic relevance and behavioral signal.
  3. Field rename: consulting_dq_applies/research_dq_applies →
     consulting_penalty_active/research_penalty_active throughout, since
     they no longer gate an absolute disqualification (see jd_processor.py).

Prior fixes retained from earlier revisions:
  • m_avail/m_loc formula: scales the PENALTY by importance, not the score
    directly (importance=0 → no penalty; importance=1 → full penalty).
  • Zero-anchor guard on domain-mismatch check.
  • Dynamic preferred_cities, framework-only-AI / non-coding-senior /
    CV-speech-without-NLP signals, YoE range penalty, honeypot safety net,
    ONNX INT8 model loading via model_engine.py — all unchanged in this pass.
"""

from __future__ import annotations

import logging
import sys
import time
import difflib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
import honeypot
from jd_processor import JDProfile
from retriever import ArtifactStore, RetrievalResult

log = logging.getLogger(__name__)
REFERENCE_DATE = date(2026, 6, 29)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate score container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateScore:
    candidate_id:     str
    array_idx:        int
    rrf_score:        float
    role_score:       float          # raw per-track cosine sim (kept for reasoning text / debug)
    cap_score:        float          # raw per-track cosine sim (kept for reasoning text / debug)
    bm25_role:        float
    bm25_cap:         float
    behavioral:       float
    domain_sim:       float   = 1.0
    role_score_norm:  float   = 0.0  # pool-normalised — used for stuffer check + penalized_score
    cap_score_norm:   float   = 0.0  # pool-normalised — used for stuffer check + penalized_score
    combined_dense_raw:  float = 0.0  # pre-normalisation, for transparency/debug
    combined_dense_norm: float = 0.0  # what actually feeds penalized_score
    is_disqualified:  bool    = False
    dq_reason:        str     = ""
    consulting_mult:  float   = 1.0   # continuous penalty (replaces old hard DQ-1)
    research_mult:    float   = 1.0   # continuous penalty (replaces old hard DQ-2)
    m_avail:          float   = 1.0
    m_loc:            float   = 1.0
    m_notice:         float   = 1.0
    m_integrity:      float   = 1.0
    m_jd:             float   = 1.0
    penalized_score:  float   = 0.0
    ce_score:         float   = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    penalty_notes:    list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 0 — Hard Disqualifiers (extension point — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────
# This JD's two formerly-hard disqualifiers (100% consulting career, pure
# research career) are now continuous Layer-5 multipliers — see
# _compute_consulting_penalty_multiplier / _compute_research_penalty_multiplier
# below. apply_hard_disqualifiers() is kept as a genuine extension point for
# any FUTURE truly-binary structural disqualifier (e.g. a strict work-
# authorization boolean) where a hard cut is actually the right model —
# it currently has nothing to check for this JD.

def _norm_co(name: str) -> str:
    return name.lower().strip().replace(".", "").replace(",", "").replace("-", " ")


def _job_recency_weight(job: dict) -> float:
    """
    Exponential recency decay: a CURRENT role weighs 1.0; a role that ended
    CAREER_RECENCY_HALF_LIFE_YEARS ago weighs ~0.5; twice that ago weighs
    ~0.25; floored at MIN_JOB_RECENCY_WEIGHT so even very old jobs retain
    some (small) influence rather than being completely erased.

    This is what lets "TCS 8 years ago" matter far less than "TCS, current
    role" in the consulting/research penalty below.
    """
    end_raw = job.get("end_date")
    try:
        end = date.fromisoformat(end_raw) if end_raw else REFERENCE_DATE   # current role -> full weight
    except (TypeError, ValueError):
        end = REFERENCE_DATE
    years_since_end = max(0.0, (REFERENCE_DATE - end).days / 365.25)
    weight = 0.5 ** (years_since_end / cfg.CAREER_RECENCY_HALF_LIFE_YEARS)
    return max(cfg.MIN_JOB_RECENCY_WEIGHT, weight)


def _compute_consulting_penalty_multiplier(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    """
    Replaces the old binary DQ-1 (100% consulting → hard 0.0) with a
    continuous, recency-weighted, content-aware multiplier in
    [CONSULTING_PENALTY_FLOOR, 1.0] — never an absolute 0.0, so a candidate
    is never eliminated purely by a company-name match against a fixed
    list. A consulting-firm job whose own description shows real
    production/infra signal (CONSULTING_PRODUCTION_REDEMPTION_WEIGHT) only
    counts at a fraction of full weight — some people do build real
    platforms on a client embed while nominally employed by a services firm.

    A candidate whose ENTIRE recency-weighted, content-checked career is
    consulting still asymptotically approaches the floor — functionally
    equivalent to disqualification for the case the JD targets — while an
    exceptional candidate with old consulting experience and years of
    recent product-company work is barely touched.
    """
    if not jd_profile.consulting_penalty_active:
        return 1.0, ""

    career = snap.get("career_history", [])
    if not career:
        return 1.0, ""

    weighted_consulting = 0.0
    weighted_total = 0.0
    for job in career:
        months = job.get("duration_months") or 0
        if months <= 0:
            continue
        recency_w = _job_recency_weight(job)
        is_consulting = _norm_co(job.get("company", "")) in cfg.CONSULTING_FIRMS
        if not is_consulting:
            effective_weight = 0.0
        else:
            desc = (job.get("description") or "").lower()
            has_prod_signal = any(kw in desc for kw in cfg.PRODUCTION_KEYWORDS)
            effective_weight = cfg.CONSULTING_PRODUCTION_REDEMPTION_WEIGHT if has_prod_signal else 1.0
        weighted_consulting += months * recency_w * effective_weight
        weighted_total += months * recency_w

    if weighted_total <= 0:
        return 1.0, ""

    intensity = weighted_consulting / weighted_total   # in [0, 1]
    floor = cfg.CONSULTING_PENALTY_FLOOR
    multiplier = 1.0 - intensity * (1.0 - floor)

    note = ""
    if intensity > 0.85:
        note = f"recency-weighted consulting intensity {intensity:.0%} (×{multiplier:.2f})"
    elif intensity > 0.50:
        note = f"moderate recency-weighted consulting intensity {intensity:.0%} (×{multiplier:.2f})"

    return float(multiplier), note


def _compute_research_penalty_multiplier(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    """
    Replaces the old binary DQ-2 (100% pure research → hard 0.0) with the
    same recency-weighted, content-aware treatment as the consulting
    multiplier above. A role counts as "research" if its title matches
    RESEARCH_TITLE_KEYWORDS; it's redeemed toward partial weight if its own
    description shows production/infra signal (someone who shipped a real
    system out of a research role isn't the case the JD is targeting).
    """
    if not jd_profile.research_penalty_active:
        return 1.0, ""

    career = snap.get("career_history", [])
    if not career:
        return 1.0, ""

    weighted_research = 0.0
    weighted_total = 0.0
    for job in career:
        months = job.get("duration_months") or 0
        if months <= 0:
            continue
        recency_w = _job_recency_weight(job)
        title = (job.get("title") or "").lower()
        is_research = any(kw in title for kw in cfg.RESEARCH_TITLE_KEYWORDS)
        if not is_research:
            effective_weight = 0.0
        else:
            desc = (job.get("description") or "").lower()
            has_prod_signal = any(kw in desc for kw in cfg.PRODUCTION_KEYWORDS)
            effective_weight = cfg.RESEARCH_PRODUCTION_REDEMPTION_WEIGHT if has_prod_signal else 1.0
        weighted_research += months * recency_w * effective_weight
        weighted_total += months * recency_w

    if weighted_total <= 0:
        return 1.0, ""

    intensity = weighted_research / weighted_total
    floor = cfg.RESEARCH_PENALTY_FLOOR
    multiplier = 1.0 - intensity * (1.0 - floor)

    note = ""
    if intensity > 0.85:
        note = f"recency-weighted pure-research intensity {intensity:.0%} (×{multiplier:.2f})"
    elif intensity > 0.50:
        note = f"moderate recency-weighted research intensity {intensity:.0%} (×{multiplier:.2f})"

    return float(multiplier), note


def _check_dq_consulting_literal(snap: dict, jd_profile: JDProfile) -> tuple[bool, str]:
    """
    DQ-1 (restored, v4): the JD's literal wording — "People who have only
    worked at consulting firms... in their entire career" — is an explicit
    "will not move forward" hard disqualifier, not a soft one. Fires ONLY
    when EVERY career_history entry matches CONSULTING_FIRMS (the literal
    100% case the JD describes), gated by jd_profile.consulting_penalty_active
    so it never fires for a JD that doesn't care about this at all.

    This is layered ALONGSIDE _compute_consulting_penalty_multiplier, not a
    replacement for it: the continuous multiplier still handles every
    candidate who has SOME consulting exposure but not literally 100% of
    their career (e.g. old consulting + years of recent product-company
    work) — exactly the edge case a blanket hard cut would wrongly catch.
    Only the literal, unambiguous 100% case gets the hard cut the JD
    explicitly asks for.
    """
    if not jd_profile.consulting_penalty_active:
        return False, ""
    career = snap.get("career_history", [])
    if not career:
        return False, ""
    consulting_jobs = [j for j in career if _norm_co(j.get("company", "")) in cfg.CONSULTING_FIRMS]
    if len(consulting_jobs) == len(career):
        firms = list({j.get("company", "") for j in consulting_jobs})
        return True, f"100% consulting career, no exceptions ({', '.join(firms[:3])})"
    return False, ""


def _check_dq_research_literal(snap: dict, jd_profile: JDProfile) -> tuple[bool, str]:
    """
    DQ-2 (restored, v4): the JD's literal wording — "If you've spent your
    career in pure research environments... without any production
    deployment — we will not move forward. We are explicit about this." —
    is an explicit hard disqualifier. Fires ONLY when EVERY career_history
    entry is research-titled AND none show production signal in its
    description (the literal 100% case), gated by
    jd_profile.research_penalty_active.

    Layered alongside _compute_research_penalty_multiplier the same way as
    the consulting check above — the continuous multiplier still protects
    a candidate with old pure-research experience and years of recent
    production ML work.
    """
    if not jd_profile.research_penalty_active:
        return False, ""
    career = snap.get("career_history", [])
    if not career:
        return False, ""
    for job in career:
        title = (job.get("title") or "").lower()
        desc  = (job.get("description") or "").lower()
        is_research = any(kw in title for kw in cfg.RESEARCH_TITLE_KEYWORDS)
        has_prod    = any(kw in desc for kw in cfg.PRODUCTION_KEYWORDS)
        if not is_research or has_prod:
            return False, ""   # at least one non-research or production-bearing role -> not the literal 100% case
    return True, "Entire career in pure research, no production deployment signal anywhere"


def apply_hard_disqualifiers(snap: dict, jd_profile: JDProfile) -> tuple[bool, str]:
    """
    Layer 0 — genuinely binary structural disqualifiers. DQ-1/DQ-2 were
    restored here (v4) at the user's explicit request, matching the JD's
    own literal "we will not move forward" wording for the LITERAL 100%
    case only. Every candidate who isn't literally 100% consulting or
    100% pure research falls through to the continuous, recency-weighted
    Layer-5 multipliers (_compute_consulting_penalty_multiplier /
    _compute_research_penalty_multiplier) instead — the two layers are
    complementary, not redundant: this layer enforces exact JD-literal
    compliance for the unambiguous edge case the JD explicitly calls out;
    the continuous layer protects every other candidate from a blunt,
    context-blind cutoff.
    """
    for check in (_check_dq_consulting_literal, _check_dq_research_literal):
        is_dq, reason = check(snap, jd_profile)
        if is_dq:
            return True, reason
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC DOMAIN MISMATCH (replaces hardcoded INCOMPATIBLE_DOMAIN_TITLES)
# ─────────────────────────────────────────────────────────────────────────────

def compute_domain_similarity(cand_role_vec: np.ndarray, jd_profile: JDProfile) -> float:
    """
    Cosine similarity between candidate's role embedding and the JD's
    auto-detected category anchor. Both are L2-normalised → dot product = cosine.

    GUARD: if anchor is all-zeros (JDProfile built without embed model),
    returns 1.0 so no penalty is applied.
    """
    anchor = jd_profile.category_anchor_vec
    if not np.any(anchor):      # all-zeros guard — BUG FIX
        return 1.0
    return float(np.dot(cand_role_vec, anchor))


def domain_mismatch_multiplier(domain_sim: float, threshold: float) -> float:
    """
    sim >= threshold              → 1.0   (no penalty)
    0.5 × threshold < sim < thresh→ 0.15–1.0 linear
    sim = 0.0                     → 0.15  (floor, not zero)
    """
    if domain_sim >= threshold:
        return 1.0
    return max(0.15, domain_sim / threshold)


# ─────────────────────────────────────────────────────────────────────────────
# NEW SIGNAL: Framework-only AI (JD: "LangChain tourists not wanted")
# ─────────────────────────────────────────────────────────────────────────────

def _check_framework_only_ai(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    """
    Detect "LangChain tourist": candidate whose AI skill set consists
    ONLY of orchestration-framework wrappers with no foundational ML skills.
    JD explicitly disqualifies: "AI experience = recent <12mo LangChain/OpenAI calls."

    Returns (deduction, note). Fires ONLY when jd_profile.flags.is_ml_ai=True.
    """
    if not jd_profile.flags.get("is_ml_ai"):
        return 0.0, ""

    skills = snap.get("skills", [])
    if not skills:
        return 0.0, ""

    all_skill_text = " ".join(s.get("name", "").lower() for s in skills)

    has_wrapper     = any(fw in all_skill_text for fw in cfg.FRAMEWORK_WRAPPER_SKILLS)
    has_foundational= any(fd in all_skill_text for fd in cfg.FOUNDATIONAL_AI_SKILLS)

    if not has_wrapper:
        return 0.0, ""  # no framework skills — not a tourist

    if not has_foundational:
        # All AI exposure is framework-only
        # Check recency: if framework skills are short-duration, penalty is harsher
        short_fw = sum(
            1 for s in skills
            if any(fw in (s.get("name") or "").lower() for fw in cfg.FRAMEWORK_WRAPPER_SKILLS)
            and isinstance(s.get("duration_months"), (int, float))
            and s["duration_months"] < 12
        )
        if short_fw > 0:
            return cfg.FRAMEWORK_ONLY_HARD_PENALTY, f"framework-only AI (<12mo, no foundational ML)"
        else:
            return cfg.FRAMEWORK_ONLY_SOFT_PENALTY, f"framework-only AI skills (no foundational ML)"

    return 0.0, ""  # has both → fine


# ─────────────────────────────────────────────────────────────────────────────
# NEW SIGNAL: 18-month non-coding senior (JD explicit disqualifier)
# ─────────────────────────────────────────────────────────────────────────────

def _check_non_coding_senior(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    """
    JD: "Senior engineer who hasn't written production code in 18 months
    because they moved into architecture/tech lead roles — not a fit."

    Detects candidates currently in executive/architecture roles for > 18 months.
    Fires ONLY for engineering JDs (jd_profile.flags.is_engineering=True).

    v4 — Founding-team bypass, added at user request: a founding-team JD
    often genuinely wants leadership-experienced people who can own
    architectural decisions, so this penalty bypasses when
    jd_profile.flags.is_founding_team is True.

    IMPORTANT — this bypass is gated by jd_profile.flags.requires_code_writing
    (an existing flag matching JD phrases like "this role writes code" /
    "hands-on coding"), and only activates when that flag is ABSENT.
    This JD's own text is the reason the gate matters: it's a founding-team
    role AND it explicitly states "This role writes code," directly
    overriding the general founding-team assumption. Without the
    requires_code_writing gate, the bypass would silently stop penalizing
    exactly the candidate profile (a non-coding Director/VP type) this JD
    explicitly says it doesn't want — for THIS JD specifically,
    requires_code_writing is True, so the bypass correctly stays inactive
    and the penalty still fires. The bypass only takes effect for a
    DIFFERENT founding-team JD that doesn't make the same explicit demand.

    Returns (deduction, note).
    """
    if not jd_profile.flags.get("is_engineering"):
        return 0.0, ""

    if jd_profile.flags.get("is_founding_team") and not jd_profile.flags.get("requires_code_writing"):
        return 0.0, ""   # founding-team bypass active — JD doesn't override it

    career = snap.get("career_history", [])
    current_jobs = [j for j in career if j.get("is_current")]

    for job in current_jobs:
        title    = (job.get("title") or "").lower()
        duration = job.get("duration_months") or 0

        is_exec = any(et in title for et in cfg.EXECUTIVE_NON_CODER_TITLES)
        if is_exec and duration > cfg.NON_CODING_SENIOR_DURATION_THRESHOLD:
            return (
                cfg.NON_CODING_SENIOR_DEDUCTION,
                f"non-coding senior role: '{job.get('title')}' for {duration}m",
            )

    return 0.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# NEW SIGNAL: CV/Speech without NLP/IR (JD explicit exclusion)
# ─────────────────────────────────────────────────────────────────────────────

def _check_cv_speech_no_nlp(snap: dict, jd_profile: JDProfile) -> tuple[float, str]:
    """
    JD: "People whose primary expertise is CV/Speech/Robotics without
    significant NLP/IR exposure — we'd be re-learning fundamentals."

    Detects domain mismatch beyond what the semantic anchor check covers:
    specifically checks if candidate's career is PRIMARILY CV/Speech
    but they have NO NLP/IR signals at all.

    Fires ONLY when JD requires NLP/IR (jd_profile.nlp_ir_required=True).
    """
    if not jd_profile.nlp_ir_required:
        return 0.0, ""

    career = snap.get("career_history", [])
    skills = snap.get("skills", [])

    # Build text blob from titles + descriptions + skills
    title_blob = " ".join(
        (j.get("title") or "") + " " + (j.get("description") or "")
        for j in career
    ).lower()
    skill_blob = " ".join(s.get("name", "") for s in skills).lower()
    combined   = title_blob + " " + skill_blob

    has_cv_speech = any(kw in combined for kw in cfg.CV_SPEECH_DOMAIN_TITLES)
    has_nlp_ir    = any(sig in combined for sig in cfg.NLP_IR_SIGNALS)

    if has_cv_speech and not has_nlp_ir:
        return (
            cfg.CV_SPEECH_WITHOUT_NLP_DEDUCTION,
            "CV/Speech/Robotics specialist with no NLP/IR background",
        )

    return 0.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Availability Multiplier (m_avail) — FORMULA FIXED
# ─────────────────────────────────────────────────────────────────────────────

def _compute_m_avail(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    notes: list[str] = []

    # P-1: Recency
    try:
        last = date.fromisoformat(str(snap.get("last_active_date", "")))
        days = (REFERENCE_DATE - last).days
    except (TypeError, ValueError):
        days = 999

    m_active = cfg.ACTIVITY_FLOOR_MULT
    for max_d, mult in cfg.ACTIVITY_TIERS:
        if days <= max_d:
            m_active = mult; break
    if days > 90:
        notes.append(f"inactive {days}d (×{m_active:.2f})")

    # P-2: Open to work
    m_open = cfg.OPEN_TO_WORK_MULT if snap.get("open_to_work_flag") else cfg.NOT_OPEN_TO_WORK_MULT
    if not snap.get("open_to_work_flag"):
        notes.append("not open_to_work (×0.80)")

    # P-3: Response rate
    rr = max(0.0, min(1.0, float(snap.get("recruiter_response_rate") or 0.5)))
    m_response = float(np.clip(
        cfg.RESPONSE_RATE_BASE + rr * cfg.RESPONSE_RATE_SCALE,
        cfg.RESPONSE_RATE_MIN, cfg.RESPONSE_RATE_MAX,
    ))
    if rr < 0.3:
        notes.append(f"low response rate {rr:.0%} (×{m_response:.2f})")

    # P-4: Interview completion
    ic = max(0.0, min(1.0, float(snap.get("interview_completion_rate") or 1.0)))
    m_interview = cfg.INTERVIEW_POOR_MULT if ic < cfg.INTERVIEW_COMPLETION_THRESHOLD else cfg.INTERVIEW_OK_MULT
    if ic < cfg.INTERVIEW_COMPLETION_THRESHOLD:
        notes.append(f"low interview completion {ic:.0%}")

    m_avail_raw = m_active * m_open * m_response * m_interview

    # BUG FIX: Scale the PENALTY, not the score.
    # Old (wrong):  m_avail = m_avail_raw × avail_importance
    # New (correct):m_avail = m_avail_raw × importance + (1 - importance)
    # When importance=0 → no penalty (m_avail=1.0)
    # When importance=1 → full penalty (m_avail=m_avail_raw)
    importance = float(jd_profile.avail_importance)
    m_avail = m_avail_raw * importance + (1.0 - importance)
    m_avail = max(0.05, min(1.0, m_avail))
    return m_avail, notes


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Location Multiplier (m_loc) — uses jd_profile.preferred_cities
# ─────────────────────────────────────────────────────────────────────────────

def _compute_m_loc(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    notes: list[str] = []
    if not jd_profile.flags.get("prefers_local"):
        return 1.0, []

    # DYNAMIC: use cities extracted from JD text, not hardcoded cfg.PREFERRED_CITIES
    pref_cities = jd_profile.preferred_cities or cfg.PREFERRED_CITIES

    location = (snap.get("location") or "").lower()
    country  = (snap.get("country")  or "in").lower()
    relocate = bool(snap.get("willing_to_relocate", False))

    in_pref  = any(c in location for c in pref_cities)
    in_india = country in ("in", "india") or any(
        c in location for c in ("india", "bengaluru", "bangalore", "mumbai",
                                  "delhi", "pune", "hyderabad", "noida", "gurgaon")
    )

    if in_pref:
        m_raw = cfg.LOC_PREFERRED
    elif in_india and relocate:
        m_raw = cfg.LOC_INDIA_RELOCATE
        notes.append(f"non-preferred Indian city, will relocate (×{m_raw:.2f})")
    elif in_india:
        m_raw = cfg.LOC_INDIA_NO_RELOCATE
        notes.append(f"non-preferred Indian city, not relocating (×{m_raw:.2f})")
    elif relocate:
        m_raw = cfg.LOC_INTL_RELOCATE
        notes.append(f"international, willing to relocate (×{m_raw:.2f})")
    else:
        m_raw = cfg.LOC_INTL_NO_RELOCATE
        notes.append(f"international, not relocating (×{m_raw:.2f})")

    # Scale by loc_importance (same formula as avail fix)
    importance = float(jd_profile.loc_importance)
    m_loc = m_raw * importance + (1.0 - importance)
    return max(0.20, min(1.0, m_loc)), notes


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Notice Period (m_notice)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_m_notice(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    """
    P-7: Notice period multiplier.
    Uses jd_profile.notice_period_days_stated when the JD explicitly states
    a number ("sub-30-day", "notice period of 45 days", etc. — extracted by
    jd_processor._extract_notice_period_days) to build the preferred-tier
    cutoff around the JD's own stated value instead of always using the
    fixed 30/60/90 fallback tiers. Falls back to the fixed tiers otherwise.
    """
    notes: list[str] = []
    if not jd_profile.flags.get("prefers_short_notice"):
        return 1.0, []

    notice = int(snap.get("notice_period_days") or 90)

    stated = jd_profile.notice_period_days_stated
    if stated is not None:
        # Build tiers around the JD's own number: at-or-below it is ideal,
        # 2x and 3x it step down, same shape as the fixed fallback tiers.
        tiers = [(stated, 1.00), (stated * 2, 0.90), (stated * 3, 0.80)]
    else:
        tiers = cfg.NOTICE_TIERS

    m_notice = cfg.NOTICE_FLOOR_MULT
    for max_d, mult in tiers:
        if notice <= max_d:
            m_notice = mult; break

    threshold = stated if stated is not None else 30
    if jd_profile.urgency > 0.60 and notice > threshold:
        m_notice = max(0.50, m_notice * (1.0 - (jd_profile.urgency - 0.60) * 0.30))
        notes.append(f"notice {notice}d in urgent JD (stated pref={stated or 'n/a'}d, ×{m_notice:.2f})")
    elif notice > threshold:
        notes.append(f"notice {notice}d (stated pref={stated or 'n/a'}d, ×{m_notice:.2f})")
    return float(m_notice), notes


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — Profile Integrity (m_integrity) — universal
# ─────────────────────────────────────────────────────────────────────────────

def _compute_m_integrity(snap: dict) -> tuple[float, list[str]]:
    notes: list[str] = []; deductions = 0.0

    # P-8: Salary inversion
    s_min, s_max = snap.get("expected_salary_min"), snap.get("expected_salary_max")
    if s_min is not None and s_max is not None:
        try:
            if float(s_min) > float(s_max):
                deductions += cfg.SALARY_INVERSION_PENALTY; notes.append("salary inversion")
        except (TypeError, ValueError): pass

    # P-9: Trust anchors
    anchors = sum([bool(snap.get("verified_email")), bool(snap.get("verified_phone")),
                   bool(snap.get("linkedin_connected"))])
    if anchors == 0:
        deductions += cfg.TRUST_ZERO_ANCHOR_PENALTY; notes.append("no trust anchors")
    elif anchors == 1:
        deductions += cfg.TRUST_ONE_ANCHOR_PENALTY; notes.append("1 trust anchor")

    # P-10: Profile completeness
    comp = float(snap.get("profile_completeness_score") or 0)
    if comp < cfg.COMPLETENESS_LOW_THRESHOLD:
        deductions += cfg.COMPLETENESS_LOW_PENALTY; notes.append(f"completeness {comp:.0f}%")
    elif comp < cfg.COMPLETENESS_MED_THRESHOLD:
        deductions += cfg.COMPLETENESS_MED_PENALTY

    # P-11: Hollow expert claims
    assessments = snap.get("skill_assessment_scores") or {}
    n_hollow = sum(
        1 for s in snap.get("skills", [])
        if (s.get("proficiency") or "").lower() == "expert"
        and int(s.get("endorsements") or 0) == 0
        and (s.get("name") or "") not in assessments
    )
    if n_hollow:
        deductions += min(n_hollow * cfg.HOLLOW_EXPERT_PER_SKILL, cfg.HOLLOW_EXPERT_DEDUCTION_CAP)
        notes.append(f"{n_hollow} hollow expert claim(s)")

    # P-21: Duplicate/near-duplicate career narratives
    dup_deduction, dup_note = _check_duplicate_career_narratives(snap)
    if dup_deduction > 0:
        deductions += dup_deduction
        notes.append(dup_note)

    return float(np.clip(1.0 - deductions, cfg.INTEGRITY_FLOOR, 1.0)), notes

def _check_duplicate_career_narratives(snap: dict) -> tuple[float, str]:
    """
    P-21: Detects word-for-word or near-identical description text reused
    across two or more DIFFERENT career_history entries — e.g. the exact
    same project narrative describing work at two different companies in
    different eras. Universal, JD-independent (no JD would ever want to
    tolerate this), so it lives in integrity scoring alongside the other
    always-on profile-quality checks, not in the JD-conditional layer.

    Uses difflib.SequenceMatcher (stdlib, no new dependency) rather than
    exact-string equality, so a trivially cosmetic edit (a swapped word,
    "50 million" vs "50M") doesn't let an otherwise-duplicated narrative
    slip past a pure equality check.
    """
    career = snap.get("career_history", [])
    descriptions = [
        (i, (job.get("description") or "").strip()) for i, job in enumerate(career)
    ]
    # Only compare substantive narratives — short one-liners are generically
    # similar across many real candidates and would false-positive here.
    descriptions = [(i, d) for i, d in descriptions if len(d) >= cfg.DUPLICATE_NARRATIVE_MIN_LENGTH]
    if len(descriptions) < 2:
        return 0.0, ""

    n_duplicate_pairs = 0
    example_pairs: list[str] = []

    for a in range(len(descriptions)):
        for b in range(a + 1, len(descriptions)):
            idx_a, text_a = descriptions[a]
            idx_b, text_b = descriptions[b]
            norm_a = " ".join(text_a.lower().split())
            norm_b = " ".join(text_b.lower().split())
            similarity = 1.0 if norm_a == norm_b else difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

            if similarity >= cfg.DUPLICATE_NARRATIVE_SIMILARITY_THRESHOLD:
                n_duplicate_pairs += 1
                co_a = career[idx_a].get("company", "?")
                co_b = career[idx_b].get("company", "?")
                if len(example_pairs) < 2:
                    example_pairs.append(f"{co_a}↔{co_b}")

    if n_duplicate_pairs == 0:
        return 0.0, ""

    deduction = min(
        n_duplicate_pairs * cfg.DUPLICATE_NARRATIVE_PENALTY_PER_PAIR,
        cfg.DUPLICATE_NARRATIVE_PENALTY_CAP,
    )
    note = f"{n_duplicate_pairs} duplicate role narrative(s) ({', '.join(example_pairs)}) — possible synthetic profile"
    return deduction, note




# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — JD-Fit Modifier (m_jd) — with new YoE, framework-only, non-coder
# ─────────────────────────────────────────────────────────────────────────────

def _compute_m_jd(snap: dict, jd_profile: JDProfile) -> tuple[float, list[str]]:
    """
    Layer 5 — JD-Fit Modifier. NOTE: the old flat-fraction "partial
    consulting" penalty (former P-13) is REMOVED from here — it's fully
    superseded by _compute_consulting_penalty_multiplier (recency-weighted,
    content-aware), applied as a direct multiplier in the main
    penalized_score formula in apply_penalties() rather than folded into
    this additive delta. Job-hopping, offer-churn, and GitHub bonus are
    unchanged; they were already soft/tiered, not binary, so they didn't
    need the same fix.
    """
    notes: list[str] = []; delta = 0.0
    career = snap.get("career_history", [])

    # P-14: Job-hopping
    completed = [j.get("duration_months") or 0 for j in career
                 if not j.get("is_current") and j.get("duration_months")]
    if completed:
        avg = sum(completed) / len(completed)
        if avg < cfg.JOB_HOP_VERY_SHORT_MONTHS:
            delta -= cfg.JOB_HOP_HARD_PENALTY; notes.append(f"avg tenure {avg:.0f}m (very short)")
        elif avg < cfg.JOB_HOP_SHORT_MONTHS:
            delta -= cfg.JOB_HOP_SOFT_PENALTY; notes.append(f"avg tenure {avg:.0f}m (short)")

    # P-15: Offer churn
    offer_rate = float(snap.get("offer_acceptance_rate") or -1)
    if offer_rate != -1.0 and offer_rate < cfg.OFFER_ACCEPTANCE_POOR_THRESHOLD:
        delta -= cfg.OFFER_ACCEPTANCE_PENALTY; notes.append(f"low offer acceptance {offer_rate:.0%}")

    # P-16: GitHub bonus (only if JD values open source)
    if jd_profile.flags.get("prefers_open_source"):
        gh = float(snap.get("github_activity_score") or -1)
        if gh >= cfg.GITHUB_HIGH_THRESHOLD:
            delta += cfg.GITHUB_HIGH_BONUS; notes.append(f"strong GitHub ({gh:.0f})")
        elif gh >= cfg.GITHUB_MED_THRESHOLD:
            delta += cfg.GITHUB_MED_BONUS

    # NEW P-17: LangChain-only framework tourist penalty
    fw_deduction, fw_note = _check_framework_only_ai(snap, jd_profile)
    if fw_deduction > 0:
        delta -= fw_deduction; notes.append(fw_note)

    # NEW P-18: 18-month non-coding senior
    nc_deduction, nc_note = _check_non_coding_senior(snap, jd_profile)
    if nc_deduction > 0:
        delta -= nc_deduction; notes.append(nc_note)

    # NEW P-19: CV/Speech without NLP/IR
    cs_deduction, cs_note = _check_cv_speech_no_nlp(snap, jd_profile)
    if cs_deduction > 0:
        delta -= cs_deduction; notes.append(cs_note)

    # NEW P-20: YoE range soft penalty (extracted from JD dynamically)
    yoe_min, yoe_max = jd_profile.yoe_min, jd_profile.yoe_max
    if yoe_min is not None or yoe_max is not None:
        cand_yoe = float(snap.get("years_of_experience") or 0)
        if cand_yoe > 0:
            if yoe_min is not None and cand_yoe < yoe_min - cfg.YOE_TOLERANCE:
                delta -= cfg.YOE_BELOW_RANGE_DEDUCTION
                notes.append(f"YoE {cand_yoe:.0f}y below JD range {yoe_min}–{yoe_max}")
            elif yoe_max is not None and cand_yoe > yoe_max + cfg.YOE_TOLERANCE:
                delta -= cfg.YOE_ABOVE_RANGE_DEDUCTION
                notes.append(f"YoE {cand_yoe:.0f}y above JD range {yoe_min}–{yoe_max}")

    return float(np.clip(1.0 + delta, cfg.JD_FIT_MIN, cfg.JD_FIT_MAX)), notes


# ─────────────────────────────────────────────────────────────────────────────
# Keyword matching for reasoning
# ─────────────────────────────────────────────────────────────────────────────

def _find_matched_keywords(snap: dict, jd_keywords: list[str]) -> list[str]:
    blob = " ".join([
        snap.get("summary", ""),
        " ".join(s.get("name", "") for s in snap.get("skills", [])),
        " ".join(j.get("description", "") for j in snap.get("career_history", [])),
    ]).lower()
    return [kw for kw in jd_keywords if kw in blob][:20]


# ─────────────────────────────────────────────────────────────────────────────
# Robust semantic-score normalization (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _robust_minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Percentile-clipped min-max normalization. Stretches `scores` to use the
    full [0,1] range based on robust low/high anchors (2nd/98th percentile
    by default — see config.SEMANTIC_NORM_LOW_PERCENTILE/HIGH_PERCENTILE)
    rather than the absolute min/max, since a single outlier candidate
    would otherwise compress everyone else's normalized spread — a known
    fragility of pure min-max that production ranking systems typically
    guard against via percentile clipping (winsorization).

    Degenerate case (near-identical scores across the whole pool, e.g. a
    tiny candidate pool or an unusual query): returns a flat 0.5 for
    everyone rather than amplifying floating-point noise into an arbitrary,
    meaningless spread.
    """
    if len(scores) == 0:
        return scores
    lo = np.percentile(scores, cfg.SEMANTIC_NORM_LOW_PERCENTILE)
    hi = np.percentile(scores, cfg.SEMANTIC_NORM_HIGH_PERCENTILE)
    if hi - lo < cfg.SEMANTIC_NORM_MIN_SPREAD:
        return np.full_like(scores, 0.5, dtype=np.float32)
    normalized = (scores - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Apply all penalties → top-200
# ─────────────────────────────────────────────────────────────────────────────

def apply_penalties(
    retrieval:   RetrievalResult,
    store:       ArtifactStore,
    jd_profile:  JDProfile,
    jd_keywords: list[str],
    top_k:       int = cfg.TOP_K_PENALTY,
) -> list[CandidateScore]:
    """
    Two-pass design:
      Pass 1 (vectorized): compute raw role/cap/combined dense scores for
        every retrieved candidate, then robust-min-max normalize each
        across the pool BEFORE any behavioral multiplier touches them —
        see _robust_minmax_normalize. This is what keeps tightly-clustered
        raw BGE cosine similarities from being swamped by behavioral
        multipliers downstream.
      Pass 2 (per-candidate): compute domain mismatch, keyword-stuffer gate
        (now evaluated on normalized scores), the continuous recency-
        weighted consulting/research multipliers, and Layers 1-5, then
        combine into penalized_score using the NORMALIZED dense score.
    """
    t0 = time.perf_counter()
    n = len(retrieval.indices)

    # ── PASS 1: vectorized raw + normalized semantic scores ──────────────────
    role_raw = retrieval.role_scores.astype(np.float64)
    cap_raw  = retrieval.cap_scores.astype(np.float64)
    combined_raw = jd_profile.role_weight * role_raw + jd_profile.cap_weight * cap_raw

    role_norm     = _robust_minmax_normalize(role_raw)
    cap_norm      = _robust_minmax_normalize(cap_raw)
    combined_norm = _robust_minmax_normalize(combined_raw)

    # ── PASS 2: per-candidate penalties on top of the normalized baseline ────
    scored: list[CandidateScore] = []
    for i, (arr_idx, cid) in enumerate(zip(retrieval.indices, retrieval.candidate_ids)):
        arr_idx = int(arr_idx)
        snap    = store.snapshot(cid)
        behav   = float(store.behaviors[arr_idx])

        cs = CandidateScore(
            candidate_id=cid, array_idx=arr_idx,
            rrf_score=float(retrieval.rrf_scores[i]),
            role_score=float(role_raw[i]), cap_score=float(cap_raw[i]),
            role_score_norm=float(role_norm[i]), cap_score_norm=float(cap_norm[i]),
            combined_dense_raw=float(combined_raw[i]), combined_dense_norm=float(combined_norm[i]),
            bm25_role=float(retrieval.bm25_role_scores[i]),
            bm25_cap=float(retrieval.bm25_cap_scores[i]),
            behavioral=behav,
        )

        # Layer 0: extension point — currently a no-op for this JD (see module docstring)
        is_dq, dq_reason = apply_hard_disqualifiers(snap, jd_profile)
        if is_dq:
            cs.is_disqualified = True; cs.dq_reason = dq_reason
            cs.penalized_score = cfg.SCORE_DISQUALIFIED
            scored.append(cs); continue

        # Semantic domain mismatch — absolute threshold by design, NOT pool-
        # normalised (see config.py comment): this is a per-candidate check
        # against a fixed semantic reference point, not a pool-relative
        # ranking signal, so normalising it would let a JD where every
        # retrieved candidate is off-domain hide that fact.
        cand_role_vec = store.role_vecs[arr_idx]
        dom_sim       = compute_domain_similarity(cand_role_vec, jd_profile)
        cs.domain_sim = dom_sim
        domain_mult   = domain_mismatch_multiplier(dom_sim, jd_profile.domain_mismatch_threshold)
        if domain_mult < 1.0:
            cs.penalty_notes.append(
                f"domain mismatch sim={dom_sim:.2f}<{jd_profile.domain_mismatch_threshold:.2f} (×{domain_mult:.2f})"
            )

        # Keyword stuffer gate — now on the NORMALISED per-track scores
        # (see config.py: thresholds recalibrated for the normalised scale)
        stuffer_mult = 1.0
        if cs.cap_score_norm >= cfg.CAP_SCORE_FAKER_THRESHOLD and cs.role_score_norm < cfg.ROLE_SCORE_FAKER_THRESHOLD:
            stuffer_mult = cfg.FAKER_PENALTY_MULTIPLIER
            cs.penalty_notes.append("keyword stuffer (×0.10, normalised scale)")

        # Continuous recency-weighted consulting/research multipliers
        # (replaces the old hard Layer-0 binary DQ — see module docstring)
        consulting_mult, consulting_note = _compute_consulting_penalty_multiplier(snap, jd_profile)
        cs.consulting_mult = consulting_mult
        if consulting_note:
            cs.penalty_notes.append(consulting_note)

        research_mult, research_note = _compute_research_penalty_multiplier(snap, jd_profile)
        cs.research_mult = research_mult
        if research_note:
            cs.penalty_notes.append(research_note)

        # Layers 1–5 (all read from jd_profile dynamically)
        m_avail, notes = _compute_m_avail(snap, jd_profile)
        cs.m_avail = m_avail; cs.penalty_notes.extend(notes)

        m_loc, notes = _compute_m_loc(snap, jd_profile)
        cs.m_loc = m_loc; cs.penalty_notes.extend(notes)

        m_notice, notes = _compute_m_notice(snap, jd_profile)
        cs.m_notice = m_notice; cs.penalty_notes.extend(notes)

        m_integrity, notes = _compute_m_integrity(snap)
        cs.m_integrity = m_integrity; cs.penalty_notes.extend(notes)

        m_jd, notes = _compute_m_jd(snap, jd_profile)
        cs.m_jd = m_jd; cs.penalty_notes.extend(notes)

        # NOTE: `behav` (the precomputed offline behavioral tensor) is stored on
        # cs.behavioral for transparency/debugging but is intentionally NOT
        # multiplied into penalized_score here. It covers the same signals
        # (recency, response rate, notice period, tenure-consistency) that
        # m_avail (Layer 1) and m_notice (Layer 3) already capture with
        # finer-grained tiering AND JD-conditional importance scaling.
        # Multiplying both in compounds the same signal twice — verified to
        # inflict a ~40% extra unintended penalty on inactive candidates
        # beyond the documented P-1..P-20 formula. m_avail/m_notice are the
        # single source of truth for availability/notice scoring.
        cs.penalized_score = (
            cs.combined_dense_norm     # <-- pool-normalised, not raw cosine
            * domain_mult
            * stuffer_mult
            * consulting_mult
            * research_mult
            * cs.m_avail
            * cs.m_loc
            * cs.m_notice
            * cs.m_integrity
            * cs.m_jd
        )
        cs.matched_keywords = _find_matched_keywords(snap, jd_keywords)
        scored.append(cs)

    # ── Selection: O(N) argpartition, not O(N log N) full sort ───────────────
    # Matches the same pattern used at every other split in the pipeline
    # (retriever.py's dense/BM25/RRF stages). At N≈1000 the wall-clock gap to
    # a Python sort is sub-millisecond, but the pattern is kept consistent
    # deliberately — it's the same algorithmic discipline this architecture
    # applies everywhere else, and it's what actually matters at the N this
    # would scale to in a larger production candidate pool.
    n_dq = sum(1 for s in scored if s.is_disqualified)
    non_dq_scored = [s for s in scored if not s.is_disqualified]

    if len(non_dq_scored) <= top_k:
        # Fewer survivors than requested — keep all, single small sort.
        # cs.behavioral (precomputed offline tensor) breaks ties when
        # penalized_score is equal — see compute_behavioral_score() in
        # offline_pipeline.py. It is NOT multiplied into penalized_score:
        # doing so double-counts the same recency/response/notice signals
        # that m_avail (Layer 1) and m_notice (Layer 3) already score more
        # precisely and JD-conditionally — verified to inflate the penalty
        # on inactive candidates by ~40% beyond the documented P-1..P-20
        # formula when both were applied. The offline tensor still earns its
        # keep as a free, JD-independent tie-breaker instead.
        non_dq = sorted(non_dq_scored, key=lambda x: (x.penalized_score, x.behavioral, x.candidate_id), reverse=True)
    else:
        scores_arr = np.array([s.penalized_score for s in non_dq_scored], dtype=np.float64)
        top_pos = np.argpartition(scores_arr, -top_k)[-top_k:]          # O(N) partition
        top_pos_sorted = sorted(                                        # O(k log k) on the slice only
            top_pos,
            key=lambda i: (scores_arr[i], non_dq_scored[i].behavioral, non_dq_scored[i].candidate_id),
            reverse=True,
        )
        non_dq = [non_dq_scored[i] for i in top_pos_sorted]

    log.info("Penalties %.3fs — %d total, %d DQ, %d → cross-encoder",
             time.perf_counter() - t0, len(scored), n_dq, len(non_dq))
    return non_dq


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Cross-encoder rerank
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is not None: return
        from model_engine import ONNXCrossEncoder
        path = str(cfg.BGE_RERANKER_MODEL_DIR) if cfg.BGE_RERANKER_MODEL_DIR.exists() else cfg.BGE_RERANKER_MODEL_ID
        log.info("Loading cross-encoder: %s", path)
        self._model = ONNXCrossEncoder(cfg.BGE_RERANKER_MODEL_DIR, fallback_model_id=cfg.BGE_RERANKER_MODEL_ID)
        log.info("Cross-encoder backend: %s", "ONNX INT8" if self._model.using_onnx else "PyTorch (fallback)")

    def rerank(
        self, scored: list[CandidateScore], store: ArtifactStore,
        raw_jd_text: str, batch_size: int = 32,
    ) -> list[CandidateScore]:
        """
        Cross-encoder reranks ALL candidates passed in (typically 200) and
        returns the FULL reordered list — not sliced to a top-k. The caller
        (rank.py) applies the honeypot safety net first, then takes the
        final top-100 from the survivors. Slicing here would discard the
        backfill candidates the safety net needs if anything gets cut.
        """
        self._load()
        t0 = time.perf_counter()
        pairs = [(raw_jd_text[:1500], _build_ce_text(store.snapshot(cs.candidate_id))) for cs in scored]
        raw = np.array(self._model.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        norm = (raw - lo) / (hi - lo + 1e-9)
        for cs, n in zip(scored, norm):
            # Floor at 0.0: combined_dense can be marginally negative for
            # very dissimilar text pairs even with real embeddings (e.g. a
            # Marketing Manager profile against an ML Engineer JD). A negative
            # number in the submission CSV reads as a bug to a human reviewer;
            # clamping preserves relative ranking order while keeping the
            # displayed score interpretable.
            cs.ce_score = max(0.0, 0.70 * float(n) + 0.30 * cs.penalized_score)

        # Deliberately a full sort here, not argpartition. At N=200 the two
        # approaches are both sub-millisecond — there is no measurable
        # performance reason to partition first. More importantly, the
        # honeypot safety net (see apply_honeypot_safety_net below) needs a
        # fully-ordered list beyond rank 100 to backfill from if anything
        # gets cut, so slicing early here would throw away exactly the
        # information that step needs. argpartition is the right tool when
        # you only need an unordered top-k and N is large (used at every
        # 2000/1000/200 split above); it stops being the right tool once you
        # need a total order over a small N, which is the case here.
        #
        # Explicit tie-breaker chain (v4): Python's sort() is stable, so
        # without an explicit secondary key, two candidates with an
        # identical ce_score (rare with floats, but not impossible —
        # clipping at the 0.0 floor or coincidental rounding can produce
        # genuine ties) would keep whatever arbitrary order they happened
        # to have after the Stage-2 argpartition, which is not a meaningful
        # tie-break and isn't guaranteed stable across runs/environments.
        #   1. ce_score          — primary rank
        #   2. penalized_score   — pre-cross-encoder score, more decimal
        #                          precision, reflects the full penalty stack
        #   3. behavioral        — precomputed offline tensor, a free
        #                          JD-independent signal (see offline_pipeline.py)
        #   4. candidate_id      — final deterministic fallback; guarantees
        #                          the sort is fully reproducible run-to-run,
        #                          which matters directly for Stage 3
        #                          (the evaluator re-runs this exact code and
        #                          compares output — any non-determinism in
        #                          tie order is a reproducibility risk)
        scored.sort(
            key=lambda x: (x.ce_score, x.penalized_score, x.behavioral, x.candidate_id),
            reverse=True,
        )
        log.info("Cross-encoder: %d pairs in %.1fs, fully ranked", len(scored), time.perf_counter() - t0)
        return scored   # NOTE: returns all reranked candidates, not sliced to top_k —
                         # caller applies the honeypot safety net before the final cut


def _build_ce_text(snap: dict) -> str:
    parts: list[str] = []
    title = snap.get("current_title", ""); company = snap.get("current_company", "")
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3.5 — Honeypot Safety Net (the final hard enforcement point)
# ═══════════════════════════════════════════════════════════════════════════
# Architecture spec: "Honeypot tags applied as a hard score floor at this
# stage (if a flagged candidate is still in top 100, demote/cut it — the
# actual enforcement point, matching the spec's '10% in top 100' rule)."
#
# Phase A's honeypot.filter_honeypots() already removed ~91,731 of 100,000
# candidates before anything was embedded. This second check is deliberately
# redundant — defense in depth, not duplicated effort:
#   • It re-runs the FULL 14-trap honeypot.is_honeypot() check, not a subset.
#   • It only touches the ~200 candidates that survived to the cross-encoder
#     stage, so the cost is negligible (sub-second) regardless of pool size.
#   • It protects against honeypot.py being updated with new traps AFTER
#     offline_pipeline.py already ran and produced a now-stale clean pool —
#     exactly the scenario where Phase A's filter alone provides no guarantee.
#   • If anything is cut, it backfills from the next-best already-scored
#     candidates (rank 101+) rather than shrinking the submission below 100
#     rows, which would itself fail format validation (Stage 1 of the
#     evaluation pipeline).

def _snapshot_to_honeypot_dict(candidate_id: str, snap: dict) -> dict:
    """
    Reconstruct a honeypot.is_honeypot()-compatible candidate dict from the
    stored snapshot. The snapshot intentionally carries every field each of
    the 14 traps reads (see offline_pipeline.py's snapshot builder) so this
    re-check is a full, not partial, re-run of honeypot.py's logic.
    """
    return {
        "candidate_id": candidate_id,
        "profile": {
            "years_of_experience": snap.get("years_of_experience", 0),
        },
        "career_history": snap.get("career_history", []),
        "education": snap.get("education", []),
        "skills": snap.get("skills", []),
        "redrob_signals": {
            "signup_date":                snap.get("signup_date", ""),
            "last_active_date":           snap.get("last_active_date", ""),
            "profile_completeness_score": snap.get("profile_completeness_score", 0),
            "recruiter_response_rate":    snap.get("recruiter_response_rate", 0.5),
            "interview_completion_rate":  snap.get("interview_completion_rate", 1.0),
            "offer_acceptance_rate":      snap.get("offer_acceptance_rate", -1),
            "github_activity_score":      snap.get("github_activity_score", -1),
            "notice_period_days":         snap.get("notice_period_days", 90),
            "avg_response_time_hours":    snap.get("avg_response_time_hours", 0),
            "profile_views_received_30d": snap.get("profile_views_received_30d", 0),
            "applications_submitted_30d": snap.get("applications_submitted_30d", 0),
            "connection_count":           snap.get("connection_count", 0),
            "endorsements_received":      snap.get("endorsements_received", 0),
            "search_appearance_30d":      snap.get("search_appearance_30d", 0),
            "saved_by_recruiters_30d":    snap.get("saved_by_recruiters_30d", 0),
        },
    }


def apply_honeypot_safety_net(
    ranked: list[CandidateScore],
    store: ArtifactStore,
    final_k: int = cfg.TOP_K_FINAL,
) -> list[CandidateScore]:
    """
    The actual Stage-3 enforcement point for the spec's honeypot rule.

    Takes the FULL cross-encoder-reranked list (not pre-sliced — see
    CrossEncoderReranker.rerank()), re-checks each candidate against all 14
    honeypot traps in order, cuts any that fire, and backfills from the
    next-best survivors so the final list still has exactly `final_k` rows.

    Returns the final, honeypot-clean top-`final_k` candidates in rank order.
    """
    t0 = time.perf_counter()
    clean: list[CandidateScore] = []
    cut: list[tuple[str, list[str]]] = []

    for cs in ranked:
        if len(clean) >= final_k:
            break   # already have enough clean candidates, no need to check further
        snap = store.snapshot(cs.candidate_id)
        hp_dict = _snapshot_to_honeypot_dict(cs.candidate_id, snap)
        result = honeypot.is_honeypot(hp_dict)
        if result["is_honeypot"]:
            cut.append((cs.candidate_id, result["triggered"]))
        else:
            clean.append(cs)

    elapsed = time.perf_counter() - t0
    if cut:
        log.warning(
            "Honeypot safety net: %d candidate(s) cut from final ranking "
            "(backfilled from rank %d+): %s",
            len(cut), final_k + 1,
            [f"{cid} ({', '.join(traps[:1])})" for cid, traps in cut[:5]],
        )
    log.info(
        "Honeypot safety net %.3fs — checked %d, cut %d, final clean count %d/%d",
        elapsed, len(clean) + len(cut), len(cut), len(clean), final_k,
    )

    if len(clean) < final_k:
        log.warning(
            "Honeypot safety net could not backfill to %d candidates "
            "(only %d available after cuts) — the reranked pool may need "
            "to be larger than %d to absorb safety-net cuts.",
            final_k, len(clean), len(ranked),
        )

    return clean