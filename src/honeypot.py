"""
honeypot.py  ·  v2
===================
Redrob Intelligent Candidate Discovery & Ranking Challenge
----------------------------------------------------------
Detects honeypot / invalid candidates via profile-integrity checks.

Each check is independent and returns a structured (fired, reasons) tuple.
is_honeypot() aggregates all checks and returns a final verdict.

TRAP REGISTRY
-------------
TRAP 1  — Fictional Companies
TRAP 2  — Company Founding Year Mismatch           (expanded company list)
TRAP 3  — Degree Field Establishment Ceilings      (strict field matching)
TRAP 4  — Impossible Degree Durations              (relaxed thresholds)
TRAP 5  — Tech Skill Invention Ceiling             (skill-name normalisation)
TRAP 6  — Instant Expert Anomaly
TRAP 7  — Master's Completed Before Bachelor's
TRAP 8  — Career Before Graduation                 (buffer: 20 months)
TRAP 9  — Date Inversions                          (+ education sub-check)
TRAP 10 — Overlapping Employment Periods           (tolerance: 30 days)
TRAP 11 — YoE Exceeds Time Since Earliest Graduation
TRAP 12 — Duration-Months vs Date-Range Inconsistency  [NEW]
TRAP 13 — Multiple Concurrent Current Jobs             [NEW]
TRAP 14 — Redrob Signal Range Violations               [NEW]

v1 → v2 changes
----------------
TRAP 1  : Pre-normalise FICTIONAL_COMPANIES set so the suffix-stripping
          normaliser produces consistent lookups ("ACME CORP" → "ACME"
          both in the set and for input — fixed the v1 bug where the set
          held "ACME CORP" but normalised input was "ACME").
TRAP 2  : Added 70+ major Indian startups to the founding-year dict.
TRAP 3  : Replaced substring match with startswith() — "Computer Science
          and Artificial Intelligence" (a CS programme) is no longer
          misflagged because it starts with "COMPUTER SCIENCE".
TRAP 4  : Split into four tiers with relaxed thresholds:
            M.TECH / M.E        ≥ 6 y  (was ≥ 4 y; allows integrated 5-y)
            M.SC / M.S / MBA    ≥ 5 y  (was ≥ 4 y)
            B.TECH / B.E        ≥ 7 y  (was ≥ 5 y; allows dual-degree 5-y)
            B.SC / BACHELOR     ≥ 6 y  (was ≥ 5 y)
TRAP 5  : Skill-name normalisation ("Llama Index" → "LLAMAINDEX", etc.)
          and alias table for common spacing/hyphenation variants.
TRAP 8  : Pre-graduation buffer extended 12 → 20 months (campus placements
          in India routinely close 14–18 months before graduation).
TRAP 9  : Added sub-check C — education date inversions (end < start) and
          impossible future graduation dates.
TRAP 10 : Overlap tolerance reduced 60 → 30 days; added early-exit
          optimisation in the inner loop.
TRAP 12 : [NEW] Flags career_history entries where the stated
          duration_months differs from the computed date-range by > 3 months.
TRAP 13 : [NEW] Flags candidates with is_current=True at > 1 distinct
          companies simultaneously.
TRAP 14 : [NEW] Flags numeric redrob_signals fields that fall outside their
          schema-defined valid range.

Reference date: 2026-06-29  (dataset snapshot — hardcoded intentionally).

Usage
-----
    from honeypot import is_honeypot, filter_honeypots

    with open("candidates.jsonl") as f:
        for line in f:
            candidate = json.loads(line)
            result = is_honeypot(candidate)
            if result["is_honeypot"]:
                print(candidate["candidate_id"], result["reasons"])
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dataset snapshot date — used as "today" for all ceiling computations.
# DO NOT replace with datetime.date.today(); the traps must stay stable.
REFERENCE_DATE = date(2026, 6, 29)


# ===========================================================================
# TRAP 1 — Fictional Companies
# ===========================================================================
# Well-known fictional organisations from TV shows and films.
# Any candidate claiming employment at one of these is a honeypot.
#
# Sources: The IT Crowd (Reynholm / Initech), Silicon Valley (Pied Piper /
# Hooli), Batman (Wayne Enterprises), The Simpsons (Globex), The Office
# (Dunder Mifflin), Iron Man (Stark Industries), ACME (Looney Tunes).
#
# v2 FIX — ACME CORP mismatch bug:
#   In v1, _normalise_company("Acme Corp") → "ACME" (strips "CORP") but the
#   set contained "ACME CORP", so the lookup always missed.  Fix: run every
#   set value through the same normaliser at import time so both sides of the
#   comparison are consistently processed.

_RAW_FICTIONAL_COMPANIES: set[str] = {
    "INITECH",
    "PIED PIPER",
    "WAYNE ENTERPRISES",
    "ACME CORP",
    "ACME CORPORATION",
    "STARK INDUSTRIES",
    "HOOLI",
    "GLOBEX",
    "GLOBEX INC",
    "GLOBEX CORPORATION",
    "DUNDER MIFFLIN",
}


def _normalise_company(name: str) -> str:
    """
    Normalise a company name for consistent set/dict lookups:
      1. Uppercase
      2. Strip non-alphanumeric chars (replace with space)
      3. Collapse whitespace
      4. Remove trailing generic corporate suffixes
         (INC, LLC, LTD, PVT, PRIVATE, LIMITED, CO, CORP, CORPORATION)
    """
    name = name.upper()
    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(
        r"\b(INC|LLC|LTD|PVT|PRIVATE|LIMITED|CO|CORP|CORPORATION)\b",
        "",
        name,
    )
    return re.sub(r"\s+", " ", name).strip()


# Pre-normalise so _normalise_company(candidate_input) can be compared
# directly.  Both sides go through the same transformation.
FICTIONAL_COMPANIES_NORMALISED: set[str] = {
    _normalise_company(c) for c in _RAW_FICTIONAL_COMPANIES
}


def check_fictional_companies(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 1 — Fictional Companies
    Flags any career_history entry whose company normalises to a known
    fictional organisation.
    """
    reasons: list[str] = []
    for job in candidate.get("career_history", []):
        normalised = _normalise_company(job.get("company", ""))
        if normalised in FICTIONAL_COMPANIES_NORMALISED:
            reasons.append(
                f"[FICTIONAL_COMPANY] '{job['company']}' is a fictional organisation "
                f"(role: {job.get('title', 'N/A')}, "
                f"period: {job.get('start_date')} → {job.get('end_date') or 'present'})"
            )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 2 — Company Founding Year Mismatch
# ===========================================================================
# Real Indian startups with documented founding years.
# A career entry whose start_date predates the company's founding is
# chronologically impossible.  Tolerance: 30 days (1 month).
#
# v2: Expanded from 21 to 90+ companies.
# Keys are already uppercase with no special chars so _normalise_company
# leaves them unchanged; candidate input is normalised before lookup.

COMPANY_FOUNDING_YEARS: dict[str, int] = {
    # ---- original entries ----
    "CRED":                         2018,
    "GLANCE":                       2019,
    "REPHRASE AI":                  2019,
    "REPHRASE":                     2019,
    "SARVAM AI":                    2023,
    "SARVAM":                       2023,
    "KRUTRIM":                      2023,
    "ZEPTO":                        2021,
    "MEESHO":                       2015,
    "RAZORPAY":                     2014,
    "SLICE":                        2016,
    "GROWW":                        2016,
    "SMALLCASE":                    2015,
    "OPEN":                         2017,
    "JUPITER":                      2019,
    "NIYO":                         2015,
    "ACKO":                         2016,
    "DIGIT":                        2016,
    "PLUM":                         2019,
    "KHATABOOK":                    2018,
    "OKCREDIT":                     2019,

    # ---- v2 additions ----
    # E-commerce / quick-commerce
    "FLIPKART":                     2007,
    "NYKAA":                        2012,
    "BIGBASKET":                    2011,
    "GROFERS":                      2013,
    "BLINKIT":                      2013,   # Grofers rebranded 2022
    "ZEPTO TECHNOLOGIES":           2021,

    # Food delivery / hyperlocal
    "ZOMATO":                       2010,
    "SWIGGY":                       2014,
    "DUNZO":                        2015,

    # Mobility
    "OLA":                          2010,
    "OLA CABS":                     2010,
    "RAPIDO":                       2015,

    # Payments / fintech
    "PAYTM":                        2010,
    "ONE97 COMMUNICATIONS":         2000,
    "PHONEPE":                      2015,
    "BHARATPE":                     2018,
    "MSWIPE":                       2011,
    "CASHFREE":                     2015,
    "CASHFREE PAYMENTS":            2015,
    "JUSPAY":                       2012,
    "SETU":                         2018,
    "DECENTRO":                     2020,
    "M2P FINTECH":                  2014,
    "LENDINGKART":                  2014,
    "INDIFI":                       2015,
    "PROGCAP":                      2018,
    "YUBI":                         2018,
    "CREDAVENUE":                   2018,

    # Insurtech
    "POLICYBAZAAR":                 2008,
    "COVERFOX":                     2013,
    "INSURANCE DEKHO":              2016,
    "DITTO INSURANCE":              2021,
    "DITTO":                        2021,

    # EdTech
    "BYJUS":                        2011,
    "BYJU":                         2011,
    "THINK AND LEARN":              2011,   # BYJU'S parent
    "UNACADEMY":                    2015,
    "VEDANTU":                      2014,
    "UPGRAD":                       2015,
    "CLASSPLUS":                    2018,
    "EXTRAMARKS":                   2007,

    # Health tech
    "PRACTO":                       2008,
    "PHARMEASY":                    2014,
    "HEALTHIFYME":                  2012,
    "CUREFIT":                      2016,
    "CULT FIT":                     2016,
    "MFINE":                        2017,
    "PRISTYN CARE":                 2018,

    # B2B / SaaS / Dev-tools
    "ZOHO":                         1996,
    "FRESHWORKS":                   2010,
    "CHARGEBEE":                    2011,
    "POSTMAN":                      2014,
    "BROWSERSTACK":                 2011,
    "HASURA":                       2017,
    "DARWINBOX":                    2015,
    "KEKA":                         2015,
    "GREYTHR":                      2012,
    "SPRINKLR":                     2009,
    "HYPERVERGE":                   2015,
    "SIGNZY":                       2015,
    "OBSERVE AI":                   2017,

    # Communication / CX
    "EXOTEL":                       2011,
    "GUPSHUP":                      2004,
    "HAPTIK":                       2013,
    "KALEYRA":                      2008,

    # B2B commerce / logistics
    "OFBUSINESS":                   2015,
    "MOGLIX":                       2015,
    "UDAAN":                        2016,
    "ZETWERK":                      2018,

    # Consumer brands
    "MAMAEARTH":                    2016,
    "LENSKART":                     2010,

    # Real estate / housing
    "NOBROKER":                     2014,
    "NESTAWAY":                     2015,
    "STANZA LIVING":                2017,

    # Travel / hospitality
    "OYO":                          2013,
    "OYO ROOMS":                    2013,
    "ZOSTEL":                       2013,
    "IXIGO":                        2006,
    "GOIBIBO":                      2009,
    "YATRA":                        2006,

    # Mobility / EV
    "ATHER ENERGY":                 2013,
    "OLA ELECTRIC":                 2017,
    "SIMPLE ENERGY":                2019,
    "PURE EV":                      2015,

    # Social / content
    "SHARECHAT":                    2015,
    "MOJ":                          2020,
    "DAILYHUNT":                    2009,
    "JOSH":                         2020,

    # Deep tech / space
    "AGNIKUL COSMOS":               2017,
    "PIXXEL":                       2019,
    "BELLATRIX AEROSPACE":          2015,
    "SKYROOT AEROSPACE":            2018,
}


def check_company_founding_year(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 2 — Company Founding Year Mismatch
    Flags career_history entries where start_date is more than 30 days
    before the company's documented founding year.
    """
    reasons: list[str] = []
    for job in candidate.get("career_history", []):
        normalised = _normalise_company(job.get("company", ""))
        if normalised not in COMPANY_FOUNDING_YEARS:
            continue
        founding_year = COMPANY_FOUNDING_YEARS[normalised]
        try:
            start = date.fromisoformat(job["start_date"])
        except (KeyError, ValueError):
            continue
        # Allow up to 30 days before Jan 1 of the founding year
        founding_date = date(founding_year, 1, 1)
        if (founding_date - start).days > 30:
            reasons.append(
                f"[FOUNDING_YEAR_MISMATCH] Career at '{job['company']}' starts "
                f"{job['start_date']} but company was founded in {founding_year}"
            )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 3 — Degree Field Establishment Ceilings
# ===========================================================================
# Formal UG/PG programmes named after AI / Data Science / ML did not exist
# before ~2018–2019 in India (AICTE approval timelines).
#
# v2 CHANGE — STRICT startswith() FIELD MATCHING:
#   v1 used `field_keyword in field` (substring).  This misflagged:
#     "Computer Science and Artificial Intelligence"  ← legitimate pre-2019 CS degree
#     "Engineering with AI Specialization"            ← legitimate pre-2019 Engg degree
#
#   v2 uses field.startswith(keyword), so only programmes whose names
#   BEGIN with the AI/ML/DS keyword are caught:
#     FLAGGED   : "Artificial Intelligence"
#                 "Artificial Intelligence and Machine Learning"
#                 "Data Science and Analytics"
#     NOT FLAGGED: "Computer Science and Artificial Intelligence"
#                  "Electronics and AI"
#                  "CSE (AI Specialization)"

ESTABLISHMENT_CEILINGS: dict[str, dict[str, int]] = {
    "B.TECH": {
        "ARTIFICIAL INTELLIGENCE": 2019,
        "DATA SCIENCE":            2019,
        "MACHINE LEARNING":        2019,
    },
    "B.SC": {
        "ARTIFICIAL INTELLIGENCE": 2019,
        "DATA SCIENCE":            2018,
        "MACHINE LEARNING":        2019,
    },
    "B.E": {
        "ARTIFICIAL INTELLIGENCE": 2019,
        "DATA SCIENCE":            2019,
        "MACHINE LEARNING":        2019,
    },
    "M.TECH": {
        "ARTIFICIAL INTELLIGENCE": 2019,
        "DATA SCIENCE":            2019,
        "MACHINE LEARNING":        2019,
    },
    "M.SC": {
        "ARTIFICIAL INTELLIGENCE": 2020,
        "DATA SCIENCE":            2018,
        "MACHINE LEARNING":        2020,
    },
    "M.E": {
        "ARTIFICIAL INTELLIGENCE": 2019,
        "DATA SCIENCE":            2019,
        "MACHINE LEARNING":        2019,
    },
    "PH.D": {
        "ARTIFICIAL INTELLIGENCE": 2015,
        "DATA SCIENCE":            2016,
        "MACHINE LEARNING":        2015,
    },
}

# Minimum programme duration in years — used in end_year consistency check.
MIN_DEGREE_DURATION: dict[str, int] = {
    "B.TECH": 4, "B.SC": 3, "B.E": 4,
    "M.TECH": 2, "M.SC": 2, "M.E": 2,
    "PH.D":   3,
}


def _degree_prefix(degree: str) -> str | None:
    """
    Return the canonical ESTABLISHMENT_CEILINGS key for a degree string,
    or None if not recognised.

    v2 FIX: v1's first loop had a broken regex-inside-startswith call
    that never matched anything.  v2 uses a single clean approach:
    strip dots from both key and degree, then do prefix matching.
    """
    deg_clean = re.sub(r"[^A-Z.]", "", degree.upper().strip().rstrip("."))
    deg_nodot = deg_clean.replace(".", "")
    for key in ESTABLISHMENT_CEILINGS:
        if deg_nodot.startswith(key.replace(".", "")):
            return key
    return None


def _field_starts_with_keyword(field_upper: str, keyword: str) -> bool:
    """
    v2 strict field matching helper.
    Returns True only if `field_upper` starts with `keyword`.

    Caller is responsible for uppercasing `field_upper` before passing in.
    """
    return field_upper.startswith(keyword)


def check_degree_field_ceilings(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 3 — Degree Field Establishment Ceilings
    Flags education entries where:
      (a) field starts with a keyword AND start_year < establishment ceiling
      (b) end_year < start_year + minimum programme duration
    """
    reasons: list[str] = []
    for edu in candidate.get("education", []):
        degree = edu.get("degree", "")
        field  = edu.get("field_of_study", "").upper().strip()
        start  = edu.get("start_year")
        end    = edu.get("end_year")

        deg_key = _degree_prefix(degree)
        if deg_key is None:
            continue

        # (a) Enrolment before programme establishment — strict startswith
        for field_keyword, ceiling_year in ESTABLISHMENT_CEILINGS.get(deg_key, {}).items():
            if _field_starts_with_keyword(field, field_keyword):
                if isinstance(start, int) and start < ceiling_year:
                    reasons.append(
                        f"[DEGREE_FIELD_CEILING] {degree} in "
                        f"'{edu['field_of_study']}' enrolled {start} but "
                        f"programme established ~{ceiling_year}"
                    )

        # (b) End-year consistency (impossible fast completion)
        min_dur = MIN_DEGREE_DURATION.get(deg_key)
        if isinstance(start, int) and isinstance(end, int) and min_dur is not None:
            if end < start + min_dur:
                reasons.append(
                    f"[DEGREE_DURATION_TOO_SHORT] {degree} in "
                    f"'{edu.get('field_of_study', '')}' completed in "
                    f"{end - start}y (min required: {min_dur}y)"
                )

    return bool(reasons), reasons


# ===========================================================================
# TRAP 4 — Impossible Degree Durations (excessive length)
# ===========================================================================
# v2 CHANGE — Relaxed and split into four tiers:
#
#   M.TECH / M.E       : flag if ≥ 6 y  (was ≥ 4 y)
#     Rationale: B.Tech+M.Tech integrated programmes = 5 y; BITS WILP
#     part-time M.Tech runs 3–4 y.  Only flag truly impossible lengths.
#
#   M.SC / M.S / MBA   : flag if ≥ 5 y  (was ≥ 4 y)
#     Rationale: Strict 2-y programmes; ≥ 5 y has no legitimate path.
#
#   B.TECH / B.E       : flag if ≥ 7 y  (was ≥ 5 y)
#     Rationale: Integrated dual degrees = 5 y; lateral-entry + back
#     papers can legitimately reach 6 y.  Flag only ≥ 7 y.
#
#   B.SC / BACHELOR    : flag if ≥ 6 y  (was ≥ 5 y)
#     Rationale: B.Sc is 3 y; ≥ 6 y is impossible regardless of delays.

DEGREE_MAX_DURATIONS: list[tuple[list[str], int]] = [
    # (degree_prefixes, flag_if_duration_years >=)
    (["M.TECH", "M.E"],         6),
    (["M.SC", "M.S", "MBA"],    5),
    (["B.TECH", "B.E"],         7),
    (["B.SC", "BACHELOR"],      6),
]

PHDS_MIN_YEARS = 3   # Ph.D in < 3 years is suspect


def check_impossible_degree_durations(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 4 — Impossible Degree Durations
    Flags education entries with durations that exceed realistic limits, or
    Ph.D entries that completed suspiciously quickly (< 3 y).
    """
    reasons: list[str] = []
    for edu in candidate.get("education", []):
        degree = edu.get("degree", "").upper().strip()
        field  = edu.get("field_of_study", "")
        start  = edu.get("start_year")
        end    = edu.get("end_year")

        if not isinstance(start, int) or not isinstance(end, int):
            continue
        duration = end - start

        # Overly long master's or undergrad
        for prefixes, threshold in DEGREE_MAX_DURATIONS:
            if any(degree.startswith(p) for p in prefixes):
                if duration >= threshold:
                    reasons.append(
                        f"[IMPOSSIBLE_DEGREE_DURATION] {degree} in '{field}' "
                        f"ran for {duration}y (flag threshold: ≥{threshold}y)"
                    )
                break   # apply only the first matching tier

        # Ph.D completed suspiciously fast
        if degree.startswith("PH.D") or "PHD" in degree.replace(".", ""):
            if 0 < duration < PHDS_MIN_YEARS:
                reasons.append(
                    f"[PHD_TOO_SHORT] {degree} in '{field}' completed in "
                    f"{duration}y (minimum expected: {PHDS_MIN_YEARS}y)"
                )

    return bool(reasons), reasons


# ===========================================================================
# TRAP 5 — Tech Skill Invention Ceiling
# ===========================================================================
# A skill cannot have been used for more months than the technology has
# existed as of REFERENCE_DATE (2026-06-29).
#
# v2 CHANGE — Skill-name normalisation pass:
#   "Llama Index" → "LLAMAINDEX"
#   "Hugging Face Transformers" → "HUGGING FACE TRANSFORMERS"
#   "Sentence Transformer" (singular) → "SENTENCE TRANSFORMERS"
#   etc.
#   This prevents fabricated profiles from dodging the ceiling by using
#   a slightly different spacing or hyphenation.

TECH_INVENTION_CEILINGS: dict[str, int] = {
    # PARAMETER-EFFICIENT FINE-TUNING
    "QLORA":                       37,    # Dettmers et al., May 2023
    "LORA":                        60,    # Microsoft paper, Jun 2021
    "PEFT":                        40,    # HuggingFace PEFT library, early 2023
    "FINE-TUNING":                108,    # widespread post-Transformers era
    "FINE TUNING":                108,
    "FINETUNING":                 108,

    # LLM FRAMEWORKS & ORCHESTRATORS
    "LANGCHAIN":                   44,    # open-sourced Oct 2022
    "LLAMAINDEX":                  43,    # Nov 2022 as gpt_index
    "HAYSTACK":                    72,    # deepset Haystack ~2020
    "PROMPT ENGINEERING":          66,    # rose with GPT-3 access (2020)

    # EMBEDDING ARCHITECTURES
    "BGE":                         34,    # BAAI General Embeddings, Aug 2023
    "E5":                          34,    # Microsoft E5 family, mid-2023
    "OPENAI EMBEDDINGS":           54,    # text-embedding-ada-002, Dec 2020
    "SENTENCE TRANSFORMERS":       79,    # SBERT paper, Nov 2019
    "SENTENCE-TRANSFORMERS":       79,

    # VECTOR DATABASES & SERVING
    "VLLM":                        36,    # PagedAttention engine, Jun 2023
    "PINECONE":                    65,    # commercial launch, early 2021
    "QDRANT":                      65,    # early 2021
    "WEAVIATE":                    65,    # early 2021
    "MILVUS":                      79,    # open-source, late 2019
    "FAISS":                       96,    # Meta open-sourced, early 2017
    "PGVECTOR":                    60,    # PostgreSQL extension, mid-2021

    # CORE LLM / RAG TERMS
    "RAG":                         66,    # Lewis et al. paper, mid-2020
    "LLM":                         72,    # phrase scaled post-GPT-3 (2020)
    "LARGE LANGUAGE MODEL":        72,

    # DEEP LEARNING FOUNDATIONS
    "TRANSFORMER":                108,    # Vaswani et al., Jun 2017
    "TRANSFORMERS":               108,
    "PYTORCH":                    116,    # released late 2016
    "HUGGINGFACE":                 96,    # traction ~late 2018
    "HUGGING FACE":                96,
    "HUGGING FACE TRANSFORMERS":   96,
    "HUGGINGFACE TRANSFORMERS":    96,
}

# Alias table: maps variant skill names → canonical key in TECH_INVENTION_CEILINGS.
_SKILL_ALIASES: dict[str, str] = {
    "LLAMA INDEX":                 "LLAMAINDEX",
    "LLAMA-INDEX":                 "LLAMAINDEX",
    "LLAMA INDEX DATA FRAMEWORK":  "LLAMAINDEX",
    "SENTENCE TRANSFORMER":        "SENTENCE TRANSFORMERS",    # singular
    "SENTENCE-TRANSFORMER":        "SENTENCE TRANSFORMERS",
    "HUGGING FACE TRANSFORMER":    "HUGGING FACE TRANSFORMERS",
    "HUGGINGFACE TRANSFORMER":     "HUGGINGFACE TRANSFORMERS",
    "OPENAI EMBEDDING":            "OPENAI EMBEDDINGS",        # singular
    "GPT FINE TUNING":             "FINE TUNING",
    "GPT FINETUNING":              "FINETUNING",
    "LLM FINE TUNING":             "FINE TUNING",
    "LLM FINETUNING":              "FINETUNING",
    "PROMPT ENGINEER":             "PROMPT ENGINEERING",
    "LARGE LANGUAGE MODELS":       "LARGE LANGUAGE MODEL",
}


def _normalise_skill(name: str) -> str:
    """
    Normalise a skill name for TECH_INVENTION_CEILINGS lookup:
      1. Uppercase and strip surrounding whitespace
      2. Collapse internal whitespace
      3. Resolve known aliases
    """
    name = re.sub(r"\s+", " ", name.upper().strip())
    return _SKILL_ALIASES.get(name, name)


def check_tech_invention_ceiling(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 5 — Tech Skill Invention Ceiling
    Flags any skill whose duration_months exceeds the months the technology
    has existed as of REFERENCE_DATE (2026-06-29).
    """
    reasons: list[str] = []
    for skill in candidate.get("skills", []):
        name     = skill.get("name", "")
        duration = skill.get("duration_months")
        if not isinstance(duration, (int, float)):
            continue
        normalised = _normalise_skill(name)
        ceiling    = TECH_INVENTION_CEILINGS.get(normalised)
        if ceiling is None:
            continue
        if duration > ceiling:
            reasons.append(
                f"[TECH_CEILING_EXCEEDED] Skill '{name}' claims {duration}m of "
                f"use but technology is at most {ceiling} months old "
                f"(as of {REFERENCE_DATE})"
            )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 6 — Instant Expert Anomaly
# ===========================================================================
# A skill listed as 'advanced' or 'expert' with duration_months == 0 is
# logically impossible: you cannot be expert in something with zero use.

EXPERT_PROFICIENCIES = {"advanced", "expert"}


def check_instant_expert(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 6 — Instant Expert Anomaly
    Flags skills where proficiency is 'advanced' or 'expert' but
    duration_months is 0.
    """
    reasons: list[str] = []
    for skill in candidate.get("skills", []):
        proficiency = skill.get("proficiency", "").lower()
        duration    = skill.get("duration_months")
        if proficiency in EXPERT_PROFICIENCIES and duration == 0:
            reasons.append(
                f"[INSTANT_EXPERT] Skill '{skill.get('name')}' is listed as "
                f"'{proficiency}' with 0 months of use"
            )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 7 — Master's Completed Before Bachelor's
# ===========================================================================
# A candidate cannot have finished a master's degree before finishing their
# undergraduate degree.  Comparison: master's end_year < bachelor's end_year.

MASTER_PREFIXES   = {"M.SC", "M.S", "MBA", "M.TECH", "M.E", "MASTER", "MTECH"}
BACHELOR_PREFIXES = {"B.TECH", "B.SC", "BACHELOR", "B.E", "B.S", "BE", "BTECH"}


def _is_master(degree: str) -> bool:
    d = degree.upper().strip().replace(".", "")
    return any(d.startswith(p.replace(".", "")) for p in MASTER_PREFIXES)


def _is_bachelor(degree: str) -> bool:
    d = degree.upper().strip().replace(".", "")
    return any(d.startswith(p.replace(".", "")) for p in BACHELOR_PREFIXES)


def check_master_before_bachelor(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 7 — Master's Completed Before Bachelor's
    Flags candidates whose master's end_year is strictly less than their
    bachelor's end_year.
    """
    reasons: list[str] = []
    education = candidate.get("education", [])
    masters   = [e for e in education if _is_master(e.get("degree", ""))]
    bachelors = [e for e in education if _is_bachelor(e.get("degree", ""))]

    for m in masters:
        for b in bachelors:
            m_end = m.get("end_year")
            b_end = b.get("end_year")
            if isinstance(m_end, int) and isinstance(b_end, int) and m_end < b_end:
                reasons.append(
                    f"[MASTERS_BEFORE_BACHELORS] {m['degree']} in "
                    f"'{m.get('field_of_study', '')}' ended {m_end} but "
                    f"{b['degree']} in '{b.get('field_of_study', '')}' "
                    f"ended {b_end} — master's completed before bachelor's"
                )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 8 — Corporate Career Started Before Graduation
# ===========================================================================
# v2 CHANGE: buffer extended from 12 → 20 months.
#
# Rationale: Indian campus placement seasons start in October/November of
# the third year (for a June fourth-year graduation), meaning legitimate
# offer letters can pre-date graduation by 14–18 months.  A 12-month
# buffer incorrectly flags these genuine candidates.  20 months retains
# a realistic ceiling while only catching truly impossible pre-graduation
# full-career histories.

GRADUATION_BUFFER_MONTHS = 20


def check_career_before_graduation(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 8 — Career Started More Than 20 Months Before Graduation
    Flags career_history entries whose start_date is more than
    GRADUATION_BUFFER_MONTHS before the candidate's earliest recorded
    graduation year.
    """
    reasons: list[str] = []
    education = candidate.get("education", [])
    if not education:
        return False, []

    end_years = [
        e.get("end_year") for e in education
        if isinstance(e.get("end_year"), int)
    ]
    if not end_years:
        return False, []

    earliest_grad      = min(end_years)
    earliest_grad_date = date(earliest_grad, 1, 1)   # conservative: Jan 1

    for job in candidate.get("career_history", []):
        try:
            start = date.fromisoformat(job["start_date"])
        except (KeyError, ValueError):
            continue
        months_before_grad = (
            (earliest_grad_date.year - start.year) * 12
            + (earliest_grad_date.month - start.month)
        )
        if months_before_grad > GRADUATION_BUFFER_MONTHS:
            reasons.append(
                f"[CAREER_BEFORE_GRADUATION] Role '{job.get('title', 'N/A')}' at "
                f"'{job.get('company', 'N/A')}' started {job['start_date']} — "
                f"more than {GRADUATION_BUFFER_MONTHS} months before earliest "
                f"graduation ({earliest_grad})"
            )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 9 — Date Inversions
# ===========================================================================
# v2 CHANGE: Added sub-check C — education date inversions.
#
# Sub-check A : career end_date < start_date
# Sub-check B : redrob_signals signup_date > last_active_date, or
#               last_active_date is after REFERENCE_DATE (future)
# Sub-check C : education end_year < start_year  OR
#               education end_year is impossibly far in the future  [NEW]

def check_date_inversions(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 9 — Date Inversions
    Sub-check A: career history end_date < start_date
    Sub-check B: platform signal date impossibilities
    Sub-check C: education year inversions / future graduation
    """
    reasons: list[str] = []

    # --- Sub-check A: career history ---
    for job in candidate.get("career_history", []):
        end_raw = job.get("end_date")
        if end_raw is None:
            continue   # current role
        try:
            start = date.fromisoformat(job["start_date"])
            end   = date.fromisoformat(end_raw)
        except (TypeError, ValueError, KeyError):
            continue
        if end < start:
            reasons.append(
                f"[DATE_INVERSION_CAREER] '{job.get('company', 'N/A')}' — "
                f"end_date {end_raw} is before start_date {job.get('start_date')}"
            )

    # --- Sub-check B: platform signal dates ---
    signals         = candidate.get("redrob_signals", {})
    signup_raw      = signals.get("signup_date")
    last_active_raw = signals.get("last_active_date")

    if signup_raw and last_active_raw:
        try:
            signup      = date.fromisoformat(signup_raw)
            last_active = date.fromisoformat(last_active_raw)
            if signup > last_active:
                reasons.append(
                    f"[DATE_INVERSION_SIGNALS] signup_date {signup_raw} is after "
                    f"last_active_date {last_active_raw} — impossible platform state"
                )
            if last_active > REFERENCE_DATE:
                reasons.append(
                    f"[FUTURE_LAST_ACTIVE] last_active_date {last_active_raw} is "
                    f"after dataset reference date {REFERENCE_DATE} — impossible"
                )
        except (TypeError, ValueError):
            pass

    # --- Sub-check C: education date inversions [NEW in v2] ---
    for edu in candidate.get("education", []):
        start_y = edu.get("start_year")
        end_y   = edu.get("end_year")
        if not isinstance(start_y, int) or not isinstance(end_y, int):
            continue

        if end_y < start_y:
            reasons.append(
                f"[DATE_INVERSION_EDUCATION] {edu.get('degree', 'N/A')} at "
                f"'{edu.get('institution', 'N/A')}' — end_year {end_y} "
                f"is before start_year {start_y}"
            )

        # end_year more than 1 year after the dataset snapshot = impossible
        # (+1 allows final-year students whose expected graduation is 2027)
        if end_y > REFERENCE_DATE.year + 1:
            reasons.append(
                f"[FUTURE_GRADUATION] {edu.get('degree', 'N/A')} at "
                f"'{edu.get('institution', 'N/A')}' — end_year {end_y} "
                f"is beyond dataset reference date {REFERENCE_DATE.year}"
            )

    return bool(reasons), reasons


# ===========================================================================
# TRAP 10 — Overlapping Employment Periods
# ===========================================================================
# Two jobs at different companies cannot have genuine date overlaps of more
# than OVERLAP_TOLERANCE_DAYS.
#
# v2 CHANGE: Tolerance reduced 60 → 30 days.
# Also added early-exit optimisation: since parsed is sorted by start date,
# once s2 >= e1 we know no further j-values can overlap with job i.

OVERLAP_TOLERANCE_DAYS = 30   # 1 month (reduced from 60 in v1)


def check_employment_overlap(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 10 — Overlapping Employment Periods
    Flags pairs of different-company jobs whose date ranges overlap by
    more than OVERLAP_TOLERANCE_DAYS (30 days).
    """
    reasons: list[str] = []
    jobs = candidate.get("career_history", [])

    # Parse dates; treat current roles as ending on REFERENCE_DATE
    parsed: list[tuple[date, date, dict]] = []
    for job in jobs:
        try:
            s = date.fromisoformat(job["start_date"])
            e = (date.fromisoformat(job["end_date"])
                 if job.get("end_date") else REFERENCE_DATE)
            parsed.append((s, e, job))
        except (KeyError, ValueError):
            continue

    parsed.sort(key=lambda x: x[0])

    for i in range(len(parsed)):
        s1, e1, job1 = parsed[i]
        norm1 = _normalise_company(job1.get("company", ""))

        for j in range(i + 1, len(parsed)):
            s2, e2, job2 = parsed[j]

            # Early exit: list is sorted by start_date; once s2 >= e1,
            # no further j can produce an overlap with job i.
            if s2 >= e1:
                break

            # Skip same-company pairs (internal transfers / parallel roles)
            norm2 = _normalise_company(job2.get("company", ""))
            if norm1 == norm2:
                continue

            # s2 < e1 is guaranteed here — compute overlap
            overlap_days = (e1 - s2).days
            if overlap_days > OVERLAP_TOLERANCE_DAYS:
                reasons.append(
                    f"[EMPLOYMENT_OVERLAP] '{job1.get('company', 'N/A')}' "
                    f"({job1['start_date']} → {job1.get('end_date', 'present')}) "
                    f"and '{job2.get('company', 'N/A')}' "
                    f"({job2['start_date']} → {job2.get('end_date', 'present')}) "
                    f"overlap by {overlap_days} days "
                    f"(tolerance: {OVERLAP_TOLERANCE_DAYS}d)"
                )

    return bool(reasons), reasons


# ===========================================================================
# TRAP 11 — Experience Exceeds Time Since Earliest Graduation (1.5× rule)
# ===========================================================================
# A candidate's stated years_of_experience cannot plausibly exceed 1.5× the
# time elapsed since their earliest graduation.

YOE_CEILING_MULTIPLIER = 1.5


def check_experience_exceeds_time_since_grad(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 11 — Years of Experience > 1.5× Time Since Earliest Graduation
    """
    reasons: list[str] = []
    education = candidate.get("education", [])
    if not education:
        return False, []

    end_years = [
        e.get("end_year") for e in education
        if isinstance(e.get("end_year"), int)
    ]
    if not end_years:
        return False, []

    earliest_grad    = min(end_years)
    # Mid-year (Jun 1) graduation is the generous assumption
    years_since_grad = (REFERENCE_DATE - date(earliest_grad, 6, 1)).days / 365.25

    stated_yoe = candidate.get("profile", {}).get("years_of_experience")
    if not isinstance(stated_yoe, (int, float)):
        return False, []

    max_plausible_yoe = years_since_grad * YOE_CEILING_MULTIPLIER
    if stated_yoe > max_plausible_yoe:
        reasons.append(
            f"[YOE_EXCEEDS_GRADUATION] Stated YoE={stated_yoe:.1f}y but only "
            f"{years_since_grad:.1f}y since earliest graduation ({earliest_grad}); "
            f"max plausible @ {YOE_CEILING_MULTIPLIER}× = {max_plausible_yoe:.1f}y"
        )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 12 — Duration-Months vs Date-Range Inconsistency  [NEW in v2]
# ===========================================================================
# Each career_history entry carries both a stated duration_months field and
# explicit start_date / end_date fields.  A discrepancy of more than
# DURATION_TOLERANCE_MONTHS between the stated and computed duration is a
# fabrication signal.
#
# Tolerance of 3 months accounts for:
#   • Rounding to the nearest whole month at data entry
#   • Part-month start / end dates

DURATION_TOLERANCE_MONTHS = 3


def check_duration_consistency(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 12 — Duration-Months vs Date-Range Inconsistency
    Computes the actual duration from start_date / end_date and compares
    against stated duration_months.  Flags discrepancies > 3 months.

    NOTE: Only applied to completed roles (is_current=False, end_date set).
    Current roles have a duration_months value recorded at data-entry time
    that will naturally diverge from the live date range — flagging them
    produces false positives on all legitimate active employees.
    """
    reasons: list[str] = []
    for job in candidate.get("career_history", []):
        # Skip current roles — duration_months is a stale snapshot for these
        if job.get("is_current") is True:
            continue
        end_raw = job.get("end_date")
        if not end_raw:
            continue   # no end_date means effectively current — skip

        claimed = job.get("duration_months")
        if not isinstance(claimed, int):
            continue
        start_raw = job.get("start_date")
        try:
            start = date.fromisoformat(start_raw)
            end   = date.fromisoformat(end_raw)
        except (TypeError, ValueError):
            continue

        actual_months = (end.year - start.year) * 12 + (end.month - start.month)
        discrepancy   = abs(claimed - actual_months)
        if discrepancy > DURATION_TOLERANCE_MONTHS:
            reasons.append(
                f"[DURATION_MISMATCH] '{job.get('company', 'N/A')}' claims "
                f"duration_months={claimed} but date range "
                f"{start_raw} → {end_raw} "
                f"implies ~{actual_months} months "
                f"(discrepancy: {discrepancy} months)"
            )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 13 — Multiple Concurrent Current Jobs  [NEW in v2]
# ===========================================================================
# A candidate cannot simultaneously hold full-time positions at two or more
# different companies.  Multiple career_history entries with is_current=True
# at distinct (normalised) employers is an impossible profile state.

def check_multiple_current_jobs(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 13 — Multiple Concurrent Current Jobs
    Flags candidates with is_current=True at more than one distinct company.
    Same-company entries are collapsed (parallel roles at one employer are
    legitimate, e.g. full-time role + internal advisory position).
    """
    reasons: list[str] = []
    current_jobs = [
        job for job in candidate.get("career_history", [])
        if job.get("is_current") is True
    ]
    if len(current_jobs) <= 1:
        return False, []

    unique_companies = {
        _normalise_company(j.get("company", "")) for j in current_jobs
    }
    if len(unique_companies) > 1:
        companies = [j.get("company", "N/A") for j in current_jobs]
        reasons.append(
            f"[MULTIPLE_CURRENT_JOBS] {len(current_jobs)} jobs marked "
            f"is_current=True at {len(unique_companies)} distinct companies: "
            f"{companies}"
        )
    return bool(reasons), reasons


# ===========================================================================
# TRAP 14 — Redrob Signal Range Violations  [NEW in v2]
# ===========================================================================
# All numeric redrob_signals fields have schema-defined valid ranges.
# Values outside those ranges cannot exist in a real platform and indicate
# fabricated data.
#
# Schema ranges (from candidate_schema.json):
#   profile_completeness_score : 0 – 100
#   recruiter_response_rate    : 0.0 – 1.0
#   interview_completion_rate  : 0.0 – 1.0
#   offer_acceptance_rate      : -1.0 – 1.0  (-1 = no prior offers)
#   github_activity_score      : -1 – 100    (-1 = no GitHub linked)
#   notice_period_days         : 0 – 180
#   avg_response_time_hours    : ≥ 0
#   profile_views_received_30d : ≥ 0
#   applications_submitted_30d : ≥ 0
#   connection_count           : ≥ 0
#   endorsements_received      : ≥ 0
#   search_appearance_30d      : ≥ 0
#   saved_by_recruiters_30d    : ≥ 0

_SIGNAL_RANGES: list[tuple[str, float, float]] = [
    # (field_name, min_inclusive, max_inclusive)
    ("profile_completeness_score",   0.0,    100.0),
    ("recruiter_response_rate",      0.0,      1.0),
    ("interview_completion_rate",    0.0,      1.0),
    ("offer_acceptance_rate",       -1.0,      1.0),
    ("github_activity_score",       -1.0,    100.0),
    ("notice_period_days",           0.0,    180.0),
    # Lower-bounded at 0; upper bound set generously (effectively ≥ 0 only)
    ("avg_response_time_hours",      0.0, 100_000.0),
    ("profile_views_received_30d",   0.0, 100_000.0),
    ("applications_submitted_30d",   0.0, 100_000.0),
    ("connection_count",             0.0, 100_000.0),
    ("endorsements_received",        0.0, 100_000.0),
    ("search_appearance_30d",        0.0, 100_000.0),
    ("saved_by_recruiters_30d",      0.0, 100_000.0),
]


def check_signal_range_violations(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    TRAP 14 — Redrob Signal Range Violations
    Flags numeric redrob_signals fields whose values fall outside their
    schema-defined valid range, or are the wrong type entirely.
    """
    reasons: list[str] = []
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return False, []

    for field, lo, hi in _SIGNAL_RANGES:
        val = signals.get(field)
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            reasons.append(
                f"[SIGNAL_WRONG_TYPE] redrob_signals.{field}={val!r} "
                f"is not numeric (expected float in [{lo}, {hi}])"
            )
            continue
        if not (lo <= float(val) <= hi):
            reasons.append(
                f"[SIGNAL_OUT_OF_RANGE] redrob_signals.{field}={val} "
                f"outside valid schema range [{lo}, {hi}]"
            )

    return bool(reasons), reasons


# ===========================================================================
# Registry — ordered list of all checks
# ===========================================================================
# Order matches the TRAP numbering above.

ALL_CHECKS: list[tuple] = [
    (check_fictional_companies,                 "TRAP 1  — Fictional Companies"),
    (check_company_founding_year,               "TRAP 2  — Company Founding Year Mismatch"),
    (check_degree_field_ceilings,               "TRAP 3  — Degree Field Establishment Ceilings"),
    (check_impossible_degree_durations,         "TRAP 4  — Impossible Degree Durations"),
    (check_tech_invention_ceiling,              "TRAP 5  — Tech Skill Invention Ceiling"),
    (check_instant_expert,                      "TRAP 6  — Instant Expert Anomaly"),
    (check_master_before_bachelor,              "TRAP 7  — Master's Before Bachelor's"),
    (check_career_before_graduation,            "TRAP 8  — Career Before Graduation"),
    (check_date_inversions,                     "TRAP 9  — Date Inversions"),
    (check_employment_overlap,                  "TRAP 10 — Overlapping Employment"),
    (check_experience_exceeds_time_since_grad,  "TRAP 11 — YoE Exceeds Time Since Grad"),
    (check_duration_consistency,                "TRAP 12 — Duration-Months Inconsistency"),
    (check_multiple_current_jobs,               "TRAP 13 — Multiple Concurrent Current Jobs"),
    (check_signal_range_violations,             "TRAP 14 — Redrob Signal Range Violations"),
]


# ===========================================================================
# Public API
# ===========================================================================

def is_honeypot(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Run all honeypot checks against a single candidate profile.

    Parameters
    ----------
    candidate : dict
        A parsed candidate object matching the Redrob candidate schema.

    Returns
    -------
    dict with keys:
        candidate_id  : str   — the candidate's ID
        is_honeypot   : bool  — True if any check fired
        triggered     : list[str]  — trap names that fired
        reasons       : list[str]  — detailed reason strings
        trap_count    : int   — number of distinct traps triggered
    """
    candidate_id    = candidate.get("candidate_id", "UNKNOWN")
    triggered_traps: list[str] = []
    all_reasons:     list[str] = []

    for check_fn, trap_name in ALL_CHECKS:
        fired, reasons = check_fn(candidate)
        if fired:
            triggered_traps.append(trap_name)
            all_reasons.extend(reasons)

    return {
        "candidate_id": candidate_id,
        "is_honeypot":  bool(triggered_traps),
        "triggered":    triggered_traps,
        "reasons":      all_reasons,
        "trap_count":   len(triggered_traps),
    }


def filter_honeypots(
    candidates: list[dict[str, Any]],
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Partition a list of candidates into clean and honeypot buckets.

    Parameters
    ----------
    candidates : list of parsed candidate dicts
    verbose    : if True, print a summary line for each honeypot detected

    Returns
    -------
    (clean_candidates, honeypot_results)
        clean_candidates : list of candidate dicts that passed all checks
        honeypot_results : list of is_honeypot() result dicts for flagged ones
    """
    clean:   list[dict] = []
    flagged: list[dict] = []

    for candidate in candidates:
        result = is_honeypot(candidate)
        if result["is_honeypot"]:
            flagged.append(result)
            if verbose:
                print(
                    f"[HONEYPOT] {result['candidate_id']} — "
                    f"{result['trap_count']} trap(s): "
                    f"{', '.join(result['triggered'])}"
                )
        else:
            clean.append(candidate)

    return clean, flagged


# ===========================================================================
# CLI — run directly against a candidates.jsonl file
# ===========================================================================

if __name__ == "__main__":
    import json
    import sys
    import csv
    import os
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(
        description="Redrob honeypot filter v2 — detect impossible candidate profiles"
    )
    parser.add_argument(
        "--candidates", "-c",
        default="../data/candidates.jsonl",
        help="Path to candidates JSONL file (default: ../data/candidates.jsonl)",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Optional: write honeypot candidate IDs to this file (one per line)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print a summary line for each honeypot detected",
    )
    args = parser.parse_args()

    # ---- Load candidates ----
    candidates_list: list[dict] = []
    try:
        with open(args.candidates, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidates_list.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"[WARN] Line {line_no}: JSON parse error — {exc}",
                          file=sys.stderr)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(candidates_list):,} candidates from '{args.candidates}'")

    # ---- Run filter ----
    clean_candidates, honeypot_results = filter_honeypots(
        candidates_list, verbose=args.verbose
    )

    pct = len(honeypot_results) / max(1, len(candidates_list)) * 100
    print(f"\nResults")
    print(f"  Clean candidates : {len(clean_candidates):,}")
    print(f"  Honeypots flagged: {len(honeypot_results):,}  ({pct:.2f}%)")

    # ---- Trap breakdown ----
    trap_counter: Counter = Counter()
    for result in honeypot_results:
        for trap_name in result["triggered"]:
            trap_counter[trap_name] += 1

    print("\nTrap breakdown (candidate counts):")
    print(f"{'-'*10} | {'-'*45}")
    print(f"{'Count':<10} | {'Trap Name'}")
    print(f"{'-'*10} | {'-'*45}")
    for trap_name, count in sorted(trap_counter.items(), key=lambda x: -x[1]):
        print(f"{count:<10,} | {trap_name}")
    print(f"{'-'*10} | {'-'*45}\n")

    # ---- Save outputs ----
    base_dir      = ".." if os.path.basename(os.getcwd()) == "src" else "."
    json_out_path = os.path.join(base_dir, "output", "verified_clean_profiles.json")
    csv_out_path  = os.path.join(base_dir, "output", "verified_clean_profiles.csv")

    # JSON — full clean profiles
    try:
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(clean_candidates, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(clean_candidates):,} clean records → {json_out_path}")
    except Exception as e:
        print(f"[ERROR] JSON save failed: {e}", file=sys.stderr)

    # CSV — flat summary table (richer columns than v1)
    try:
        if clean_candidates:
            with open(csv_out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "candidate_id",
                    "anonymized_name",
                    "current_title",
                    "current_company",
                    "years_of_experience",
                    "last_active_date",
                    "open_to_work_flag",
                    "notice_period_days",
                    "willing_to_relocate",
                    "preferred_work_mode",
                ])
                for cand in clean_candidates:
                    profile = cand.get("profile", {})
                    signals = cand.get("redrob_signals", {})
                    writer.writerow([
                        cand.get("candidate_id", ""),
                        profile.get("anonymized_name", ""),
                        profile.get("current_title", ""),
                        profile.get("current_company", ""),
                        profile.get("years_of_experience", ""),
                        signals.get("last_active_date", ""),
                        signals.get("open_to_work_flag", ""),
                        signals.get("notice_period_days", ""),
                        signals.get("willing_to_relocate", ""),
                        signals.get("preferred_work_mode", ""),
                    ])
            print(f"Saved spreadsheet → {csv_out_path}")
    except Exception as e:
        print(f"[ERROR] CSV save failed: {e}", file=sys.stderr)

    # Optional ID list
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                for result in honeypot_results:
                    f.write(result["candidate_id"] + "\n")
            print(f"Honeypot IDs written → {args.out}")
        except Exception as e:
            print(f"[ERROR] ID file save failed: {e}", file=sys.stderr)