"""
config.py  — Production Dynamic Brain (v3)
============================================
All hyperparameters, thresholds, and dynamic structures.

v3 CHANGES (this revision) — three architectural fixes:

  1. TEXT-NATIVE DOMAIN INDEPENDENCE (replaces winner-take-all bucket lookup)
     ROLE_CATEGORY_ANCHORS are no longer used to pick ONE winning category.
     jd_processor.py now computes a softmax-weighted BLEND across all 10
     anchors' similarity scores, and every downstream weight (role_weight,
     cap_weight, avail_importance, loc_importance, domain_mismatch_threshold)
     is a similarity-weighted average across categories, not a single
     category's fixed profile. A JD that doesn't strongly resemble any of
     the 10 anchors (e.g. "Quantum Cryptographer", "Chef") naturally
     degrades toward a near-uniform blend (close to the "default" profile)
     rather than being force-fit into the least-wrong bucket.
     Why not a separate zero-shot classifier model instead: a second model
     (even a small one) adds disk/RAM/loading-time/correctness surface for
     marginal benefit over blending similarities I'm already computing —
     under a 5GB disk / 16GB RAM / 5-min budget that's reproduced in a
     sandboxed Docker container, that tradeoff isn't worth it. The blend
     achieves the same graceful-degradation property with zero added cost.

  2. RECENCY-WEIGHTED CONSULTING/RESEARCH PENALTY (replaces hard binary DQ)
     consulting_dq_applies / research_dq_applies are RENAMED to
     consulting_penalty_active / research_penalty_active and no longer
     trigger an absolute Layer-0 score=0.0 cut. Instead, guardrails.py
     computes a continuous multiplier from (a) how RECENT the
     consulting/research-only time was — a job ending 8 years ago carries
     far less weight than a current role — and (b) whether the role's own
     description shows production/infra signal despite the company being a
     nominal "consulting firm" (some people do build real platforms while
     technically employed by a services company). A candidate whose entire
     career is CURRENTLY, RECENTLY, and PURELY consulting/research with zero
     production signal still gets crushed toward the floor — functionally
     equivalent to disqualification for the case the JD actually targets —
     but an exceptional candidate with old consulting experience and years
     of recent product-company work is never zeroed out by a blunt rule.

  3. ROBUST MIN-MAX NORMALIZATION of the combined dense (semantic) score
     BGE-small cosine similarities cluster tightly across a candidate pool
     (e.g. top matches all sitting in a narrow 0.78-0.86 band). Multiplying
     that narrow band directly by behavioral multipliers (which can range
     from ~0.05 to 1.10) lets behavioral signal dominate the ranking and
     drown out actual technical/role relevance. guardrails.py now stretches
     the combined dense score to use the full [0,1] range via percentile-
     clipped min-max normalization (2nd/98th percentile, not raw min/max —
     a single outlier candidate shouldn't compress everyone else's spread)
     BEFORE any behavioral multiplier is applied.
"""

from __future__ import annotations
import re as _re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"

CANDIDATES_PATH     = DATA_DIR / "candidates.jsonl"
ROLE_VECTORS_PATH   = ARTIFACTS_DIR / "role_vectors.npy"
CAP_VECTORS_PATH    = ARTIFACTS_DIR / "cap_vectors.npy"
ROLE_BM25_PATH      = ARTIFACTS_DIR / "role_bm25.pkl"
CAP_BM25_PATH       = ARTIFACTS_DIR / "cap_bm25.pkl"
BEHAVIORS_PATH      = ARTIFACTS_DIR / "behaviors.npy"
CANDIDATE_META_PATH = ARTIFACTS_DIR / "candidate_meta.json"

BGE_EMBED_MODEL_DIR    = MODELS_DIR / "bge-small-en-v1.5"
BGE_RERANKER_MODEL_DIR = MODELS_DIR / "bge-reranker-base"
BGE_EMBED_MODEL_ID     = "BAAI/bge-small-en-v1.5"
BGE_RERANKER_MODEL_ID  = "BAAI/bge-reranker-base"
OUTPUT_CSV_PATH        = BASE_DIR / "submission.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING & PIPELINE SIZES
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM        = 384
EMBED_BATCH_SIZE = 128
MAX_SEQ_LEN      = 512

TOP_K_DENSE   = 2_000
TOP_K_SPARSE  = 2_000
TOP_K_RRF     = 1_000
TOP_K_PENALTY = 200
TOP_K_FINAL   = 100
RRF_K         = 60

# Keyword-stuffer thresholds — operate on the NORMALISED [0,1] scale (see
# change #3 above), not raw BGE cosine similarity. Set slightly more
# conservative (wider gap required) than the old raw-scale values, because
# normalisation gives scores genuine room to spread out across the full
# range, where previously they were compressed into a narrow native band.
CAP_SCORE_FAKER_THRESHOLD  = 0.65
ROLE_SCORE_FAKER_THRESHOLD = 0.25
FAKER_PENALTY_MULTIPLIER   = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLE CATEGORY SEMANTIC ANCHORS (10 domains)
# ─────────────────────────────────────────────────────────────────────────────
# These are no longer used for winner-take-all classification. jd_processor.py
# computes cosine similarity against all 10, then blends every downstream
# weight by a softmax over those similarities — see CATEGORY_BLEND_TEMPERATURE
# below. Adding an 11th domain here automatically participates in the blend;
# no other code changes needed.
ROLE_CATEGORY_ANCHORS: dict[str, str] = {
    "ml_ai_engineering": (
        "machine learning engineer artificial intelligence embeddings vector retrieval "
        "ranking recommendation llm nlp search information retrieval production ml "
        "fine-tuning evaluation transformer bert"
    ),
    "software_engineering": (
        "software engineer developer backend frontend full stack java python javascript "
        "node react api microservices system design distributed computing rest"
    ),
    "data_science": (
        "data scientist analytics statistics modeling experimentation sql pandas tableau "
        "power bi business intelligence reporting ab testing"
    ),
    "data_engineering": (
        "data engineer pipeline etl elt spark kafka airflow snowflake dbt warehouse "
        "bigquery streaming batch ingestion orchestration"
    ),
    "product_management": (
        "product manager product owner agile scrum roadmap stakeholder requirements "
        "user story okr metrics strategy market research discovery"
    ),
    "marketing": (
        "marketing manager growth digital campaigns seo sem content strategy brand "
        "demand generation lead acquisition email performance"
    ),
    "sales": (
        "sales account executive business development revenue b2b enterprise saas "
        "crm pipeline closing quota customer acquisition deal"
    ),
    "ux_design": (
        "ux designer ui product designer user research figma sketch prototyping "
        "interaction design accessibility usability wireframe design system"
    ),
    "devops_infrastructure": (
        "devops site reliability infrastructure cloud aws gcp azure kubernetes docker "
        "terraform ci cd pipeline monitoring observability sre platform"
    ),
    "engineering_management": (
        "engineering manager tech lead people management team building hiring "
        "mentoring organizational design cross functional delivery"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CATEGORY WEIGHT PROFILES
# ─────────────────────────────────────────────────────────────────────────────
# Each profile is now a BLEND INPUT, not a selected output. jd_processor.py
# computes softmax(similarities / temperature) across all 10, then takes the
# weighted average of every numeric field below per JD. A JD sitting cleanly
# in one category (high max similarity, low entropy) ends up very close to
# that category's raw profile, same as the old behaviour. A JD that doesn't
# resemble any category well spreads weight more evenly and regresses
# toward "default" — exactly the graceful degradation a true zero-shot
# system needs, achieved here without a second model.
CATEGORY_WEIGHT_PROFILES: dict[str, dict] = {
    "ml_ai_engineering": {
        "role_weight": 0.35, "cap_weight": 0.65,
        "avail_importance": 0.85, "loc_importance": 0.75,
        "domain_mismatch_threshold": 0.28,
    },
    "software_engineering": {
        "role_weight": 0.40, "cap_weight": 0.60,
        "avail_importance": 0.82, "loc_importance": 0.75,
        "domain_mismatch_threshold": 0.25,
    },
    "data_science": {
        "role_weight": 0.38, "cap_weight": 0.62,
        "avail_importance": 0.80, "loc_importance": 0.70,
        "domain_mismatch_threshold": 0.25,
    },
    "data_engineering": {
        "role_weight": 0.38, "cap_weight": 0.62,
        "avail_importance": 0.80, "loc_importance": 0.70,
        "domain_mismatch_threshold": 0.25,
    },
    "product_management": {
        "role_weight": 0.58, "cap_weight": 0.42,
        "avail_importance": 0.88, "loc_importance": 0.78,
        "domain_mismatch_threshold": 0.22,
    },
    "marketing": {
        "role_weight": 0.62, "cap_weight": 0.38,
        "avail_importance": 0.90, "loc_importance": 0.82,
        "domain_mismatch_threshold": 0.18,
    },
    "sales": {
        "role_weight": 0.68, "cap_weight": 0.32,
        "avail_importance": 0.92, "loc_importance": 0.85,
        "domain_mismatch_threshold": 0.15,
    },
    "ux_design": {
        "role_weight": 0.50, "cap_weight": 0.50,
        "avail_importance": 0.80, "loc_importance": 0.72,
        "domain_mismatch_threshold": 0.22,
    },
    "devops_infrastructure": {
        "role_weight": 0.42, "cap_weight": 0.58,
        "avail_importance": 0.82, "loc_importance": 0.70,
        "domain_mismatch_threshold": 0.25,
    },
    "engineering_management": {
        "role_weight": 0.62, "cap_weight": 0.38,
        "avail_importance": 0.85, "loc_importance": 0.78,
        "domain_mismatch_threshold": 0.20,
    },
    # "default" is included as one of the blend candidates with a small but
    # non-zero base weight (see CATEGORY_BLEND_DEFAULT_PRIOR below) so that
    # even a JD with a clear winning category is pulled slightly toward
    # generic, balanced weights — and a JD that matches nothing pulls
    # strongly toward this profile.
    "default": {
        "role_weight": 0.45, "cap_weight": 0.55,
        "avail_importance": 0.80, "loc_importance": 0.72,
        "domain_mismatch_threshold": 0.20,
    },
}

# Softmax temperature for blending category similarities into weights.
# Lower = sharper (closer to old winner-take-all behaviour).
# Higher = flatter (more even blending, more conservative/generic weights).
# 0.12 keeps a JD that's clearly ML (sim ~0.55) close to ml_ai_engineering's
# profile while still meaningfully blending in neighbours, and flattens out
# for a JD with no strong match (sims all ~0.15-0.25) toward "default".
CATEGORY_BLEND_TEMPERATURE: float = 0.12

# Constant similarity score assigned to "default" before the softmax, acting
# as a flat prior so it always has SOME pull, growing relatively stronger
# precisely when no real category fits well (since the real categories'
# similarities stay low while default's stays fixed).
CATEGORY_BLEND_DEFAULT_PRIOR: float = 0.20

# If the single highest category similarity is below this, log a low-
# confidence notice — the JD doesn't strongly resemble any known category
# and the system is relying on the blend's graceful-degradation behaviour.
CATEGORY_LOW_CONFIDENCE_THRESHOLD: float = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 5. MICRO-ADJUSTMENT FACTORS
# ─────────────────────────────────────────────────────────────────────────────
SENIORITY_ROLE_WEIGHT_BOOST:      float = 0.08
TECHNICAL_DEPTH_CAP_WEIGHT_BOOST: float = 0.08
URGENCY_AVAIL_IMPORTANCE_BOOST:   float = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# 6. UNIVERSAL DOMAIN KNOWLEDGE (not JD-specific)
# ─────────────────────────────────────────────────────────────────────────────
CONSULTING_FIRMS: set[str] = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree", "mindtree",
    "l&t infotech", "niit technologies", "zensar", "cyient",
}

PRODUCTION_KEYWORDS: set[str] = {
    "production", "deployed", "deployment", "users", "scale", "latency",
    "serving", "api", "pipeline", "real-time", "streaming", "inference",
    "endpoint", "traffic", "throughput",
}

RESEARCH_TITLE_KEYWORDS: set[str] = {
    "researcher", "research engineer", "research scientist",
    "scientist", "faculty", "postdoc", "postdoctoral",
    "professor", "lab engineer", "research associate",
    "research intern", "visiting researcher", "research fellow",
}

FRAMEWORK_WRAPPER_SKILLS: frozenset[str] = frozenset([
    "langchain", "llamaindex", "llama index", "haystack",
    "openai api", "openai", "anthropic api", "chatgpt api",
    "gpt api", "gpt wrapper", "langsmith", "flowise",
])
FOUNDATIONAL_AI_SKILLS: frozenset[str] = frozenset([
    "pytorch", "tensorflow", "keras", "jax", "numpy", "scikit-learn",
    "sklearn", "transformers", "bert", "embedding", "embeddings",
    "retrieval", "bm25", "faiss", "milvus", "pinecone", "weaviate",
    "qdrant", "pgvector", "recommendation", "ranking", "reranking",
    "neural network", "deep learning", "machine learning", "nlp",
    "information retrieval", "xgboost", "lightgbm", "catboost",
    "fine-tuning", "lora", "qlora", "peft", "mlops", "kubeflow",
    "sentence-transformers", "huggingface", "hugging face",
    "vector database", "vector search", "dense retrieval",
    "sparse retrieval", "hybrid search", "re-ranking", "cross-encoder",
])

EXECUTIVE_NON_CODER_TITLES: frozenset[str] = frozenset([
    "vp of engineering", "vp engineering", "vice president engineering",
    "cto", "chief technology officer", "chief architect",
    "engineering director", "director of engineering",
    "head of engineering", "head of technology",
    "principal architect", "solution architect", "enterprise architect",
    "tech lead", "technical lead", "architecture lead",
])
NON_CODING_SENIOR_DURATION_THRESHOLD: int   = 18   # months
NON_CODING_SENIOR_DEDUCTION:          float = 0.15

CV_SPEECH_DOMAIN_TITLES: frozenset[str] = frozenset([
    "computer vision", "cv engineer", "image processing", "image recognition",
    "object detection", "ocr", "speech recognition", "asr", "tts",
    "text-to-speech", "speech synthesis", "robotics engineer",
    "autonomous vehicle", "self-driving", "slam", "robot",
])
NLP_IR_SIGNALS: frozenset[str] = frozenset([
    "nlp", "natural language", "information retrieval", "ranking",
    "embeddings", "retrieval", "search", "recommendation",
    "language model", "bert", "transformer", "rag", "llm",
    "semantic search", "text classification", "named entity",
])
CV_SPEECH_WITHOUT_NLP_DEDUCTION: float = 0.18

# ─────────────────────────────────────────────────────────────────────────────
# 7. RECENCY-WEIGHTED CONSULTING / RESEARCH PENALTY  (replaces hard DQ)
# ─────────────────────────────────────────────────────────────────────────────
# A job's weight in the consulting/research intensity calculation decays
# with how long ago it ENDED — a current role weighs 1.0, a role that ended
# CAREER_RECENCY_HALF_LIFE_YEARS ago weighs 0.5, twice that ago weighs 0.25,
# and so on, floored so even very old jobs retain some (small) influence.
CAREER_RECENCY_HALF_LIFE_YEARS: float = 5.0
MIN_JOB_RECENCY_WEIGHT:         float = 0.15

# A "consulting firm" job whose own description shows real production/infra
# signal (see PRODUCTION_KEYWORDS) is only counted at this fraction of full
# consulting weight — some people do build real platforms while nominally
# employed by a services company on a client embed.
CONSULTING_PRODUCTION_REDEMPTION_WEIGHT: float = 0.40
RESEARCH_PRODUCTION_REDEMPTION_WEIGHT:   float = 0.40

# Multiplier floor: even a candidate whose ENTIRE recency-weighted career is
# consulting/research with zero production signal is never multiplied by
# exactly 0.0 — but the floor is steep enough that this case is functionally
# indistinguishable from disqualification once combined with the rest of
# the penalty chain, without ever using an absolute, context-blind cutoff.
CONSULTING_PENALTY_FLOOR: float = 0.08
RESEARCH_PENALTY_FLOOR:   float = 0.08

# ─────────────────────────────────────────────────────────────────────────────
# 8. AVAILABILITY MULTIPLIER (m_avail) — universal
# ─────────────────────────────────────────────────────────────────────────────
ACTIVITY_TIERS: list[tuple[int, float]] = [
    (30, 1.00), (60, 0.85), (90, 0.65), (180, 0.40),
]
ACTIVITY_FLOOR_MULT:   float = 0.10
OPEN_TO_WORK_MULT:     float = 1.00
NOT_OPEN_TO_WORK_MULT: float = 0.80
RESPONSE_RATE_BASE:    float = 0.50
RESPONSE_RATE_SCALE:   float = 0.50
RESPONSE_RATE_MIN:     float = 0.50
RESPONSE_RATE_MAX:     float = 1.00
INTERVIEW_COMPLETION_THRESHOLD: float = 0.50
INTERVIEW_POOR_MULT:   float = 0.75
INTERVIEW_OK_MULT:     float = 1.00

# ─────────────────────────────────────────────────────────────────────────────
# 9. LOCATION MULTIPLIER (m_loc)
# ─────────────────────────────────────────────────────────────────────────────
PREFERRED_CITIES: set[str] = {
    "pune", "noida", "delhi", "delhi ncr", "new delhi",
    "mumbai", "hyderabad", "gurgaon", "gurugram",
    "bengaluru", "bangalore",
}

KNOWN_CITY_POOL: frozenset[str] = frozenset([
    "pune", "noida", "delhi", "delhi ncr", "new delhi", "mumbai",
    "hyderabad", "gurgaon", "gurugram", "bengaluru", "bangalore",
    "chennai", "kolkata", "ahmedabad", "jaipur", "surat", "lucknow",
    "nagpur", "bhopal", "visakhapatnam", "indore", "patna",
    "thane", "navi mumbai", "kochi", "chandigarh", "coimbatore",
    "london", "new york", "san francisco", "singapore", "dubai",
    "amsterdam", "berlin", "toronto", "sydney", "tokyo", "paris",
    "seattle", "austin", "boston", "chicago", "los angeles",
    "hong kong", "beijing", "shanghai", "stockholm", "zurich",
])
CITY_POSITIVE_CONTEXTS: frozenset[str] = frozenset([
    "preferred", "welcome", "office", "location", "based in",
    "open to", "accepted", "considered", "our team", "join us",
])

LOC_PREFERRED:         float = 1.00
LOC_INDIA_RELOCATE:    float = 0.90
LOC_INDIA_NO_RELOCATE: float = 0.55
LOC_INTL_RELOCATE:     float = 0.70
LOC_INTL_NO_RELOCATE:  float = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 10. NOTICE PERIOD (m_notice)
# ─────────────────────────────────────────────────────────────────────────────
# Fallback tiers, used when the JD doesn't state an explicit notice-period
# number. If it does (e.g. "sub-30-day", "notice period of 45 days"),
# jd_processor.py extracts that number directly from the text and
# guardrails.py builds tiers around it instead — see
# NOTICE_TEXT_EXTRACTION_PATTERN below. This is the "numbers near the word
# notice" text-native extraction from change #1.
NOTICE_TIERS: list[tuple[int, float]] = [
    (30, 1.00), (60, 0.90), (90, 0.80),
]
NOTICE_FLOOR_MULT: float = 0.65

# Regex to pull an explicit notice-period number directly from JD text.
# Matches: "sub-30-day", "30 day notice", "notice period of 45 days",
#          "notice period: 60 days", "within 30 days"
NOTICE_TEXT_EXTRACTION_PATTERN: str = (
    r"(?:sub-?|notice\s+period\s+(?:of|is|:)?\s*|within\s+|up\s+to\s+)?"
    r"(\d{1,3})\s*-?\s*days?\s*(?:notice)?"
)

# ─────────────────────────────────────────────────────────────────────────────
# 11. PROFILE INTEGRITY (m_integrity)
# ─────────────────────────────────────────────────────────────────────────────
SALARY_INVERSION_PENALTY:    float = 0.15
TRUST_ZERO_ANCHOR_PENALTY:   float = 0.20
TRUST_ONE_ANCHOR_PENALTY:    float = 0.08
COMPLETENESS_LOW_THRESHOLD:    int = 40
COMPLETENESS_MED_THRESHOLD:    int = 60
COMPLETENESS_LOW_PENALTY:    float = 0.15
COMPLETENESS_MED_PENALTY:    float = 0.07
HOLLOW_EXPERT_PER_SKILL:     float = 0.05
HOLLOW_EXPERT_DEDUCTION_CAP: float = 0.20
INTEGRITY_FLOOR:             float = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# 12. JD-FIT (m_jd)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: the old flat-fraction "partial consulting" penalty (former P-13) is
# REMOVED from here — it's fully superseded by the recency-weighted,
# content-aware consulting multiplier in Section 7 above, applied directly
# in the main penalized_score formula rather than folded into this additive
# delta. Job-hopping, offer-churn, and GitHub bonus are unaffected; they
# were already soft/tiered, not binary, so they didn't need this fix.
JOB_HOP_VERY_SHORT_MONTHS: int   = 12
JOB_HOP_SHORT_MONTHS:      int   = 18
JOB_HOP_HARD_PENALTY:      float = 0.20
JOB_HOP_SOFT_PENALTY:      float = 0.10

OFFER_ACCEPTANCE_POOR_THRESHOLD: float = 0.30
OFFER_ACCEPTANCE_PENALTY:        float = 0.10

GITHUB_HIGH_THRESHOLD: int   = 70
GITHUB_MED_THRESHOLD:  int   = 40
GITHUB_HIGH_BONUS:     float = 0.05
GITHUB_MED_BONUS:      float = 0.02

FRAMEWORK_ONLY_HARD_PENALTY: float = 0.20
FRAMEWORK_ONLY_SOFT_PENALTY: float = 0.10

YOE_BELOW_RANGE_DEDUCTION: float = 0.10
YOE_ABOVE_RANGE_DEDUCTION: float = 0.05
YOE_TOLERANCE:               int  = 2

JD_FIT_MIN: float = 0.40
JD_FIT_MAX: float = 1.10
SCORE_DISQUALIFIED: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 13. SEMANTIC SCORE NORMALIZATION  (NEW — change #3)
# ─────────────────────────────────────────────────────────────────────────────
# Percentile anchors for the robust min-max stretch of the combined dense
# score (and the per-track role/cap scores) across the retrieved candidate
# pool, applied before any behavioral multiplier. 2/98 rather than 0/100 so
# one outlier candidate can't compress everyone else's normalised spread.
SEMANTIC_NORM_LOW_PERCENTILE:  float = 2.0
SEMANTIC_NORM_HIGH_PERCENTILE: float = 98.0

# If the pool's raw score spread is smaller than this (near-identical
# scores across the board — e.g. a tiny candidate pool, or a degenerate
# query), normalization returns a flat 0.5 for everyone rather than
# amplifying floating-point noise into an arbitrary, meaningless spread.
SEMANTIC_NORM_MIN_SPREAD: float = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# 14. JD LEXICAL FLAG PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
JD_FLAG_PATTERNS: dict[str, tuple[str, bool]] = {
    "is_engineering":        (r"\b(engineer|engineering|developer|architect)\b",              False),
    "is_ml_ai":              (r"\b(machine\s*learning|ml\b|llm|embeddings?|retrieval|nlp)\b", False),
    "is_senior":              (r"\b(senior|sr\.?|lead|principal|founding|staff)\b",            False),
    "is_founding_team":       (r"\b(founding\s+team|founding\s+member|first\s+hire)\b",        False),
    "is_management":          (r"\b(manager|management|director|vp\b|head\s+of)\b",            False),

    "requires_python":        (r"\bpython\b",                                                  False),
    "requires_vector_db":     (r"\b(vector\s*(?:db|database)|faiss|pinecone|qdrant|weaviate|milvus|pgvector|opensearch|elasticsearch)\b", False),
    "requires_embeddings":    (r"\b(embeddings?|sentence.transformer|bge|e5)\b",               False),
    "requires_rag":           (r"\b(rag\b|retrieval.augmented|hybrid\s+search)\b",             False),
    "requires_llm":           (r"\b(llm\b|large\s+language|gpt|mistral|llama|openai|fine.tun)\b", False),
    "requires_ranking":       (r"\b(ranking|ranker|learning.to.rank|ltr|ndcg|mrr|map)\b",      False),
    "requires_eval_framework":(r"\b(ndcg|mrr\b|map\b|a/b\s+test|offline\s+bench|eval\w*\s+framework)\b", False),
    "requires_production":    (r"\b(production|deployed|real\s+users|at\s+scale|latency)\b",   False),
    "prefers_product_company":(r"\b(product\s+company|startup|scale.up)\b",                   False),
    "requires_startup_exp":   (r"\b(series\s+[abcd]|seed\s+stage|early.stage|startup)\b",     False),
    "prefers_open_source":    (r"\b(open.source|github|contribution)\b",                       False),

    "explicitly_rejects_consulting": (
        r"(?:only\s+work(?:ed)?\s+at|career.*?consulting|consulting.*?only|tcs|infosys|wipro).{0,150}"
        r"(?:not\s+move|won.t|will\s+not|disqualif|reject)|"
        r"(?:will\s+not\s+move|won.t\s+move|do\s+not).{0,150}(?:consulting|services?\s+firm)",
        False,
    ),
    "explicitly_rejects_research": (
        r"(?:pure\s+research|academic|research.only).{0,150}(?:not\s+move|won.t|will\s+not)|"
        r"(?:will\s+not\s+move|won.t).{0,150}(?:research|academic)",
        False,
    ),
    "explicitly_rejects_framework_only": (
        r"(?:langchain|framework).{0,100}(?:only|primarily|tourist|enthusiast)",
        False,
    ),
    "requires_code_writing": (
        r"\b(?:this\s+role\s+writes?\s+code|hands.?on\s+coding|write\s+production\s+code)\b",
        False,
    ),

    "prefers_local":          (r"\b(on.?site|in.?office|local|relocation|location)\b",         False),
    "prefers_short_notice":   (r"\b(sub.30|notice\s+period|buy.?out|immediate)\b",             False),
    "no_visa_sponsorship":    (r"\b(no\s+visa|don.t\s+sponsor|cannot\s+sponsor)\b",            False),
}


def extract_jd_flags(raw_jd_text: str) -> dict[str, bool]:
    """Evaluate all JD_FLAG_PATTERNS against raw JD text."""
    text_lower = raw_jd_text.lower()
    flags: dict[str, bool] = {}
    for flag_name, (pattern, negate) in JD_FLAG_PATTERNS.items():
        matched = bool(_re.search(pattern, text_lower, _re.IGNORECASE | _re.DOTALL))
        flags[flag_name] = (not matched) if negate else matched
    return flags

# ─────────────────────────────────────────────────────────────────────────────
# 15. BM25 / TOKENIZATION
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS: frozenset[str] = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","not","we","our","you","your","they","their",
    "this","that","these","those","it","its","who","which","what","when",
    "where","how","why","if","then","than","so","also","about","into",
    "out","up","more","most","all","both","each","any","some","such","no",
    "nor","only","own","same","very","just","over","after","i","me","my",
    "he","she","his","her","him","us","them","work","working","years",
    "experience","strong","good","great","ability","skills","skill",
    "knowledge","understanding","new","role","team","company","looking",
    "seeking","candidate","position","using","used","use","well","need",
    "required","must","preferred","nice","including","like","etc",
})

ROLE_SECTION_HEADERS: list[str] = [
    "responsibilities", "what you", "the role", "your role",
    "about the role", "position overview", "what we need",
    "key responsibilities", "what you will do", "what you'd actually be doing",
]
CAPABILITY_SECTION_HEADERS: list[str] = [
    "requirements", "qualifications", "skills", "you'll need",
    "must have", "what we're looking for", "technical skills",
    "tech stack", "technical requirements", "the skills inventory",
    "things you absolutely need",
]