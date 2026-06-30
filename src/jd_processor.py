"""
jd_processor.py — Dynamic JD Profile + 4-Way Multi-Cast
===============================================================
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# JDProfile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JDProfile:
    role_category:        str
    category_confidence:  float
    category_low_confidence: bool
    seniority_level:      float
    technical_depth:      float
    urgency:              float

    role_weight:               float
    cap_weight:                float
    avail_importance:          float
    loc_importance:            float
    domain_mismatch_threshold: float

    consulting_penalty_active: bool
    research_penalty_active:   bool
    production_is_required:    bool

    preferred_cities: set[str] = field(default_factory=set)
    yoe_min: int | None = None
    yoe_max: int | None = None
    nlp_ir_required: bool = False
    notice_period_days_stated: int | None = None

    # Dynamic Behavioral Flags
    requires_high_integrity: bool = False
    penalize_job_hoppers: bool = False
    enforce_availability: bool = False

    category_anchor_vec: np.ndarray = field(default_factory=lambda: np.zeros(cfg.EMBED_DIM, dtype=np.float32))
    flags: dict[str, bool] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_SENIORITY_EXEC_KW   = frozenset(["vp of", "vice president", "director of", "head of", "chief ", "c-suite"])
_SENIORITY_SENIOR_KW = frozenset(["principal", "staff engineer", "founding team", "founding member", "senior", " sr ", "sr."])
_SENIORITY_MID_KW    = frozenset(["mid-level", "intermediate", "3-5 years", "3 to 5", "associate"])
_SENIORITY_JUNIOR_KW = frozenset(["junior", "jr.", "entry level", "entry-level", "0-2 years", "fresher", "new graduate"])

def _detect_seniority(text_lower: str) -> float:
    if any(kw in text_lower for kw in _SENIORITY_EXEC_KW):   return 1.00
    if any(kw in text_lower for kw in _SENIORITY_SENIOR_KW): return 0.78
    if any(kw in text_lower for kw in _SENIORITY_MID_KW):    return 0.50
    if any(kw in text_lower for kw in _SENIORITY_JUNIOR_KW): return 0.22
    return 0.65


_TECH_DEEP_SIGNALS = frozenset([
    "algorithm", "latency", "throughput", "distributed system", "machine learning",
    "neural network", "embeddings", "vector", "kubernetes", "microservice",
    "database", "api", "backend", "compiler", "runtime", "system design",
    "architecture", "pipeline", "inference", "model", "python", "java", "c++",
    "sql", "nosql", "ci/cd", "devops", "framework", "tensor", "gradient",
    "fine-tuning", "tokenizer", "retrieval", "ranking",
])
_SOFT_SIGNALS = frozenset([
    "stakeholder", "strategy", "roadmap", "campaign", "brand", "customer",
    "sales", "marketing", "growth", "revenue", "fundraising", "partnership",
    "communication", "negotiation", "business development", "influencing",
])

def _detect_technical_depth(text_lower: str) -> float:
    tech = sum(1 for s in _TECH_DEEP_SIGNALS if s in text_lower)
    soft = sum(1 for s in _SOFT_SIGNALS if s in text_lower)
    return (tech / (tech + soft)) if (tech + soft) > 0 else 0.50


_URGENCY_HIGH = frozenset(["immediate joiner", "immediate start", "asap", "urgently"])
_URGENCY_MED  = frozenset(["sub-30", "30 days notice", "buy out notice", "notice period < 30", "sub 30"])
_URGENCY_LOW  = frozenset(["flexible start", "notice period negotiable", "3 months notice", "6 months"])

def _detect_urgency(text_lower: str) -> float:
    if any(kw in text_lower for kw in _URGENCY_HIGH): return 1.00
    if any(kw in text_lower for kw in _URGENCY_MED):  return 0.65
    if any(kw in text_lower for kw in _URGENCY_LOW):  return 0.20
    return 0.40


def _extract_dq_conditions(text_lower: str) -> dict[str, bool]:
    dq_verbs = r"(?:will\s+not|won.t|do\s+not|don.t|cannot|not\s+a\s+fit|disqualif|reject|eliminat)"
    consulting_relevant = bool(re.search(
        rf"(?:consulting|services?\s+firm|tcs|infosys|wipro|accenture|cognizant).{{0,200}}{dq_verbs}|"
        rf"{dq_verbs}.{{0,200}}(?:consulting|services?\s+firm|only\s+work)",
        text_lower, re.DOTALL
    )) or bool(re.search(r"people\s+who\s+have\s+only\s+work\w+\s+at\s+consulting", text_lower))

    research_relevant = bool(re.search(
        rf"(?:pure\s+research|academic\s+lab|research.only|without.*?production).{{0,200}}{dq_verbs}|"
        rf"{dq_verbs}.{{0,200}}(?:pure\s+research|academic|no\s+production)",
        text_lower, re.DOTALL
    ))
    production_required = bool(re.search(r"\b(?:production|deployed|real\s+users|at\s+scale)\b", text_lower))

    return {
        "consulting_penalty_active": consulting_relevant,
        "research_penalty_active":   research_relevant,
        "production_is_required":    production_required,
    }


def _extract_preferred_cities(jd_text: str) -> set[str]:
    text_lower = jd_text.lower()
    found: set[str] = set()
    for city in cfg.KNOWN_CITY_POOL:
        if city not in text_lower:
            continue
        idx     = text_lower.find(city)
        context = text_lower[max(0, idx - 100): idx + 120]
        if any(pos in context for pos in cfg.CITY_POSITIVE_CONTEXTS):
            found.add(city)
    return found if found else set(cfg.PREFERRED_CITIES)


def _detect_yoe_range(jd_text: str) -> tuple[int | None, int | None]:
    """Capture multiple range sequences, choose the widest layout bounds, map open-ended to 99."""
    text_lower = jd_text.lower()
    min_y, max_y = None, None
    
    # Standard ranges (e.g., 5-9 years)
    range_matches = re.findall(r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)", text_lower)
    if range_matches:
        min_y = min(int(m[0]) for m in range_matches)
        max_y = max(int(m[1]) for m in range_matches)
        
    # Open-ended ranges (e.g., 10+ years, min 5 years)
    open_matches = re.findall(r"(?:minimum|at\s+least|min\.?)\s*(\d+)\s*(?:years?|yrs?)|(\d+)\+\s*(?:years?|yrs?)", text_lower)
    if open_matches:
        extracted_mins = [int(m[0] or m[1]) for m in open_matches]
        open_min = min(extracted_mins)
        if min_y is None or open_min < min_y: 
            min_y = open_min
        if max_y is None or max_y < 99:
            max_y = 99
            
    return min_y, max_y


def _detect_nlp_ir_required(text_lower: str) -> bool:
    NLP_IR_REQUIRED_SIGNALS = frozenset([
        "nlp", "natural language processing", "information retrieval",
        "ranking system", "retrieval system", "search system",
        "recommendation system", "embedding", "semantic search",
    ])
    return any(sig in text_lower for sig in NLP_IR_REQUIRED_SIGNALS)


def _extract_notice_period_days(jd_text: str) -> int | None:
    """Contextual cascade that handles explicit tags, string spans, and a defensive baseline."""
    text_lower = jd_text.lower()
    
    # 1. Explicit hard tags
    if re.search(r'\bsub-?30\b', text_lower):
        return 30
    if re.search(r'\bimmediate\s+joiner\b', text_lower):
        return 0
        
    # 2. String spans / standard formats
    for m in re.finditer(cfg.NOTICE_TEXT_EXTRACTION_PATTERN, text_lower):
        window = text_lower[max(0, m.start() - 40): m.end() + 20]
        if "notice" in window or "join" in window or "buy" in window:
            days = int(m.group(1))
            if 0 < days <= 365:
                return days
                
    return None


def _detect_behavioral_toggles(text_lower: str) -> dict[str, bool]:
    """Identify if the JD explicitly cares about dynamic behavioral traits."""
    return {
        "requires_high_integrity": bool(re.search(r"\b(reliable|verified|integrity|trustworthy)\b", text_lower)),
        "penalize_job_hoppers": bool(re.search(r"\b(stable\s+tenure|long-term\s+commitment|track\s+record)\b", text_lower)),
        "enforce_availability": bool(re.search(r"\b(active|urgently\s+looking|available\s+now)\b", text_lower)),
    }


def _compute_dynamic_weights(
    category: str, seniority: float, technical_depth: float, urgency: float,
) -> dict:
    base = dict(cfg.CATEGORY_WEIGHT_PROFILES.get(category, cfg.CATEGORY_WEIGHT_PROFILES["default"]))
    sen_delta  = (seniority       - 0.50) * cfg.SENIORITY_ROLE_WEIGHT_BOOST
    tech_delta = (technical_depth - 0.50) * cfg.TECHNICAL_DEPTH_CAP_WEIGHT_BOOST
    base["role_weight"] = float(base["role_weight"]) + sen_delta - tech_delta
    base["cap_weight"]  = float(base["cap_weight"])  - sen_delta + tech_delta
    if urgency > 0.60:
        base["avail_importance"] = float(base["avail_importance"]) + (urgency - 0.60) * cfg.URGENCY_AVAIL_IMPORTANCE_BOOST
    base["role_weight"] = max(0.20, min(0.80, float(base["role_weight"])))
    base["cap_weight"]  = max(0.20, min(0.80, float(base["cap_weight"])))
    total = base["role_weight"] + base["cap_weight"]
    base["role_weight"] /= total
    base["cap_weight"]  /= total
    return base


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def _compute_blended_weights(
    category_sims: dict[str, float],
    seniority: float,
    technical_depth: float,
    urgency: float,
    low_confidence: bool,
) -> tuple[dict, np.ndarray, dict[str, float]]:
    categories = list(cfg.CATEGORY_WEIGHT_PROFILES.keys())
    sims = np.array([category_sims.get(c, cfg.CATEGORY_BLEND_DEFAULT_PRIOR) for c in categories], dtype=np.float64)
    weights = _softmax(sims, cfg.CATEGORY_BLEND_TEMPERATURE)

    field_names = ["role_weight", "cap_weight", "avail_importance", "loc_importance", "domain_mismatch_threshold"]
    blended: dict[str, float] = {f: 0.0 for f in field_names}

    for cat, w in zip(categories, weights):
        cat_weights = _compute_dynamic_weights(cat, seniority, technical_depth, urgency)
        for f in field_names:
            blended[f] += w * cat_weights[f]

    total = blended["role_weight"] + blended["cap_weight"]
    blended["role_weight"] /= total
    blended["cap_weight"]  /= total

    # Expand bounds to prevent stripping the matrix on out-of-distribution roles
    if low_confidence:
        blended["domain_mismatch_threshold"] = 0.05

    distribution = {cat: float(w) for cat, w in zip(categories, weights)}
    return blended, weights, distribution


class JDProcessor:
    def __init__(self, jd_path: str | Path) -> None:
        self._path    = Path(jd_path)
        self._raw:    str | None        = None
        self._flags:  dict | None       = None
        self._rvec:   np.ndarray | None = None
        self._cvec:   np.ndarray | None = None
        self._rkw:    list[str] | None  = None
        self._ckw:    list[str] | None  = None
        self._profile:JDProfile | None  = None
        self._model   = None
        _ = self.raw_text

    @property
    def raw_text(self) -> str:
        if self._raw is None:
            self._raw = self._ingest()
        return self._raw

    def _ingest(self) -> str:
        suffix = self._path.suffix.lower()
        if suffix == ".docx":
            try:
                from docx import Document
            except ImportError:
                raise ImportError("pip install python-docx")
            doc   = Document(self._path)
            lines = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            lines.append(cell.text.strip())
            return "\n".join(lines)
        return self._path.read_text(encoding="utf-8")

    @property
    def flags(self) -> dict[str, bool]:
        if self._flags is None:
            self._flags = cfg.extract_jd_flags(self.raw_text)
        return self._flags

    def _load_model(self):
        if self._model is None:
            from model_engine import ONNXEmbedder
            self._model = ONNXEmbedder(cfg.BGE_EMBED_MODEL_DIR, fallback_model_id=cfg.BGE_EMBED_MODEL_ID)
        return self._model

    def _encode(self, text: str) -> np.ndarray:
        return self._load_model().encode(text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def _split_sections(self) -> tuple[str, str]:
        lines = self.raw_text.split("\n")
        role_l: list[str] = []
        cap_l:  list[str] = []
        cur: str | None   = None
        for line in lines:
            ll = line.lower().strip()
            if any(h in ll for h in cfg.ROLE_SECTION_HEADERS):       cur = "role"
            elif any(h in ll for h in cfg.CAPABILITY_SECTION_HEADERS): cur = "cap"
            if cur == "role": role_l.append(line)
            elif cur == "cap": cap_l.append(line)
        return ("\n".join(role_l).strip() or self.raw_text[:500],
                "\n".join(cap_l).strip()  or self.raw_text)

    @property
    def role_vector(self) -> np.ndarray:
        if self._rvec is None: self._compute_vecs()
        return self._rvec

    @property
    def cap_vector(self) -> np.ndarray:
        if self._cvec is None: self._compute_vecs()
        return self._cvec

    def _compute_vecs(self) -> None:
        r, c = self._split_sections()
        self._rvec = self._encode(r)
        self._cvec = self._encode(c)

    @property
    def role_keywords(self) -> list[str]:
        if self._rkw is None: self._rkw = self._tok(self._split_sections()[0])
        return self._rkw

    @property
    def cap_keywords(self) -> list[str]:
        if self._ckw is None: self._ckw = self._tok(self._split_sections()[1])
        return self._ckw

    @property
    def keywords(self) -> list[str]:
        return self._tok(self.raw_text)

    @staticmethod
    def _tok(text: str) -> list[str]:
        text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
        seen: set[str] = set(); result: list[str] = []
        for t in text.split():
            t = t.strip("-")
            if len(t) > 2 and t not in cfg.STOPWORDS and t not in seen:
                seen.add(t); result.append(t)
        return result

    @property
    def profile(self) -> JDProfile:
        if self._profile is None:
            self._profile = self._build_profile()
        return self._profile

    def _build_profile(self) -> JDProfile:
        text_lower = self.raw_text.lower()
        seniority       = _detect_seniority(text_lower)
        technical_depth = _detect_technical_depth(text_lower)
        urgency         = _detect_urgency(text_lower)

        jd_role_vec = self.role_vector
        category_sims, anchor_vecs = self._compute_category_similarities(jd_role_vec)

        best_cat = max(category_sims, key=category_sims.get)
        best_sim = category_sims[best_cat]
        low_confidence = best_sim < cfg.CATEGORY_LOW_CONFIDENCE_THRESHOLD

        blended, blend_weights, distribution = _compute_blended_weights(
            category_sims, seniority, technical_depth, urgency, low_confidence
        )

        blended_anchor = np.zeros(cfg.EMBED_DIM, dtype=np.float32)
        for (cat, _sim), w in zip(category_sims.items(), blend_weights):
            blended_anchor += w * anchor_vecs[cat]
        norm = np.linalg.norm(blended_anchor)
        if norm > 1e-9:
            blended_anchor = (blended_anchor / norm).astype(np.float32)

        dq = _extract_dq_conditions(text_lower)
        preferred_cities = _extract_preferred_cities(self.raw_text)
        yoe_min, yoe_max = _detect_yoe_range(self.raw_text)
        nlp_ir_required = _detect_nlp_ir_required(text_lower)
        notice_stated = _extract_notice_period_days(self.raw_text)
        behavioral_toggles = _detect_behavioral_toggles(text_lower)

        return JDProfile(
            role_category=best_cat,
            category_confidence=best_sim,
            category_low_confidence=low_confidence,
            seniority_level=seniority, technical_depth=technical_depth, urgency=urgency,
            role_weight=float(blended["role_weight"]), cap_weight=float(blended["cap_weight"]),
            avail_importance=float(blended["avail_importance"]),
            loc_importance=float(blended["loc_importance"]),
            domain_mismatch_threshold=float(blended["domain_mismatch_threshold"]),
            consulting_penalty_active=dq["consulting_penalty_active"],
            research_penalty_active=dq["research_penalty_active"],
            production_is_required=dq["production_is_required"],
            preferred_cities=preferred_cities,
            yoe_min=yoe_min, yoe_max=yoe_max,
            nlp_ir_required=nlp_ir_required,
            notice_period_days_stated=notice_stated,
            requires_high_integrity=behavioral_toggles["requires_high_integrity"],
            penalize_job_hoppers=behavioral_toggles["penalize_job_hoppers"],
            enforce_availability=behavioral_toggles["enforce_availability"],
            category_anchor_vec=blended_anchor,
            flags=self.flags,
        )

    def _compute_category_similarities(
        self, jd_role_vec: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        model = self._load_model()
        sims: dict[str, float] = {}
        anchor_vecs: dict[str, np.ndarray] = {}
        for cat_name, anchor_text in cfg.ROLE_CATEGORY_ANCHORS.items():
            av = model.encode(anchor_text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            sims[cat_name] = float(np.dot(jd_role_vec, av))
            anchor_vecs[cat_name] = av
        anchor_vecs["default"] = np.zeros(cfg.EMBED_DIM, dtype=np.float32)
        sims["default"] = cfg.CATEGORY_BLEND_DEFAULT_PRIOR
        return sims, anchor_vecs

    def summary(self) -> dict[str, Any]:
        p = self.profile
        # Safe string conversions prevent interpolation loops from crashing on open-ended Nones
        return {
            "jd_file": self._path.name,
            "role_category": p.role_category, "confidence": f"{p.category_confidence:.3f}",
            "low_confidence_blend": p.category_low_confidence,
            "seniority": f"{p.seniority_level:.2f}", "tech_depth": f"{p.technical_depth:.2f}",
            "urgency": f"{p.urgency:.2f}",
            "weights": {"role": f"{p.role_weight:.2f}", "cap": f"{p.cap_weight:.2f}"},
            "preferred_cities": sorted(p.preferred_cities),
            "yoe_range": f"{str(p.yoe_min)}–{str(p.yoe_max)}",
            "nlp_ir_required": p.nlp_ir_required,
            "notice_period_days_stated": p.notice_period_days_stated,
            "penalty_active": {"consulting": p.consulting_penalty_active, "research": p.research_penalty_active},
            "flags_true": [k for k, v in p.flags.items() if v],
        }