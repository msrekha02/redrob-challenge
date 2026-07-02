"""
rank.py — CLI Entry Point (v5)
================================
Usage:
    # Default JD (bundled with this submission) — what Stage 3 runs:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

    # Explicit different JD — dynamic JD processing remains fully available:
    python rank.py --jd_path other_role.docx --candidates ./candidates.jsonl --out ./submission.csv

v5 changes:

  1. --jd_path is now OPTIONAL, not required. The Stage 3 evaluator's
     literal reproduction command is:
         python rank.py --candidates ./candidates.jsonl --out ./submission.csv
     with no --jd_path at all. A required argument here would make that
     exact command fail outright — disqualifying the submission at Stage 3
     regardless of composite score. When --jd_path is omitted, rank.py
     resolves a default JD via DEFAULT_JD_SOURCES below: first the bundled
     data/job_description.docx file, then (only if that's missing for any
     reason — wrong working directory, file excluded from a packaging
     step) an embedded verbatim copy of the same JD text as a last-resort
     fallback, so ranking can proceed even in a maximally stripped-down
     reproduction environment. The dynamic JD-processing capability itself
     is untouched: passing --jd_path still fully overrides the default and
     works exactly as before for any other JD.

  2. --out added as the primary output flag name, matching the evaluator's
     exact example command. --output is kept as a backward-compatible
     alias (whichever is actually passed wins; if neither is passed, the
     config default is used).

  3. jd_profile.category_low_confidence is now threaded into retrieve(),
     activating the BM25-bias routing in retriever.py for out-of-
     distribution JDs (see retriever.py's module docstring).
"""

from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [RANK] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from jd_processor import JDProcessor
from retriever import ArtifactStore, retrieve
from guardrails import CrossEncoderReranker, apply_penalties, apply_honeypot_safety_net
from reasoner import export_csv


# ─────────────────────────────────────────────────────────────────────────────
# Default JD resolution (v5)
# ─────────────────────────────────────────────────────────────────────────────

# Candidate filenames checked, in order, when --jd_path isn't supplied.
# All resolved relative to cfg.DATA_DIR (Path(__file__)-based, so this is
# robust regardless of the working directory the script is invoked from).
DEFAULT_JD_FILENAMES: list[str] = [
    "job_description.docx",
    "jd.docx",
    "JD.docx",
    "job_description.txt",
]

# Last-resort fallback: the verbatim Redrob Senior AI Engineer JD text,
# embedded directly so ranking can proceed even if no JD file exists
# anywhere on disk (e.g. a packaging step accidentally excluded the
# data/ folder's .docx file). Kept word-for-word identical to the source
# JD — jd_processor.py's detection regexes depend on exact phrasing
# ("we will not move forward", "5-9 years", "sub-30-day", etc.), so a
# paraphrased version would silently change what gets detected.
_EMBEDDED_DEFAULT_JD_TEXT = """Job Description: Senior AI Engineer — Founding Team

Company: Redrob AI (Series A AI-native talent intelligence platform)

Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities

Employment Type: Full-time

Experience Required: 5–9 years (see "what we mean by this" below)

Let's be honest about this role

We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.

If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.

If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product or recruiter workflows or eval frameworks, this also isn't it.

We need someone who is simultaneously comfortable with two things that sound contradictory:

Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.

Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.

These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.

What you'd actually be doing

The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.

In practical terms, your first 90 days will probably look like:

Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.

Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.

Weeks 9-12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind.

Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.

What we mean by "5-9 years"

This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5-9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.

That said, here are the disqualifiers we actually apply:

If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We are explicit about this. We've tried it twice and it didn't work for either side.

If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking before it became fashionable.

If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" or "tech lead" roles — we will probably not move forward. This role writes code.

The skills inventory (please read carefully)

Most JDs list 20 skills and you're supposed to have all of them. We're going to do this differently.

Things you absolutely need

Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.

Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.

Strong Python. Yes really, we care about code quality.

Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.

Things we'd like you to have but won't reject you for

LLM fine-tuning experience (LoRA, QLoRA, PEFT)

Experience with learning-to-rank models (XGBoost-based or neural)

Prior exposure to HR-tech, recruiting tech, or marketplace products

Background in distributed systems or large-scale inference optimization

Open-source contributions in the AI/ML space

Things we explicitly do NOT want

This is the section most JDs skip but we think it's the most important:

Title-chasers. If your career trajectory shows you optimizing for "Senior" → "Staff" → "Principal" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years.

Framework enthusiasts. If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" — that's fine but it's not what we need. We need people who think about systems, not frameworks.

People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine.

People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here.

People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think.

On location, comp, and logistics

Location: Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas.

Notice period: We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.

The vibe check

We genuinely believe culture-fit matters more at this stage than skills-fit. Skills are teachable; the rest mostly isn't.

We work async-first and write a lot. If you find writing painful, you'll find this role painful.

We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.

We move fast and break things, with the caveat that "things" are usually our internal assumptions, not user-facing systems. If you need a stable, mature codebase to be productive, you'll find this role unstable.

How to read between the lines

The "ideal candidate" we're imagining is roughly:

6-8 years total experience, of which 4-5 are in applied ML/AI roles at product companies (not pure services).

Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.

Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt) — and can defend them with reference to systems they actually built.

Located in or willing to relocate to Noida or Pune.

Active on Redrob platform (or has clear signal of being in the job market) so we can actually talk to them.

We are aware this is a narrow profile. We're not expecting to find many matches in a 100K candidate pool. We're explicitly OK with that — we'd rather see 10 great matches than 1000 maybes.

Final note for the participants of the Redrob hackathon

If you're reading this in the context of the Intelligent Candidate Discovery & Ranking Challenge:

The "right answer" to this JD is not "find candidates whose skills section contains the most AI keywords." That's a trap we've explicitly built into the dataset.

The right answer involves reasoning about the gap between what the JD says and what the JD means. A Tier 5 candidate may not use the words "RAG" or "Pinecone" in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit. A candidate who has all the AI keywords listed as skills but whose title is "Marketing Manager" is not a fit, no matter how perfect their skill list looks.

Your ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately.

Good luck."""


def _resolve_jd_path(explicit_path: str | None) -> Path:
    """
    Three-tier resolution, in priority order:
      1. --jd_path explicitly given -> use it (full dynamic JD support unchanged)
      2. A bundled default JD file found at one of DEFAULT_JD_FILENAMES
         under cfg.DATA_DIR
      3. Last resort: write _EMBEDDED_DEFAULT_JD_TEXT to a stable location
         inside cfg.ARTIFACTS_DIR (not /tmp — some sandboxed reproduction
         environments restrict writes outside the project tree) and use that

    Tier 3 only activates if NEITHER an explicit path NOR any bundled file
    was found, so it never silently overrides a real file that exists.
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            log.error("JD not found at explicitly-provided --jd_path: %s", p)
            sys.exit(1)
        log.info("Using explicitly-provided JD: %s", p)
        return p

    for filename in DEFAULT_JD_FILENAMES:
        candidate = cfg.DATA_DIR / filename
        if candidate.exists():
            log.info("No --jd_path given — using bundled default JD: %s", candidate)
            return candidate

    log.warning(
        "No --jd_path given and no bundled default JD file found at any of %s "
        "under %s — falling back to embedded JD text (last resort).",
        DEFAULT_JD_FILENAMES, cfg.DATA_DIR,
    )
    fallback_path = cfg.ARTIFACTS_DIR / "default_jd_embedded_fallback.txt"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text(_EMBEDDED_DEFAULT_JD_TEXT, encoding="utf-8")
    log.info("Embedded fallback JD written to: %s", fallback_path)
    return fallback_path


class _T:
    def __init__(self, label): self.label = label; self._t = 0.0
    def __enter__(self): self._t = time.perf_counter(); return self
    def __exit__(self, *_): log.info("  ✓ %-46s %.2fs", self.label, time.perf_counter() - self._t)


def run_ranking(jd_path: Path, output_path: Path) -> None:
    t0 = time.perf_counter()
    log.info("=" * 64)
    log.info("Redrob Ranker — Final Dynamic Pipeline")
    log.info("=" * 64)

    # ── STEP 1: JD ingestion + auto-detection ────────────────────────────────
    log.info("STEP 1 — JD Processing (4-way cast + auto-detection)")
    with _T("JD ingest + flags"): jdp = JDProcessor(jd_path); flags = jdp.flags
    with _T("Dense vectors (role + cap)"): rv = jdp.role_vector; cv = jdp.cap_vector
    with _T("Category detection (10 anchors)"): profile = jdp.profile

    log.info("")
    log.info("  ── AUTO-DETECTED JD PROFILE ─────────────────────────────────")
    log.info(
        "  Role category       : %s  (top-match sim=%.3f%s)",
        profile.role_category, profile.category_confidence,
        ", LOW CONFIDENCE — weights blend toward default, BM25-biased retrieval active" if profile.category_low_confidence else "",
    )
    log.info("  Seniority           : %.2f   Tech depth: %.2f   Urgency: %.2f",
             profile.seniority_level, profile.technical_depth, profile.urgency)
    log.info("  Blended weights     : role=%.2f  cap=%.2f  (similarity-weighted across all categories, not a single bucket)",
             profile.role_weight, profile.cap_weight)
    log.info("  Avail importance    : %.2f   Loc importance: %.2f", profile.avail_importance, profile.loc_importance)
    log.info("  Domain threshold    : %.3f", profile.domain_mismatch_threshold)
    log.info("  Preferred cities    : %s", sorted(profile.preferred_cities))
    log.info("  YoE range           : %s – %s years", profile.yoe_min, profile.yoe_max)
    log.info("  NLP/IR required     : %s", profile.nlp_ir_required)
    log.info("  Notice (JD-stated)  : %s days", profile.notice_period_days_stated)
    log.info("  Consulting penalty active : %s  (hard DQ for literal 100%%; continuous elsewhere)", profile.consulting_penalty_active)
    log.info("  Research penalty active   : %s  (hard DQ for literal 100%%; continuous elsewhere)", profile.research_penalty_active)
    log.info("  Production required : %s", profile.production_is_required)
    log.info("  Flags (True)        : %s", [k for k, v in flags.items() if v])
    log.info("  ─────────────────────────────────────────────────────────────")

    with _T("Keywords (BM25 query)"):
        rtok = jdp.role_keywords; ctok = jdp.cap_keywords; all_kw = jdp.keywords
    log.info("  Keywords: role=%d  cap=%d", len(rtok), len(ctok))

    # ── STEP 2: Artifacts ────────────────────────────────────────────────────
    log.info("")
    log.info("STEP 2 — Loading offline artifacts")
    with _T("Artifact store"):
        store = ArtifactStore(); store.load()
    log.info("  Pool: %d clean candidates", store.n)

    # ── STEP 3: Retrieval → top-1000 ─────────────────────────────────────────
    log.info("")
    log.info("STEP 3 — Dual-track retrieval → top-%d", cfg.TOP_K_RRF)
    with _T(f"Retrieve (4-list RRF → {cfg.TOP_K_RRF})"):
        retrieval = retrieve(
            jd_role_vec=rv, jd_cap_vec=cv,
            jd_role_tokens=rtok, jd_cap_tokens=ctok,
            store=store, top_k_rrf=cfg.TOP_K_RRF,
            role_weight=profile.role_weight, cap_weight=profile.cap_weight,
            category_low_confidence=profile.category_low_confidence,
        )
    log.info("  Retrieved: %d", len(retrieval.indices))

    # ── STEP 4: Penalty gate → top-200 ───────────────────────────────────────
    log.info("")
    log.info("STEP 4 — Penalty gate (20 signals) → top-%d", cfg.TOP_K_PENALTY)
    with _T(f"apply_penalties → {cfg.TOP_K_PENALTY}"):
        penalized = apply_penalties(retrieval, store, profile, all_kw, cfg.TOP_K_PENALTY)
    log.info("  Penalty output: %d", len(penalized))

    # ── STEP 5: Cross-encoder deep read ──────────────────────────────────────
    log.info("")
    log.info("STEP 5 — Cross-encoder deep read (%d candidates, fully ranked)", len(penalized))
    with _T(f"CrossEncoder rerank ({len(penalized)} pairs)"):
        reranked = CrossEncoderReranker().rerank(penalized, store, jdp.raw_text)
    log.info("  Reranked: %d candidates", len(reranked))

    # ── STEP 5.5: Honeypot safety net — the actual Stage-3 enforcement point ──
    log.info("")
    log.info("STEP 5.5 — Honeypot safety net (final enforcement, all 14 traps)")
    with _T(f"Safety net → exactly {cfg.TOP_K_FINAL}"):
        final = apply_honeypot_safety_net(reranked, store, final_k=cfg.TOP_K_FINAL)
    log.info("  Final clean: %d candidates", len(final))

    # ── STEP 6: Reasoning + CSV ───────────────────────────────────────────────
    log.info("")
    log.info("STEP 6 — Reasoning + CSV")
    with _T("export_csv"):
        export_csv(final, store, flags, all_kw, profile, output_path)

    elapsed = time.perf_counter() - t0
    log.info("")
    log.info("=" * 64)
    log.info("Done %.1fs (%.1f min) — %s", elapsed, elapsed / 60, output_path)
    log.info("=" * 64)
    if elapsed > 270:
        log.warning("⚠ %.1fs approaching 5-min budget!", elapsed)

    log.info("\nTop-10:")
    for i, cs in enumerate(final[:10], 1):
        snap = store.snapshot(cs.candidate_id)
        log.info("  %2d. %-15s | %-35s @ %-20s | CE=%.3f dom=%.2f",
                 i, cs.candidate_id,
                 (snap.get("current_title") or "")[:35],
                 (snap.get("current_company") or "")[:20],
                 cs.ce_score, cs.domain_sim)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Redrob Final Ranker",
        epilog="If --jd_path is omitted, a bundled default JD is used automatically "
               "(see DEFAULT_JD_FILENAMES) — full dynamic JD processing remains "
               "available any time --jd_path is supplied.",
    )
    p.add_argument(
        "--jd_path", default=None,
        help="Path to a JD file (.docx/.txt/.md). Optional — if omitted, a bundled "
             "default JD is used automatically so the single-command Stage 3 "
             "reproduction case (no --jd_path) still works.",
    )
    p.add_argument("--candidates", default=str(cfg.CANDIDATES_PATH))
    p.add_argument(
        "--out", "--output", dest="output", default=str(cfg.OUTPUT_CSV_PATH),
        help="Output CSV path. --out matches the evaluator's example command; "
             "--output is kept as a backward-compatible alias.",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)

    jd = _resolve_jd_path(args.jd_path)

    if not cfg.CANDIDATE_META_PATH.exists():
        log.error("Artifacts missing. Run: python offline_pipeline.py --candidates %s", args.candidates)
        sys.exit(1)
    run_ranking(jd_path=jd, output_path=Path(args.output))


if __name__ == "__main__":
    main()