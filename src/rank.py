"""
rank.py — Final CLI Entry Point
================================
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

class _T:
    def __init__(self, label): self.label = label; self._t = 0.0
    def __enter__(self): self._t = time.perf_counter(); return self
    def __exit__(self, *_): log.info("  ✓ %-46s %.2fs", self.label, time.perf_counter() - self._t)

def run_ranking(jd_path: Path, output_path: Path) -> None:
    t0 = time.perf_counter()
    
    # 1. Instantiate the Zero-Shot JdProcessor Natively
    with _T("JD ingest + flags"): jdp = JDProcessor(jd_path); flags = jdp.flags
    with _T("Dense vectors (role + cap)"): rv = jdp.role_vector; cv = jdp.cap_vector
    with _T("Category detection (10 anchors)"): profile = jdp.profile

    with _T("Keywords (BM25 query)"):
        rtok = jdp.role_keywords; ctok = jdp.cap_keywords; all_kw = jdp.keywords

    with _T("Artifact store"):
        store = ArtifactStore(); store.load()

    # 2. Flow the Dynamic Variables directly into the Retriever
    with _T(f"Retrieve (4-list RRF → {cfg.TOP_K_RRF})"):
        retrieval = retrieve(
            jd_role_vec=rv, jd_cap_vec=cv,
            jd_role_tokens=rtok, jd_cap_tokens=ctok,
            store=store, top_k_rrf=cfg.TOP_K_RRF,
            role_weight=profile.role_weight, cap_weight=profile.cap_weight,
            jd_profile=profile,  # Native tracking injection for adaptive fallbacks
        )

    # 3. Execute the Post-Retrieval Scoring and Neural Pruning Ring
    with _T(f"apply_penalties → {cfg.TOP_K_PENALTY}"):
        penalized = apply_penalties(retrieval, store, profile, all_kw, cfg.TOP_K_PENALTY)

    with _T(f"CrossEncoder rerank ({len(penalized)} pairs)"):
        reranked = CrossEncoderReranker().rerank(penalized, store, jdp.raw_text)

    with _T(f"Safety net → exactly {cfg.TOP_K_FINAL}"):
        final = apply_honeypot_safety_net(reranked, store, final_k=cfg.TOP_K_FINAL)

    with _T("export_csv"):
        export_csv(final, store, flags, all_kw, profile, output_path)

    elapsed = time.perf_counter() - t0
    if elapsed > 270:
        log.warning("⚠ %.1fs approaching 5-min budget!", elapsed)


def main() -> None:
    p = argparse.ArgumentParser(description="Redrob Final Ranker")
    p.add_argument("--jd_path",    required=True)
    p.add_argument("--candidates", default=str(cfg.CANDIDATES_PATH))
    p.add_argument("--output",     default=str(cfg.OUTPUT_CSV_PATH))
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    jd = Path(args.jd_path)
    if not jd.exists(): sys.exit(1)
    if not cfg.CANDIDATE_META_PATH.exists(): sys.exit(1)
    run_ranking(jd_path=jd, output_path=Path(args.output))

if __name__ == "__main__":
    main()