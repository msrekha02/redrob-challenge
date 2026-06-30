"""
retriever.py  v2 — Dual-Track Retrieval with Dynamic Weights
=============================================================
"""

from __future__ import annotations
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

log = logging.getLogger(__name__)

# Hold the raw structural scores for downstream re-ranking accuracy
@dataclass
class RetrievalResult:
    indices:          np.ndarray
    rrf_scores:       np.ndarray
    role_scores:      np.ndarray
    cap_scores:       np.ndarray
    bm25_role_scores: np.ndarray
    bm25_cap_scores:  np.ndarray
    candidate_ids:    list[str] = field(default_factory=list)

class ArtifactStore:
    def __init__(self) -> None:
        self._loaded = False
    def load(self) -> None:
        if self._loaded: return
        with open(cfg.CANDIDATE_META_PATH, "r", encoding="utf-8") as f: self._meta = json.load(f)
        N = self._meta["n_candidates"]
        self._role_vecs = np.memmap(cfg.ROLE_VECTORS_PATH, dtype="float32", mode="r", shape=(N, cfg.EMBED_DIM))
        self._cap_vecs  = np.memmap(cfg.CAP_VECTORS_PATH,  dtype="float32", mode="r", shape=(N, cfg.EMBED_DIM))
        with open(cfg.ROLE_BM25_PATH, "rb") as f: self._role_bm25 = pickle.load(f)
        with open(cfg.CAP_BM25_PATH,  "rb") as f: self._cap_bm25  = pickle.load(f)
        self._behaviors = np.load(cfg.BEHAVIORS_PATH)
        self._loaded    = True
    @property
    def meta(self): self.load(); return self._meta
    @property
    def role_vecs(self): self.load(); return self._role_vecs
    @property
    def cap_vecs(self): self.load(); return self._cap_vecs
    @property
    def role_bm25(self): self.load(); return self._role_bm25
    @property
    def cap_bm25(self): self.load(); return self._cap_bm25
    @property
    def behaviors(self): self.load(); return self._behaviors
    @property
    def n(self) -> int: return self.meta["n_candidates"]
    def candidate_id(self, idx: int) -> str: return self.meta["index_to_id"][idx]
    def snapshot(self, cid: str) -> dict: return self.meta["snapshots"][cid]

def _dense_topk(qvec: np.ndarray, corpus: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = corpus @ qvec
    k = min(k, len(scores))
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top, scores[top]

def _bm25_topk(model, tokens: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    if not tokens: return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    scores = np.array(model.get_scores(tokens), dtype=np.float32)
    k = min(k, int((scores > 0).sum()), len(scores))
    if k == 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top, scores[top]

def _normalise(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0: return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo: return np.ones_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)

def _rrf_fuse(lists: list[tuple[np.ndarray, np.ndarray]], k: int, rrf_k: int = cfg.RRF_K) -> tuple[np.ndarray, np.ndarray]:
    acc: dict[int, float] = {}
    for indices, _ in lists:
        for rank, idx in enumerate(indices):
            acc[int(idx)] = acc.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
    if not acc: return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    all_idx = np.array(list(acc.keys()), dtype=np.int64)
    all_sc  = np.array(list(acc.values()), dtype=np.float32)
    k = min(k, len(all_idx))
    top = np.argpartition(all_sc, -k)[-k:]
    top = top[np.argsort(all_sc[top])[::-1]]
    return all_idx[top], all_sc[top]

def retrieve(
    jd_role_vec:    np.ndarray,
    jd_cap_vec:     np.ndarray,
    jd_role_tokens: list[str],
    jd_cap_tokens:  list[str],
    store:          ArtifactStore,
    top_k_rrf:      int = cfg.TOP_K_RRF,
    role_weight:    float = cfg.CATEGORY_WEIGHT_PROFILES["default"]["role_weight"],
    cap_weight:     float = cfg.CATEGORY_WEIGHT_PROFILES["default"]["cap_weight"],
    jd_profile      = None,
) -> RetrievalResult:
    rd_idx, rd_sc = _dense_topk(jd_role_vec, store.role_vecs, cfg.TOP_K_DENSE)
    cd_idx, cd_sc = _dense_topk(jd_cap_vec,  store.cap_vecs,  cfg.TOP_K_DENSE)
    rb_idx, rb_sc = _bm25_topk(store.role_bm25, jd_role_tokens, cfg.TOP_K_SPARSE)
    cb_idx, cb_sc = _bm25_topk(store.cap_bm25,  jd_cap_tokens,  cfg.TOP_K_SPARSE)

    # Adaptive Fallback Routing
    if jd_profile and jd_profile.category_low_confidence:
        fused_idx, fused_sc = _rrf_fuse([(rb_idx, rb_sc), (cb_idx, cb_sc)], k=top_k_rrf)
    else:
        fused_idx, fused_sc = _rrf_fuse([(rd_idx, rd_sc), (cd_idx, cd_sc), (rb_idx, rb_sc), (cb_idx, cb_sc)], k=top_k_rrf)

    # Extract raw structural scores for downstream re-ranking calculation
    full_rd = store.role_vecs @ jd_role_vec
    full_cd = store.cap_vecs  @ jd_cap_vec
    full_rb = np.zeros(store.n, dtype=np.float32)
    full_cb = np.zeros(store.n, dtype=np.float32)
    if len(rb_idx): full_rb[rb_idx] = _normalise(rb_sc)
    if len(cb_idx): full_cb[cb_idx] = _normalise(cb_sc)

    return RetrievalResult(
        indices=fused_idx,
        rrf_scores=fused_sc,
        role_scores=full_rd[fused_idx].astype(np.float32),
        cap_scores=full_cd[fused_idx].astype(np.float32),
        bm25_role_scores=full_rb[fused_idx],
        bm25_cap_scores=full_cb[fused_idx],
        candidate_ids=[store.candidate_id(int(i)) for i in fused_idx],
    )