"""
model_engine.py — ONNX INT8 Model Serving Layer
=================================================
Architecture requirement (stated twice in the spec): both the embedding
model and the cross-encoder run as ONNX INT8, not raw PyTorch, for CPU
speed and artifact-size reasons under the 5-minute / 16GB budget.

Two classes, both with the SAME design pattern:
    1. Try to load a quantized ONNX model from disk (model_int8.onnx).
    2. If that file doesn't exist, or onnxruntime isn't installed, or
       loading fails for any reason — silently fall back to the PyTorch
       sentence-transformers path that was already proven correct.

This makes ONNX a pure performance optimization layer: if the export step
in setup_models.py never ran, or ran on a machine without AVX2 support, or
optimum/onnxruntime aren't installed, the system still produces correct
rankings — just slower. Correctness is never gated on ONNX working.

Pooling-strategy correctness note:
    BGE models' correct pooling mode (CLS-token vs mean-pooling) is read
    directly from the sentence-transformers model's own 1_Pooling/config.json
    rather than hardcoded, since getting this wrong silently produces worse
    (but not obviously broken) embeddings. Reading the model's own config is
    the only way to be certain this matches what the already-correct PyTorch
    SentenceTransformer path does internally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_pooling_mode(model_dir: Path) -> str:
    """
    Read the model's own sentence-transformers pooling config.
    Returns 'cls' or 'mean'. Defaults to 'mean' if no config is found —
    the standard, safe default for most transformer embedding models.
    """
    cfg_path = model_dir / "1_Pooling" / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                pooling_cfg = json.load(f)
            if pooling_cfg.get("pooling_mode_cls_token"):
                return "cls"
            if pooling_cfg.get("pooling_mode_mean_tokens"):
                return "mean"
        except (json.JSONDecodeError, OSError):
            pass
    return "mean"


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Attention-mask-weighted mean pooling. last_hidden: (B,T,H), mask: (B,T)."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1e-9, norm)
    return (vecs / norm).astype(np.float32)


def _onnx_input_names(session) -> set[str]:
    return {inp.name for inp in session.get_inputs()}


# ─────────────────────────────────────────────────────────────────────────────
# ONNXEmbedder
# ─────────────────────────────────────────────────────────────────────────────

class ONNXEmbedder:
    """
    Drop-in replacement for SentenceTransformer.encode(), backed by an
    ONNX Runtime INT8 session when available.

    Usage (identical call signature to the SentenceTransformer path it replaces):
        embedder = ONNXEmbedder(model_dir, fallback_model_id)
        vecs = embedder.encode(["text one", "text two"])   # (2, 384) float32, L2-normalised
    """

    def __init__(self, model_dir: str | Path, fallback_model_id: str) -> None:
        self._model_dir = Path(model_dir)
        self._fallback_model_id = fallback_model_id
        self._tokenizer = None
        self._session: Any = None
        self._pooling_mode: str | None = None
        self._st_fallback = None
        self._using_onnx = False
        self._try_load_onnx()

    def _try_load_onnx(self) -> None:
        onnx_path = self._model_dir / "model_int8.onnx"
        if not onnx_path.exists():
            log.info("No ONNX INT8 artifact at %s — using PyTorch fallback", onnx_path)
            return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._pooling_mode = _detect_pooling_mode(self._model_dir)
            self._using_onnx = True
            log.info(
                "ONNX INT8 embedder loaded: %s (pooling=%s)",
                onnx_path.name, self._pooling_mode,
            )
        except Exception as e:
            log.warning(
                "ONNX embedder load failed (%s) — falling back to PyTorch. "
                "Rankings will still be correct, just slower.", e,
            )
            self._session = None
            self._using_onnx = False

    def _load_fallback(self):
        if self._st_fallback is None:
            from sentence_transformers import SentenceTransformer
            path = str(self._model_dir) if self._model_dir.exists() else self._fallback_model_id
            log.info("Loading PyTorch fallback embedding model: %s", path)
            self._st_fallback = SentenceTransformer(path)
        return self._st_fallback

    @property
    def using_onnx(self) -> bool:
        return self._using_onnx

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 128,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,   # kept for call-signature parity; always normalised
        convert_to_numpy: bool = True,        # kept for call-signature parity; always numpy
    ) -> np.ndarray:
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)

        if self._session is None:
            result = self._load_fallback().encode(
                text_list, batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=show_progress_bar, convert_to_numpy=True,
            ).astype(np.float32)
            return result[0] if single else result

        all_vecs: list[np.ndarray] = []
        valid_inputs = _onnx_input_names(self._session)
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="np",
            )
            onnx_inputs = {k: np.asarray(v) for k, v in enc.items() if k in valid_inputs}
            # np.asarray() is a defensive, near-zero-cost cast: return_tensors="np"
            # already requests genuine np.ndarray from the tokenizer, so this is a
            # no-op in the normal case, but it guarantees session.run() never receives
            # a list/tensor type from a tokenizer implementation or version that
            # behaves differently than expected.
            outputs = self._session.run(None, onnx_inputs)
            last_hidden = outputs[0]   # (batch, seq_len, hidden)

            if self._pooling_mode == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                pooled = _mean_pool(last_hidden, enc["attention_mask"])

            all_vecs.append(_l2_normalize(pooled))

        result = np.vstack(all_vecs).astype(np.float32)
        return result[0] if single else result


# ─────────────────────────────────────────────────────────────────────────────
# ONNXCrossEncoder
# ─────────────────────────────────────────────────────────────────────────────

class ONNXCrossEncoder:
    """
    Drop-in replacement for sentence_transformers.CrossEncoder.predict(),
    backed by an ONNX Runtime INT8 session when available.

    Usage:
        ce = ONNXCrossEncoder(model_dir, fallback_model_id)
        scores = ce.predict([(jd_text, cand_text), ...])   # (N,) float32
    """

    def __init__(self, model_dir: str | Path, fallback_model_id: str) -> None:
        self._model_dir = Path(model_dir)
        self._fallback_model_id = fallback_model_id
        self._tokenizer = None
        self._session: Any = None
        self._st_fallback = None
        self._using_onnx = False
        self._try_load_onnx()

    def _try_load_onnx(self) -> None:
        onnx_path = self._model_dir / "model_int8.onnx"
        if not onnx_path.exists():
            log.info("No ONNX INT8 artifact at %s — using PyTorch fallback", onnx_path)
            return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._using_onnx = True
            log.info("ONNX INT8 cross-encoder loaded: %s", onnx_path.name)
        except Exception as e:
            log.warning(
                "ONNX cross-encoder load failed (%s) — falling back to PyTorch. "
                "Rankings will still be correct, just slower.", e,
            )
            self._session = None
            self._using_onnx = False

    def _load_fallback(self):
        if self._st_fallback is None:
            from sentence_transformers import CrossEncoder
            path = str(self._model_dir) if self._model_dir.exists() else self._fallback_model_id
            log.info("Loading PyTorch fallback cross-encoder: %s", path)
            try:
                self._st_fallback = CrossEncoder(path, max_length=512)
            except Exception as e:
                log.warning("Primary cross-encoder failed (%s) — MiniLM fallback", e)
                self._st_fallback = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        return self._st_fallback

    @property
    def using_onnx(self) -> bool:
        return self._using_onnx

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if self._session is None:
            return np.array(
                self._load_fallback().predict(
                    pairs, batch_size=batch_size, show_progress_bar=show_progress_bar,
                ),
                dtype=np.float32,
            )

        all_scores: list[np.ndarray] = []
        valid_inputs = _onnx_input_names(self._session)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            texts_a = [p[0] for p in batch]
            texts_b = [p[1] for p in batch]
            enc = self._tokenizer(
                texts_a, texts_b, padding=True, truncation=True, max_length=512, return_tensors="np",
            )
            onnx_inputs = {k: np.asarray(v) for k, v in enc.items() if k in valid_inputs}
            # np.asarray() is a defensive, near-zero-cost cast: return_tensors="np"
            # already requests genuine np.ndarray from the tokenizer, so this is a
            # no-op in the normal case, but it guarantees session.run() never receives
            # a list/tensor type from a tokenizer implementation or version that
            # behaves differently than expected.
            outputs = self._session.run(None, onnx_inputs)
            logits = outputs[0]
            batch_scores = logits[:, 0] if logits.ndim > 1 else logits
            all_scores.append(batch_scores.astype(np.float32))

        return np.concatenate(all_scores)