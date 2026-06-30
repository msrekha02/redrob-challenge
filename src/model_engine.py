"""
model_engine.py — ONNX INT8 Model Serving Layer
=================================================
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

log = logging.getLogger(__name__)

def _detect_pooling_mode(model_dir: Path) -> str:
    cfg_path = model_dir / "1_Pooling" / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                pooling_cfg = json.load(f)
            if pooling_cfg.get("pooling_mode_cls_token"): return "cls"
            if pooling_cfg.get("pooling_mode_mean_tokens"): return "mean"
        except (json.JSONDecodeError, OSError): pass
    return "mean"

def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
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


class ONNXEmbedder:
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
        if not onnx_path.exists(): return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._pooling_mode = _detect_pooling_mode(self._model_dir)
            self._using_onnx = True
        except Exception:
            self._session = None
            self._using_onnx = False

    def _load_fallback(self):
        if self._st_fallback is None:
            from sentence_transformers import SentenceTransformer
            path = str(self._model_dir) if self._model_dir.exists() else self._fallback_model_id
            self._st_fallback = SentenceTransformer(path)
        return self._st_fallback

    @property
    def using_onnx(self) -> bool: return self._using_onnx

    def encode(self, texts: str | list[str], batch_size: int = 128, show_progress_bar: bool = False, normalize_embeddings: bool = True, convert_to_numpy: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)

        if self._session is None:
            result = self._load_fallback().encode(text_list, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=show_progress_bar, convert_to_numpy=True).astype(np.float32)
            return result[0] if single else result

        all_vecs: list[np.ndarray] = []
        valid_inputs = _onnx_input_names(self._session)
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="np")
            # Explicitly cast to numpy array ensuring memory alignment for ONNX
            onnx_inputs = {k: np.array(v) for k, v in enc.items() if k in valid_inputs}
            outputs = self._session.run(None, onnx_inputs)
            last_hidden = outputs[0]
            if self._pooling_mode == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                pooled = _mean_pool(last_hidden, enc["attention_mask"])
            all_vecs.append(_l2_normalize(pooled))
        result = np.vstack(all_vecs).astype(np.float32)
        return result[0] if single else result


class ONNXCrossEncoder:
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
        if not onnx_path.exists(): return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._using_onnx = True
        except Exception:
            self._session = None
            self._using_onnx = False

    def _load_fallback(self):
        if self._st_fallback is None:
            from sentence_transformers import CrossEncoder
            path = str(self._model_dir) if self._model_dir.exists() else self._fallback_model_id
            try: self._st_fallback = CrossEncoder(path, max_length=512)
            except Exception: self._st_fallback = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        return self._st_fallback

    @property
    def using_onnx(self) -> bool: return self._using_onnx

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        if self._session is None:
            return np.array(self._load_fallback().predict(pairs, batch_size=batch_size, show_progress_bar=show_progress_bar), dtype=np.float32)

        all_scores: list[np.ndarray] = []
        valid_inputs = _onnx_input_names(self._session)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            texts_a, texts_b = [p[0] for p in batch], [p[1] for p in batch]
            enc = self._tokenizer(texts_a, texts_b, padding=True, truncation=True, max_length=512, return_tensors="np")
            # Explicitly cast to numpy array ensuring memory alignment for ONNX
            onnx_inputs = {k: np.array(v) for k, v in enc.items() if k in valid_inputs}
            outputs = self._session.run(None, onnx_inputs)
            logits = outputs[0]
            batch_scores = logits[:, 0] if logits.ndim > 1 else logits
            all_scores.append(batch_scores.astype(np.float32))
        return np.concatenate(all_scores)