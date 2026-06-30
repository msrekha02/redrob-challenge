"""
setup_models.py — One-Time Model Download + ONNX INT8 Export
===============================================================
"""

from __future__ import annotations
import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SETUP] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# 1. Add the Extractor Model Constants
JD_EXTRACTOR_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def download_model(model_id: str, save_dir: Path, model_type: str = "encoder") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    if model_type == "encoder":
        log.info("Downloading embedding model: %s → %s", model_id, save_dir)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        model.save(str(save_dir))
    elif model_type == "cross-encoder":
        log.info("Downloading cross-encoder: %s → %s", model_id, save_dir)
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_id, max_length=512)
        model.save(str(save_dir))

def export_to_onnx_int8(model_dir: Path, model_type: str) -> bool:
    onnx_target = model_dir / "model_int8.onnx"
    if onnx_target.exists():
        return True
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        return False

    onnx_tmp_dir = model_dir / "_onnx_export_tmp"
    try:
        if model_type == "encoder":
            ort_model = ORTModelForFeatureExtraction.from_pretrained(str(model_dir), export=True)
        else:
            ort_model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), export=True)
        ort_model.save_pretrained(str(onnx_tmp_dir))
        quantizer = ORTQuantizer.from_pretrained(onnx_tmp_dir)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=str(model_dir), quantization_config=qconfig)
        quantized_candidates = sorted(model_dir.glob("*quantized*.onnx"))
        if not quantized_candidates: return False
        quantized_candidates[0].rename(onnx_target)
        for leftover in model_dir.glob("*.onnx"):
            if leftover != onnx_target: leftover.unlink(missing_ok=True)
        return True
    except Exception as e:
        return False
    finally:
        shutil.rmtree(onnx_tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models + export to ONNX INT8")
    parser.add_argument("--models-dir", default=str(cfg.MODELS_DIR), help="Directory to save models")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export, PyTorch only")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    embed_dir = models_dir / "bge-small-en-v1.5"
    if not (embed_dir.exists() and list(embed_dir.glob("*.json"))):
        download_model(model_id=cfg.BGE_EMBED_MODEL_ID, save_dir=embed_dir, model_type="encoder")

    reranker_dir = models_dir / "bge-reranker-base"
    if not (reranker_dir.exists() and list(reranker_dir.glob("*.json"))):
        download_model(model_id=cfg.BGE_RERANKER_MODEL_ID, save_dir=reranker_dir, model_type="cross-encoder")

    # 2. Include the Model in the Download Loop
    extractor_dir = models_dir / "jd-extractor"
    if not (extractor_dir.exists() and list(extractor_dir.glob("*.json"))):
        download_model(model_id=JD_EXTRACTOR_MODEL_ID, save_dir=extractor_dir, model_type="cross-encoder")

    if not args.skip_onnx:
        embed_ok    = export_to_onnx_int8(embed_dir, "encoder")
        reranker_ok = export_to_onnx_int8(reranker_dir, "cross-encoder")
        # 2. Include the Model in the ONNX Export Loop
        extractor_ok = export_to_onnx_int8(extractor_dir, "cross-encoder")

if __name__ == "__main__":
    main()