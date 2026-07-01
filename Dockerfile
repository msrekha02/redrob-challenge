FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
# build-essential: needed for some pip C extensions (numpy wheels, etc.)
# git: needed by huggingface_hub's model download internals
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python env ────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/artifacts/.hf_cache

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install requirements first so this layer is cached on rebuilds
# that only change source code (not requirements.txt).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install "gradio>=4.0.0" "pandas>=2.0.0"

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/  ./src/
COPY data/ ./data/

# ── Model download — runs once at BUILD time, baked into the image ────────────
# --skip-onnx keeps build time reasonable (~3-5 min):
# downloads BGE-small (~90 MB) and BGE-reranker-base (~140 MB) as PyTorch
# models. model_engine.py's PyTorch fallback is correct; ONNX is a
# speed-only optimization and not needed for correctness on ≤100 candidates.
#
# This is what keeps ranking network-free at runtime, which is BOTH the
# Stage 3 constraint AND what makes this demo behave identically to the
# real full-scale run.
RUN python src/setup_models.py --skip-onnx

# ── Runtime directories ───────────────────────────────────────────────────────
# offline_pipeline.py writes here at runtime. Must exist and be writable.
RUN mkdir -p artifacts

# ── App ───────────────────────────────────────────────────────────────────────
COPY app.py .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

CMD ["python", "app.py"]