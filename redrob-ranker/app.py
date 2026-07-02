#!/usr/bin/env python3
"""
app.py — Redrob Ranker Sandbox (HuggingFace Spaces)
=====================================================
This file contains ZERO ranking logic.
It is pure UI plumbing: collect an input file, call the existing
offline_pipeline.py and rank.py as subprocesses, return the CSV.

Keeping it this way means the code under evaluation in the sandbox
is bit-for-bit identical to the code in the GitHub repo — exactly
what Stage 3 requires.
"""

import sys
import subprocess
import time
import shutil
from pathlib import Path

import gradio as gr
import pandas as pd

# ── Paths (must match config.py's BASE_DIR derivation) ────────────────────────
BASE_DIR          = Path("/app")
SRC_DIR           = BASE_DIR / "src"
DATA_DIR          = BASE_DIR / "data"
ARTIFACTS_DIR     = BASE_DIR / "artifacts"
DEFAULT_CANDIDATES = DATA_DIR / "candidates_sample.jsonl"
OUTPUT_CSV         = BASE_DIR / "submission.csv"


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(candidates_file):
    """
    Two-phase pipeline, each phase is a subprocess call to the real scripts.

    Phase A: offline_pipeline.py — honeypot filter → dual-track embeddings
             → BM25 indexes → behavioral tensor → candidate metadata
    Phase B: rank.py — JD ingestion → RRF retrieval → penalty gate (20 signals)
             → BGE cross-encoder → honeypot safety net → ranked CSV

    Returns: (log_text, preview_dataframe_or_None, csv_path_or_None)
    """
    # ── Input resolution ───────────────────────────────────────────────────────
    if candidates_file is not None:
        candidates_path = candidates_file.name
        source_label = f"uploaded file ({Path(candidates_path).name})"
    else:
        candidates_path = str(DEFAULT_CANDIDATES)
        source_label = f"pre-loaded sample ({DEFAULT_CANDIDATES.name})"

    log_lines: list[str] = [
        "━" * 56,
        f"  Redrob Candidate Ranker — Sandbox Run",
        "━" * 56,
        f"  Candidates source : {source_label}",
        f"  JD                : data/job_description.docx (bundled default)",
        f"  Python            : {sys.version.split()[0]}",
        "━" * 56,
    ]

    def _run(cmd: list[str], label: str) -> int:
        log_lines.append(f"\n▶  {label}")
        log_lines.append("─" * 48)
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        elapsed = time.perf_counter() - t0
        combined = (result.stdout + result.stderr).strip()
        if combined:
            # Indent each log line for readability in the Gradio textbox
            for line in combined.splitlines():
                log_lines.append("   " + line)
        status_char = "✓" if result.returncode == 0 else "✗"
        log_lines.append(f"\n{status_char}  Completed in {elapsed:.1f}s  (return code {result.returncode})")
        return result.returncode

    # ── Phase A ────────────────────────────────────────────────────────────────
    # Clear any stale artifacts from a previous run so we start fresh.
    # Without this, a second run with different candidates would use the
    # first run's memmapped vectors and produce wrong results.
    for stale in ARTIFACTS_DIR.glob("*.npy"):
        stale.unlink(missing_ok=True)
    for stale in ARTIFACTS_DIR.glob("*.pkl"):
        stale.unlink(missing_ok=True)
    for stale in ["candidate_meta.json"]:
        p = ARTIFACTS_DIR / stale
        if p.exists():
            p.unlink()

    rc = _run(
        [
            sys.executable,
            str(SRC_DIR / "offline_pipeline.py"),
            "--candidates", candidates_path,
        ],
        "Phase A — Offline artifacts (honeypot filter → embeddings → BM25 → behavioral tensor)",
    )
    if rc != 0:
        log_lines.append("\n✗  Phase A failed — see log above for details.")
        return "\n".join(log_lines), None, None

    # ── Phase B ────────────────────────────────────────────────────────────────
    rc = _run(
        [
            sys.executable,
            str(SRC_DIR / "rank.py"),
            "--out", str(OUTPUT_CSV),
        ],
        "Phase B — Ranking (JD → retrieval → penalty gate → cross-encoder → CSV)",
    )
    if rc != 0:
        log_lines.append("\n✗  Phase B failed — see log above for details.")
        return "\n".join(log_lines), None, None

    # ── Output ─────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(OUTPUT_CSV)
        log_lines.append(
            f"\n✓  Done — {len(df)} candidates ranked, submission.csv ready."
        )
        log_lines.append(
            "   Download the full CSV below, or preview the top rows in the table."
        )
        return "\n".join(log_lines), df.head(20), str(OUTPUT_CSV)
    except Exception as exc:
        log_lines.append(f"\n✗  Could not read output CSV: {exc}")
        return "\n".join(log_lines), None, None


# ── Gradio UI ──────────────────────────────────────────────────────────────────

_DESCRIPTION = """
## Redrob Intelligent Candidate Ranker — Sandbox

Upload a `candidates.jsonl` file (≤ 100 candidates) **or** click **Run** to use
the pre-loaded 100-candidate sample bundled with this space.

The full two-phase pipeline runs end-to-end:

| Phase | What runs |
|-------|-----------|
| A (offline) | Honeypot filter → BGE-small dual-track embeddings → dual BM25 index → behavioral tensor |
| B (online)  | JD ingestion → dual-track RRF retrieval (1000) → 20-signal penalty gate (200) → BGE cross-encoder (100) → CSV |

**Expected runtime:** 60–120 s on this CPU-only sandbox for ≤ 100 candidates.
"""

with gr.Blocks(title="Redrob Candidate Ranker", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(
                label="candidates.jsonl  (optional — ≤ 100 candidates)",
                file_types=[".jsonl"],
            )
            run_btn = gr.Button("▶  Run Ranking", variant="primary", size="lg")
            gr.Markdown(
                "_If no file is uploaded, the pre-loaded 100-candidate sample is used._"
            )

        with gr.Column(scale=2):
            log_box = gr.Textbox(
                label="Pipeline log",
                lines=22,
                max_lines=50,
                show_copy_button=True,
            )

    with gr.Row():
        results_table = gr.DataFrame(
            label="Top-20 preview (ranked by score)",
            wrap=True,
        )

    with gr.Row():
        download_btn = gr.File(
            label="⬇  Download full submission.csv",
        )

    run_btn.click(
        fn=run_pipeline,
        inputs=[upload],
        outputs=[log_box, results_table, download_btn],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )