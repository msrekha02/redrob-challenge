# Redrob Intelligent Candidate Discovery & Ranking

A two-phase candidate ranking pipeline built for the Redrob AI Hackathon challenge.

Ranks 100,000 candidates against a job description and produces a top-100 CSV — on CPU, in under three minutes, without any external API calls.

---

## The Problem

Finding the right candidate in a pool of 100,000 profiles is not a search problem — it is a filtering problem with multiple failure modes:

- Keyword matching surfaces candidates who list the right skills but have the wrong career trajectory
- Behavioral signals (inactivity, low response rates) are invisible to pure semantic search
- Synthetic and honeypot profiles inflate pools with noise
- A candidate who built a recommendation system at a product company may never use the word "RAG" — and gets missed

A good ranking system needs to handle all of these simultaneously, within strict compute constraints, and without retraining when the job description changes.

---

## The Solution

Our Solution is a two-phase ranking pipeline that accepts any job description and any candidate pool — without retraining, without reconfiguration, and without external API calls. Drop in a new JD and the system extracts its requirements, recalibrates its scoring weights, and ranks candidates against it automatically. Replace the candidate pool and only the offline indexing step reruns. Every stage runs on CPU within the challenge's 5-minute, 16 GB constraint making it scalable and production ready code — not as a workaround, but by design.

---

## How It Works

The pipeline separates offline work (done once per candidate pool) from online work (done per job description).

```
PHASE A — offline, run once
─────────────────────────────────────────────────────
candidates.jsonl
      │
      ├── Honeypot filter (14 traps)
      │         synthetic profiles removed
      │         clean profiles remain
      │
      ├── ROLE TRACK                CAP TRACK
      │   titles · companies    skills · tools
      │   industries · edu      project desc · certs
      │        │                      │
      │   BGE-small INT8         BGE-small INT8
      │   + BM25Okapi            + BM25Okapi
      │        │                      │
      │   role_vectors.npy       cap_vectors.npy
      │   role_bm25.pkl          cap_bm25.pkl
      │
      └── behaviors.npy · candidate_meta.json

artifacts/  6 files · ~400 MB


PHASE B — online, run per JD · ≤5 min · CPU only
─────────────────────────────────────────────────────
job_description.docx
      │
      ├── JD profiling
      │   regex extraction → YoE, cities, notice period, DQ flags
      │   BGE-small encode → role vector + capability vector
      │   10-anchor softmax blend → role/cap weights
      │
      ├── Hybrid retrieval → top-1,000
      │   4 ranked lists (role dense, cap dense, role BM25, cap BM25)
      │   RRF fusion
      │
      ├── Penalty gate → top-200
      │   20 signals across 5 layers
      │   Percentile-clipped score normalisation
      │
      ├── Cross-encoder reranking → top-100
      │   BGE-reranker-base ONNX INT8
      │   Full sequence attention on (JD, candidate) pairs
      │
      └── Honeypot safety net
          14 traps re-run · backfill from 101+ if needed

submission.csv
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and export models (run once)
python src/setup_models.py

# 3. Build offline artifacts (run once per candidate pool)
python src/offline_pipeline.py --candidates ./data/candidates.jsonl

# 4. Rank candidates against the default JD
python src/rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 4b. Or rank against a different JD
python src/rank.py --jd_path ./my_jd.docx --candidates ./data/candidates.jsonl --out ./submission.csv
```

The `--jd_path` flag is optional. When omitted, the system uses the bundled default JD automatically — which is what the Stage 3 evaluator does.

---

## Repository Structure

```text
.
├── src/                              # Core source code
│   ├── config.py                     # Configuration and scoring parameters
│   ├── honeypot.py                   # Candidate validation and honeypot detection
│   ├── model_engine.py               # Embedding and cross-encoder model loader
│   ├── setup_models.py               # Downloads and prepares models
│   ├── offline_pipeline.py           # Builds offline artifacts
│   ├── jd_processor.py               # Processes job descriptions
│   ├── retriever.py                  # Hybrid candidate retrieval
│   ├── guardrails.py                 # Re-ranking, penalties, and safety checks
│   ├── reasoner.py                   # Generates ranking explanations and CSV output
│   └── rank.py                       # Main ranking pipeline entry point
│
├── data/
│   ├── candidates.jsonl              # Candidate dataset (excluded from repository)
│   └── job_description.docx          # Sample job description
│
├── artifacts/                        # Generated offline assets (gitignored)
│   ├── role_vectors.npy
│   ├── cap_vectors.npy
│   ├── role_bm25.pkl
│   ├── cap_bm25.pkl
│   ├── behaviors.npy
│   ├── candidate_meta.json
│   └── models/                       # Downloaded embedding model files
│
├── output/
│   └── submission.csv                # Final ranked candidates
│
├── redrob-ranker/                    # Hugging Face Spaces deployment
│   ├── src/
│   ├── data/
│   ├── app.py                        # Gradio application
│   └── README.md
│
├── Dockerfile
├── requirements.txt
├── submission_metadata.yaml
└── validate_submission.py
```

---

## Ranking Pipeline

### Stage 0 — Honeypot Filter

Runs offline. 14 independent trap checks cover:

- Fictional company names
- Impossible date ranges
- Proficiency contradictions (Expert skill, 0 months experience)
- Behavioral signal impossibilities (100% response rate, instant platform signup)
- Synthetic signal clusters

Removes synthetic candidates from the 100K pool. Re-runs on the final top-200 as a safety net, with backfill from rank 101+ if anything is cut.

---

### Stage 1 — Dual-Track Embedding

Candidates are indexed on two independent tracks:

| Track | Encodes | Catches |
|---|---|---|
| Role | Job titles, companies, career trajectory | Whether the career arc fits the role |
| Capability | Skills, tools, project descriptions | Whether the skill set matches the JD |

Scoring both independently allows detection of keyword stuffers — a candidate with a perfect AI skill list but a Marketing Manager career arc scores high on capability but near-zero on role. A stuffer gate (cap ≥ 0.65 AND role < 0.25 on normalised scores) applies a ×0.10 multiplier.

A candidate who "built a recommendation system at a product company" scores high on the capability track through semantic similarity, even without writing "RAG" or "Pinecone".

---

### Stage 2 — Hybrid Retrieval (1,000 candidates)

Four ranked lists are built independently:

```
role_vecs  @ jd_role_vec   argpartition O(N) → top-2,000
cap_vecs   @ jd_cap_vec    argpartition O(N) → top-2,000
role_bm25  get_scores()    argpartition O(N) → top-2,000
cap_bm25   get_scores()    argpartition O(N) → top-2,000
```

Fused via Reciprocal Rank Fusion (k=60). Score-independent fusion handles the incompatible scales between dense and sparse scores.

If the JD does not resemble any known role category well (`category_low_confidence = True`), BM25 list weights increase and dense weights decrease — BM25 is more reliable when the JD sits outside the embedding model's familiar distribution.

---

### Stage 3 — Penalty Gate (200 candidates)

Two passes:

**Pass 1** — pool-wide normalisation. Raw BGE cosine similarities cluster tightly (e.g. 0.78–0.86 across 1,000 candidates). Percentile-clipped normalisation (2nd/98th) stretches scores to [0, 1] before any multiplier is applied. This prevents behavioral signals from overwhelming semantic relevance.

**Pass 2** — per-candidate, 20 signals across 5 layers:

```
Layer 0   Hard disqualifiers
          100% consulting career          → eliminated
          100% pure research, no prod     → eliminated

Layer 1   Availability (universal)
          Platform inactivity · open to work flag
          Recruiter response rate · interview completion

Layer 2   Location (JD-conditional)
          Preferred city tiers

Layer 3   Notice period (JD-conditional)
          Matched against JD-extracted preference

Layer 4   Profile integrity (universal)
          Salary inversion · unverified identity
          Hollow expert claims · duplicate career narratives

Layer 5   JD-fit (JD-conditional)
          LangChain-only AI tourist · non-coding senior
          CV/Speech specialist without NLP/IR
          YoE out of range · job-hopping · consulting history
```

JD-conditional signals only activate when the JD text contains relevant phrases. Load a different JD — the penalties reconfigure automatically.

Score formula:

```
penalized_score = combined_dense_norm
                × domain_mult × stuffer_mult
                × consulting_mult × research_mult
                × m_avail × m_loc × m_notice
                × m_integrity × m_jd

semantic_floor  = combined_dense_norm × 0.40
penalized_score = max(penalized_score, semantic_floor)
```

The semantic floor ensures a strong semantic match is never buried by compounding soft penalties before the cross-encoder sees it.

---

### Stage 4 — Cross-Encoder Reranking

`BAAI/bge-reranker-base` reads the JD and candidate profile together in one forward pass — full sequence-to-sequence attention. Applied only to top-200 (bi-encoder retrieval handles the earlier filtering).

```
ce_score = 0.70 × CE_norm + 0.30 × penalized_score
```

Final sort is deterministic:
```
ce_score → penalized_score → behavioral_tensor → candidate_id
```

Same input always produces identical output.

---

### Stage 5 — Reasoning and Export

Each candidate gets a 1–2 sentence justification generated programmatically from their actual profile fields — no LLM, no hallucination possible.

Tone is calibrated to rank position: top 15% get confident language; bottom 30% get honest, hedged language. A rank-92 candidate never reads like a rank-3 candidate.

---

## Key Design Decisions

**Why two tracks instead of one embedding?**
A single pooled embedding cannot separately evaluate "does this person's career look like an AI engineer?" and "do they have the right technical skills?". Separating them blocks keyword stuffers at the retrieval stage, before they consume cross-encoder compute.

**Why BM25 alongside dense retrieval?**
BGE-small misses exact string matches for rare tokens ("qdrant", "PEFT", "FAISS"). BM25 guarantees those candidates are never lost from the pool.

**Why argpartition at every funnel stage?**
`np.argpartition` is O(N) compared to O(N log N) for a full sort. Applied consistently at the 2,000 / 1,000 / 200 splits. Full sort only at Stage 4, where a total order is required for the honeypot backfill.

**Why cross-encoder only on top-200?**
At 150ms per pair, running the cross-encoder on 8,269 candidates would take ~20 minutes. On 200 candidates it takes ~30 seconds. The penalty gate is designed to ensure the 200 candidates reaching Stage 4 are already the most relevant.

**Why is the behavioral tensor a tie-breaker and not a score multiplier?**
The behavioral signals (recency, response rate, notice period) are already captured with better precision by m_avail and m_notice. Multiplying the tensor into the score would apply these signals twice — verified to over-penalise inactive candidates by ~40%.

**Why is the JD processing dynamic?**
All JD parameters — YoE range, preferred cities, notice period, disqualifier conditions — are extracted from the raw JD text at runtime via regex and embedding similarity. No templates, no hardcoded fields. A different JD reconfigures the entire pipeline automatically.

---

## Technologies

| Component | Technology | Why |
|---|---|---|
| Candidate embedding | BAAI/bge-small-en-v1.5 ONNX INT8 | Best BEIR benchmark performance at its size class. Fits CPU cache. Full-precision alternatives would not complete Phase A in time. |
| Cross-encoder | BAAI/bge-reranker-base ONNX INT8 | Reads JD and candidate jointly. ONNX INT8 gives 2–4× speedup on CPU. |
| Sparse retrieval | rank-bm25 (BM25Okapi) | Exact lexical recall for rare tokens that dense models miss. |
| Vector storage | numpy memmap | O(1) random access without loading the full matrix into RAM. |
| Top-K selection | numpy argpartition | O(N) instead of O(N log N). Applied at every funnel split. |
| JD text parsing | stdlib re, python-docx | No extra model needed for structured parameter extraction. |
| Duplicate detection | stdlib difflib | SequenceMatcher at ≥0.90 threshold catches copy-pasted narratives. |
| Inference fallback | sentence-transformers | Silent PyTorch fallback if ONNX export fails. Rankings are identical. |

---

## Compute Constraints

| Constraint | Requirement | Actual |
|---|---|---|
| Runtime | ≤ 5 min | ~90–140s |
| Memory | ≤ 16 GB | ~2–3 GB |
| Compute | CPU only | ONNX INT8 |
| Network | Off during ranking | Models baked at setup time |
| Disk | ≤ 5 GB | ~400 MB artifacts |

---

## Output

`submission.csv` with four columns:

| Column | Description |
|---|---|
| `candidate_id` | Platform candidate identifier |
| `rank` | Final rank (1 = best) |
| `score` | Combined score (0–100) |
| `reasoning` | 1–2 sentence justification from actual profile fields |

---

## Evaluation Criteria

| Metric | Weight | What it measures |
|---|---|---|
| NDCG@10 | 50% | Top 10 precision and ordering |
| NDCG@50 | 30% | Quality across the full top-50 |
| MAP | 15% | Relevant candidates surfaced early |
| P@10 | 5% | At least 10 relevant candidates in top 10 |
| Honeypot rate | Hard limit | >10% honeypots in top-100 = disqualified |

---

## Sandbox

A live demo is available at:
`https://huggingface.co/spaces/msrekha/redrob-ranker`

Upload a `candidates.jsonl` file (≤ 100 candidates) or click Run to use the pre-loaded sample. The full pipeline runs end-to-end and returns a downloadable CSV.

The sandbox runs on HuggingFace Spaces `cpu-basic` (2 vCPU, 16 GB RAM) — the same resource envelope as the Stage 3 constraint.

---

## Extending the System

| Change | What to touch |
|---|---|
| Add a new ranking signal | One function in `guardrails.py` |
| Add a new role domain | One entry in `config.py ROLE_CATEGORY_ANCHORS` |
| Swap the embedding model | Two lines in `config.py` + re-run `setup_models.py` |
| Support a new JD format | One method in `jd_processor.py` |
| Change a threshold | One constant in `config.py` |

No stage knows the internal implementation of any other stage. Every boundary is a typed Python dataclass.

---

## Project Structure Philosophy

Each file has one clearly defined responsibility and is forbidden from knowing the internals of adjacent files. Stages communicate exclusively through typed dataclasses:

- `JDProfile` — carries all extracted JD parameters through Phases B
- `RetrievalResult` — carries 6 score arrays from retriever to penalty gate
- `CandidateScore[]` — carries all intermediate scores from penalty gate to reasoner
- `ArtifactStore` — shared read-only access to Phase A outputs across all Phase B stages