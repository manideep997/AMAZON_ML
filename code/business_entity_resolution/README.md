# Business Entity Resolution — Reproduction Guide

## Overview

End-to-end pipeline for Amazon ML Challenge 2026: Business Entity Resolution.

**Task:** Match records from Source 2 and Source 3 to each Source 1 entity using only the provided training data.

**Metric:** Macro-averaged F_β (β=0.5) — precision-weighted.

---

## Environment Setup

```bash
pip install -r requirements.txt
```

**Python version:** 3.10+

**Key dependencies:**
- `lightgbm>=4.3.0` — matching model
- `scikit-learn>=1.4.0` — TF-IDF, calibration, CV
- `rapidfuzz>=3.6.0` — fast string similarity (C-optimized)
- `jellyfish>=1.0.0` — Soundex phonetic blocking
- `indic-transliteration>=2.3.0` — Devanagari/Kannada → IAST romanization
- `anyascii>=0.3.2` — Unicode → ASCII fallback (French accents, etc.)
- `datasketch>=1.6.4` — MinHash LSH blocking

---

## Running the Pipeline

**All commands are run from the `student_resource/` directory.**

### Submission #1 — Baseline (TF-IDF cosine, no ML, ~60-90 min)

```bash
python run_baseline.py
```

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`

### Submission #2 / #3 — Full LightGBM Pipeline (~3-4 hours)

```bash
python run_train.py
```

To skip training and use a saved model:

```bash
python run_train.py --predict-only
```

### Validate before submitting

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## Pipeline Architecture

```
Raw TSVs
   │
   ▼ 01_normalize.py (vectorized pandas str operations, ~2 min for 12M rows)
   │  - Legal suffix expansion (Pvt→Private, Ltd→Limited, &→and, etc.)
   │  - Address abbreviation expansion (Rd→Road, Blvd→Boulevard, etc.)
   │  - Indic script → IAST romanization (indic-transliteration)
   │  - anyascii fallback (French accents, remaining non-ASCII)
   │  - ZIP token extraction for blocking
   │  - Cached as Parquet after first run
   │
   ▼ 02_blocking.py (5 keys, OR-union, ~40-60 min for full test set)
   │  B1: TF-IDF char-ngram cosine, chunked sparse matmul, top-50/S1
   │  B2: MinHash LSH on name trigrams (Jaccard threshold=0.25)
   │  B3: ZIP/PIN exact + 4-digit prefix merge (vectorized)
   │  B4: Country + name-prefix merge (vectorized)
   │  B5: Soundex phonetic key merge (vectorized)
   │  → target: ≥99% blocking recall, avg ~50 candidates/S1
   │
   ▼ 03_features.py (15 features, rapidfuzz + multiprocessing)
   │  Name: token Jaccard, trigram Jaccard, edit distance, Jaro-Winkler,
   │        TF-IDF cosine (reused from blocking), Soundex match, token sort ratio
   │  Addr: token Jaccard, edit distance, numeric overlap, ZIP exact, ZIP prefix
   │  Meta: country match, source pair (S2 vs S3), name length ratio
   │
   ▼ 04_train.py (LightGBM with 5-fold stratified CV)
   │  - Hard-negative mining: 1:3:1 ratio (pos:hard-neg:random-neg)
   │    Hard negatives sampled proportional to TF-IDF blocking score
   │  - 5-fold CV stratified by (country × match-count bucket)
   │  - Isotonic probability calibration on OOF predictions
   │  - Macro F_0.5 threshold sweep [0.10, 0.85] on calibrated OOF scores
   │
   ▼ 05_predict.py
      - LightGBM inference on test candidate pairs
      - Isotonic calibration applied
      - Threshold from training applied
      - Auto-bump if multi-claim rate > 15%
      - Conflict monitoring (multi-claimed S23 records)
```

---

## Source File Index

| File | Purpose |
|---|---|
| `run_baseline.py` | Submission #1: TF-IDF cosine baseline |
| `run_train.py` | Submission #2+: Full LightGBM pipeline |
| `src/01_normalize.py` | Vectorized text normalization |
| `src/02_blocking.py` | 5-key blocking / candidate generation |
| `src/03_features.py` | 15-feature engineering per candidate pair |
| `src/04_train.py` | LightGBM training, CV, calibration |
| `src/05_predict.py` | Inference, output writing, conflict monitoring |

---

## Reproducing Output

1. Place `dataset/train/` and `dataset/test/` as provided
2. `pip install -r requirements.txt`
3. `python run_baseline.py` → `output/matching_results.tsv` (Submission #1)
4. `python run_train.py` → `output/matching_results.tsv` (Submission #2+)
5. Validate with `utils/validate_submission.py`

Parquet caches are created in `dataset/train/` and `dataset/test/` after first run.
Delete `*_norm.parquet` files to force re-normalization.

---

## Constraints Satisfied

| Constraint | How |
|---|---|
| Model ≤ 8B params | LightGBM (no size limit in practice; feature vectors are 15-dim) |
| MIT/Apache 2.0 license | LightGBM (MIT), rapidfuzz (MIT), jellyfish (MIT), indic-transliteration (MIT), anyascii (Apache-2.0), scikit-learn (BSD-3) |
| No external data/APIs | All computation is deterministic and local |
| France (zero-shot) | Country is an open-set string feature; no hardcoding |
