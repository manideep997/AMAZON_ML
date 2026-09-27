# 🏆 Amazon ML Challenge 2026 — Business Entity Resolution

> **Challenge:** Given business records from 3 independent data sources with noisy and inconsistent fields, determine which records across sources refer to the same real-world business entity.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Problem Overview](#-problem-overview)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset Setup](#-dataset-setup)
- [Running the Pipeline](#-running-the-pipeline)
- [Validating Output](#-validating-output)
- [Evaluation Metric](#-evaluation-metric)
- [Dependencies](#-dependencies)

---

## 🎯 Problem Overview

Large-scale commercial platforms receive business identity data from multiple independent sources — each contributing partial, noisy fragments about the same real-world entities. These fragments share **no common identifiers**.

**Task:** Match records from Source 2 and Source 3 to each Source 1 entity.

- **Source 1** is the deduplicated reference source
- A Source 1 entity may match **zero, one, or many** records from Source 2 and Source 3
- Noise patterns include: name abbreviations, address variations, transliterations, typos, and format differences
- Training data covers **US** and **India**; test set additionally includes **France** (zero-shot)

---

## 📂 Repository Structure

```
student_resource/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── 01_normalize.py     # Text normalization (legal suffix, address abbrev, transliteration)
│       │   ├── 02_blocking.py      # 5-key candidate generation (TF-IDF, MinHash LSH, ZIP, Soundex)
│       │   ├── 03_features.py      # 15-feature engineering (rapidfuzz + multiprocessing)
│       │   ├── 04_train.py         # LightGBM with 5-fold CV + isotonic calibration
│       │   └── 05_predict.py       # Inference, threshold application, output writing
│       ├── README.md               # Reproduction guide
│       └── requirements.txt        # Pinned dependencies
├── dataset/                        # ⚠️ Not tracked in git — download separately (see below)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/                         # Generated after running pipeline
│   ├── matching_results.tsv        # ← Submit this to the leaderboard
│   └── candidate_pairs.tsv         # Blocking candidate set (for audit)
├── utils/
│   └── validate_submission.py      # Local validation (stdlib only, no deps)
├── run_baseline.py                 # Submission #1: TF-IDF cosine baseline (~60-90 min)
├── run_train.py                    # Submission #2+: Full LightGBM pipeline (~3-4 hrs)
├── smoke_test.py                   # Quick sanity check on a mini dataset slice
└── Documentation_template.md       # Methodology write-up template
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/manideep997/AMAZON_ML.git
cd AMAZON_ML
```

### 2. Set up environment (Python 3.10+)

```bash
# Create and activate virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

### 3. Place your dataset files

Download the challenge dataset and place files as follows:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv

dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

> ⚠️ All files are **tab-separated** (`.tsv`). Always use `sep="\t"` when reading with pandas:
> ```python
> df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
> ```

### 4. Run the pipeline

**All commands are run from the `student_resource/` directory.**

```bash
# Option A: Baseline (TF-IDF cosine, no ML, ~60-90 min)
python run_baseline.py

# Option B: Full LightGBM pipeline (~3-4 hours)
python run_train.py

# Option C: Skip training, use a saved model for prediction only
python run_train.py --predict-only
```

---

## 🏗️ Pipeline Architecture

```
Raw TSVs
   │
   ▼  01_normalize.py  (~2 min for 12M rows)
   │  • Legal suffix expansion: Pvt→Private, Ltd→Limited, &→and
   │  • Address abbreviations: Rd→Road, Blvd→Boulevard, St→Street
   │  • Indic script → IAST romanization (indic-transliteration)
   │  • Unicode → ASCII fallback (anyascii) — handles French accents
   │  • ZIP token extraction for blocking
   │  • Cached as Parquet after first run
   │
   ▼  02_blocking.py  (~40-60 min for full test set)
   │  B1: TF-IDF char-ngram cosine (chunked sparse matmul, top-50/S1)
   │  B2: MinHash LSH on name trigrams (Jaccard threshold=0.25)
   │  B3: ZIP/PIN exact + 4-digit prefix merge (vectorized)
   │  B4: Country + name-prefix merge (vectorized)
   │  B5: Soundex phonetic key merge (vectorized)
   │  → Target: ≥99% blocking recall, avg ~50 candidates/S1
   │
   ▼  03_features.py  (15 features, rapidfuzz + multiprocessing)
   │  Name:    token Jaccard, trigram Jaccard, edit distance,
   │           Jaro-Winkler, TF-IDF cosine, Soundex match, token sort ratio
   │  Address: token Jaccard, edit distance, numeric overlap,
   │           ZIP exact, ZIP prefix
   │  Meta:    country match, source pair (S2 vs S3), name length ratio
   │
   ▼  04_train.py  (LightGBM + 5-fold stratified CV)
   │  • Hard-negative mining: 1:3:1 ratio (pos:hard-neg:random-neg)
   │    Hard negatives sampled proportional to TF-IDF blocking score
   │  • 5-fold CV stratified by (country × match-count bucket)
   │  • Isotonic probability calibration on OOF predictions
   │  • Macro F_0.5 threshold sweep [0.10, 0.85] on calibrated OOF scores
   │
   ▼  05_predict.py
      • LightGBM inference on test candidate pairs
      • Isotonic calibration applied
      • Threshold from training applied
      • Auto-bump if multi-claim rate > 15%
      • Conflict monitoring (multi-claimed S2/S3 records)
```

---

## 📊 Dataset Setup

The dataset files are **not tracked in this repository** (too large for GitHub). They should be obtained from the Amazon ML Challenge portal.

| File | Size (approx) | Description |
|---|---|---|
| `train_source1.tsv` | ~200 MB | Reference source (train) |
| `train_source2.tsv` | ~470 MB | Source 2 (train) |
| `train_source3.tsv` | ~480 MB | Source 3 (train) |
| `train_ground_truth.tsv` | ~121 MB | Ground truth labels |
| `test_source1.tsv` | ~167 MB | Reference source (test) |
| `test_source2.tsv` | ~486 MB | Source 2 (test) |
| `test_source3.tsv` | ~483 MB | Source 3 (test) |

> **Parquet caches** (`*_norm.parquet`) are generated automatically after the first normalization run. Delete them to force re-normalization.

---

## ✅ Validating Output

Before submitting to the leaderboard, validate your output files locally:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Prints `PASS` (exit 0) if all rules are satisfied, or a numbered list of issues (exit 1).

---

## 📏 Evaluation Metric

Submissions are evaluated using **F_β Score (β = 0.5)** — precision-heavy (precision weighted 2× over recall):

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed as a **macro-average** over all Source 1 entities (singletons included).

**Example:**
- Predicted: `[S2-00047, S2-00193, S3-00812]`
- Ground truth: `[S2-00047, S3-00812]`
- Precision = 2/3, Recall = 1.0 → **F_0.5 = 0.714**

---

## 📦 Output Format

Two tab-separated files in `output/`:

**`matching_results.tsv`** ← upload to leaderboard
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**`candidate_pairs.tsv`** ← blocking audit
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

---

## 🔧 Dependencies

```
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
rapidfuzz>=3.6.0
jellyfish>=1.0.0
tqdm>=4.66.0
pyarrow>=15.0.0
indic-transliteration>=2.3.0
anyascii>=0.3.2
datasketch>=1.6.4
```

Install with:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

**License compliance:** All models/libraries are MIT or Apache-2.0 licensed as required by the challenge constraints (≤8B parameters, MIT/Apache-2.0).

---

## 🧪 Smoke Test

A quick sanity check that runs the pipeline on a mini slice of data:

```bash
python smoke_test.py
```

---

## ⚠️ Academic Integrity

This solution uses **only the provided training data** — no external databases, APIs, geocoding services, or internet lookups. All computation is deterministic and local.
