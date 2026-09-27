"""
run_train.py — Submission #2 / #3: Full LightGBM Pipeline
==========================================================

Trains LightGBM on extracted features, with:
  - 5-fold stratified CV (country × match-count)
  - Hard-negative mining (ratio 1:3:1 pos:hard-neg:random-neg)
  - Isotonic probability calibration on OOF predictions
  - Macro F_0.5 threshold sweep on calibrated OOF
  - Inference on test set

Usage:
    # First time (trains from scratch):
    python run_train.py

    # Predict only (model already trained):
    python run_train.py --predict-only
"""
import sys, io, os, time, json, pickle, shutil, random, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'code/business_entity_resolution/src')

import importlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import normalize as sk_norm

# Tee: write to both stdout and a log file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            if f is sys.__stdout__ or f is sys.__stderr__:
                try:
                    f.write(data)
                except UnicodeEncodeError:
                    f.write(data.encode('ascii', errors='replace').decode('ascii'))
            else:
                f.write(data)
            f.flush()
    def flush(self):
        for f in self.files: f.flush()

log_file = open('train_run.log', 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

_n  = importlib.import_module('01_normalize')
_b  = importlib.import_module('02_blocking')
_f  = importlib.import_module('03_features')
_t  = importlib.import_module('04_train')
_p  = importlib.import_module('05_predict')

normalize_sources      = _n.normalize_sources
run_blocking           = _b.run_blocking
build_tfidf            = _b.build_tfidf
write_candidate_pairs  = _b.write_candidate_pairs
build_feature_matrix   = _f.build_feature_matrix
df_to_lookup           = _f.df_to_lookup
FEATURE_NAMES          = _f.FEATURE_NAMES
run_training           = _t.run_training
load_model_artefacts   = _t.load_model_artefacts
sweep_threshold        = _t.sweep_threshold
macro_f05_from_pairs   = _t.macro_f05_from_pairs
predict_with_model     = _p.predict_with_model
monitor_conflicts      = _p.monitor_conflicts
write_matching_results = _p.write_matching_results

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--predict-only', action='store_true')
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--n-folds', type=int, default=5)
parser.add_argument('--no-b2', action='store_true')
parser.add_argument('--low-memory', action='store_true',
                    help='Process train/test sequentially to stay under 30 GB RAM '
                         '(recommended for Kaggle free tier). Automatically disables B2.')
args = parser.parse_args()

if args.low_memory:
    print('[LOW-MEMORY MODE] Train and test data will be processed sequentially.')
    print('[LOW-MEMORY MODE] MinHash LSH (B2) disabled to save RAM.')
    args.no_b2 = True

OUT_DIR   = Path('output')
MODEL_DIR = Path('models')
DATA_DIR  = Path('dataset')
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

t_total = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Normalize all data
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 60)
print('[Step 1] Normalizing ...')

if args.low_memory and not args.predict_only:
    # In low-memory mode: load train first, run train pipeline, then free RAM,
    # then load test for blocking + inference. Never hold both in RAM together.
    print('  [LOW-MEM] Loading TRAIN data only for now ...')
    train_srcs = normalize_sources(DATA_DIR / 'train', split='train')
    s1_train   = train_srcs['s1']
    s23_train  = pd.concat([train_srcs['s2'], train_srcs['s3']], ignore_index=True)
    del train_srcs
    gc.collect()
    print(f'Train — S1: {len(s1_train):,}  S23: {len(s23_train):,}')
    # Placeholders — will be loaded later in Step 6
    s1_test  = None
    s23_test = None
else:
    test_srcs  = normalize_sources(DATA_DIR / 'test',  split='test')
    train_srcs = normalize_sources(DATA_DIR / 'train', split='train')
    s1_test   = test_srcs['s1']
    s23_test  = pd.concat([test_srcs['s2'],  test_srcs['s3']],  ignore_index=True)
    del test_srcs
    s1_train  = train_srcs['s1']
    s23_train = pd.concat([train_srcs['s2'], train_srcs['s3']], ignore_index=True)
    del train_srcs
    gc.collect()
    print(f'Train — S1: {len(s1_train):,}  S23: {len(s23_train):,}')
    print(f'Test  — S1: {len(s1_test):,}   S23: {len(s23_test):,}')

# Load ground truth
gt_df = pd.read_csv(DATA_DIR / 'train' / 'train_ground_truth.tsv',
                    sep='\t', encoding='utf-8')
ground_truth = {}
for _, row in gt_df.iterrows():
    raw = row['matched_entity_ids']
    ground_truth[row['source1_entity_id']] = (
        set(str(raw).split(',')) if pd.notna(raw) and str(raw).strip() else set()
    )

if not args.predict_only:
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Block on training data
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[Step 2] Blocking on training data ...')
    train_cand_cache = OUT_DIR / 'candidate_pairs_train.tsv'

    if train_cand_cache.exists():
        print(f'  Loading cached train candidates ...')
        cdf = pd.read_csv(train_cand_cache, sep='\t', encoding='utf-8')
        cands_train = {}
        for _, row in cdf.iterrows():
            raw = row['candidate_entity_ids']
            cands_train[row['source1_entity_id']] = (
                set(str(raw).split(',')) if pd.notna(raw) and str(raw).strip() else set()
            )
    else:
        cands_train, vec_train = run_blocking(
            s1_train, s23_train,
            top_k=args.top_k,
            run_b2=not args.no_b2,
            run_b3=True, run_b4=True, run_b5=True,
        )
        write_candidate_pairs(cands_train, train_cand_cache)
        with open(MODEL_DIR / 'tfidf_vec_train.pkl', 'wb') as f:
            pickle.dump(vec_train, f)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Compute TF-IDF scores for training pairs (for hard-neg weighting)
    # ─────────────────────────────────────────────────────────────────────────
    print('\n[Step 3] Computing TF-IDF scores for training pairs ...')
    # Refit or reload vectorizer
    vec_path = MODEL_DIR / 'tfidf_vec_train.pkl'
    if vec_path.exists():
        with open(vec_path, 'rb') as f:
            vec_train = pickle.load(f)
    else:
        all_texts = list(s1_train['name_addr'].fillna('')) + list(s23_train['name_addr'].fillna(''))
        vec_train = build_tfidf(all_texts)
        with open(vec_path, 'wb') as f:
            pickle.dump(vec_train, f)

    print('  Transforming training S1 (for tfidf scores) ...')
    X_tr_s1    = sk_norm(vec_train.transform(s1_train['name_addr'].fillna('')), norm='l2')
    s1_tr_idx  = {eid: i for i, eid in enumerate(s1_train['entity_id'])}
    # Use text-lookup for S23 to avoid holding 10M-row matrix in RAM
    s23_tr_text = dict(zip(s23_train['entity_id'], s23_train['name_addr'].fillna('')))

    print('  Building tfidf_scores dict for candidate pairs ...')
    tfidf_scores_train = {}
    all_train_pairs = [(s1id, cid)
                       for s1id, cands in cands_train.items()
                       for cid in cands]
    BATCH = 20_000
    from tqdm import tqdm
    for b in tqdm(range(0, len(all_train_pairs), BATCH), desc='  Scoring train pairs'):
        batch     = all_train_pairs[b:b+BATCH]
        s1_rows   = [s1_tr_idx.get(p[0]) for p in batch]
        s23_texts = [s23_tr_text.get(p[1], '') for p in batch]
        valid     = [(k, s1_rows[k]) for k in range(len(batch)) if s1_rows[k] is not None]
        if not valid:
            continue
        s23_chunk = sk_norm(vec_train.transform([s23_texts[k] for k, _ in valid]), norm='l2')
        for idx, (k, r1) in enumerate(valid):
            score = float(X_tr_s1[r1].dot(s23_chunk[idx].T).toarray().flatten()[0])
            tfidf_scores_train[batch[k]] = score

    print(f'  Scored {len(tfidf_scores_train):,} training pairs')

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Build feature matrix
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[Step 4] Building feature matrix (training) ...')

    s1_lookup_train  = df_to_lookup(s1_train)
    s23_lookup_train = df_to_lookup(s23_train)
    country_map      = dict(zip(s1_train['entity_id'], s1_train['country_clean']))
    all_s1_train_ids = s1_train['entity_id'].tolist()

    X_train, y_train, pair_ids_train = build_feature_matrix(
        cands_train,
        s1_lookup_train,
        s23_lookup_train,
        ground_truth=ground_truth,
        tfidf_scores=tfidf_scores_train,
        neg_ratio_hard=3,
        neg_ratio_random=1,
        n_workers=max(1, (os.cpu_count() or 2) - 1),
        chunk_size=50_000,
    )
    print(f'  Feature matrix: {X_train.shape}  pos_rate={y_train.mean():.3f}')
    # Save for reproducibility
    np.save(MODEL_DIR / 'X_train.npy', X_train)
    np.save(MODEL_DIR / 'y_train.npy', y_train)
    with open(MODEL_DIR / 'pair_ids_train.pkl', 'wb') as f:
        pickle.dump(pair_ids_train, f)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 — Train LightGBM (5-fold CV + isotonic calibration)
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[Step 5] Training LightGBM ...')

    results = run_training(
        X_train, y_train, pair_ids_train,
        ground_truth, country_map, all_s1_train_ids,
        model_dir=MODEL_DIR,
        n_folds=args.n_folds,
        feature_names=FEATURE_NAMES,
    )
    print(f'\n  OOF macro F_0.5 = {results["oof_f05"]:.4f}')
    print(f'  Best threshold  = {results["best_threshold"]}')

    if args.low_memory:
        # Free ALL train data from RAM before loading test data
        print('\n  [LOW-MEM] Freeing train data from RAM ...')
        del s1_train, s23_train, s1_lookup_train, s23_lookup_train
        del X_train, y_train, pair_ids_train, cands_train
        del X_tr_s1, tfidf_scores_train, all_train_pairs, vec_train
        gc.collect()
        gc.collect()  # double-collect to flush cyclic refs
        print('  [LOW-MEM] Train RAM freed. Loading test data ...')
        test_srcs = normalize_sources(DATA_DIR / 'test', split='test')
        s1_test   = test_srcs['s1']
        s23_test  = pd.concat([test_srcs['s2'], test_srcs['s3']], ignore_index=True)
        del test_srcs
        gc.collect()
        print(f'  Test — S1: {len(s1_test):,}  S23: {len(s23_test):,}')

else:
    print('  --predict-only: skipping training, loading saved model.')

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Block on test set
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('[Step 6] Blocking on test set ...')

test_cand_cache = OUT_DIR / 'candidate_pairs_test.tsv'
if test_cand_cache.exists():
    print(f'  Loading cached test candidates ...')
    cdf = pd.read_csv(test_cand_cache, sep='\t', encoding='utf-8')
    cands_test = {}
    for _, row in cdf.iterrows():
        raw = row['candidate_entity_ids']
        cands_test[row['source1_entity_id']] = (
            set(str(raw).split(',')) if pd.notna(raw) and str(raw).strip() else set()
        )
else:
    cands_test, _ = run_blocking(
        s1_test, s23_test,
        top_k=args.top_k,
        run_b2=not args.no_b2,
        run_b3=True, run_b4=True, run_b5=True,
    )
    write_candidate_pairs(cands_test, test_cand_cache)

all_s1_test_ids = s1_test['entity_id'].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Build feature matrix for test candidates
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('[Step 7] Building feature matrix (test) ...')

# Load vec for tfidf scores
vec_path = MODEL_DIR / 'tfidf_vec_train.pkl'
if vec_path.exists():
    with open(vec_path, 'rb') as f:
        vec_for_test = pickle.load(f)
    X_te_s1    = sk_norm(vec_for_test.transform(s1_test['name_addr'].fillna('')), norm='l2')
    s1_te_idx  = {eid: i for i, eid in enumerate(s1_test['entity_id'])}
    # Use text-lookup to avoid holding 10M-row S23 matrix in RAM
    s23_te_text = dict(zip(s23_test['entity_id'], s23_test['name_addr'].fillna('')))
    tfidf_scores_test = {}
    all_test_pairs = [(s1id, cid) for s1id, cands in cands_test.items() for cid in cands]
    for b in tqdm(range(0, len(all_test_pairs), BATCH), desc='  Scoring test pairs'):
        batch     = all_test_pairs[b:b+BATCH]
        s1r       = [s1_te_idx.get(p[0]) for p in batch]
        s23_texts = [s23_te_text.get(p[1], '') for p in batch]
        valid     = [(k, s1r[k]) for k in range(len(batch)) if s1r[k] is not None]
        if not valid:
            continue
        s23_chunk = sk_norm(vec_for_test.transform([s23_texts[k] for k, _ in valid]), norm='l2')
        for idx, (k, r1) in enumerate(valid):
            score = float(X_te_s1[r1].dot(s23_chunk[idx].T).toarray().flatten()[0])
            tfidf_scores_test[batch[k]] = score
else:
    tfidf_scores_test = None

s1_lookup_test  = df_to_lookup(s1_test)
s23_lookup_test = df_to_lookup(s23_test)

X_test, _, pair_ids_test = build_feature_matrix(
    cands_test, s1_lookup_test, s23_lookup_test,
    ground_truth=None,
    tfidf_scores=tfidf_scores_test,
    n_workers=max(1, (os.cpu_count() or 2) - 1),
)
print(f'  Test feature matrix: {X_test.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Predict + output
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('[Step 8] Predicting ...')

model, calibrator, train_results = load_model_artefacts(MODEL_DIR)
threshold = train_results['best_threshold']
print(f'  Using threshold: {threshold}  (OOF F_0.5: {train_results["oof_f05"]:.4f})')

matching = predict_with_model(X_test, pair_ids_test, model, calibrator,
                               threshold, all_s1_test_ids)

# Conflict monitor
diag = monitor_conflicts(matching)
if diag['alert']:
    print(f'  ⚠ Multi-claim alert — consider raising threshold')
    # Auto-bump threshold if multi-claim > 15%
    if diag['multi_claimed_pct'] > 15:
        new_t = min(threshold + 0.05, 0.80)
        print(f'  Auto-bumping threshold from {threshold:.3f} → {new_t:.3f}')
        matching = predict_with_model(X_test, pair_ids_test, model, calibrator,
                                       new_t, all_s1_test_ids)
        train_results['best_threshold'] = new_t

write_matching_results(matching, all_s1_test_ids, OUT_DIR / 'matching_results.tsv')
shutil.copy(test_cand_cache, OUT_DIR / 'candidate_pairs.tsv')

elapsed = (time.time() - t_total) / 60
print(f'\n{"=" * 60}')
print(f'DONE in {elapsed:.1f} min')
print(f'\nValidate:')
print(f'  python utils/validate_submission.py \\')
print(f'    --matching output/matching_results.tsv \\')
print(f'    --candidate output/candidate_pairs.tsv \\')
print(f'    --test-dir dataset/test')
print(f'{"=" * 60}')

log_file.close()
