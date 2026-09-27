"""
run_baseline.py — Submission #1: Fast TF-IDF Baseline
======================================================
Uses full 5-key blocking + cosine threshold tuned on training validation.
No ML model needed.

Usage:  python run_baseline.py
"""
import sys, os, time, pickle, json, shutil, random, gc
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as sk_norm
from tqdm import tqdm

sys.path.insert(0, 'code/business_entity_resolution/src')
import importlib
_n = importlib.import_module('01_normalize')
_b = importlib.import_module('02_blocking')

normalize_sources     = _n.normalize_sources
run_blocking          = _b.run_blocking
build_tfidf           = _b.build_tfidf
write_candidate_pairs = _b.write_candidate_pairs

OUT_DIR  = Path('output')
DATA_DIR = Path('dataset')
OUT_DIR.mkdir(exist_ok=True)

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

log_file = open('baseline_run.log', 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

t_total = time.time()

# ─────────────────────────────────────────────────────────────────────────────
print('=' * 60)
print(f'Baseline pipeline started: {time.strftime("%H:%M:%S")}')
print('=' * 60)

# ── Step 1: Normalize (SEQUENTIAL — load train first, test later to save RAM) ──
print('\n[Step 1] Normalizing data ...')
t0 = time.time()
# Load TRAIN only — test data loaded later after train RAM is freed
train_srcs = normalize_sources(DATA_DIR / 'train', split='train')
s1_train  = train_srcs['s1']
s23_train = pd.concat([train_srcs['s2'], train_srcs['s3']], ignore_index=True)
del train_srcs
gc.collect()
print(f'Train — S1: {len(s1_train):,}  S23: {len(s23_train):,}')
print(f'Normalize (train) done in {(time.time()-t0)/60:.1f} min')

# ── Step 2: Load ground truth + sample val set ────────────────────────────────
print('\n[Step 2] Loading ground truth + sampling val set ...')
gt_df = pd.read_csv(DATA_DIR / 'train' / 'train_ground_truth.tsv',
                    sep='\t', encoding='utf-8')
ground_truth = {}
for _, row in gt_df.iterrows():
    raw = row['matched_entity_ids']
    ground_truth[row['source1_entity_id']] = (
        set(str(raw).split(',')) if pd.notna(raw) and str(raw).strip() else set()
    )
del gt_df
gc.collect()
print(f'Ground truth loaded: {len(ground_truth):,} S1 entities')

random.seed(42)
VAL_N   = 30_000
val_ids = random.sample(list(ground_truth.keys()), min(VAL_N, len(ground_truth)))
val_gt  = {k: ground_truth[k] for k in val_ids}
val_s1  = s1_train[s1_train['entity_id'].isin(set(val_ids))].copy()
print(f'Val sample: {len(val_ids):,} S1 entities')

# Subsample S23 for val tuning — 2M rows is plenty for threshold calibration
# This avoids the 10M-row TF-IDF transform that OOMs on machines with <30GB
S23_VAL_SAMPLE = 2_000_000
if len(s23_train) > S23_VAL_SAMPLE:
    # Include all records that are true matches for our val sample
    true_match_ids = set(mid for s in val_gt.values() for mid in s)
    s23_true = s23_train[s23_train['entity_id'].isin(true_match_ids)]
    n_random  = S23_VAL_SAMPLE - len(s23_true)
    s23_rest  = s23_train[~s23_train['entity_id'].isin(true_match_ids)]
    s23_rand  = s23_rest.sample(min(n_random, len(s23_rest)), random_state=42)
    s23_val   = pd.concat([s23_true, s23_rand], ignore_index=True)
    print(f'S23 val subset: {len(s23_val):,} rows (includes {len(s23_true):,} true matches)')
    del s23_true, s23_rest, s23_rand
    gc.collect()
else:
    s23_val = s23_train

# ── Step 3: Block on val set (for threshold tuning) ────────────────────────────
print('\n[Step 3] Blocking on val set (for threshold tuning) ...')
t0 = time.time()
val_cands, val_vec = run_blocking(
    val_s1, s23_val,
    top_k=50,
    run_b2=False,   # skip slow MinHash for quick tuning
    run_b3=True, run_b4=True, run_b5=True,
)
print(f'Val blocking done in {(time.time()-t0)/60:.1f} min')

# Blocking recall check
recall_n, recall_d = 0, 0
for s1_id in val_ids:
    true = val_gt.get(s1_id, set())
    cand = val_cands.get(s1_id, set())
    recall_n += len(true & cand)
    recall_d += len(true)
br = recall_n / recall_d if recall_d > 0 else 0.0
print(f'Blocking recall @ top-50 (val): {br:.4f}')

# ── Step 4: Score val pairs + sweep threshold ─────────────────────────────────
print('\n[Step 4] Scoring val candidate pairs ...')
t0 = time.time()

X_val_s1  = sk_norm(val_vec.transform(val_s1['name_addr'].fillna('')), norm='l2')
X_val_s23 = sk_norm(val_vec.transform(s23_val['name_addr'].fillna('')), norm='l2')
s1_val_idx  = {eid: i for i, eid in enumerate(val_s1['entity_id'])}
s23_val_idx = {eid: i for i, eid in enumerate(s23_val['entity_id'])}

# Build flat pair list
val_pairs  = [(s1_id, cid) for s1_id, cands in val_cands.items() for cid in cands]
val_labels = [1 if cid in val_gt.get(s1_id, set()) else 0
              for s1_id, cid in val_pairs]
print(f'Val pairs: {len(val_pairs):,}  positives: {sum(val_labels):,}')

# Score in batches
BATCH = 20_000
val_scores = np.zeros(len(val_pairs), dtype=np.float32)
for b in tqdm(range(0, len(val_pairs), BATCH), desc='  Scoring val', unit='batch'):
    batch = val_pairs[b:b+BATCH]
    s1rs  = [s1_val_idx.get(p[0])  for p in batch]
    s23rs = [s23_val_idx.get(p[1]) for p in batch]
    valid = [(k, s1rs[k], s23rs[k]) for k in range(len(batch))
             if s1rs[k] is not None and s23rs[k] is not None]
    if not valid:
        continue
    _, vr1, vr2 = zip(*valid)
    scores = np.array(X_val_s1[list(vr1)].multiply(X_val_s23[list(vr2)]).sum(axis=1)).flatten()
    for j, (k, _, _) in enumerate(valid):
        val_scores[b + k] = scores[j]

print(f'Scoring done in {(time.time()-t0):.1f}s')

# Sweep threshold
print('\nSweeping threshold [0.10 → 0.70] ...')
def macro_f05_fast(pairs, scores, gt, all_ids, thresh):
    pred = defaultdict(set)
    for (s1_id, cid), sc in zip(pairs, scores):
        if sc >= thresh:
            pred[s1_id].add(cid)
    f_vals = []
    for s1_id in all_ids:
        p, t = pred.get(s1_id, set()), gt.get(s1_id, set())
        if not t and not p:   f_vals.append(1.0)
        elif not t:           f_vals.append(0.0)
        else:
            tp = len(p & t)
            pr = tp / len(p) if p else 0.0
            rc = tp / len(t)
            dn = 0.25 * pr + rc
            f_vals.append(1.25 * pr * rc / dn if dn else 0.0)
    return float(np.mean(f_vals))

best_t, best_f = 0.35, 0.0
for t in np.arange(0.10, 0.71, 0.02):
    f = macro_f05_fast(val_pairs, val_scores, val_gt, val_ids, float(t))
    marker = ' ← best' if f > best_f else ''
    print(f'  threshold={t:.2f}  F_0.5={f:.4f}{marker}')
    if f > best_f:
        best_f, best_t = f, float(t)

print(f'\nBest threshold: {best_t:.2f}  (val F_0.5 = {best_f:.4f})')
with open(OUT_DIR / 'tuning_results.json', 'w') as fp:
    json.dump({'best_threshold': best_t, 'val_f05': best_f,
               'blocking_recall': br}, fp, indent=2)

# Free ALL train/val data before loading test — critical for Kaggle 30 GB limit
print('\n[Memory] Freeing ALL train/val data before loading test set ...')
del val_cands, val_vec, val_s1, s23_val, val_pairs, val_scores
del X_val_s1, X_val_s23, val_gt, ground_truth
del s1_train, s23_train
gc.collect()
gc.collect()  # double-pass to flush cyclic refs
print('[Memory] Train data freed. Now loading test data ...')

# Load test data NOW (after train freed) — never hold both in RAM at once
t0 = time.time()
test_srcs = normalize_sources(DATA_DIR / 'test', split='test')
s1_test   = test_srcs['s1']
s23_test  = pd.concat([test_srcs['s2'], test_srcs['s3']], ignore_index=True)
del test_srcs
gc.collect()
print(f'Test  — S1: {len(s1_test):,}  S23: {len(s23_test):,}')
print(f'Normalize (test) done in {(time.time()-t0)/60:.1f} min')

# ── Step 5: Block on test set ─────────────────────────────────────────────────
print('\n[Step 5] Blocking on test set ...')
cand_cache = OUT_DIR / 'candidate_pairs_test.tsv'

if cand_cache.exists():
    print(f'Loading cached candidates: {cand_cache}')
    cdf = pd.read_csv(cand_cache, sep='\t', encoding='utf-8')
    candidates = {}
    for _, row in cdf.iterrows():
        raw = row['candidate_entity_ids']
        candidates[row['source1_entity_id']] = (
            set(str(raw).split(',')) if pd.notna(raw) and str(raw).strip() else set()
        )
    # Rebuild vec for test scoring using test texts only
    all_texts = (list(s1_test['name_addr'].fillna('')) +
                 list(s23_test['name_addr'].fillna('')))
    test_vec = build_tfidf(all_texts)
    del all_texts; gc.collect()
else:
    t0 = time.time()
    candidates, test_vec = run_blocking(
        s1_test, s23_test,
        top_k=50,
        run_b2=False,   # MinHash LSH disabled: iterrows on 10M S23 rows = OOM + hours
        run_b3=True, run_b4=True, run_b5=True,
    )
    print(f'Test blocking done in {(time.time()-t0)/60:.1f} min')
    write_candidate_pairs(candidates, cand_cache)
    with open(OUT_DIR / 'tfidf_vec.pkl', 'wb') as fp:
        pickle.dump(test_vec, fp)

all_s1_test_ids = s1_test['entity_id'].tolist()
n_pairs = sum(len(v) for v in candidates.values())
print(f'Total test candidate pairs: {n_pairs:,}  avg/S1: {n_pairs/max(1,len(candidates)):.1f}')

# ── Step 6: Score test pairs + apply threshold ────────────────────────────────
print(f'\n[Step 6] Scoring test candidate pairs (threshold={best_t:.2f}) ...')
t0 = time.time()

# Transform S1 (small, ~1.7M × 50k sparse = manageable)
X_te_s1 = sk_norm(test_vec.transform(s1_test['name_addr'].fillna('')), norm='l2')
s1_te_idx = {eid: i for i, eid in enumerate(s1_test['entity_id'])}

# Build lookup: s23_id → text (to score on-demand without loading full matrix)
s23_text_lookup = dict(zip(s23_test['entity_id'], s23_test['name_addr'].fillna('')))

all_test_pairs = [(s1_id, cid) for s1_id, cands in candidates.items() for cid in cands]
print(f'Scoring {len(all_test_pairs):,} pairs ...')

matching = defaultdict(set)
for b in tqdm(range(0, len(all_test_pairs), BATCH), desc='  Scoring test', unit='batch'):
    batch      = all_test_pairs[b:b+BATCH]
    s1rs       = [s1_te_idx.get(p[0]) for p in batch]
    s23_texts  = [s23_text_lookup.get(p[1], '') for p in batch]
    valid      = [(k, s1rs[k]) for k in range(len(batch)) if s1rs[k] is not None]
    if not valid:
        continue
    # Transform only this batch's S23 texts (not all 10M)
    s23_chunk = sk_norm(test_vec.transform([s23_texts[k] for k, _ in valid]), norm='l2')
    for idx, (k, r1) in enumerate(valid):
        score = float(X_te_s1[r1].dot(s23_chunk[idx].T).toarray().flatten()[0])
        if score >= best_t:
            matching[batch[k][0]].add(batch[k][1])

# Ensure all S1 present
for s1_id in all_s1_test_ids:
    if s1_id not in matching:
        matching[s1_id] = set()

print(f'Scoring done in {(time.time()-t0)/60:.1f} min')

# ── Step 7: Conflict monitor ──────────────────────────────────────────────────
claim_counts = Counter()
for matches in matching.values():
    claim_counts.update(matches)
multi_claimed = sum(1 for v in claim_counts.values() if v > 1)
multi_pct = 100.0 * multi_claimed / max(1, len(claim_counts))
print(f'\n[Conflict Monitor]')
print(f'  Total S23 records matched : {len(claim_counts):,}')
print(f'  Multi-claimed (>1 S1)     : {multi_claimed:,}  ({multi_pct:.2f}%)')
if multi_pct > 15:
    print('  ⚠  High multi-claim — consider raising threshold for next submission')

# ── Step 8: Write outputs ─────────────────────────────────────────────────────
print('\n[Step 7] Writing output files ...')
rows = [{'source1_entity_id': s1_id,
         'matched_entity_ids': ','.join(sorted(matching.get(s1_id, set())))}
        for s1_id in all_s1_test_ids]
out_df = pd.DataFrame(rows, columns=['source1_entity_id', 'matched_entity_ids'])
out_df.to_csv(OUT_DIR / 'matching_results.tsv', sep='\t', index=False, encoding='utf-8')
shutil.copy(cand_cache, OUT_DIR / 'candidate_pairs.tsv')

n_nonempty = (out_df['matched_entity_ids'] != '').sum()
total_out  = out_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0).sum()
print(f'  matching_results.tsv : {len(out_df):,} rows')
print(f'  Non-empty (matched)  : {n_nonempty:,} ({100*n_nonempty/len(out_df):.1f}%)')
print(f'  Total matched pairs  : {total_out:,}')
print(f'  candidate_pairs.tsv  : copied')

elapsed = (time.time() - t_total) / 60
print(f'\n{"=" * 60}')
print(f'DONE in {elapsed:.1f} min  [{time.strftime("%H:%M:%S")}]')
print(f'\nNext — validate then submit:')
print(f'  python utils/validate_submission.py \\')
print(f'    --matching output/matching_results.tsv \\')
print(f'    --candidate output/candidate_pairs.tsv \\')
print(f'    --test-dir dataset/test')
print(f'{"=" * 60}')
log_file.close()
