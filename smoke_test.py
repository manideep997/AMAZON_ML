"""
Quick end-to-end smoke test on 5000 rows from train.
Validates: normalize → blocking → scoring → output format.
Should complete in < 2 minutes.
"""
import sys, io, os, time, random, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'code/business_entity_resolution/src')

import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import normalize as sk_norm

_n = importlib.import_module('01_normalize')
_b = importlib.import_module('02_blocking')

t0 = time.time()
NROWS = 5000   # small slice

print(f'Reading {NROWS} rows from each training source ...')
s1  = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', encoding='utf-8', nrows=NROWS)
s2  = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', encoding='utf-8', nrows=NROWS)
s3  = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', encoding='utf-8', nrows=NROWS)
gt  = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', encoding='utf-8', nrows=NROWS)

# Normalize
print('Normalizing ...')
s1_n  = _n.normalize_df(s1,  desc='s1')
s2_n  = _n.normalize_df(s2,  desc='s2')
s3_n  = _n.normalize_df(s3,  desc='s3')
s23_n = pd.concat([s2_n, s3_n], ignore_index=True)
print(f'Normalize done in {time.time()-t0:.1f}s')

# Blocking — B1 + B3 + B4 + B5 only (no B2 for speed)
print('\nBlocking ...')
t1 = time.time()
candidates, vec = _b.run_blocking(s1_n, s23_n, top_k=20,
                                  run_b2=False, run_b3=True, run_b4=True, run_b5=True)
print(f'Blocking done in {time.time()-t1:.1f}s')

# Ground truth
ground_truth = {}
for _, row in gt.iterrows():
    raw = row['matched_entity_ids']
    ground_truth[row['source1_entity_id']] = (
        set(str(raw).split(',')) if pd.notna(raw) and str(raw).strip() else set()
    )

# Blocking recall
recall_n, recall_d = 0, 0
for s1_id in s1_n['entity_id']:
    true = ground_truth.get(s1_id, set())
    cand = candidates.get(s1_id, set())
    # Only count true matches that exist in our s2/s3 slice
    valid_true = true & set(s23_n['entity_id'])
    recall_n += len(valid_true & cand)
    recall_d += len(valid_true)
br = recall_n / recall_d if recall_d > 0 else 0.0
print(f'\nBlocking recall (within slice): {br:.4f}')

# Score pairs
print('\nScoring candidate pairs ...')
t2 = time.time()
X_s1  = sk_norm(vec.transform(s1_n['name_addr'].fillna('')), norm='l2')
X_s23 = sk_norm(vec.transform(s23_n['name_addr'].fillna('')), norm='l2')
s1_idx  = {eid: i for i, eid in enumerate(s1_n['entity_id'])}
s23_idx = {eid: i for i, eid in enumerate(s23_n['entity_id'])}

# Apply threshold 0.35 and compute F_0.5
threshold = 0.35
matching = defaultdict(set)
for s1_id, cands in candidates.items():
    si = s1_idx.get(s1_id)
    if si is None: continue
    for cid in cands:
        ci = s23_idx.get(cid)
        if ci is None: continue
        score = float(X_s1[si].multiply(X_s23[ci]).sum())
        if score >= threshold:
            matching[s1_id].add(cid)

# Compute F_0.5
f_scores = []
for s1_id in s1_n['entity_id']:
    pred = matching.get(s1_id, set())
    true = ground_truth.get(s1_id, set()) & set(s23_n['entity_id'])
    if not true and not pred:
        f_scores.append(1.0)
    elif not true:
        f_scores.append(0.0)
    else:
        tp   = len(pred & true)
        prec = tp / len(pred) if pred else 0.0
        rec  = tp / len(true)
        den  = 0.25 * prec + rec
        f_scores.append(1.25 * prec * rec / den if den > 0 else 0.0)

print(f'Scoring done in {time.time()-t2:.1f}s')
print(f'\n=== Mini-validation results ===')
print(f'  Blocking recall : {br:.4f}')
print(f'  Threshold       : {threshold}')
print(f'  Macro F_0.5     : {np.mean(f_scores):.4f}')
print(f'  Non-empty preds : {sum(1 for v in matching.values() if v):,} / {len(s1_n):,}')

# Check output format
print('\nTesting output format ...')
Path('output').mkdir(exist_ok=True)
rows = [{'source1_entity_id': eid,
         'matched_entity_ids': ','.join(sorted(matching.get(eid, set())))}
        for eid in s1_n['entity_id']]
out = pd.DataFrame(rows, columns=['source1_entity_id', 'matched_entity_ids'])
test_match = Path('output/test_mini_matching.tsv')
test_cand  = Path('output/test_mini_candidates.tsv')
out.to_csv(test_match, sep='\t', index=False, encoding='utf-8')
_b.write_candidate_pairs(candidates, test_cand)
print(f'  Wrote {test_match}')
print(f'  Wrote {test_cand}')

print(f'\nTotal smoke test time: {(time.time()-t0):.1f}s')
print('ALL OK — ready to launch full run_baseline.py')
