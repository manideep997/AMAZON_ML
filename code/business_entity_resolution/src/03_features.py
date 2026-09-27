"""
03_features.py — Feature Engineering for Candidate Pairs

Computes 15 features per (S1, S2/S3) candidate pair using rapidfuzz (C-optimized)
and jellyfish. Designed for scale: batched multiprocessing over chunks of pairs.

Features:
  Name (7):
    f01  token_jaccard_name         — word-level Jaccard
    f02  trigram_jaccard_name       — char-trigram Jaccard
    f03  edit_dist_name             — Levenshtein normalized similarity
    f04  jaro_winkler_name          — Jaro-Winkler similarity
    f05  tfidf_cosine_name_addr     — TF-IDF cosine from blocking (reused)
    f06  soundex_match              — Soundex of first word matches (binary)
    f07  token_sort_ratio           — rapidfuzz token_sort_ratio (handles word-order)

  Address (5):
    f08  token_jaccard_addr         — word-level Jaccard on address
    f09  edit_dist_addr             — Levenshtein normalized similarity on address
    f10  numeric_token_overlap      — intersection of numeric tokens (house no / PIN)
    f11  zip_exact_match            — any shared ZIP/PIN token (binary)
    f12  zip_prefix_match           — any shared 4-digit ZIP/PIN prefix (binary)

  Meta (3):
    f13  country_match              — exact country string match (binary)
    f14  source_pair                — 0 = S1-S2, 1 = S1-S3
    f15  name_len_ratio             — min/max of name token lengths (similar length = higher)
"""

import re
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    from rapidfuzz import distance as rf_dist
    from rapidfuzz.fuzz import token_sort_ratio as rf_token_sort
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    print('WARNING: rapidfuzz not installed. Feature computation will be slow.')

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _trigram_jaccard(a: str, b: str) -> float:
    def tgrams(s):
        s = '_' + s.replace(' ', '_') + '_'
        return set(s[i:i+3] for i in range(len(s)-2)) if len(s) >= 3 else set()
    ta, tb = tgrams(a), tgrams(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _edit_sim(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        return rf_dist.Levenshtein.normalized_similarity(a, b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Pure Python fallback (slow)
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (ca != cb), dp[j+1] + 1, ndp[-1] + 1))
        dp = ndp
    return 1.0 - dp[-1] / max(la, lb)


def _jaro_winkler(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        return rf_dist.JaroWinkler.normalized_similarity(a, b)
    if _HAS_JELLYFISH:
        try:
            return jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            return 0.0
    return 0.0


def _token_sort(a: str, b: str) -> float:
    if _HAS_RAPIDFUZZ:
        return rf_token_sort(a, b) / 100.0
    return _token_jaccard(a, b)


def _soundex_match(a: str, b: str) -> float:
    if not _HAS_JELLYFISH:
        return 0.0
    wa = a.split()
    wb = b.split()
    if not wa or not wb:
        return 0.0
    try:
        return float(jellyfish.soundex(wa[0]) == jellyfish.soundex(wb[0]))
    except Exception:
        return 0.0


def _numeric_tokens(addr: str) -> set:
    return set(re.findall(r'\b\d{3,}\b', addr))


def _numeric_overlap(a: str, b: str) -> float:
    ta, tb = _numeric_tokens(a), _numeric_tokens(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def _zip_exact(a_zips: list, b_zips: list) -> float:
    return float(bool(set(a_zips) & set(b_zips)))


def _zip_prefix(a_zips: list, b_zips: list) -> float:
    a4 = {z[:4] for z in a_zips}
    b4 = {z[:4] for z in b_zips}
    return float(bool(a4 & b4))


def _name_len_ratio(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    if max(la, lb) == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


# ─────────────────────────────────────────────────────────────────────────────
# Single pair → feature vector
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'f01_token_jaccard_name',
    'f02_trigram_jaccard_name',
    'f03_edit_dist_name',
    'f04_jaro_winkler_name',
    'f05_tfidf_cosine',
    'f06_soundex_match',
    'f07_token_sort_name',
    'f08_token_jaccard_addr',
    'f09_edit_dist_addr',
    'f10_numeric_token_overlap',
    'f11_zip_exact',
    'f12_zip_prefix',
    'f13_country_match',
    'f14_source_pair',
    'f15_name_len_ratio',
]


def compute_pair_features(
    s1_row: dict,
    s23_row: dict,
    tfidf_score: float = 0.0,
) -> list[float]:
    """
    Compute all 15 features for a single (S1, S2/S3) candidate pair.

    Parameters:
        s1_row:       dict-like with keys: norm_name, norm_addr, zip_tokens, country_clean
        s23_row:      same
        tfidf_score:  pre-computed TF-IDF cosine from blocking step (f05)
    """
    nn1, nn2 = s1_row['norm_name'],  s23_row['norm_name']
    na1, na2 = s1_row['norm_addr'],  s23_row['norm_addr']
    z1,  z2  = s1_row['zip_tokens'], s23_row['zip_tokens']
    c1,  c2  = s1_row['country_clean'], s23_row['country_clean']

    # Determine source pair: S1-S2=0, S1-S3=1
    src_pair = 0.0 if s23_row['entity_id'].startswith('S2-') else 1.0

    return [
        _token_jaccard(nn1, nn2),       # f01
        _trigram_jaccard(nn1, nn2),     # f02
        _edit_sim(nn1, nn2),            # f03
        _jaro_winkler(nn1, nn2),        # f04
        float(tfidf_score),             # f05
        _soundex_match(nn1, nn2),       # f06
        _token_sort(nn1, nn2),          # f07
        _token_jaccard(na1, na2),       # f08
        _edit_sim(na1, na2),            # f09
        _numeric_overlap(na1, na2),     # f10
        _zip_exact(z1, z2),             # f11
        _zip_prefix(z1, z2),            # f12
        float(c1 == c2),               # f13
        src_pair,                       # f14
        _name_len_ratio(nn1, nn2),      # f15
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Batch feature computation (multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_batch(args):
    """Worker function: compute features for a batch of pair tuples."""
    pair_batch = args
    results = []
    for s1r, s23r, tfidf_score, label in pair_batch:
        feat = compute_pair_features(s1r, s23r, tfidf_score)
        results.append((s1r['entity_id'], s23r['entity_id'], label, feat))
    return results


def build_feature_matrix(
    candidates: dict[str, set],
    s1_lookup: dict,     # entity_id → row dict
    s23_lookup: dict,    # entity_id → row dict
    ground_truth: dict | None = None,   # s1_id → set of true match ids (None = inference)
    tfidf_scores: dict | None = None,   # (s1_id, s23_id) → float
    neg_ratio_hard: int = 3,
    neg_ratio_random: int = 1,
    n_workers: int = max(1, os.cpu_count() - 1),
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """
    Build feature matrix X and label vector y for training OR inference.

    Training mode (ground_truth is not None):
      - Positive pairs: (s1, true_match)
      - Hard negatives: blocking candidates that are NOT true matches,
                        sampled proportionally to tfidf_score
      - Random negatives: random S2/S3 from the whole pool

    Inference mode (ground_truth is None):
      - All candidates are included with label=0 (ignored at inference)

    Returns:
      X: (n_pairs, 15) float32 array
      y: (n_pairs,) int8 array  (0/1 for training, all 0 for inference)
      pair_ids: list of (s1_id, s23_id) tuples (in same order as X rows)
    """
    from multiprocessing import Pool
    import random

    all_s23_ids = list(s23_lookup.keys())
    pairs = []    # list of (s1_row, s23_row, tfidf_score, label)

    print(f'  Building pairs (n_workers={n_workers}) ...')

    for s1_id, cands in candidates.items():
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue

        if ground_truth is not None:
            # ─── Training mode ─────────────────────────────────────────────
            true_matches = ground_truth.get(s1_id, set())

            # Positive pairs
            for match_id in true_matches:
                s23_row = s23_lookup.get(match_id)
                if s23_row is None:
                    continue
                score = (tfidf_scores or {}).get((s1_id, match_id), 0.0)
                pairs.append((s1_row, s23_row, score, 1))

            # Hard negatives (blocking candidates that are wrong)
            hard_neg_pool = list(cands - true_matches)
            if hard_neg_pool and true_matches:
                # Weight by tfidf score (harder negatives first)
                if tfidf_scores:
                    weights = np.array([
                        (tfidf_scores or {}).get((s1_id, cid), 0.01)
                        for cid in hard_neg_pool
                    ], dtype=np.float32)
                    weights = weights / weights.sum()
                else:
                    weights = None

                n_hard = min(len(hard_neg_pool), len(true_matches) * neg_ratio_hard)
                chosen_hard = np.random.choice(
                    len(hard_neg_pool),
                    size=n_hard,
                    replace=False,
                    p=weights,
                )
                for idx in chosen_hard:
                    s23_id = hard_neg_pool[idx]
                    s23_row = s23_lookup.get(s23_id)
                    if s23_row is None:
                        continue
                    score = (tfidf_scores or {}).get((s1_id, s23_id), 0.0)
                    pairs.append((s1_row, s23_row, score, 0))

            # Random negatives (broad diversity)
            n_rand = len(true_matches) * neg_ratio_random
            rand_ids = random.sample(all_s23_ids, min(n_rand, len(all_s23_ids)))
            for s23_id in rand_ids:
                if s23_id in true_matches or s23_id in cands:
                    continue
                s23_row = s23_lookup.get(s23_id)
                if s23_row is None:
                    continue
                pairs.append((s1_row, s23_row, 0.0, 0))

        else:
            # ─── Inference mode ────────────────────────────────────────────
            for s23_id in cands:
                s23_row = s23_lookup.get(s23_id)
                if s23_row is None:
                    continue
                score = (tfidf_scores or {}).get((s1_id, s23_id), 0.0)
                pairs.append((s1_row, s23_row, score, 0))

    print(f'  Total pairs to feature: {len(pairs):,}')

    # Chunk into batches for multiprocessing
    batches = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]

    X_rows, y_rows, pair_ids = [], [], []

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for batch_result in pool.imap_unordered(_worker_batch, batches):
                for s1_id, s23_id, label, feat in batch_result:
                    X_rows.append(feat)
                    y_rows.append(label)
                    pair_ids.append((s1_id, s23_id))
    else:
        from tqdm import tqdm
        for batch in tqdm(batches, desc='  Features'):
            for s1r, s23r, tfidf_score, label in batch:
                feat = compute_pair_features(s1r, s23r, tfidf_score)
                X_rows.append(feat)
                y_rows.append(label)
                pair_ids.append((s1r['entity_id'], s23r['entity_id']))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int8)
    return X, y, pair_ids


def df_to_lookup(df: pd.DataFrame) -> dict:
    """Convert DataFrame to dict of entity_id → row_dict for fast lookup."""
    records = df.to_dict('records')
    return {r['entity_id']: r for r in records}


if __name__ == '__main__':
    # Quick smoke test on 100 dummy pairs
    s1r  = {'entity_id': 'S1-1', 'norm_name': 'ram marketing private limited',
             'norm_addr': '123 road delhi', 'zip_tokens': ['110001'],
             'country_clean': 'india'}
    s23r = {'entity_id': 'S2-1', 'norm_name': 'ram marketing pvt ltd',
            'norm_addr': '123 road new delhi', 'zip_tokens': ['110001'],
            'country_clean': 'india'}
    feats = compute_pair_features(s1r, s23r, tfidf_score=0.72)
    for name, val in zip(FEATURE_NAMES, feats):
        print(f'  {name:<30} {val:.4f}')
