"""
02_blocking.py — Multi-Key Candidate Generation (VECTORIZED)

Blocking keys (OR-union):
  B1: TF-IDF char-ngram cosine, chunked sparse matmul, top-K per S1
  B2: MinHash LSH on name character trigrams (approximate Jaccard)
  B3: PIN/ZIP exact + 4-digit prefix hash join    ← pandas merge (fast)
  B4: Country + name-prefix hash join              ← pandas merge (fast)
  B5: Soundex phonetic key join                    ← pandas merge (fast)

B3/B4/B5 use pandas merge instead of iterrows — 100× faster on 5M rows.

MEMORY NOTE:
  B1 streams S23 in S23_CHUNK-row batches (outer loop) and processes S1 in
  S1_BLOCK-row batches (inner loop).  Each S23 chunk is transformed EXACTLY ONCE.
  Dense matmul output = S1_BLOCK × S23_CHUNK × 4B = 1k × 100k × 4B = 400MB.
"""

import sys
import re
import time
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# B1: TF-IDF Chunked Sparse Top-K
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(all_texts, max_features: int = 50_000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Sample up to 2M texts for vocab fitting — enough for all relevant n-grams
    sample = all_texts
    if len(all_texts) > 2_000_000:
        import random
        rng = random.Random(42)
        sample = rng.sample(all_texts, 2_000_000)
    print(f'  Fitting TF-IDF on {len(sample):,} texts (sampled from {len(all_texts):,}) ...')
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 3),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )
    vec.fit(sample)
    del sample
    print(f'  TF-IDF fit: {time.time()-t0:.1f}s  vocab={len(vec.vocabulary_):,}')
    return vec


def tfidf_top_k_blocking(
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    vec,
    top_k: int = 50,
) -> dict:
    """
    Memory-safe chunked sparse matmul.

    OUTER loop = S23 chunks — each S23 chunk is transformed ONCE.
    INNER loop = S1 blocks  — only matmul, no re-transforms.

    For test set (1.7M S1, 10M S23, S23_CHUNK=100k):
      - S23 transforms: 100    (was 340,000 in old code — 3400x faster!)
      - Dense matmul:   1k × 100k × 4B = 400MB per step (safe on 30GB RAM)
    """
    from sklearn.preprocessing import normalize as sk_norm
    from tqdm import tqdm

    S1_BLOCK  = 1_000    # S1 rows per inner step
    S23_CHUNK = 100_000  # S23 rows per outer step → 400MB dense per matmul

    s1_ids    = s1['entity_id'].values
    s23_ids   = s23['entity_id'].values
    s23_texts = s23['name_addr'].fillna('').tolist()
    n_s1, n_s23 = len(s1_ids), len(s23_ids)

    print(f'  [B1] Transforming S1 ({n_s1:,}) ...')
    X_s1 = sk_norm(vec.transform(s1['name_addr'].fillna('')), norm='l2')

    # Global top-K storage for all S1 entities
    all_best_scores = np.full((n_s1, top_k), -np.inf, dtype=np.float32)
    all_best_idx    = np.zeros((n_s1, top_k), dtype=np.int64)

    n_s23_chunks = (n_s23 + S23_CHUNK - 1) // S23_CHUNK
    pbar = tqdm(total=n_s23_chunks, desc='  B1 TF-IDF', unit='S23-chunk')

    for j in range(0, n_s23, S23_CHUNK):
        # Transform this S23 chunk exactly once, then reuse for all S1 blocks
        Xs23_chunk = sk_norm(vec.transform(s23_texts[j:j+S23_CHUNK]), norm='l2')
        chunk_size = Xs23_chunk.shape[0]

        for i in range(0, n_s1, S1_BLOCK):
            nb        = min(S1_BLOCK, n_s1 - i)
            Xs1_block = X_s1[i:i+nb]

            sim = (Xs1_block @ Xs23_chunk.T).toarray().astype(np.float32)  # nb × chunk_size

            k_local   = min(top_k, chunk_size)
            local_idx = np.argpartition(sim, -k_local, axis=1)[:, -k_local:]
            local_sc  = np.take_along_axis(sim, local_idx, axis=1)
            del sim

            combined_s = np.concatenate([all_best_scores[i:i+nb], local_sc],      axis=1)
            combined_i = np.concatenate([all_best_idx[i:i+nb],    local_idx + j], axis=1)

            k_m = min(top_k, combined_s.shape[1])
            pos = np.argpartition(combined_s, -k_m, axis=1)[:, -k_m:]
            all_best_scores[i:i+nb] = np.take_along_axis(combined_s, pos, axis=1)
            all_best_idx[i:i+nb]    = np.take_along_axis(combined_i, pos, axis=1)

        del Xs23_chunk
        gc.collect()
        pbar.update(1)

    pbar.close()

    candidates = defaultdict(set)
    for i, s1_id in enumerate(s1_ids):
        valid = all_best_idx[i][all_best_scores[i] > -np.inf]
        if len(valid):
            candidates[s1_id] = set(s23_ids[valid])

    del all_best_scores, all_best_idx, X_s1
    gc.collect()
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# B2: MinHash LSH
# ─────────────────────────────────────────────────────────────────────────────

def minhash_lsh_blocking(s1: pd.DataFrame, s23: pd.DataFrame,
                          threshold: float = 0.25, num_perm: int = 128) -> dict:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print('  [B2] datasketch not installed, skipping.')
        return defaultdict(set)

    def trigrams(text: str) -> list:
        t = '_' + text.replace(' ', '_') + '_'
        return [t[i:i+3].encode('utf-8') for i in range(max(1, len(t)-2))]

    print(f'  [B2] Building LSH index ({len(s23):,} S23 records) ...')
    t0 = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for _, row in s23.iterrows():
        m = MinHash(num_perm=num_perm)
        for g in trigrams(row['norm_name']): m.update(g)
        try: lsh.insert(row['entity_id'], m)
        except ValueError: pass
    print(f'  [B2] Index built in {time.time()-t0:.1f}s')

    candidates = defaultdict(set)
    for _, row in s1.iterrows():
        q = MinHash(num_perm=num_perm)
        for g in trigrams(row['norm_name']): q.update(g)
        res = lsh.query(q)
        if res: candidates[row['entity_id']] |= set(res)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# B3: PIN/ZIP Hash Join — VECTORIZED via pandas merge/explode
# ─────────────────────────────────────────────────────────────────────────────

def zip_blocking(s1: pd.DataFrame, s23: pd.DataFrame) -> dict:
    """Vectorized: explode zip_tokens, merge on exact match + 4-digit prefix."""
    print(f'  [B3] ZIP/PIN blocking (vectorized) ...')
    t0 = time.time()

    def extract_zips(df):
        d = df[['entity_id', 'zip_tokens']].copy()
        d['zip_tokens'] = d['zip_tokens'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        return d.explode('zip_tokens').dropna(subset=['zip_tokens'])

    s1z  = extract_zips(s1).rename(columns={'entity_id': 's1_id',  'zip_tokens': 'zip'})
    s23z = extract_zips(s23).rename(columns={'entity_id': 's23_id', 'zip_tokens': 'zip'})

    # Exact match
    exact = s1z.merge(s23z, on='zip')[['s1_id', 's23_id']].drop_duplicates()

    # 4-digit prefix match
    s1z['zip4']  = s1z['zip'].str[:4]
    s23z['zip4'] = s23z['zip'].str[:4]
    prefix = (s1z[s1z['zip4'].str.len() == 4]
              .merge(s23z[s23z['zip4'].str.len() == 4], on='zip4')
              [['s1_id', 's23_id']].drop_duplicates())

    merged = pd.concat([exact, prefix]).drop_duplicates()

    # Cap hot keys
    counts = merged.groupby('s1_id')['s23_id'].count()
    heavy  = counts[counts > 500].index
    if len(heavy) > 0:
        light = merged[~merged['s1_id'].isin(heavy)]
        heavy_sample = (merged[merged['s1_id'].isin(heavy)]
                        .groupby('s1_id')
                        .apply(lambda g: g.sample(min(500, len(g)), random_state=42))
                        .reset_index(drop=True))
        merged = pd.concat([light, heavy_sample])

    candidates = merged.groupby('s1_id')['s23_id'].apply(set).to_dict()
    print(f'  [B3] Done in {time.time()-t0:.1f}s — {len(merged):,} candidate pairs')
    return defaultdict(set, candidates)


# ─────────────────────────────────────────────────────────────────────────────
# B4: Country + Name Prefix — VECTORIZED via pandas merge
# ─────────────────────────────────────────────────────────────────────────────

def name_prefix_blocking(s1: pd.DataFrame, s23: pd.DataFrame) -> dict:
    """Vectorized: group by country + first-3-chars of name, then merge."""
    print(f'  [B4] Name-prefix blocking (vectorized) ...')
    t0 = time.time()

    def prefix_key(df):
        d = df[['entity_id', 'norm_name', 'country_clean']].copy()
        d['prefix'] = d['norm_name'].str[:3].fillna('')
        d['key']    = d['country_clean'].fillna('') + '_' + d['prefix']
        return d

    s1p  = prefix_key(s1).rename(columns={'entity_id': 's1_id'})
    s23p = prefix_key(s23).rename(columns={'entity_id': 's23_id'})

    merged = (s1p[['s1_id', 'key']]
              .merge(s23p[['s23_id', 'key']], on='key')
              [['s1_id', 's23_id']].drop_duplicates())

    # Cap heavy keys
    counts = merged.groupby('s1_id')['s23_id'].count()
    heavy  = counts[counts > 500].index
    if len(heavy) > 0:
        light = merged[~merged['s1_id'].isin(heavy)]
        heavy_sample = (merged[merged['s1_id'].isin(heavy)]
                        .groupby('s1_id')
                        .apply(lambda g: g.sample(min(500, len(g)), random_state=42))
                        .reset_index(drop=True))
        merged = pd.concat([light, heavy_sample])

    candidates = merged.groupby('s1_id')['s23_id'].apply(set).to_dict()
    print(f'  [B4] Done in {time.time()-t0:.1f}s — {len(merged):,} candidate pairs')
    return defaultdict(set, candidates)


# ─────────────────────────────────────────────────────────────────────────────
# B5: Phonetic (Soundex) Join — VECTORIZED via pandas merge
# ─────────────────────────────────────────────────────────────────────────────

def phonetic_blocking(s1: pd.DataFrame, s23: pd.DataFrame) -> dict:
    """Vectorized: compute soundex key on entire column, then merge."""
    try:
        import jellyfish
    except ImportError:
        print('  [B5] jellyfish not installed, skipping.')
        return defaultdict(set)

    print(f'  [B5] Phonetic blocking (vectorized) ...')
    t0 = time.time()

    def soundex_key(norm_name: pd.Series, country: pd.Series) -> pd.Series:
        first_word = norm_name.str.split().str[0].fillna('')
        sx = first_word.apply(lambda w: jellyfish.soundex(w) if w else 'Z000')
        return country + '_' + sx

    s1_copy  = s1[['entity_id', 'norm_name', 'country_clean']].copy()
    s23_copy = s23[['entity_id', 'norm_name', 'country_clean']].copy()

    s1_copy['phon_key']  = soundex_key(s1_copy['norm_name'],  s1_copy['country_clean'])
    s23_copy['phon_key'] = soundex_key(s23_copy['norm_name'], s23_copy['country_clean'])

    s1_k  = s1_copy[['entity_id', 'phon_key']].rename(columns={'entity_id': 's1_id'})
    s23_k = s23_copy[['entity_id', 'phon_key']].rename(columns={'entity_id': 's23_id'})

    merged = s1_k.merge(s23_k, on='phon_key')[['s1_id', 's23_id']].drop_duplicates()

    # Cap hot phonetic codes
    counts = merged.groupby('s1_id')['s23_id'].count()
    heavy  = counts[counts > 300].index
    if len(heavy) > 0:
        light = merged[~merged['s1_id'].isin(heavy)]
        heavy_sample = (merged[merged['s1_id'].isin(heavy)]
                        .groupby('s1_id')
                        .apply(lambda g: g.sample(min(300, len(g)), random_state=42))
                        .reset_index(drop=True))
        merged = pd.concat([light, heavy_sample])

    candidates = merged.groupby('s1_id')['s23_id'].apply(set).to_dict()
    print(f'  [B5] Done in {time.time()-t0:.1f}s — {len(merged):,} candidate pairs')
    return defaultdict(set, candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Merge + cap
# ─────────────────────────────────────────────────────────────────────────────

def merge_candidates(*dicts, max_per_entity: int = 500, s1_ids=None) -> dict:
    merged = defaultdict(set)
    for d in dicts:
        for eid, cands in d.items():
            merged[eid] |= cands
    for eid in list(merged):
        if len(merged[eid]) > max_per_entity:
            merged[eid] = set(list(merged[eid])[:max_per_entity])
    if s1_ids is not None:
        for eid in s1_ids:
            if eid not in merged:
                merged[eid] = set()
    return dict(merged)


def write_candidate_pairs(candidates: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'source1_entity_id': s1, 'candidate_entity_ids': ','.join(sorted(c))}
            for s1, c in candidates.items()]
    pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False, encoding='utf-8')
    print(f'  Wrote {len(rows):,} rows → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_blocking(s1, s23, s1_all_texts=None, top_k=50,
                 run_b2=True, run_b3=True, run_b4=True, run_b5=True):
    t_start = time.time()
    print(f'\n[Blocking] S1={len(s1):,}  S23={len(s23):,}  top_k={top_k}')

    # Fit TF-IDF vocab on available texts (sampled to 2M max inside build_tfidf)
    all_texts = list(s1['name_addr'].fillna('')) + list(s23['name_addr'].fillna(''))
    if s1_all_texts:
        all_texts += s1_all_texts
    vec = build_tfidf(all_texts)
    del all_texts
    gc.collect()

    print('\n  B1: TF-IDF top-K ...')
    b1 = tfidf_top_k_blocking(s1, s23, vec, top_k=top_k)

    b2 = minhash_lsh_blocking(s1, s23) if run_b2 else defaultdict(set)
    b3 = zip_blocking(s1, s23)          if run_b3 else defaultdict(set)
    b4 = name_prefix_blocking(s1, s23)  if run_b4 else defaultdict(set)
    b5 = phonetic_blocking(s1, s23)     if run_b5 else defaultdict(set)

    print('\n  Merging (OR-union) ...')
    candidates = merge_candidates(b1, b2, b3, b4, b5, s1_ids=s1['entity_id'].tolist())

    n_pairs = sum(len(v) for v in candidates.values())
    print(f'  Total candidate pairs : {n_pairs:,}')
    print(f'  Avg per S1 entity     : {n_pairs/max(1,len(candidates)):.1f}')
    print(f'  [Blocking] Total time : {(time.time()-t_start)/60:.1f} min')
    return candidates, vec


if __name__ == '__main__':
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib
    _n = importlib.import_module('01_normalize')

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test')
    parser.add_argument('--data-dir', default='dataset')
    parser.add_argument('--out-dir', default='output')
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()

    srcs = _n.normalize_sources(Path(args.data_dir) / args.split, split=args.split)
    s23  = pd.concat([srcs['s2'], srcs['s3']], ignore_index=True)
    cands, _ = run_blocking(srcs['s1'], s23, top_k=args.top_k)
    write_candidate_pairs(cands, Path(args.out_dir) / 'candidate_pairs.tsv')
