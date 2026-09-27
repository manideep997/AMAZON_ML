"""
05_predict.py — Inference, Conflict Monitoring, and Output File Writing

Steps:
  1. Load model + calibrator + threshold from training artefacts
  2. Score all test candidate pairs (inference mode)
  3. Apply calibrated threshold → matching_results
  4. Conflict monitoring: log multi-claim rate
  5. Write matching_results.tsv and candidate_pairs.tsv (if not already written)
  6. Baseline mode: pure TF-IDF cosine threshold (no ML model, for Submission #1)
"""

import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Conflict monitoring
# ─────────────────────────────────────────────────────────────────────────────

def monitor_conflicts(matching: dict[str, set], threshold_pct: float = 5.0) -> dict:
    """
    Count how many S2/S3 records are claimed by more than one S1 entity.
    Multi-claim rate > threshold_pct → threshold is likely too low.

    Returns diagnostic dict.
    """
    claim_counts = Counter()
    for s1_id, matches in matching.items():
        for m in matches:
            claim_counts[m] += 1

    multi_claimed = {k: v for k, v in claim_counts.items() if v > 1}
    total_matched = sum(len(v) for v in matching.values())
    multi_pct = 100.0 * len(multi_claimed) / max(1, len(claim_counts))

    result = {
        'total_s23_matched': len(claim_counts),
        'multi_claimed_count': len(multi_claimed),
        'multi_claimed_pct': round(multi_pct, 2),
        'max_claims_on_single_s23': max(claim_counts.values()) if claim_counts else 0,
        'alert': multi_pct > threshold_pct,
    }

    print(f'\n[Conflict Monitor]')
    print(f'  S2/S3 records matched (total): {result["total_s23_matched"]:,}')
    print(f'  Multi-claimed (>1 S1):         {result["multi_claimed_count"]:,}  '
          f'({result["multi_claimed_pct"]:.2f}%)')
    if result['alert']:
        print(f'  ⚠ Alert: multi-claim rate > {threshold_pct}% — consider raising threshold')
    else:
        print(f'  ✓ Multi-claim rate within normal range')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

def write_matching_results(
    matching: dict[str, set],
    all_s1_ids: list[str],
    out_path: Path,
) -> None:
    """
    Write matching_results.tsv.
    Every S1 entity must appear. Empty = no matches (singleton prediction).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id in all_s1_ids:
        matches = matching.get(s1_id, set())
        rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(sorted(matches)),
        })
    df = pd.DataFrame(rows, columns=['source1_entity_id', 'matched_entity_ids'])
    df.to_csv(out_path, sep='\t', index=False, encoding='utf-8')

    n_non_empty = (df['matched_entity_ids'] != '').sum()
    total_matches = df['matched_entity_ids'].apply(
        lambda x: len(x.split(',')) if x else 0).sum()
    print(f'\n[Output] {len(df):,} S1 entities → {out_path}')
    print(f'  Non-empty (has matches): {n_non_empty:,}  '
          f'({100*n_non_empty/len(df):.1f}%)')
    print(f'  Total matched pairs: {total_matches:,}  '
          f'(avg {total_matches/max(1,n_non_empty):.2f} per non-singleton)')


# ─────────────────────────────────────────────────────────────────────────────
# Baseline predictor (Submission #1 — no ML model)
# ─────────────────────────────────────────────────────────────────────────────

def predict_baseline(
    candidates: dict[str, set],
    tfidf_scores: dict,          # (s1_id, s23_id) → float
    threshold: float,
    all_s1_ids: list[str],
) -> dict[str, set]:
    """
    Baseline: predict a match if TF-IDF cosine score ≥ threshold.
    No ML model needed. Fast, used for Submission #1.
    """
    matching = {}
    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, set())
        matched = {
            cid for cid in cands
            if tfidf_scores.get((s1_id, cid), 0.0) >= threshold
        }
        matching[s1_id] = matched
    return matching


def tune_baseline_threshold(
    candidates: dict[str, set],
    tfidf_scores: dict,
    ground_truth: dict,
    all_s1_ids: list[str],
    lo: float = 0.20,
    hi: float = 0.80,
    step: float = 0.01,
) -> tuple[float, float]:
    """Tune threshold for baseline predictor on validation set."""
    from train import macro_f05_from_pairs, sweep_threshold

    # Build pair_ids and probs for the sweep function
    pair_ids = []
    probs = []
    for s1_id, cands in candidates.items():
        for cid in cands:
            pair_ids.append((s1_id, cid))
            probs.append(tfidf_scores.get((s1_id, cid), 0.0))

    return sweep_threshold(pair_ids, np.array(probs, dtype=np.float32),
                           ground_truth, all_s1_ids, lo, hi, step)


# ─────────────────────────────────────────────────────────────────────────────
# ML model predictor (Submission #2, #3, #4)
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_model(
    X: np.ndarray,
    pair_ids: list[tuple],       # (s1_id, s23_id)
    model,
    calibrator,
    threshold: float,
    all_s1_ids: list[str],
) -> dict[str, set]:
    """
    Score candidate pairs with LightGBM + isotonic calibration.
    Apply threshold → matching dict.
    """
    print(f'\n[Predict] Scoring {len(X):,} pairs with LightGBM ...')
    t0 = time.time()

    raw_probs  = model.predict_proba(X)[:, 1].astype(np.float32)
    cal_probs  = calibrator.predict(raw_probs).astype(np.float32)

    print(f'  Scored in {time.time()-t0:.1f}s')
    print(f'  Prob stats — raw: mean={raw_probs.mean():.3f} '
          f'| calibrated: mean={cal_probs.mean():.3f}')
    print(f'  Threshold: {threshold:.4f} → '
          f'{(cal_probs >= threshold).sum():,} positive pairs')

    matching = defaultdict(set)
    for (s1_id, s23_id), prob in zip(pair_ids, cal_probs):
        if prob >= threshold:
            matching[s1_id].add(s23_id)

    # Ensure all S1 IDs present
    for s1_id in all_s1_ids:
        if s1_id not in matching:
            matching[s1_id] = set()

    return dict(matching)


# ─────────────────────────────────────────────────────────────────────────────
# Run full predict pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(
    candidates: dict[str, set],
    s1: pd.DataFrame,
    s23: pd.DataFrame,
    all_s1_ids: list[str],
    out_dir: Path,
    model_dir: Path | None = None,
    tfidf_scores: dict | None = None,
    baseline_threshold: float | None = None,
) -> dict[str, set]:
    """
    Unified predict entry point.

    If model_dir is provided: use LightGBM model + calibrator.
    Otherwise: use baseline TF-IDF cosine threshold.
    """
    from features import build_feature_matrix, df_to_lookup, FEATURE_NAMES

    out_dir.mkdir(parents=True, exist_ok=True)

    if model_dir and (model_dir / 'lgbm_model.pkl').exists():
        # ── ML mode ─────────────────────────────────────────────────────────
        print('\n[Predict] ML mode (LightGBM)')
        from train import load_model_artefacts
        model, calibrator, results = load_model_artefacts(model_dir)
        threshold = results['best_threshold']
        print(f'  Loaded model. Threshold from training: {threshold}')

        s1_lookup  = df_to_lookup(s1)
        s23_lookup = df_to_lookup(s23)

        X, _, pair_ids = build_feature_matrix(
            candidates, s1_lookup, s23_lookup,
            ground_truth=None,       # inference mode
            tfidf_scores=tfidf_scores,
            n_workers=max(1, (int(os.cpu_count()) or 2) - 1) if 'os' in dir() else 1,
        )

        matching = predict_with_model(X, pair_ids, model, calibrator, threshold, all_s1_ids)

    else:
        # ── Baseline mode ────────────────────────────────────────────────────
        print('\n[Predict] Baseline mode (TF-IDF cosine threshold)')
        t = baseline_threshold if baseline_threshold is not None else 0.35
        print(f'  Using threshold: {t}')
        matching = predict_baseline(candidates, tfidf_scores or {}, t, all_s1_ids)

    # Conflict monitoring
    diagnostics = monitor_conflicts(matching)
    with open(out_dir / 'conflict_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)

    # Write output
    write_matching_results(matching, all_s1_ids, out_dir / 'matching_results.tsv')

    return matching


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True,
                        help='Path to candidate_pairs.tsv')
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--out-dir', default='output')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override threshold (baseline mode)')
    args = parser.parse_args()

    # Load candidates
    cdf = pd.read_csv(args.candidates, sep='\t', encoding='utf-8')
    candidates = {}
    for _, row in cdf.iterrows():
        ids = set(row['candidate_entity_ids'].split(',')) if pd.notna(row['candidate_entity_ids']) and row['candidate_entity_ids'] else set()
        candidates[row['source1_entity_id']] = ids

    all_s1_ids = list(candidates.keys())
    print(f'Loaded {len(candidates):,} S1 entities with candidates')
    print('Note: run via run_pipeline.py for full integrated mode.')
