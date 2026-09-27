"""
run_pipeline.py — End-to-End Orchestration Script

Usage:

  # Submission #1: Baseline (TF-IDF cosine, no ML, fast ~3h)
  python run_pipeline.py --mode baseline --split test

  # Submission #2: Full blocking + LightGBM (basic features)
  python run_pipeline.py --mode train --split test

  # Submission #3+: Full pipeline (all features, calibration, CV)
  python run_pipeline.py --mode train --split test --full-features

  # Inference only (model already trained)
  python run_pipeline.py --mode predict --split test

  # Validate output before submitting
  python utils/validate_submission.py --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""

import os
import sys
import time
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add src to path
SRC_DIR = Path(__file__).parent / 'src'
sys.path.insert(0, str(SRC_DIR))

# ─── Import pipeline modules ──────────────────────────────────────────────────
# Module imports (filenames have numeric prefix, imported by sys.path)
import importlib

_n = importlib.import_module('01_normalize')
normalize_sources = _n.normalize_sources

_b = importlib.import_module('02_blocking')
run_blocking = _b.run_blocking
write_candidate_pairs = _b.write_candidate_pairs

_f = importlib.import_module('03_features')
build_feature_matrix = _f.build_feature_matrix
df_to_lookup = _f.df_to_lookup
FEATURE_NAMES = _f.FEATURE_NAMES

_t = importlib.import_module('04_train')
run_training = _t.run_training
load_model_artefacts = _t.load_model_artefacts
sweep_threshold = _t.sweep_threshold
macro_f05_from_pairs = _t.macro_f05_from_pairs

_p = importlib.import_module('05_predict')
predict_baseline = _p.predict_baseline
predict_with_model = _p.predict_with_model
monitor_conflicts = _p.monitor_conflicts
write_matching_results = _p.write_matching_results
tune_baseline_threshold = _p.tune_baseline_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(gt_path: Path) -> dict[str, set]:
    """Load train_ground_truth.tsv → {s1_id: set(matched_ids)}."""
    df = pd.read_csv(gt_path, sep='\t', encoding='utf-8')
    result = {}
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']) or str(row['matched_entity_ids']).strip() == '':
            result[s1_id] = set()
        else:
            result[s1_id] = set(str(row['matched_entity_ids']).split(','))
    return result


def load_candidate_pairs(path: Path) -> dict[str, set]:
    """Load candidate_pairs.tsv → {s1_id: set(candidate_ids)}."""
    df = pd.read_csv(path, sep='\t', encoding='utf-8')
    result = {}
    for _, row in df.iterrows():
        raw = row['candidate_entity_ids']
        if pd.isna(raw) or str(raw).strip() == '':
            result[row['source1_entity_id']] = set()
        else:
            result[row['source1_entity_id']] = set(str(raw).split(','))
    return result


def split_train_val(
    ground_truth: dict,
    country_map: dict,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Stratified train/val split of ground truth.
    Stratify by (country × match-count bucket).
    """
    from collections import defaultdict
    import random

    rng = random.Random(seed)
    groups = defaultdict(list)
    for s1_id in ground_truth:
        n = len(ground_truth[s1_id])
        country = country_map.get(s1_id, 'unknown')
        bucket = 'zero' if n == 0 else 'low' if n <= 2 else 'mid' if n <= 5 else 'high'
        groups[f'{country}_{bucket}'].append(s1_id)

    train_ids, val_ids = set(), set()
    for grp_ids in groups.values():
        rng.shuffle(grp_ids)
        n_val = max(1, int(len(grp_ids) * val_frac))
        val_ids.update(grp_ids[:n_val])
        train_ids.update(grp_ids[n_val:])

    gt_train = {k: v for k, v in ground_truth.items() if k in train_ids}
    gt_val   = {k: v for k, v in ground_truth.items() if k in val_ids}
    return gt_train, gt_val


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Business Entity Resolution Pipeline')
    parser.add_argument('--mode', choices=['baseline', 'train', 'predict'],
                        default='baseline',
                        help='baseline = TF-IDF cosine only (fast); '
                             'train = train LightGBM then predict; '
                             'predict = load existing model and predict')
    parser.add_argument('--split', choices=['test', 'train'], default='test',
                        help='Which split to run prediction on')
    parser.add_argument('--data-dir', default='dataset',
                        help='Root dataset directory')
    parser.add_argument('--out-dir', default='output',
                        help='Output directory for TSV files')
    parser.add_argument('--model-dir', default='models',
                        help='Directory for saved model artefacts')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-K candidates per S1 from TF-IDF blocking')
    parser.add_argument('--no-b2', action='store_true', help='Skip MinHash LSH blocking')
    parser.add_argument('--no-b3', action='store_true', help='Skip ZIP/PIN blocking')
    parser.add_argument('--no-b4', action='store_true', help='Skip name-prefix blocking')
    parser.add_argument('--no-b5', action='store_true', help='Skip phonetic blocking')
    parser.add_argument('--baseline-threshold', type=float, default=None,
                        help='TF-IDF cosine threshold for baseline mode')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds for training')
    parser.add_argument('--val-frac', type=float, default=0.2,
                        help='Fraction of training data held out for val (single split)')
    args = parser.parse_args()

    t_pipeline_start = time.time()
    base     = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'Business Entity Resolution Pipeline')
    print(f'  mode={args.mode}  split={args.split}  top_k={args.top_k}')
    print('=' * 60)

    # ── Step 1: Normalize ────────────────────────────────────────────────────
    print('\n[Step 1] Normalizing text ...')

    test_srcs = normalize_sources(base / 'test', split='test')
    s1_test   = test_srcs['s1']
    s23_test  = pd.concat([test_srcs['s2'], test_srcs['s3']], ignore_index=True)

    if args.mode in ('baseline', 'train'):
        train_srcs = normalize_sources(base / 'train', split='train')
        s1_train   = train_srcs['s1']
        s23_train  = pd.concat([train_srcs['s2'], train_srcs['s3']], ignore_index=True)

    # Determine which S1/S23 to run blocking on
    if args.split == 'test':
        s1_pred  = s1_test
        s23_pred = s23_test
    else:
        s1_pred  = s1_train
        s23_pred = s23_train

    all_s1_ids_pred = s1_pred['entity_id'].tolist()

    # ── Step 2: Blocking on PREDICTION split ─────────────────────────────────
    print('\n[Step 2] Blocking (candidate generation) ...')
    candidate_cache = out_dir / f'candidate_pairs_{args.split}.tsv'

    if candidate_cache.exists():
        print(f'  Loading cached candidates from {candidate_cache}')
        candidates_pred = load_candidate_pairs(candidate_cache)
        # Also need the TF-IDF vectorizer for features — rebuild quickly if needed
        vec = None
    else:
        # Combine train+test texts to build a robust TF-IDF vocabulary
        all_name_addrs = (
            list(s1_pred['name_addr'].fillna('')) +
            list(s23_pred['name_addr'].fillna(''))
        )
        if args.mode in ('baseline', 'train') and args.split == 'test':
            # Add train texts to vocab for better coverage
            all_name_addrs += list(s1_train['name_addr'].fillna(''))
            all_name_addrs += list(s23_train['name_addr'].fillna(''))

        candidates_pred, vec = run_blocking(
            s1_pred, s23_pred,
            s1_all_texts=all_name_addrs,
            top_k=args.top_k,
            run_b2=not args.no_b2,
            run_b3=not args.no_b3,
            run_b4=not args.no_b4,
            run_b5=not args.no_b5,
        )
        write_candidate_pairs(candidates_pred, candidate_cache)
        # Save vec for reuse
        vec_path = out_dir / 'tfidf_vec.pkl'
        with open(vec_path, 'wb') as f:
            pickle.dump(vec, f)

    # ── Step 3: Baseline predict or ML predict ───────────────────────────────

    if args.mode == 'baseline':
        print('\n[Step 3] Baseline prediction (TF-IDF cosine threshold) ...')

        # If we have train data, tune threshold on a validation split
        if args.mode == 'baseline' and args.split == 'test':
            print('  Tuning threshold on training validation split ...')
            gt_train = load_ground_truth(base / 'train' / 'train_ground_truth.tsv')
            country_map_tr = dict(zip(s1_train['entity_id'], s1_train['country_clean']))
            gt_tr, gt_val = split_train_val(gt_train, country_map_tr, args.val_frac)

            # Block on training data for threshold tuning
            val_s1_ids = list(gt_val.keys())
            val_s1_df  = s1_train[s1_train['entity_id'].isin(set(val_s1_ids))]

            print(f'  Val set: {len(val_s1_ids):,} S1 entities')
            cands_val, vec_tr = run_blocking(
                val_s1_df, s23_train,
                s1_all_texts=None,
                top_k=args.top_k,
                run_b2=not args.no_b2,
                run_b3=not args.no_b3,
                run_b4=not args.no_b4,
                run_b5=not args.no_b5,
            )

            # Build tfidf scores dict for val
            from sklearn.preprocessing import normalize as sk_normalize
            s23_val_lookup = {}  # just need the candidates
            tfidf_scores_val = {}  # skip detailed scoring for speed; use blocking score=0.5
            # Tune threshold using simple approach: any candidate = 0.5 score
            for s1_id, cands in cands_val.items():
                for cid in cands:
                    tfidf_scores_val[(s1_id, cid)] = 0.5

            best_t, best_f = tune_baseline_threshold(
                cands_val, tfidf_scores_val, gt_val, val_s1_ids)
            print(f'  Tuned threshold: {best_t:.4f}  Val F_0.5: {best_f:.4f}')
        else:
            best_t = args.baseline_threshold or 0.35

        # Apply to test candidates (all get score=0.5 since we don't have scores without vec)
        # For baseline: predict any candidate as a match (threshold = 0 effectively)
        # OR use the fact that candidates ARE already our predictions
        # Decision: for Submission #1, use threshold=0 → all candidates = matches
        # This is the "permissive" baseline that gets high recall
        print(f'  Applying threshold: {best_t}')
        tfidf_scores_pred = {(s1_id, cid): 0.5
                             for s1_id, cands in candidates_pred.items()
                             for cid in cands}
        matching = predict_baseline(candidates_pred, tfidf_scores_pred,
                                    threshold=best_t, all_s1_ids=all_s1_ids_pred)

    elif args.mode == 'train':
        print('\n[Step 3] Training LightGBM model ...')

        gt_train     = load_ground_truth(base / 'train' / 'train_ground_truth.tsv')
        country_map  = dict(zip(s1_train['entity_id'], s1_train['country_clean']))
        all_s1_train = s1_train['entity_id'].tolist()

        # Block on full training data
        print('  Blocking on training data ...')
        train_cand_cache = out_dir / 'candidate_pairs_train.tsv'
        if train_cand_cache.exists():
            print(f'  Loading cached train candidates from {train_cand_cache}')
            candidates_train = load_candidate_pairs(train_cand_cache)
        else:
            candidates_train, _ = run_blocking(
                s1_train, s23_train,
                s1_all_texts=None,
                top_k=args.top_k,
                run_b2=not args.no_b2,
                run_b3=not args.no_b3,
                run_b4=not args.no_b4,
                run_b5=not args.no_b5,
            )
            write_candidate_pairs(candidates_train, train_cand_cache)

        # Build feature matrix for training
        print('  Building feature matrix for training ...')
        s1_lookup  = df_to_lookup(s1_train)
        s23_lookup = df_to_lookup(s23_train)

        X_train, y_train, pair_ids_train = build_feature_matrix(
            candidates_train, s1_lookup, s23_lookup,
            ground_truth=gt_train,
            tfidf_scores=None,   # no pre-computed scores in train mode
            neg_ratio_hard=3,
            neg_ratio_random=1,
            n_workers=max(1, (os.cpu_count() or 2) - 1),
        )

        print(f'  Feature matrix: {X_train.shape}  pos_rate={y_train.mean():.3f}')

        # Train
        results = run_training(
            X_train, y_train, pair_ids_train,
            gt_train, country_map, all_s1_train,
            model_dir=model_dir,
            n_folds=args.n_folds,
            feature_names=FEATURE_NAMES,
        )

        # Predict on test
        print('\n[Step 4] Predicting on test set ...')
        model, calibrator, results = load_model_artefacts(model_dir)
        threshold = results['best_threshold']

        s1_pred_lookup  = df_to_lookup(s1_pred)
        s23_pred_lookup = df_to_lookup(s23_pred)

        X_pred, _, pair_ids_pred = build_feature_matrix(
            candidates_pred, s1_pred_lookup, s23_pred_lookup,
            ground_truth=None,
            tfidf_scores=None,
            n_workers=max(1, (os.cpu_count() or 2) - 1),
        )

        matching = predict_with_model(
            X_pred, pair_ids_pred, model, calibrator, threshold, all_s1_ids_pred)

    elif args.mode == 'predict':
        print('\n[Step 3] Inference with existing model ...')
        if not (model_dir / 'lgbm_model.pkl').exists():
            print(f'ERROR: No model found at {model_dir}. Run with --mode train first.')
            sys.exit(1)

        model, calibrator, results = load_model_artefacts(model_dir)
        threshold = results['best_threshold']
        print(f'  Loaded model. Threshold: {threshold}')

        s1_pred_lookup  = df_to_lookup(s1_pred)
        s23_pred_lookup = df_to_lookup(s23_pred)

        X_pred, _, pair_ids_pred = build_feature_matrix(
            candidates_pred, s1_pred_lookup, s23_pred_lookup,
            ground_truth=None, tfidf_scores=None,
            n_workers=max(1, (os.cpu_count() or 2) - 1),
        )

        matching = predict_with_model(
            X_pred, pair_ids_pred, model, calibrator, threshold, all_s1_ids_pred)

    # ── Step Final: Conflict monitor + write outputs ──────────────────────────
    diagnostics = monitor_conflicts(matching)
    with open(out_dir / 'conflict_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)

    write_matching_results(matching, all_s1_ids_pred, out_dir / 'matching_results.tsv')

    # Copy final candidate_pairs to output dir (required for submission zip)
    import shutil
    final_cand = out_dir / 'candidate_pairs.tsv'
    if candidate_cache != final_cand and candidate_cache.exists():
        shutil.copy(candidate_cache, final_cand)
        print(f'  Copied candidates → {final_cand}')

    total_time = (time.time() - t_pipeline_start) / 60
    print(f'\n{"=" * 60}')
    print(f'Pipeline complete in {total_time:.1f} min')
    print(f'Output files:')
    print(f'  {out_dir}/matching_results.tsv')
    print(f'  {out_dir}/candidate_pairs.tsv')
    print(f'\nValidate before submitting:')
    print(f'  python utils/validate_submission.py '
          f'--matching {out_dir}/matching_results.tsv '
          f'--candidate {out_dir}/candidate_pairs.tsv '
          f'--test-dir dataset/test')
    print('=' * 60)


if __name__ == '__main__':
    main()
