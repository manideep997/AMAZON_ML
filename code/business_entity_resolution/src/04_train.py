"""
04_train.py — LightGBM Training with Calibration and 5-Fold CV

Pipeline:
  1. Load training data (normalized + blocked + featured)
  2. 5-fold stratified CV (by country × match-count bucket)
  3. Train LightGBM binary classifier
  4. Isotonic calibration on out-of-fold (OOF) predictions
  5. Threshold sweep on calibrated OOF scores to maximize macro F_0.5
  6. Save model + calibrator + best threshold

Key design decisions:
  - Hard negatives sampled proportional to blocking TF-IDF score
  - Ratio: 1 positive : 3 hard-neg : 1 random-neg  (20% / 60% / 20%)
  - scale_pos_weight compensates remaining imbalance
  - Isotonic calibration before threshold sweep (raw LGB probs not well-calibrated)
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────────────────────────────────────────
# F_0.5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def f05_score(tp: int, pred_pos: int, true_pos: int) -> float:
    if pred_pos == 0 and true_pos == 0:
        return 1.0
    if pred_pos == 0:
        return 0.0
    p = tp / pred_pos
    r = tp / true_pos if true_pos else 0.0
    denom = 0.25 * p + r
    return (1.25 * p * r / denom) if denom > 0 else 0.0


def macro_f05_from_pairs(
    pair_ids: list[tuple],   # (s1_id, s23_id)
    probs: np.ndarray,
    ground_truth: dict,      # s1_id → set of true match ids
    threshold: float,
    all_s1_ids: list[str],
) -> float:
    """
    Compute macro-averaged F_0.5 given predicted probabilities and a threshold.
    Includes all S1 entities (including those with no candidates).
    """
    # Build prediction dict from pairs
    pred_dict = defaultdict(set)
    for (s1_id, s23_id), prob in zip(pair_ids, probs):
        if prob >= threshold:
            pred_dict[s1_id].add(s23_id)

    scores = []
    for s1_id in all_s1_ids:
        pred_set = pred_dict.get(s1_id, set())
        true_set = ground_truth.get(s1_id, set())

        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set:
            scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            scores.append(f05_score(tp, len(pred_set), len(true_set)))

    return float(np.mean(scores))


def sweep_threshold(
    pair_ids: list[tuple],
    probs: np.ndarray,
    ground_truth: dict,
    all_s1_ids: list[str],
    lo: float = 0.20,
    hi: float = 0.85,
    step: float = 0.01,
) -> tuple[float, float]:
    """Find threshold maximizing macro F_0.5. Returns (best_threshold, best_f05)."""
    best_t, best_f = 0.5, 0.0
    thresholds = np.arange(lo, hi + step, step)
    for t in thresholds:
        f = macro_f05_from_pairs(pair_ids, probs, ground_truth, t, all_s1_ids)
        if f > best_f:
            best_f, best_t = f, t
    return round(float(best_t), 4), round(float(best_f), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Stratification helper
# ─────────────────────────────────────────────────────────────────────────────

def make_strat_key(ground_truth: dict, country_map: dict, s1_id: str) -> str:
    """Country + match-count bucket → stratification key for CV."""
    n_matches = len(ground_truth.get(s1_id, set()))
    country   = country_map.get(s1_id, 'unknown')
    if n_matches == 0:
        bucket = 'zero'
    elif n_matches <= 2:
        bucket = 'low'
    elif n_matches <= 5:
        bucket = 'mid'
    else:
        bucket = 'high'
    return f'{country}_{bucket}'


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM training
# ─────────────────────────────────────────────────────────────────────────────

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 255,          # ↑ 127→255: deeper trees, better on 15+ features
    'min_child_samples': 20,    # ↓ 50→20: captures rarer match patterns
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.05,          # ↓ lighter L1 reg
    'lambda_l2': 0.05,          # ↓ lighter L2 reg
    'min_split_gain': 0.01,     # prevent overly small splits
    'n_estimators': 2000,       # ↑ 1000→2000: early stopping caps it anyway
    'n_jobs': -1,               # use ALL available cores
    'verbose': -1,
    'random_state': 42,
}


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = LGBM_PARAMS,
) -> lgb.LGBMClassifier:
    """Train a LightGBM model with early stopping on val logloss."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0

    model_params = dict(params)
    model_params['scale_pos_weight'] = scale_pos

    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(100)],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Full training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: list[tuple],          # (s1_id, s23_id)
    ground_truth: dict,             # s1_id → set of true s23_ids
    country_map: dict,              # s1_id → country string
    all_s1_ids: list[str],          # all S1 entity IDs in training set
    model_dir: Path,
    n_folds: int = 5,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Full training loop:
      1. 5-fold stratified CV
      2. OOF predictions for calibration
      3. Isotonic calibration
      4. Threshold sweep on calibrated OOF scores
      5. Final model on full data
      6. Save model, calibrator, threshold, feature importances

    Returns: results dict with best_threshold, oof_f05, feature_importances
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Stratification keys per sample (based on S1 entity of each pair)
    strat_keys = [
        make_strat_key(ground_truth, country_map, s1_id)
        for s1_id, _ in pair_ids
    ]
    # Encode as integers for StratifiedKFold
    key_set   = sorted(set(strat_keys))
    key_enc   = {k: i for i, k in enumerate(key_set)}
    strat_enc = np.array([key_enc[k] for k in strat_keys])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_probs    = np.zeros(len(y), dtype=np.float32)
    fold_models  = []

    print(f'\n[Training] {n_folds}-fold stratified CV on {len(X):,} pairs ...')
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, strat_enc)):
        print(f'\n  Fold {fold+1}/{n_folds}  '
              f'train={len(tr_idx):,}  val={len(val_idx):,}  '
              f'pos_rate={y[tr_idx].mean():.3f}')

        model = train_lgbm(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx])
        fold_probs = model.predict_proba(X[val_idx])[:, 1]
        oof_probs[val_idx] = fold_probs
        fold_models.append(model)

        val_f05 = macro_f05_from_pairs(
            [pair_ids[i] for i in val_idx],
            fold_probs,
            ground_truth,
            threshold=0.5,
            all_s1_ids=all_s1_ids,
        )
        print(f'  Fold {fold+1} F_0.5 @ 0.5: {val_f05:.4f}')

    # ── Isotonic calibration ──────────────────────────────────────────────────
    print('\n  Fitting isotonic calibration on OOF probabilities ...')
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_probs, y)
    cal_probs = calibrator.predict(oof_probs)

    # ── Threshold sweep on calibrated OOF ────────────────────────────────────
    print('  Sweeping threshold [0.20, 0.85] ...')
    best_t, best_f = sweep_threshold(pair_ids, cal_probs, ground_truth, all_s1_ids)
    print(f'  Best threshold: {best_t:.4f}  →  OOF macro F_0.5 = {best_f:.4f}')

    # ── Feature importances (average across folds) ────────────────────────────
    if feature_names:
        importances = np.zeros(len(feature_names))
        for m in fold_models:
            importances += m.feature_importances_
        importances /= n_folds
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        print('\n  Top feature importances:')
        print(fi_df.to_string(index=False))
        fi_df.to_csv(model_dir / 'feature_importances.csv', index=False)

    # ── Train final model on full data ───────────────────────────────────────
    print('\n  Training final model on full dataset ...')
    final_model = train_lgbm(X, y, X[-max(1, len(X)//10):], y[-max(1, len(y)//10):])

    # ── Save artefacts ────────────────────────────────────────────────────────
    with open(model_dir / 'lgbm_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    with open(model_dir / 'calibrator.pkl', 'wb') as f:
        pickle.dump(calibrator, f)

    results = {
        'best_threshold': best_t,
        'oof_f05': best_f,
        'n_folds': n_folds,
        'n_pairs': len(y),
        'pos_rate': float(y.mean()),
        'training_time_min': round((time.time() - t_start) / 60, 1),
    }
    with open(model_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n[Training] Done in {results["training_time_min"]:.1f} min')
    print(f'  Saved model → {model_dir / "lgbm_model.pkl"}')
    print(f'  Best threshold: {best_t}   OOF F_0.5: {best_f:.4f}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Load saved artefacts
# ─────────────────────────────────────────────────────────────────────────────

def load_model_artefacts(model_dir: Path):
    """Load model, calibrator, and results from a previous training run."""
    with open(model_dir / 'lgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(model_dir / 'calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)
    with open(model_dir / 'training_results.json') as f:
        results = json.load(f)
    return model, calibrator, results


if __name__ == '__main__':
    # Quick sanity check: generate dummy data and run training
    np.random.seed(42)
    n = 10000
    X = np.random.rand(n, 15).astype(np.float32)
    y = (np.random.rand(n) > 0.8).astype(np.int8)

    pair_ids = [(f'S1-{i}', f'S2-{i}') for i in range(n)]
    gt = {f'S1-{i}': {f'S2-{i}'} if y[i] == 1 else set() for i in range(n)}
    cm = {f'S1-{i}': 'us' for i in range(n)}
    all_s1 = [f'S1-{i}' for i in range(n)]

    results = run_training(
        X, y, pair_ids, gt, cm, all_s1,
        model_dir=Path('models'),
        n_folds=3,
    )
    print(results)
