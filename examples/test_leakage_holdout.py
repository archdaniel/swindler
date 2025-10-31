# tests/test_leakage_holdout.py
"""
Unit test reproducer for leakage + holdout protection.

Creates a toy dataset where one feature perfectly encodes the target (leakage).
Verifies:
 - ModelDataProfiler.detect_leakage_and_proxies finds that feature
 - run_pipeline with auto_drop_leakage=True removes the leaking feature and returns a model
   whose holdout metrics are not perfect (i.e., no leakage).
"""
import pandas as pd
import numpy as np
from src.main import run_pipeline
from src.auto_definitions import ModelDataProfiler

def make_leaky_df(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    # binary target
    y = rng.integers(0, 2, size=n)
    # non-leaky signal
    x1 = rng.normal(size=n)
    # leaking feature (exact copy of y)
    leak = y.copy()
    df = pd.DataFrame({
        'x1': x1,
        'leak': leak,
        'target': y
    })
    return df

def test_leakage_detection_and_holdout():
    df = make_leaky_df(1000, seed=123)
    profiler = ModelDataProfiler(df, target='target', verbose=False)
    report, model, X, y, df_fixed = profiler.profile()
    # profiler should flag 'leak' as suspicious
    leakage_keys = list(report.get('leakage_flags', {}).keys())
    assert 'leak' in leakage_keys or any('leak' in k for k in leakage_keys), f"Profiler did not flag leakage: {leakage_keys}"

    # run pipeline with auto_drop_leakage True and small holdout
    payload = run_pipeline(df, target='target', verbose=False, holdout_size=0.2, max_profile_samples=500, output_sparse=False, auto_drop_leakage=True)
    holdout_metrics = payload['run_report']['holdout_metrics']
    # Since leak was removed, accuracy should NOT be perfect (should be ~0.5 for random)
    acc = holdout_metrics.get('accuracy') or 0.0
    assert acc < 0.99, f"Holdout accuracy suspiciously high: {acc}"
    # also assert training_index and holdout_index are returned and disjoint
    train_idx = set(payload['training_index'])
    hold_idx = set(payload['holdout_index'])
    assert train_idx.isdisjoint(hold_idx), "Training and holdout indices overlap!"