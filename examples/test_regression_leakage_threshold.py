import pandas as pd
import numpy as np
from src.auto_definitions import ModelDataProfiler

def make_regression_leaky(n=500):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    # make a feature weakly correlated, and one nearly deterministic
    weak = x + rng.normal(scale=10.0, size=n)
    strong = x * 10.0  # near perfect predictor
    y = strong + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({'weak': weak, 'strong': strong, 'y': y})
    return df

def test_regression_leakage_strict_threshold():
    df = make_regression_leaky(300)
    profiler = ModelDataProfiler(df, target='y', verbose=False)
    flags = profiler.detect_leakage_and_proxies(df, target='y', min_count=10, purity_threshold=0.99, verbose=False)
    # Expect only 'strong' to be flagged (not 'weak'), due to high R2 requirement
    assert 'strong' in flags or any('strong' in k for k in flags.keys())
    assert 'weak' not in flags