import pandas as pd
import numpy as np
from src.auto_definitions import ModelDataProfiler

def test_categorical_fillna_handling_in_predictive_check():
    # Create df with a categorical column that has NaNs and categorical dtype
    n = 200
    rng = np.random.default_rng(1)
    cat_vals = ['a', 'b', 'c']
    cats = rng.choice(cat_vals, size=n)
    cats[rng.choice(n, size=20, replace=False)] = None  # introduce NaNs
    df = pd.DataFrame({
        'cat_col': pd.Categorical(cats, categories=cat_vals),
        'num_col': rng.normal(size=n),
        'target': rng.integers(0, 2, size=n)
    })
    profiler = ModelDataProfiler(df, target='target', verbose=False)
    # Should not raise when running detection
    flags = profiler.detect_leakage_and_proxies(df, target='target', min_count=5, purity_threshold=0.9, accuracy_threshold=0.9, verbose=False)
    # flags is a dict (possibly empty); just asserting the call completed without exception
    assert isinstance(flags, dict)