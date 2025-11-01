import pandas as pd
import numpy as np
from src.auto_definitions import ModelDataProfiler

def make_categorical_target_df():
    # Create a dataset where a categorical feature strongly correlates with a categorical target
    rng = np.random.default_rng(42)
    n = 500
    # categorical target with 3 classes
    y = rng.choice(['A', 'B', 'C'], size=n, p=[0.5, 0.3, 0.2])
    # feature that leaks (almost 1:1 mapping for 'A')
    f1 = []
    for v in y:
        if v == 'A':
            f1.append('x_special')
        else:
            f1.append(rng.choice(['u', 'v', 'w']))
    df = pd.DataFrame({
        'f_leak': pd.Categorical(f1),
        'f_rand': rng.normal(size=n),
        'target': pd.Categorical(y)
    })
    return df

def test_detect_leakage_on_categorical_target():
    df = make_categorical_target_df()
    profiler = ModelDataProfiler(df, target='target', verbose=False)
    flags = profiler.detect_leakage_and_proxies(df, target='target', min_count=10, purity_threshold=0.9, accuracy_threshold=0.95, verbose=False)
    # Expect the leaking feature to be flagged (either as category_purity or single_feature_acc)
    keys = list(flags.keys())
    assert 'f_leak' in keys or any('f_leak' in k for k in keys), f"Leakage not detected: {keys}"