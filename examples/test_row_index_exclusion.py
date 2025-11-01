import pandas as pd
import numpy as np
from src.preprocessing import AutoPreprocessor

def test_row_index_not_seen_as_id():
    n = 100
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        '_swindler_row_index_': np.arange(n),
        'a': rng.normal(size=n),
        'b': rng.integers(0, 5, size=n)
    })
    pre = AutoPreprocessor(output_sparse=True, verbose=False)
    pre.fit(df.drop(columns=['_swindler_row_index_']), None)
    # ensure id detection did not mark internal index as id
    assert '_swindler_row_index_' not in pre.id_features_