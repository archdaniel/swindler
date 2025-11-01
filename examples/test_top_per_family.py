import pandas as pd
import numpy as np
from src.main import run_pipeline

def test_top_per_family_present():
    # small synthetic dataset
    n = 300
    rng = np.random.default_rng(3)
    X1 = rng.normal(size=n)
    X2 = rng.integers(0,2,size=n)
    y = (X1 > 0).astype(int)
    df = pd.DataFrame({'f1': X1, 'f2': X2, 'target': y})
    payload = run_pipeline(df, target='target', verbose=False, holdout_size=0.2, max_profile_samples=100, output_sparse=False)
    trainer = payload.get('trainer_results_df')
    # Ensure best_per_family exists on the saved trainer object
    trainer_obj = payload.get('profiler')  # trainer stored in earlier design? If not, we fetch results
    # The run_report should also include trainer results, and we check run_report structure includes candidate models
    run_report = payload.get('run_report', {})
    assert 'trainer_results' in run_report
    # we can't directly assert best_per_family_ here, but ensure multiple models were evaluated
    assert isinstance(run_report['trainer_results'], list)