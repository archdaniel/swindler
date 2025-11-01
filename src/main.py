# src/main.py
"""
Main orchestration for profiling, preprocessing, training, and reporting.
Now supports:
 - returning training indices,
 - options to produce sparse hashed output from preprocessor,
 - automatic dropping of profiler-detected leakage features prior to training,
 - returning fitted transformer for reuse,
 - a runtime estimator that benchmarks small fits and extrapolates total runtime.
"""
import os
import json
import time
import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from auto_definitions import ModelDataProfiler
from preprocessing import ModelAwarePreprocessor
from model_definitions import ModelTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse as sp

def _infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        return "regression"
    else:
        return "classification"

def _human_time(seconds: float) -> str:
    if seconds is None:
        return "unknown"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}d {h}h {m}m"
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def _estimate_runtime_via_benchmark(
    preproc: ModelAwarePreprocessor,
    work_df: pd.DataFrame,
    target_col: str,
    model_candidates: Dict[str, Any],
    sample_n: int = 2000,
    cv_folds: int = 5,
    parallel_workers: int = None
) -> Dict[str, Any]:
    """
    Simple estimator:
     - sample a small subset, transform it with fitted preproc
     - run 1 fit for LogisticRegression and 1 for RandomForest on the sample, measure times
     - extrapolate per-fit time to full dataset by linear scaling on rows
     - compute total_fits from model_candidates (grid size * cv_folds or hyperopt approximations)
     - divide by parallel_workers (approx) to estimate wall time.
    Returns dict with seconds estimate and breakdown.
    """
    if parallel_workers is None:
        try:
            import os
            parallel_workers = max(1, os.cpu_count() or 1)
        except Exception:
            parallel_workers = 1

    n_full = len(work_df)
    n_sample = min(sample_n, n_full)
    sample = work_df.sample(n=n_sample, random_state=42).reset_index(drop=True)

    # transform the sample (use preproc.transform which is already fitted)
    transformed_sample = preproc.transform(sample.drop(columns=[target_col], errors='ignore'))
    if isinstance(transformed_sample, tuple):
        X_sample, feature_names = transformed_sample
    else:
        X_sample = transformed_sample

    y_sample = sample[target_col].values

    # If sparse matrix, keep as-is; some estimators accept sparse
    def time_fit(model, X, y, repeats=1):
        t0 = time.time()
        for _ in range(repeats):
            model.fit(X, y)
        return (time.time() - t0) / repeats

    # Benchmark two representative models
    t_log = None
    t_rf = None
    try:
        t_log = time_fit(LogisticRegression(max_iter=200, solver='liblinear'), X_sample, y_sample, repeats=1)
    except Exception:
        t_log = None

    try:
        t_rf = time_fit(RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42), X_sample, y_sample, repeats=1)
    except Exception:
        t_rf = None

    # Count total fits
    def count_combinations(params):
        # params expected to be dict of lists (Grid) or hyperopt spaces. If hyperopt present, assume 10 evals.
        if params is None:
            return 1
        # hyperopt detection: values have attribute 'name' or are not list-like
        if hasattr(list(params.values())[0], "name") if params else False:
            return 10
        # otherwise treat as grid of lists
        total = 1
        for v in params.values():
            try:
                total *= len(v)
            except Exception:
                total *= 1
        return max(1, total)

    total_fits = 0
    breakdown = {}
    for name, (model, params) in model_candidates.items():
        combos = count_combinations(params) if isinstance(params, dict) else 1
        fits = combos * cv_folds
        breakdown[name] = {'combos': combos, 'fits': fits}
        total_fits += fits

    # Choose per-fit time proxy depending on model type name
    per_fit_times = []
    for name in breakdown:
        if 'Logistic' in name or 'Ridge' in name:
            proxy = t_log if t_log is not None else (t_rf or 1.0)
        else:
            proxy = t_rf if t_rf is not None else (t_log or 1.0)
        # extrapolate sample->full by linear scaling
        est_per_fit_full = proxy * (n_full / max(1, n_sample))
        per_fit_times.append((name, est_per_fit_full, breakdown[name]['fits']))

    total_seconds = sum(t * fits for (_, t, fits) in per_fit_times)
    wall_seconds = total_seconds / max(1, parallel_workers)

    return {
        "n_full_rows": n_full,
        "n_sample_rows": n_sample,
        "parallel_workers": parallel_workers,
        "sample_timings": {"logistic_sec": t_log, "random_forest_sec": t_rf},
        "per_model_estimates": [{ "model": m, "est_per_fit_sec": t, "n_fits": f } for (m,t,f) in per_fit_times],
        "total_fits": total_fits,
        "total_seconds_est": total_seconds,
        "wall_seconds_est": wall_seconds,
        "human_readable_wall_time_est": _human_time(wall_seconds)
    }

def run_pipeline(
    data: Any,
    target: str,
    output_dir: str = "pipeline_reports",
    verbose: bool = True,
    use_mlflow: bool = False,
    holdout_size: float = 0.10,
    holdout_random_state: int = 505,
    max_profile_samples: Optional[int] = 5000,
    random_state: int = 505,
    preproc_params: Optional[dict] = None,
    trainer_params: Optional[dict] = None,
    output_sparse: bool = True,
    auto_drop_leakage: bool = True,
    return_transformer: bool = True,
    benchmark_sample_n: int = 2000
) -> Dict[str, Any]:
    """
    Orchestrate the pipeline. Adds runtime estimator into run_report.
    """
    preproc_params = preproc_params or {}
    trainer_params = trainer_params or {}

    # load data
    if isinstance(data, str):
        if data.endswith(".csv"):
            df = pd.read_csv(data)
        elif data.endswith(".parquet") or data.endswith(".pq"):
            df = pd.read_parquet(data)
        else:
            raise ValueError("Unsupported data path format - pass pandas.DataFrame or .csv/.parquet path")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("`data` must be a pandas DataFrame or a path to csv/parquet")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not present in data")

    df = df.reset_index(drop=True)
    df['_swindler_row_index_'] = df.index

    # split holdout
    stratify = df[target] if (not pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 1) else None
    try:
        work_df, holdout_df = train_test_split(
            df,
            test_size=holdout_size,
            random_state=holdout_random_state,
            stratify=stratify
        )
    except Exception:
        work_df, holdout_df = train_test_split(df, test_size=holdout_size, random_state=holdout_random_state)

    if verbose:
        print(f"Data sizes - total: {len(df)}, work: {len(work_df)}, holdout: {len(holdout_df)}")

    # profiling (sampled)
    profile_df = work_df
    if max_profile_samples and len(work_df) > max_profile_samples:
        profile_df = work_df.sample(n=max_profile_samples, random_state=random_state)
        if verbose:
            print(f"Profiling on sample of workset: {len(profile_df)} rows (max_profile_samples={max_profile_samples})")

    profiler = ModelDataProfiler(profile_df.reset_index(drop=True), target=target, verbose=verbose)
    profiler_report, profiler_model, X_prof, y_prof, df_prof = profiler.profile()

    # auto-drop leakage (never drop target)
    if auto_drop_leakage and getattr(profiler, "leakage_flags", None):
        leakage_cols = [c for c in profiler.leakage_flags.keys() if c != target]
        work_df = work_df.drop(columns=[c for c in leakage_cols if c in work_df.columns], errors='ignore')
        if verbose:
            print(f"Auto-removed {len(leakage_cols)} profiler-flagged leakage columns before preprocessing: {leakage_cols[:6]}{'...' if len(leakage_cols)>6 else ''}")

    # preprocessor
    preproc_params = dict(preproc_params)
    preproc_params.update({"output_sparse": output_sparse})
    preproc = ModelAwarePreprocessor(verbose=verbose, **preproc_params)
    preproc.configure_from_profiler(profiler_report)
    preproc.fit(work_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'), work_df[target])

    # estimate runtime before full transform/training:
    # build temporary trainer to obtain model candidates
    temp_trainer = ModelTrainer(data=None, X=np.zeros((1,1)), y=np.array([0]), feature_names=['f0'], task_type=_infer_task_type(work_df[target]), model_type=profiler_report.get("model_type", "non-linear"), verbose=False)
    # populate candidates by defining them
    temp_trainer._define_candidate_models()
    runtime_est = _estimate_runtime_via_benchmark(preproc, work_df, target, temp_trainer.models_, sample_n=benchmark_sample_n, cv_folds=temp_trainer.cv_folds)

    # transform workset and prepare trainer
    transformed_work = preproc.transform(work_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'))
    if isinstance(transformed_work, tuple):
        X_work, feature_names = transformed_work
        trainer = ModelTrainer(X=X_work, y=work_df[target].values, feature_names=feature_names, task_type=_infer_task_type(work_df[target]), model_type=profiler_report.get("model_type", "non-linear"), use_mlflow=use_mlflow, verbose=verbose, **trainer_params)
    else:
        X_work = transformed_work
        df_for_model = pd.concat([X_work.reset_index(drop=True), work_df[target].reset_index(drop=True)], axis=1)
        trainer = ModelTrainer(data=df_for_model, target_col=target, task_type=_infer_task_type(work_df[target]), model_type=profiler_report.get("model_type", "non-linear"), use_mlflow=use_mlflow, verbose=verbose, **trainer_params)

    trainer.run()

    training_index = work_df['_swindler_row_index_'].tolist()

    # evaluate on holdout
    transformed_holdout = preproc.transform(holdout_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'))
    if isinstance(transformed_holdout, tuple):
        X_hold, _ = transformed_holdout
    else:
        X_hold = transformed_holdout

    def _compute_holdout_metrics(task_type, model, X, y):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, r2_score, mean_squared_error
        metrics = {}
        y_pred = model.predict(X)
        if task_type == "classification":
            metrics['accuracy'] = float(accuracy_score(holdout_df[target], y_pred))
            metrics['f1'] = float(f1_score(holdout_df[target], y_pred, zero_division=0))
            metrics['precision'] = float(precision_score(holdout_df[target], y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(holdout_df[target], y_pred, zero_division=0))
            try:
                y_proba = model.predict_proba(X)[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(holdout_df[target], y_proba))
            except Exception:
                metrics['roc_auc'] = None
        else:
            metrics['r2'] = float(r2_score(holdout_df[target], y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(holdout_df[target], y_pred)))
        return metrics

    best_model = trainer.best_model_
    holdout_metrics = _compute_holdout_metrics(_infer_task_type(work_df[target]), best_model, X_hold if 'X_hold' in locals() else X_work, holdout_df[target].values)

    unique_id_columns = {
        'categorical_keys': getattr(profiler, "unique_categorical_keys_", []),
        'numeric_ids': getattr(profiler, "unique_numerical_ids_", [])
    }
    unique_id_values_in_holdout = {}
    for c in unique_id_columns.get('categorical_keys', []) + unique_id_columns.get('numeric_ids', []):
        if c in holdout_df.columns:
            unique_id_values_in_holdout[c] = holdout_df[c].dropna().unique().tolist()
        else:
            unique_id_values_in_holdout[c] = []

    # Save report and include runtime_est
    os.makedirs(output_dir, exist_ok=True)
    run_report = {
        "chosen_model_name": trainer.best_model_name_,
        "chosen_model_type": profiler_report.get("model_type", "non-linear"),
        "task_type": _infer_task_type(work_df[target]),
        "trainer_results": trainer.results_summary_.to_dict(orient="records") if hasattr(trainer.results_summary_, "to_dict") else str(trainer.results_summary_),
        "profiler_summary": profiler_report,
        "preprocessing": {
            "high_cardinality_strategy": preproc.high_cardinality_strategy,
            "n_hashing_features": preproc.n_hashing_features,
            "rare_category_threshold": preproc.rare_category_threshold,
            "scale_numeric": getattr(preproc, "scale_numeric", False),
            "final_feature_count": X_work.shape[1] if hasattr(X_work, "shape") else None
        },
        "holdout_metrics": holdout_metrics,
        "holdout_count": len(holdout_df),
        "training_index": training_index,
        "holdout_index": holdout_df['_swindler_row_index_'].tolist(),
        "unique_id_columns": unique_id_columns,
        "unique_id_values_in_holdout": unique_id_values_in_holdout,
        "runtime_estimate": runtime_est
    }

    run_path = os.path.join(output_dir, f"pipeline_run_{trainer.best_model_name_}.json")
    def _safe(obj):
        try:
            import pandas as _pd, numpy as _np
            if isinstance(obj, _pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, _pd.Series):
                return obj.tolist()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            return str(obj)
        except Exception:
            return str(obj)
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, default=_safe)

    payload = {
        "run_report_path": run_path,
        "run_report": run_report,
        "trainer_results_df": trainer.results_summary_,
        "best_model": trainer.best_model_,
        "best_model_name": trainer.best_model_name_,
        "preprocessor": preproc,
        "profiler": profiler,
        "X_transformed_shape": X_work.shape if hasattr(X_work, "shape") else None,
        "training_index": training_index,
        "holdout_index": holdout_df['_swindler_row_index_'].tolist(),
        "unique_id_columns": unique_id_columns,
        "unique_id_values_in_holdout": unique_id_values_in_holdout
    }

    if verbose:
        print("Pipeline finished. Report written to:", run_path)
        print("Best model:", trainer.best_model_name_)
        print("Holdout metrics:", holdout_metrics)
        print("Estimated wall-time for sweep:", runtime_est.get("human_readable_wall_time_est"))

    return payload