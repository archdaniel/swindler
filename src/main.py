# src/main.py
"""
Main orchestration for profiling, preprocessing, training, and reporting.
Now supports:
 - returning training indices,
 - options to produce sparse hashed output from preprocessor,
 - automatic dropping of profiler-detected leakage features prior to training,
 - returning fitted transformer for reuse.
"""
import os
import json
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from auto_definitions import ModelDataProfiler
from preprocessing import ModelAwarePreprocessor
from model_definitions import ModelTrainer
from sklearn import __version__ as sklearn_version

def _infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        return "regression"
    else:
        return "classification"

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
    # new options requested by user:
    output_sparse: bool = True,           # (A) return sparse hashed output from preprocessor
    auto_drop_leakage: bool = True,       # (B) automatically drop profiler-detected leakage features before training
    return_transformer: bool = True       # (C) always return fitted transformer by default
) -> Dict[str, Any]:
    """
    Orchestrate the pipeline. New behavior:
    - output_sparse (A): if True, preprocessor will emit sparse matrix for hashed features.
    - auto_drop_leakage (B): if True, remove profiler-detected leakage columns from workset before training.
    - A fitted transformer (preprocessor) is returned so callers can reuse preprocessing (C).
    """
    preproc_params = preproc_params or {}
    trainer_params = trainer_params or {}

    # load
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

    # ensure stable indexing to return indices used in training
    df = df.reset_index(drop=True)
    df['_swindler_row_index_'] = df.index

    # create holdout
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

    # profiling (sampled if requested)
    profile_df = work_df
    if max_profile_samples and len(work_df) > max_profile_samples:
        profile_df = work_df.sample(n=max_profile_samples, random_state=random_state)
        if verbose:
            print(f"Profiling on sample of workset: {len(profile_df)} rows (max_profile_samples={max_profile_samples})")

    profiler = ModelDataProfiler(profile_df.reset_index(drop=True), target=target, verbose=verbose)
    profiler_report, profiler_model, X_prof, y_prof, df_prof = profiler.profile()

    # Optionally drop profiler-detected leakage columns from the workset before fitting preprocessor
    if auto_drop_leakage and getattr(profiler, "leakage_flags", None):
        leakage_cols = [c for c in profiler.leakage_flags.keys() if c != target]
        # remove those columns from work_df (and from profile_df representation)
        work_df = work_df.drop(columns=[c for c in leakage_cols if c in work_df.columns], errors='ignore')
        if verbose:
            print(f"Auto-removed {len(leakage_cols)} profiler-flagged leakage columns before preprocessing: {leakage_cols[:6]}{'...' if len(leakage_cols)>6 else ''}")

    # configure preprocessor and fit on the full workset (not holdout)
    preproc_params = dict(preproc_params)
    preproc_params.update({"output_sparse": output_sparse})
    preproc = ModelAwarePreprocessor(verbose=verbose, **preproc_params)
    preproc.configure_from_profiler(profiler_report)
    # fit using only features (drop target and our internal index)
    preproc.fit(work_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'), work_df[target])

    transformed_work = preproc.transform(work_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'))

    # If preproc returned sparse tuple (sparse_matrix, feature_names)
    if isinstance(transformed_work, tuple):
        X_work, feature_names = transformed_work
        # pass X and y to ModelTrainer using the X,y signature
        trainer = ModelTrainer(X=X_work, y=work_df[target].values, feature_names=feature_names, task_type=_infer_task_type(work_df[target]), model_type=profiler_report.get("model_type", "non-linear"), use_mlflow=use_mlflow, verbose=verbose, **trainer_params)
    else:
        # DataFrame path
        X_work = transformed_work
        df_for_model = pd.concat([X_work.reset_index(drop=True), work_df[target].reset_index(drop=True)], axis=1)
        trainer = ModelTrainer(data=df_for_model, target_col=target, task_type=_infer_task_type(work_df[target]), model_type=profiler_report.get("model_type", "non-linear"), use_mlflow=use_mlflow, verbose=verbose, **trainer_params)

    trainer.run()

    # store training indices that were used (entire workset indices, not internal trainer split)
    training_index = work_df['_swindler_row_index_'].tolist()

    # Evaluate best model on holdout using the fitted preprocessor
    best_model = trainer.best_model_
    transformed_holdout = preproc.transform(holdout_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'))
    if isinstance(transformed_holdout, tuple):
        X_hold, _ = transformed_holdout
    else:
        X_hold = transformed_holdout

    # Compute holdout metrics (simple)
    def _compute_holdout_metrics(task_type, model, X, y):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, r2_score, mean_squared_error
        metrics = {}
        try:
            y_pred = model.predict(X)
        except Exception:
            y_pred = model.predict(X)  # try anyway
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

    holdout_metrics = _compute_holdout_metrics(_infer_task_type(work_df[target]), best_model, X_work if 'X_hold' not in locals() else X_hold, holdout_df[target].values)

    # unique id columns and values in holdout for downstream exclusion
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

    # Save compact run report
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
    return payload