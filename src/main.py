# src/main.py
"""
Main orchestration for profiling, preprocessing, training, and reporting.

Key changes:
- Split incoming dataframe into workset (for profiling, preprocessing, training) and holdout (final evaluation).
- Profiler runs on workset (optionally sampled for memory), preprocessor is fitted on workset only.
- Trainer trains on workset; after selecting best model, we evaluate it on holdout using the fitted transformer.
- Returns the fitted transformer and metadata needed to ensure no overlap (holdout_index, unique id columns/values).
"""
import os
import json
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error
)

# local imports (assumes this file is placed in src/)
from auto_definitions import ModelDataProfiler
from preprocessing import ModelAwarePreprocessor
from model_definitions import ModelTrainer

def _infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        return "regression"
    else:
        return "classification"

def _compute_holdout_metrics(task_type: str, model, X_hold, y_hold) -> Dict[str, Any]:
    """Compute simple evaluation metrics on holdout set."""
    metrics = {}
    if task_type == "classification":
        # models may or may not have predict_proba
        y_pred = model.predict(X_hold)
        try:
            y_proba = model.predict_proba(X_hold)[:, 1]
        except Exception:
            # fallback: if no predict_proba, use decision_function if available
            try:
                scores = model.decision_function(X_hold)
                # convert to probabilities via logistic-ish mapping for rough AUC
                y_proba = 1 / (1 + np.exp(-scores))
            except Exception:
                y_proba = None

        metrics['accuracy'] = float(accuracy_score(y_hold, y_pred))
        metrics['f1'] = float(f1_score(y_hold, y_pred, zero_division=0))
        metrics['precision'] = float(precision_score(y_hold, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_hold, y_pred, zero_division=0))
        if y_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_hold, y_proba))
            except Exception:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None

    else:
        y_pred = model.predict(X_hold)
        metrics['r2'] = float(r2_score(y_hold, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_hold, y_pred)))

    return metrics

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
    preproc_params: dict = None,
    trainer_params: dict = None,
) -> Dict[str, Any]:
    """
    Orchestrate the full pipeline with a strict holdout split to prevent leakage.

    Parameters
    ----------
    - data: DataFrame or path to csv/parquet
    - target: name of target column
    - holdout_size: fraction to reserve as final holdout (default 0.10)
    - max_profile_samples: if set and workset larger than this, profiler will run on a sample of workset
      (profiling is only sampling; training uses the full workset)
    - preproc_params: dict passed to ModelAwarePreprocessor(...)
    - trainer_params: dict passed to ModelTrainer(...)
    """
    if preproc_params is None:
        preproc_params = {}
    if trainer_params is None:
        trainer_params = {}

    # Load dataframe
    if isinstance(data, str):
        # accept csv/parquet path
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

    # 0) Create a reproducible index if none present; we use it to guarantee non-overlap and to return back to user
    df = df.reset_index(drop=True)
    df['_swindler_row_index_'] = df.index  # preserved unique row ids for tracing

    # 1) Create final holdout (completely unseen by profiler/preprocessor/trainer)
    stratify = df[target] if (pd.api.types.is_numeric_dtype(df[target]) is False and df[target].nunique() > 1) else None
    # Use stratify for classification when possible
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() <= 10:
        # treat small-unique numeric as categorical
        stratify = df[target]
    try:
        work_df, holdout_df = train_test_split(
            df,
            test_size=holdout_size,
            random_state=holdout_random_state,
            stratify=stratify
        )
    except Exception:
        # fallback if stratify invalid
        work_df, holdout_df = train_test_split(df, test_size=holdout_size, random_state=holdout_random_state)

    if verbose:
        print(f"Data loaded: total={len(df):,}, workset={len(work_df):,}, holdout={len(holdout_df):,}")

    # 2) Run profiler on workset. Optionally sample workset for profiling (saves memory/time)
    profile_df_for_run = work_df
    if max_profile_samples is not None and len(work_df) > max_profile_samples:
        profile_df_for_run = work_df.sample(n=max_profile_samples, random_state=random_state)
        if verbose:
            print(f"Profiling on sample of workset: {len(profile_df_for_run):,} rows (max_profile_samples={max_profile_samples})")

    profiler = ModelDataProfiler(profile_df_for_run.reset_index(drop=True), target=target, verbose=verbose)
    profiler_report, _, _, _, _ = profiler.profile()

    # 3) Configure preprocessor based on profiler and fit it on the *full* workset (NOT on holdout).
    preproc = ModelAwarePreprocessor(verbose=verbose, **preproc_params)
    preproc.configure_from_profiler(profiler_report)

    # Fit preprocessor on workset (full workset)
    preproc.fit(work_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'), work_df[target])

    # Transform workset (for trainer) and keep fitted transformer for later holdout transform
    transformed_work = preproc.transform(work_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'))
    if isinstance(transformed_work, tuple):
        X_work_transformed, cat_feature_list = transformed_work
    else:
        X_work_transformed = transformed_work
        cat_feature_list = []

    y_work = work_df[target].reset_index(drop=True)

    df_for_model = pd.concat([X_work_transformed.reset_index(drop=True), y_work.reset_index(drop=True)], axis=1)

    # 4) Decide task and model_type (from profiler diagnostics or inferred)
    task_type = _infer_task_type(y_work)
    diag = profiler_report.get("diagnostics", {})
    rec_type = diag.get("recommendation_type", None) or diag.get("recommendation", None)
    if isinstance(rec_type, str) and "non" in rec_type.lower():
        chosen_model_type = "non-linear"
    elif isinstance(rec_type, str) and "param" in rec_type.lower():
        chosen_model_type = "linear"
    else:
        # fallback: use profiler.model_type if present
        chosen_model_type = profiler_report.get("model_type", "non-linear") or "non-linear"
        chosen_model_type = "linear" if "param" in str(chosen_model_type).lower() else "non-linear"

    if verbose:
        print(f"Training models for task={task_type}, chosen_model_type={chosen_model_type}")

    # 5) Train models (trainer sees only df_for_model -> workset)
    trainer = ModelTrainer(
        data=df_for_model,
        target_col=target,
        task_type=task_type,
        model_type=chosen_model_type,
        use_mlflow=use_mlflow,
        verbose=verbose,
        **trainer_params
    )
    trainer.run()

    # 6) After selecting best model, evaluate it on the separated holdout (never seen by profiler/preproc/trainer)
    best_model = trainer.best_model_
    # transform holdout using the same fitted preprocessor
    transformed_holdout = preproc.transform(holdout_df.drop(columns=[target, '_swindler_row_index_'], errors='ignore'))
    if isinstance(transformed_holdout, tuple):
        X_hold_transformed, _ = transformed_holdout
    else:
        X_hold_transformed = transformed_holdout
    y_hold = holdout_df[target].reset_index(drop=True)

    # Some models (catboost) expect numpy; ensure we pass arrays or DataFrame as needed
    holdout_metrics = _compute_holdout_metrics(task_type, best_model, X_hold_transformed, y_hold)

    # 7) Gather unique id columns and values in holdout so user can remove overlap later
    unique_id_columns = {
        'categorical_keys': getattr(profiler, "unique_categorical_keys_", []),
        'numeric_ids': getattr(profiler, "unique_numerical_ids_", [])
    }
    # map id columns -> list of unique values found in holdout
    unique_id_values_in_holdout = {}
    for c in unique_id_columns.get('categorical_keys', []) + unique_id_columns.get('numeric_ids', []):
        if c in holdout_df.columns:
            unique_id_values_in_holdout[c] = holdout_df[c].dropna().unique().tolist()
        else:
            unique_id_values_in_holdout[c] = []

    # 8) Save run report + profiler details with JSON-safe serializer
    os.makedirs(output_dir, exist_ok=True)
    run_report = {
        "chosen_model_name": trainer.best_model_name_,
        "chosen_model_type": chosen_model_type,
        "task_type": task_type,
        "trainer_results": trainer.results_summary_.to_dict(orient="records") if hasattr(trainer.results_summary_, "to_dict") else str(trainer.results_summary_),
        "profiler_summary": profiler_report,
        "preprocessing": {
            "high_cardinality_strategy": preproc.high_cardinality_strategy,
            "n_hashing_features": preproc.n_hashing_features,
            "rare_category_threshold": preproc.rare_category_threshold,
            "scale_numeric": getattr(preproc, "scale_numeric", False),
            "categorical_feature_names_for_catboost": getattr(preproc, "categorical_feature_names_", []),
            "final_feature_count": X_work_transformed.shape[1] if hasattr(X_work_transformed, "shape") else None
        },
        "holdout_metrics": holdout_metrics,
        "holdout_count": len(holdout_df),
        "unique_id_columns": unique_id_columns,
        "unique_id_values_in_holdout": unique_id_values_in_holdout,
    }

    def _safe(obj):
        try:
            import pandas as _pd
            import numpy as _np
            if isinstance(obj, _pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, _pd.Series):
                return obj.tolist()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            return str(obj)
        except Exception:
            return str(obj)

    run_path = os.path.join(output_dir, f"pipeline_run_{trainer.best_model_name_}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, default=_safe)

    payload = {
        "run_report_path": run_path,
        "run_report": run_report,
        "trainer_results_df": trainer.results_summary_,
        "best_model": best_model,
        "best_model_name": trainer.best_model_name_,
        "preprocessor": preproc,            # fitted transformer returned for reuse
        "profiler": profiler,               # profiler object (for further inspection)
        "X_transformed_shape": X_work_transformed.shape if hasattr(X_work_transformed, "shape") else None,
        "holdout_index": holdout_df['_swindler_row_index_'].tolist(),
        "unique_id_columns": unique_id_columns,
        "unique_id_values_in_holdout": unique_id_values_in_holdout
    }

    if verbose:
        print("Pipeline finished. Report written to:", run_path)
        print("Best model:", trainer.best_model_name_)
        print("Holdout metrics:", holdout_metrics)

    return payload