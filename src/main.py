# import other functionalities, with redefinitions if necessary. 
# load trained and warm started models to this file. 
# call for APIs and with the endpoints exposed. 

#  url=https://github.com/archdaniel/swindler/blob/main/src/main.py
# src/main.py
"""
Main orchestration for profiling, preprocessing, training, and reporting.

Usage:
    from main import run_pipeline
    report, model, X, y, df = run_pipeline(dataframe_or_path, target, verbose=True)

This script:
- runs ModelDataProfiler to gather diagnostics and a recommendation
- configures ModelAwarePreprocessor accordingly
- preprocesses the data
- runs ModelTrainer to train and tune candidate models
- returns a consolidated report and saves JSON output
"""
import os
import pandas as pd
from typing import Any, Dict, Tuple

# local imports (assumes this file is placed in src/)
from auto_definitions import ModelDataProfiler
from preprocessing import ModelAwarePreprocessor
from model_definitions import ModelTrainer

def _infer_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        return "regression"
    else:
        return "classification"

def run_pipeline(data: Any, target: str, output_dir: str = "pipeline_reports", verbose: bool = True, use_mlflow: bool = False) -> Dict[str, Any]:
    """
    Orchestrate the full pipeline.

    Returns a dictionary with keys:
      - profiler_report: dict from ModelDataProfiler
      - preprocessing_info: dict about chosen preprocessing
      - trainer_report: trainer.results_summary_ as DataFrame (converted to dict)
      - best_model_name, best_params, model_object
    """
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

    # 1) Profile data and model assumptions
    profiler = ModelDataProfiler(df, target=target, verbose=verbose)
    profiler_report, profiler_model, X_prof, y_prof, df_prof = profiler.profile()

    # 2) Configure preprocessor from profiler suggestion
    preproc = ModelAwarePreprocessor(verbose=verbose)
    # set some sensible defaults / carry over thresholds if present
    # let configure_from_profiler set the strategy
    preproc.configure_from_profiler(profiler_report)

    # Fit & transform
    preproc.fit(df_prof.drop(columns=[target], errors='ignore'), df_prof[target])
    transformed = preproc.transform(df_prof.drop(columns=[target], errors='ignore'))
    # transform may return (X, cat_feature_list) if CatBoost-style requested
    if isinstance(transformed, tuple):
        X_transformed, cat_feature_list = transformed
    else:
        X_transformed = transformed
        cat_feature_list = []

    y = df_prof[target].reset_index(drop=True)
    # assemble final dataframe for ModelTrainer
    df_for_model = pd.concat([X_transformed.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    # 3) Decide task and model_type
    task_type = _infer_task_type(y)
    # Map profiler recommendation to ModelTrainer.model_type
    diag = profiler_report.get("diagnostics", {})
    rec_type = diag.get("recommendation_type", None) or diag.get("recommendation", None)
    if isinstance(rec_type, str) and "parametric" in rec_type.lower():
        chosen_model_type = "linear"
    else:
        # default to non-linear (trees)
        chosen_model_type = "non-linear"

    # 4) Train and tune models
    trainer = ModelTrainer(
        data=df_for_model,
        target_col=target,
        task_type=task_type,
        model_type=chosen_model_type,
        use_mlflow=use_mlflow,
        verbose=verbose
    )
    trainer.run()

    # 5) Collate results
    trainer_report_df = trainer.results_summary_
    best_model_name = trainer.best_model_name_
    best_model = trainer.best_model_
    model_metrics = trainer_report_df[trainer_report_df['model'] == best_model_name].to_dict('records')

    # Save a compact run report using profiler's save_report and a local JSON file
    os.makedirs(output_dir, exist_ok=True)
    run_report = {
        "chosen_model_name": best_model_name,
        "chosen_model_type": chosen_model_type,
        "task_type": task_type,
        "model_metrics": model_metrics,
        "profiler_summary": profiler_report,
        "preprocessing": {
            "high_cardinality_strategy": preproc.high_cardinality_strategy,
            "n_hashing_features": preproc.n_hashing_features,
            "rare_category_threshold": preproc.rare_category_threshold,
            "scale_numeric": preproc.scale_numeric,
            "categorical_feature_names_for_catboost": getattr(preproc, "categorical_feature_names_", []),
            "final_feature_count": X_transformed.shape[1]
        }
    }

    # Use profiler.save_report (which we updated to be JSON-safe) to save profiler details
    try:
        # profiler.save_report will save to reports/ by default (and use serializer)
        profiler.save_report(output_dir, prefix=f"model_profile_{best_model_name}")
    except Exception as e:
        if verbose:
            print(f"⚠️ profiler.save_report failed: {e}")

    # Write run_report to JSON (safe serializer)
    import json
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

    run_path = os.path.join(output_dir, f"pipeline_run_{best_model_name}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, default=_safe)

    # Final returned payload
    payload = {
        "run_report_path": run_path,
        "run_report": run_report,
        "trainer_results_df": trainer_report_df,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "preprocessor": preproc,
        "profiler": profiler,
        "X_transformed_shape": X_transformed.shape
    }

    if verbose:
        print("Pipeline finished. Report written to:", run_path)
        print("Best model:", best_model_name)

    return payload