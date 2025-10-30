"""
Lightweight API exposing ModelDataProfiler and pipeline endpoints.

Endpoints:
- POST /diagnose : Upload CSV (multipart/form-data) + form field 'target'
    → runs ModelDataProfiler and returns textual summary + report dict
- POST /run_pipeline : Upload CSV + 'target' (and optional use_mlflow flag)
    → runs full pipeline via main.run_pipeline and returns compact run summary
- GET /health : returns {"status":"ok"}

Usage:
    uvicorn src.api:app --reload --port 8000
"""
import io
import os
import json
import traceback
import contextlib
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# local imports (assumes package layout as in your repo)
from auto_definitions import ModelDataProfiler
from main import run_pipeline

app = FastAPI(title="Swindler Profiler / Pipeline API")

def _safe_json_default(obj):
    """JSON default to handle pandas / numpy / Timestamps."""
    try:
        import pandas as _pd
        import numpy as _np
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, _pd.Series):
            return obj.tolist()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, )):
            return int(obj)
        if isinstance(obj, (np.floating, )):
            return float(obj)
        if isinstance(obj, (np.bool_, )):
            return bool(obj)
    except Exception:
        pass
    return str(obj)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...), target: str = Form(...)):
    """
    Upload a CSV file and the name of the target column. Runs ModelDataProfiler and
    returns:
      - summary_text: captured output of profiler.summarize_profile(detailed=True)
      - report: profiler.to_report_dict() (JSON-serializable)
    """
    try:
        contents = await file.read()
        # try to read as CSV first (user asked CSV)
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        # fallback: try parquet
        try:
            df = pd.read_parquet(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not read uploaded file as CSV or Parquet: {e}")

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target '{target}' not found in uploaded data columns: {df.columns.tolist()}")

    # Run profiler
    profiler = ModelDataProfiler(df, target=target, verbose=False)
    report, model, X, y, df_fixed = profiler.profile()

    # Capture textual detailed summary
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            profiler.summarize_profile(detailed=True)
        except Exception:
            # fallback to concise printing if summarize_profile raises
            print("Failed to print full summary:", traceback.format_exc())
    summary_text = buf.getvalue()

    # Return JSON-safe report
    try:
        report_dict = profiler.to_report_dict()
    except Exception:
        # fallback: build a simpler report
        report_dict = {
            "model_type": report.get("model_type"),
            "diagnostics": report.get("diagnostics"),
            "date_features": report.get("date_features"),
            "leakage_flags_count": len(report.get("leakage_flags", {})),
        }

    return JSONResponse({"summary_text": summary_text, "report": report_dict}, default=_safe_json_default)

@app.post("/run_pipeline")
async def run_pipeline_endpoint(
    file: UploadFile = File(...),
    target: str = Form(...),
    use_mlflow: bool = Form(False)
):
    """
    Upload a CSV and target column. Runs the full pipeline (profiling => preprocessing => training)
    using main.run_pipeline. Returns a compact run_report and the path to saved artifacts.

    WARNING: This endpoint runs model training synchronously — it can be slow.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        # try parquet
        try:
            df = pd.read_parquet(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not read uploaded file as CSV or Parquet: {e}")

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target '{target}' not found in uploaded data columns: {df.columns.tolist()}")

    try:
        # run the pipeline (this will train models)
        payload = run_pipeline(df, target=target, output_dir="pipeline_reports", verbose=False, use_mlflow=use_mlflow)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}\n{tb}")

    # Extract a compact summary to return (avoid returning non-serializable model objects)
    run_report = payload.get("run_report", {})
    trainer_results = payload.get("trainer_results_df")
    if hasattr(trainer_results, "to_dict"):
        trainer_results_json = trainer_results.to_dict(orient="records")
    else:
        trainer_results_json = str(trainer_results)

    response = {
        "run_report_path": payload.get("run_report_path"),
        "run_report": run_report,
        "trainer_results": trainer_results_json,
        "best_model_name": payload.get("best_model_name"),
        "x_transformed_shape": payload.get("X_transformed_shape"),
    }

    return JSONResponse(response, default=_safe_json_default)