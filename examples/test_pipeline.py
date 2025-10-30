"""
Example script to test the pipeline locally with a toy CSV dataset.

Usage:
    python examples/test_pipeline.py --data path/to/toy.csv --target target_column

This script:
- Loads the toy CSV
- Calls the pipeline runner (main.run_pipeline)
- Prints a short summary and the location of saved run report
"""
import argparse
import pandas as pd
from main import run_pipeline

def main(data_path: str, target: str):
    print("Loading dataset:", data_path)
    df = pd.read_csv(data_path)
    print("Data shape:", df.shape)
    print("Target:", target)

    print("Running pipeline — this may take a while depending on models and data size...")
    payload = run_pipeline(df, target=target, output_dir="pipeline_reports", verbose=True, use_mlflow=False)

    print("\n=== PIPELINE SUMMARY ===")
    print("Run report path:", payload.get("run_report_path"))
    run_report = payload.get("run_report", {})
    print("Chosen model:", run_report.get("chosen_model_name"))
    print("Chosen model type:", run_report.get("chosen_model_type"))
    print("Task type:", run_report.get("task_type"))
    print("Final feature count:", run_report.get("preprocessing", {}).get("final_feature_count"))
    print("Trainer results (summary):")
    print(payload.get("trainer_results_df"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to toy CSV dataset")
    parser.add_argument("--target", required=True, help="Name of the target column")
    args = parser.parse_args()
    main(args.data, args.target)