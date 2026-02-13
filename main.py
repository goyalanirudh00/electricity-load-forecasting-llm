#!/usr/bin/env python3
"""CLI entry point: load utilities, run evaluation, report metrics."""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from data import load_and_prepare
from evaluator import evaluate_utility_baseline, evaluate_utility_with_llm
from llm_forecaster import forecast as llm_forecast
from logger import (
    log_data_loaded,
    log_mean_metrics,
    log_metrics,
    log_results_saved,
    log_run_start,
    log_section,
)


def _llm_forecast_fn(origin_summary, hod_profile, recent_history):
    """Wrapper that falls back to NotImplementedError when API key is missing."""
    try:
        return llm_forecast(origin_summary, hod_profile, recent_history)
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            raise NotImplementedError("OPENAI_API_KEY not set; using seasonal naive") from e
        raise


def main() -> None:
    """Load all 3 utilities, run evaluation, print metrics, save results."""
    log_run_start()
    base = Path(__file__).resolve().parent
    csv_patterns = [
        "Utility_1_data.csv",
        "Utility_2_data.csv",
        "Utility_3_data.csv",
    ]

    baseline_results: dict = {}
    llm_results: dict = {}

    for pattern in csv_patterns:
        path = base / pattern
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {pattern}")

        df = load_and_prepare(path)
        utility_name = df["utility_name"].iloc[0]
        log_data_loaded(path.name, utility_name)

        # Baseline evaluation
        base_metrics, _, _ = evaluate_utility_baseline(df, utility_name)
        baseline_results[utility_name] = base_metrics

        # LLM evaluation (falls back to seasonal naive if OPENAI_API_KEY not set)
        llm_metrics, _, _ = evaluate_utility_with_llm(
            df,
            utility_name,
            forecast_fn=_llm_forecast_fn,
        )
        llm_results[utility_name] = llm_metrics

    # Mean metrics across utilities
    def mean_metrics(results: dict) -> dict:
        mae_vals = [r["mae"] for r in results.values()]
        rmse_vals = [r["rmse"] for r in results.values()]
        mape_vals = [r["mape"] for r in results.values()]
        return {
            "mae": sum(mae_vals) / len(mae_vals),
            "rmse": sum(rmse_vals) / len(rmse_vals),
            "mape": sum(mape_vals) / len(mape_vals),
        }

    baseline_mean = mean_metrics(baseline_results)
    llm_mean = mean_metrics(llm_results)

    # Print results
    log_section("Baseline (Seasonal Naive)")
    for name, m in baseline_results.items():
        log_metrics(name, m["mae"], m["rmse"], m["mape"])
    log_mean_metrics(baseline_mean["mae"], baseline_mean["rmse"], baseline_mean["mape"])

    log_section("LLM Forecaster")
    for name, m in llm_results.items():
        log_metrics(name, m["mae"], m["rmse"], m["mape"])
    log_mean_metrics(llm_mean["mae"], llm_mean["rmse"], llm_mean["mape"])

    out_path = base / "results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "baseline": {"per_utility": baseline_results, "mean": baseline_mean},
                "llm": {"per_utility": llm_results, "mean": llm_mean},
            },
            f,
            indent=2,
        )
    log_results_saved(str(out_path))


if __name__ == "__main__":
    main()
