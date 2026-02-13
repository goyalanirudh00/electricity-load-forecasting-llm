"""Minimal progress logging for the forecasting pipeline."""


def _ts_str(ts) -> str:
    """Format timestamp for display."""
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M")
    return str(ts)


def log(msg: str) -> None:
    """Print a log line."""
    print(msg)


def log_section(title: str) -> None:
    """Print a section header."""
    print(f"\n=== {title} ===\n")


def log_data_loaded(path: str, utility_name: str) -> None:
    """Log that a utility's data was loaded."""
    print(f"  Loaded {utility_name} from {path}")


def log_baseline_start(utility_name: str) -> None:
    """Log baseline evaluation start for a utility."""
    print(f"  {utility_name} baseline ... ", end="", flush=True)


def log_baseline_done() -> None:
    """Log baseline evaluation complete (same line)."""
    print("done", flush=True)


def log_llm_start(utility_name: str, n_origins: int) -> None:
    """Log LLM evaluation start for a utility."""
    print(f"  {utility_name} ({n_origins} forecasts):")


def log_forecast_start(
    utility_name: str,
    origin_ts,
    current: int,
    total: int,
) -> None:
    """Log a forecast in progress (call before API)."""
    ts = _ts_str(origin_ts)
    print(f"    [{current}/{total}] {utility_name} | {ts} | ... ", end="", flush=True)


def log_forecast_done() -> None:
    """Log forecast complete (same line, after log_forecast_start)."""
    print("✓", flush=True)


def log_metrics(name: str, mae: float, rmse: float, mape: float) -> None:
    """Log metric line for a utility."""
    print(f"  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")


def log_mean_metrics(mae: float, rmse: float, mape: float) -> None:
    """Log mean metrics line."""
    print(f"  Mean: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")


def log_results_saved(path: str) -> None:
    """Log that results were saved."""
    print(f"\nResults saved to {path}", flush=True)


def log_run_start() -> None:
    """Log pipeline start."""
    print("Electricity load forecast evaluation\n", flush=True)
