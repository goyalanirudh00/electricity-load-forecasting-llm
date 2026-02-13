"""Rolling-origin evaluation: select origins, run forecasts, aggregate metrics."""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from baseline import seasonal_naive_forecast
from data import (
    get_daily_origins,
    get_window_at,
    select_last_n_valid_origins,
)
from logger import log_baseline_done, log_baseline_start, log_forecast_done, log_forecast_start, log_llm_start
from metrics import EPSILON, compute_all_metrics

N_ORIGINS = 14
MIN_HISTORY_HOURS = 336  # 14 days
MIN_FUTURE_HOURS = 24
HORIZON = 24


def get_actual_next_24h(df: pd.DataFrame, origin_idx: int) -> List[float]:
    """Extract actual load values for the next 24 hours after origin.

    Args:
        df: Full DataFrame.
        origin_idx: Index of origin (midnight).

    Returns:
        List of 24 actual load values.
    """
    start = origin_idx + 1
    end = origin_idx + 1 + HORIZON
    return df.iloc[start:end]["load"].astype(float).tolist()


def evaluate_utility_baseline(
    df: pd.DataFrame,
    utility_name: str,
) -> Tuple[Dict[str, float], List[float], List[float]]:
    """Evaluate seasonal naive baseline for one utility.

    Args:
        df: Prepared DataFrame for one utility.
        utility_name: Name of the utility (for logging).

    Returns:
        Tuple of (metrics_dict, all_actual, all_predicted).
    """
    origins = get_daily_origins(df)
    valid_origins = select_last_n_valid_origins(
        df,
        origins,
        n_origins=N_ORIGINS,
        min_history_hours=MIN_HISTORY_HOURS,
        min_future_hours=MIN_FUTURE_HOURS,
    )

    all_actual: List[float] = []
    all_predicted: List[float] = []

    log_baseline_start(utility_name)
    for origin_idx in valid_origins:
        actual = get_actual_next_24h(df, origin_idx)
        predicted = seasonal_naive_forecast(df, origin_idx, horizon=HORIZON)
        all_actual.extend(actual)
        all_predicted.extend(predicted)
    log_baseline_done()

    metrics = compute_all_metrics(all_actual, all_predicted, epsilon=EPSILON)
    return metrics, all_actual, all_predicted


def evaluate_utility_with_llm(
    df: pd.DataFrame,
    utility_name: str,
    forecast_fn: Callable[
        [
            str,
            List[float],
            List[Tuple[str, float]],
        ],
        List[float],
    ],
) -> Tuple[Dict[str, float], List[float], List[float]]:
    """Evaluate LLM forecaster for one utility.

    For each origin: build origin summary, HOD profile, recent history; run forecast;
    compare to actual next 24 hours. Falls back to seasonal naive if forecast_fn
    raises NotImplementedError.

    Args:
        df: Prepared DataFrame for one utility.
        utility_name: Name of the utility.
        forecast_fn: Function (origin_summary, hod_profile, recent_history) -> 24 floats.

    Returns:
        Tuple of (metrics_dict, all_actual, all_predicted) for the LLM model.
    """
    from context import (
        build_hour_of_day_profile,
        build_recent_history,
        build_origin_summary,
    )

    origins = get_daily_origins(df)
    valid_origins = select_last_n_valid_origins(
        df,
        origins,
        n_origins=N_ORIGINS,
        min_history_hours=MIN_HISTORY_HOURS,
        min_future_hours=MIN_FUTURE_HOURS,
    )

    all_actual: List[float] = []
    all_predicted: List[float] = []

    n_origins = len(valid_origins)
    log_llm_start(utility_name, n_origins)

    for i, origin_idx in enumerate(valid_origins):
        origin_ts = df.iloc[origin_idx]["timestamp"]
        window_14d = get_window_at(df, origin_idx, n_hours=336)
        origin_summary = build_origin_summary(window_14d)
        hod_profile = build_hour_of_day_profile(window_14d)
        recent_history = build_recent_history(df, origin_idx)

        actual = get_actual_next_24h(df, origin_idx)

        log_forecast_start(utility_name, origin_ts, i + 1, n_origins)
        try:
            predicted = forecast_fn(origin_summary, hod_profile, recent_history)
        except NotImplementedError:
            predicted = seasonal_naive_forecast(df, origin_idx, horizon=HORIZON)
        log_forecast_done()

        if len(predicted) != HORIZON:
            raise ValueError(
                f"Forecast must return {HORIZON} values, got {len(predicted)}"
            )

        all_actual.extend(actual)
        all_predicted.extend(predicted)

    metrics = compute_all_metrics(all_actual, all_predicted, epsilon=EPSILON)
    return metrics, all_actual, all_predicted
