"""Evaluation metrics for load forecasting: MAE, RMSE, MAPE."""

from typing import List, Optional

import numpy as np

EPSILON = 1e-6


def mae(actual: List[float], predicted: List[float]) -> float:
    """Mean Absolute Error.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        MAE value.
    """
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    if len(actual_arr) != len(pred_arr):
        raise ValueError(
            f"Length mismatch: actual={len(actual_arr)}, predicted={len(pred_arr)}"
        )
    return float(np.mean(np.abs(actual_arr - pred_arr)))


def rmse(actual: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        RMSE value.
    """
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    if len(actual_arr) != len(pred_arr):
        raise ValueError(
            f"Length mismatch: actual={len(actual_arr)}, predicted={len(pred_arr)}"
        )
    return float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))


def mape(
    actual: List[float],
    predicted: List[float],
    epsilon: float = EPSILON,
) -> float:
    """Mean Absolute Percentage Error with epsilon safeguard.

    MAPE = mean(|actual - predicted| / max(|actual|, epsilon)) * 100

    Uses epsilon to avoid division by zero when actual values are very small.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.
        epsilon: Minimum denominator to avoid division by zero (default 1e-6).

    Returns:
        MAPE value as a percentage (e.g. 5.0 for 5%).
    """
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    if len(actual_arr) != len(pred_arr):
        raise ValueError(
            f"Length mismatch: actual={len(actual_arr)}, predicted={len(pred_arr)}"
        )
    denom = np.maximum(np.abs(actual_arr), epsilon)
    return float(np.mean(np.abs(actual_arr - pred_arr) / denom) * 100.0)


def compute_all_metrics(
    actual: List[float],
    predicted: List[float],
    epsilon: float = EPSILON,
) -> dict:
    """Compute MAE, RMSE, and MAPE.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.
        epsilon: Epsilon for MAPE.

    Returns:
        Dict with keys 'mae', 'rmse', 'mape'.
    """
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted, epsilon=epsilon),
    }
