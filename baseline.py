"""Seasonal naive baseline forecast for electricity load."""

from typing import List

import numpy as np
import pandas as pd


def seasonal_naive_forecast(
    df: pd.DataFrame,
    origin_idx: int,
    horizon: int = 24,
    season_length: int = 24,
) -> List[float]:
    """Produce seasonal naive forecast for the next horizon hours.

    For each h from 1 to horizon:
    Predicted load at T+h = actual load at T+h-24 (same hour, previous day).

    Uses only data at or before origin_idx (no leakage).

    Args:
        df: Sorted DataFrame with 'load' column.
        origin_idx: Index of the origin timestamp (T). Row at origin_idx has load at T.
        horizon: Number of hours to forecast (default 24).
        season_length: Seasonal period in hours (default 24 for daily seasonality).

    Returns:
        List of horizon predicted values (floats).
    """
    loads = df["load"].values
    n = len(loads)
    predictions: List[float] = []
    for h in range(1, horizon + 1):
        lookback_idx = origin_idx + h - season_length
        if lookback_idx < 0:
            raise ValueError(
                f"Seasonal naive requires at least {season_length} hours before origin. "
                f"For h={h}, need index {lookback_idx} which is negative."
            )
        pred = loads[lookback_idx]
        predictions.append(float(pred))
    return predictions
