"""Context builder: level, trend, seasonality, weekend, ramp features for LLM input."""

from typing import List, Tuple

import numpy as np
import pandas as pd

TREND_THRESHOLD_PCT = 0.003  # 0.3% per day
WEEKEND_THRESHOLD_PCT = 0.03  # 3%


def _last_load(window: pd.DataFrame) -> float:
    """Last load value (at T)."""
    return float(window["load"].iloc[-1])


def _mean_last_24h(window: pd.DataFrame) -> float:
    """Mean of last 24 hours ending at T."""
    tail = window.tail(24)
    return float(tail["load"].mean())


def _mean_last_7d(window: pd.DataFrame) -> float:
    """Mean of last 168 hours (7 days) ending at T."""
    tail = window.tail(168)
    return float(tail["load"].mean())


def _daily_means_14d(window: pd.DataFrame) -> np.ndarray:
    """Daily average load for each of the 14 days. Returns 14 values."""
    df = window.copy()
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["load"].mean()
    return daily.values


def _trend_slope_and_pct(daily_means: np.ndarray) -> Tuple[float, float]:
    """Fit line, return slope per day and slope/mean as fraction."""
    x = np.arange(len(daily_means))
    coeffs = np.polyfit(x, daily_means, 1)
    slope = coeffs[0]
    mean_level = np.mean(daily_means)
    pct = slope / mean_level if mean_level != 0 else 0.0
    return slope, pct


def _hour_of_day_means(window: pd.DataFrame) -> np.ndarray:
    """Mean load per hour 0..23. Returns 24 values."""
    df = window.copy()
    df["hour"] = df["timestamp"].dt.hour
    hod = df.groupby("hour")["load"].mean()
    result = np.zeros(24)
    for h in range(24):
        result[h] = hod.get(h, 0.0)
    return result


def _peak_trough_amplitude(hod_means: np.ndarray) -> Tuple[int, int, float]:
    """Peak hour, trough hour, amplitude (peak - trough)."""
    peak_hour = int(np.argmax(hod_means))
    trough_hour = int(np.argmin(hod_means))
    amplitude = hod_means[peak_hour] - hod_means[trough_hour]
    return peak_hour, trough_hour, float(amplitude)


def _weekday_weekend_means(window: pd.DataFrame) -> Tuple[float, float]:
    """Weekday mean and weekend mean. Saturday=5, Sunday=6 are weekend."""
    df = window.copy()
    df["dow"] = df["timestamp"].dt.dayofweek
    weekday_mask = (df["dow"] >= 0) & (df["dow"] <= 4)
    weekend_mask = (df["dow"] >= 5)
    weekday_mean = float(df.loc[weekday_mask, "load"].mean())
    weekend_mean = float(df.loc[weekend_mask, "load"].mean())
    return weekday_mean, weekend_mean


def _weekend_delta_pct(weekday_mean: float, weekend_mean: float) -> float:
    """(weekend_mean - weekday_mean) / weekday_mean."""
    if weekday_mean == 0:
        return 0.0
    return (weekend_mean - weekday_mean) / weekday_mean


def _ramp_percentiles(window: pd.DataFrame) -> Tuple[float, float]:
    """p95 and p99 of absolute hour-to-hour differences."""
    loads = window["load"].values
    diffs = np.abs(np.diff(loads))
    p95 = float(np.percentile(diffs, 95))
    p99 = float(np.percentile(diffs, 99))
    return p95, p99


def build_origin_summary(window: pd.DataFrame) -> str:
    """Build deterministic textual summary of system state at forecast origin.

    Uses 14-day window (336 hours) ending at T inclusive.
    All features computed from data <= T only.

    Args:
        window: DataFrame with last 336 rows ending at T.

    Returns:
        Concise bullet-form origin summary.
    """
    lines: List[str] = []
    lines.append("Data: hourly electricity load. Forecast next 24 hours.")

    last = _last_load(window)
    mean_24h = _mean_last_24h(window)
    mean_7d = _mean_last_7d(window)
    lines.append(f"Level: last={last:.1f}, mean_24h={mean_24h:.1f}, mean_7d={mean_7d:.1f}")

    hod_means = _hour_of_day_means(window)
    peak_h, trough_h, amplitude = _peak_trough_amplitude(hod_means)
    lines.append(
        f"Seasonality: peak_hour={peak_h}, trough_hour={trough_h}, amplitude≈{amplitude:.1f}"
    )

    daily_means = _daily_means_14d(window)
    slope, pct = _trend_slope_and_pct(daily_means)
    if abs(pct) > TREND_THRESHOLD_PCT:
        lines.append(
            f"Trend: slope≈{slope:.1f} units/day (≈{pct*100:.2f}%/day)"
        )

    weekday_mean, weekend_mean = _weekday_weekend_means(window)
    weekend_delta = _weekend_delta_pct(weekday_mean, weekend_mean)
    if abs(weekend_delta) > WEEKEND_THRESHOLD_PCT:
        lines.append(
            f"Weekend: weekend≈{weekend_delta*100:.1f}% vs weekday"
        )

    _, p99_ramp = _ramp_percentiles(window)
    lines.append(
        f"Smoothness: typical ramp ≤ p99≈{p99_ramp:.1f} per hour. Avoid unrealistic single-hour jumps."
    )
    lines.append("Constraints: output 24 non-negative numbers")

    return "\n".join(lines)


def build_hour_of_day_profile(window: pd.DataFrame) -> List[float]:
    """Build 24-value hour-of-day average profile from 14-day window.

    Args:
        window: DataFrame with 336 hours ending at T inclusive.

    Returns:
        List of 24 floats ordered by hour index 0..23.
    """
    hod_means = _hour_of_day_means(window)
    return [float(x) for x in hod_means]


def build_recent_history(
    df: pd.DataFrame,
    origin_idx: int,
    n_hours: int = 168,
) -> List[Tuple[str, float]]:
    """Build recent 7-day hourly history (168 timestamp-value pairs).

    Args:
        df: Full DataFrame.
        origin_idx: Index of origin T.
        n_hours: Number of hours (default 168 = 7 days).

    Returns:
        List of (timestamp_iso_utc, load_float) pairs, chronologically ordered.
    """
    window = df.iloc[max(0, origin_idx - n_hours + 1) : origin_idx + 1]
    result: List[Tuple[str, float]] = []
    for _, row in window.iterrows():
        ts = row["timestamp"]
        iso_utc = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        result.append((iso_utc, float(row["load"])))
    return result
