"""Load, parse, validate, and provide rolling window helpers for electricity load data."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a single utility CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with columns: utility_name, timestamp, load.
    """
    df = pd.read_csv(path)
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp column to datetime.

    Expects format like '05/01/2022 01:00' or '05/01/2022 00:00'.
    Parses as UTC and converts to timezone-naive for consistent behavior
    across environments (e.g. Docker vs local).

    Args:
        df: DataFrame with 'timestamp' column as string.

    Returns:
        DataFrame with 'timestamp' as datetime64.
    """
    df = df.copy()
    # Parse as UTC for consistent behavior across environments (Docker vs local)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M", utc=True)
    return df


def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by timestamp ascending.

    Args:
        df: DataFrame with 'timestamp' column.

    Returns:
        Sorted DataFrame.
    """
    return df.sort_values("timestamp").reset_index(drop=True)


def validate_hourly_continuity(df: pd.DataFrame) -> None:
    """Validate that timestamps are hourly with no gaps.

    Uses pandas diff() for compatibility across environments (Docker, local).

    Raises:
        ValueError: If hourly continuity is violated.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    ts = df["timestamp"]
    diffs = ts.diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    gaps = diffs[diffs != one_hour].index.tolist()
    # gap index i means diff between row i-1 and i is wrong
    gap_indices = [ts.index.get_loc(i) for i in gaps]
    if gap_indices:
        raise ValueError(
            f"Hourly continuity violated at indices {gap_indices}, "
            f"expected 1-hour steps"
        )


def validate_no_duplicates(df: pd.DataFrame) -> None:
    """Validate no duplicate timestamps.

    Raises:
        ValueError: If duplicates exist.
    """
    dupes = df["timestamp"].duplicated()
    if dupes.any():
        raise ValueError(f"Duplicate timestamps found: {df.loc[dupes, 'timestamp'].tolist()}")


def validate_non_negative(df: pd.DataFrame) -> None:
    """Validate load values are non-negative.

    Raises:
        ValueError: If any load is negative.
    """
    neg = (df["load"] < 0).any()
    if neg:
        raise ValueError("Found negative load values")


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Run all validations and return prepared DataFrame.

    Args:
        df: Raw loaded DataFrame.

    Returns:
        Validated, parsed, sorted DataFrame.
    """
    df = parse_timestamps(df)
    df = sort_by_timestamp(df)
    validate_hourly_continuity(df)
    validate_no_duplicates(df)
    validate_non_negative(df)
    return df


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load CSV, validate, and return prepared DataFrame.

    Args:
        path: Path to CSV file.

    Returns:
        Validated DataFrame with columns utility_name, timestamp, load.
    """
    df = load_csv(path)
    return validate_and_prepare(df)


def get_window_at(df: pd.DataFrame, origin_idx: int, n_hours: int) -> pd.DataFrame:
    """Get the last n_hours of data ending at (and including) the row at origin_idx.

    Strict no-leak: uses only data at indices <= origin_idx.

    Args:
        df: Sorted DataFrame indexed 0..N-1.
        origin_idx: Index of the origin (inclusive end).
        n_hours: Number of hours to include.

    Returns:
        DataFrame slice with n_hours rows ending at origin_idx.
    """
    start_idx = max(0, origin_idx - n_hours + 1)
    return df.iloc[start_idx : origin_idx + 1].copy()


def get_daily_origins(df: pd.DataFrame) -> np.ndarray:
    """Get indices of all midnight (hour==0) timestamps.

    Args:
        df: DataFrame with 'timestamp' column.

    Returns:
        Array of row indices where hour == 0.
    """
    hours = df["timestamp"].dt.hour.values
    return np.where(hours == 0)[0]


def select_last_n_valid_origins(
    df: pd.DataFrame,
    origins: np.ndarray,
    n_origins: int,
    min_history_hours: int = 336,
    min_future_hours: int = 24,
) -> np.ndarray:
    """Select the last n valid origins with sufficient history and future data.

    A valid origin at index i satisfies:
    - At least min_history_hours rows exist before (and including) i
    - At least min_future_hours rows exist after i

    Args:
        df: Full DataFrame.
        origins: Array of origin indices (e.g. from get_daily_origins).
        n_origins: Number of origins to return.
        min_history_hours: Minimum hours before origin (e.g. 336 = 14 days).
        min_future_hours: Minimum hours after origin (e.g. 24).

    Returns:
        Array of the last n valid origin indices.
    """
    n = len(df)
    valid = []
    for idx in origins:
        if idx + 1 >= min_history_hours and idx + min_future_hours < n:
            valid.append(idx)
    valid = np.array(valid)
    if len(valid) < n_origins:
        raise ValueError(
            f"Need {n_origins} valid origins but found {len(valid)}. "
            f"Ensure sufficient data (≥{min_history_hours} hours history, "
            f"≥{min_future_hours} hours future)."
        )
    return valid[-n_origins:]
