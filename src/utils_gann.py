from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


# extended square numbers: classic, half/double variants
SQUARE_NUMBERS = [25, 36, 49, 64, 81, 100, 121,
                  50, 72, 98, 128]


def is_square_number(n: int,
                     tol: int = 4) -> bool:
    for s in SQUARE_NUMBERS:
        if abs(n - s) <= tol:
            return True
    return False


def classify_square(delta_points: float,
                    delta_bars: int,
                    delta_days: int,
                    slope_tol: float = 0.25,
                    tol_bars: int = 4,
                    tol_days: int = 4) -> Optional[str]:
    """
    Decide whether we have:
    - time-based square (bars)
    - date-based square (calendar days)
    - both

    Returns "time", "date", "both" or None.
    """
    if delta_bars <= 0 or delta_days <= 0:
        return None

    slope_bars = delta_points / delta_bars
    slope_days = delta_points / delta_days

    time_ok = (abs(slope_bars - 1.0) <= slope_tol) and is_square_number(delta_bars, tol_bars)
    date_ok = (abs(slope_days - 1.0) <= slope_tol) and is_square_number(delta_days, tol_days)

    if time_ok and date_ok:
        return "both"
    if time_ok:
        return "time"
    if date_ok:
        return "date"
    return None


def find_square_from_swing_low(df: pd.DataFrame,
                               i0: int,
                               date_col: str,
                               close_col: str,
                               slope_tol: float = 0.25,
                               max_lookahead: int = 160) -> Tuple[Optional[int], Optional[str]]:
    """
    From swing low at index i0, look forward for first up-move
    where price-time and/or price-date is squared.
    Return (index, square_type) or (None, None).
    """
    n = len(df)
    p0 = df.loc[i0, close_col]
    d0 = df.loc[i0, date_col]

    for t in range(i0 + 5, min(i0 + max_lookahead, n)):
        delta_bars = t - i0
        d_t = df.loc[t, date_col]
        delta_days = (d_t - d0).days
        delta_p = df.loc[t, close_col] - p0
        if delta_p <= 0:
            continue

        sq_type = classify_square(abs(delta_p), delta_bars, delta_days, slope_tol=slope_tol)
        if sq_type is not None:
            return t, sq_type

    return None, None


def find_square_from_swing_high(df: pd.DataFrame,
                                i0: int,
                                date_col: str,
                                close_col: str,
                                slope_tol: float = 0.25,
                                max_lookahead: int = 160) -> Tuple[Optional[int], Optional[str]]:
    """
    From swing high at index i0, look forward for first down-move
    where price-time and/or price-date is squared.
    Return (index, square_type) or (None, None).
    """
    n = len(df)
    p0 = df.loc[i0, close_col]
    d0 = df.loc[i0, date_col]

    for t in range(i0 + 5, min(i0 + max_lookahead, n)):
        delta_bars = t - i0
        d_t = df.loc[t, date_col]
        delta_days = (d_t - d0).days
        delta_p = df.loc[t, close_col] - p0
        if delta_p >= 0:
            continue

        sq_type = classify_square(abs(delta_p), delta_bars, delta_days, slope_tol=slope_tol)
        if sq_type is not None:
            return t, sq_type

    return None, None
