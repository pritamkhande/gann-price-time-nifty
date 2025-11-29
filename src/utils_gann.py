import math
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


GANN_SQUARE_LEVELS = [25, 36, 49, 64, 81, 100, 121, 50, 72, 98, 128]


def _classify_square(
    dP: float,
    d_bars: int,
    d_days: int,
    slope_tol: float,
) -> Optional[Tuple[str, float]]:
    """
    Decide if (dP, d_bars, d_days) forms a Gann-like square.

    Returns:
      (square_type, error_score) or None
    """
    if dP <= 0:
        return None

    best_type = None
    best_err = float("inf")

    # Price-Time (bars)
    if d_bars > 0:
        ratio_bt = dP / d_bars
        err_bt = abs(ratio_bt - 1.0)
        if err_bt <= slope_tol and err_bt < best_err:
            best_type = "price_time"
            best_err = err_bt

    # Price-Date (calendar days)
    if d_days > 0:
        ratio_dd = dP / d_days
        err_dd = abs(ratio_dd - 1.0)
        if err_dd <= slope_tol and err_dd < best_err:
            best_type = "price_date"
            best_err = err_dd

    # Prefer around known Gann square numbers for dP
    # (this is a soft filter; we do not require it, but we reward closeness)
    if best_type is not None:
        nearest_square = min(GANN_SQUARE_LEVELS, key=lambda x: abs(x - dP))
        square_err = abs(nearest_square - dP) / max(nearest_square, 1.0)
        # combine errors: geometric-like mix
        total_err = best_err + 0.5 * square_err
        return best_type, total_err

    return None


def _scan_forward_for_square(
    df: pd.DataFrame,
    start_idx: int,
    date_col: str,
    close_col: str,
    slope_tol: float,
    max_lookahead: int,
    direction: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Generic forward scanner for squares from a swing point.

    direction: "up" or "down"
    """
    n = len(df)
    base_close = float(df.loc[start_idx, close_col])
    base_date = df.loc[start_idx, date_col]

    best_idx = None
    best_type = None
    best_err = float("inf")

    max_idx = min(n - 1, start_idx + max_lookahead)

    for j in range(start_idx + 1, max_idx + 1):
        c = float(df.loc[j, close_col])
        if direction == "up" and c <= base_close:
            continue
        if direction == "down" and c >= base_close:
            continue

        dP = abs(c - base_close)
        d_bars = j - start_idx
        d_days = (df.loc[j, date_col] - base_date).days

        res = _classify_square(dP, d_bars, d_days, slope_tol)
        if res is None:
            continue
        sq_type, err = res
        if err < best_err:
            best_err = err
            best_type = sq_type
            best_idx = j

    return best_idx, best_type


def find_square_from_swing_low(
    df: pd.DataFrame,
    swing_idx: int,
    date_col: str,
    close_col: str,
    slope_tol: float,
    max_lookahead: int,
) -> Tuple[Optional[int], Optional[str]]:
    """
    From a swing low, scan forward for an up-move that forms
    a Gann-like price-time / price-date square.
    """
    return _scan_forward_for_square(
        df,
        swing_idx,
        date_col=date_col,
        close_col=close_col,
        slope_tol=slope_tol,
        max_lookahead=max_lookahead,
        direction="up",
    )


def find_square_from_swing_high(
    df: pd.DataFrame,
    swing_idx: int,
    date_col: str,
    close_col: str,
    slope_tol: float,
    max_lookahead: int,
) -> Tuple[Optional[int], Optional[str]]:
    """
    From a swing high, scan forward for a down-move that forms
    a Gann-like price-time / price-date square.
    """
    return _scan_forward_for_square(
        df,
        swing_idx,
        date_col=date_col,
        close_col=close_col,
        slope_tol=slope_tol,
        max_lookahead=max_lookahead,
        direction="down",
    )
