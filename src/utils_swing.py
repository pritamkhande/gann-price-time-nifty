import numpy as np
import pandas as pd


def detect_swings(
    df: pd.DataFrame,
    low_col: str = "Low",
    high_col: str = "High",
    lookback_main: int = 1,
    lookback_fractal: int = 2,
) -> pd.DataFrame:
    """
    Detects swing highs/lows with two logics:
    - main: local min/max over +/- lookback_main bars (default 1 → many swings)
    - fractal: Williams-style +/- lookback_fractal bars

    Adds boolean columns:
      swing_low, swing_high
    """

    n = len(df)
    lows = df[low_col].values
    highs = df[high_col].values

    swing_low_main = np.zeros(n, dtype=bool)
    swing_high_main = np.zeros(n, dtype=bool)
    swing_low_fractal = np.zeros(n, dtype=bool)
    swing_high_fractal = np.zeros(n, dtype=bool)

    # main +/- lookback_main
    for i in range(lookback_main, n - lookback_main):
        window_lows = lows[i - lookback_main : i + lookback_main + 1]
        window_highs = highs[i - lookback_main : i + lookback_main + 1]

        if lows[i] == window_lows.min():
            swing_low_main[i] = True
        if highs[i] == window_highs.max():
            swing_high_main[i] = True

    # fractal +/- lookback_fractal
    for i in range(lookback_fractal, n - lookback_fractal):
        left_lows = lows[i - lookback_fractal : i]
        right_lows = lows[i + 1 : i + 1 + lookback_fractal]
        left_highs = highs[i - lookback_fractal : i]
        right_highs = highs[i + 1 : i + 1 + lookback_fractal]

        if lows[i] < left_lows.min() and lows[i] < right_lows.min():
            swing_low_fractal[i] = True
        if highs[i] > left_highs.max() and highs[i] > right_highs.max():
            swing_high_fractal[i] = True

    swing_low = swing_low_main | swing_low_fractal
    swing_high = swing_high_main | swing_high_fractal

    df = df.copy()
    df["swing_low"] = swing_low
    df["swing_high"] = swing_high
    return df
