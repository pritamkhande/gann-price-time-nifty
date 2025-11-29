import pandas as pd


def detect_swings(
    df: pd.DataFrame,
    low_col: str = "Low",
    high_col: str = "High",
    lookback_main: int = 1,
    lookback_fractal: int = 2,
) -> pd.DataFrame:
    """
    Detect swing highs and lows using:
    - Tight +/- lookback_main pivot
    - Williams-style fractal with lookback_fractal (e.g. 2 → 5-bar fractal)

    Adds boolean columns:
      - swing_high
      - swing_low
    """
    n = len(df)
    highs = df[high_col].values
    lows = df[low_col].values

    swing_high = [False] * n
    swing_low = [False] * n

    # Tight pivots
    for i in range(lookback_main, n - lookback_main):
        window_h = highs[i - lookback_main : i + lookback_main + 1]
        window_l = lows[i - lookback_main : i + lookback_main + 1]
        h = highs[i]
        l = lows[i]
        if h == max(window_h) and window_h.tolist().count(h) == 1:
            swing_high[i] = True
        if l == min(window_l) and window_l.tolist().count(l) == 1:
            swing_low[i] = True

    # Fractals (5-bar if lookback_fractal=2)
    if lookback_fractal > 0:
        for i in range(lookback_fractal, n - lookback_fractal):
            window_h = highs[i - lookback_fractal : i + lookback_fractal + 1]
            window_l = lows[i - lookback_fractal : i + lookback_fractal + 1]
            h = highs[i]
            l = lows[i]
            if h == max(window_h) and window_h.tolist().count(h) == 1:
                swing_high[i] = True or swing_high[i]
            if l == min(window_l) and window_l.tolist().count(l) == 1:
                swing_low[i] = True or swing_low[i]

    df = df.copy()
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df
