import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

DATA_PATH = "data/nifty_daily.csv"
EARLY_DATA_PATH = "Early_Data/nifty_early_close.csv"
OUT_HTML = "docs/live-signal.html"

DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

ATR_PERIOD = 14
SLOPE_TOL = 0.25
MAX_LOOKAHEAD = 160

SAFE_MARGIN_THRESHOLD_PCT = 0.60  # your observed last-10-min volatility (~0.6%)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    high = df[HIGH_COL]
    low = df[LOW_COL]
    close = df[CLOSE_COL]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period, min_periods=1).mean()
    return df


def load_early_close() -> pd.DataFrame | None:
    if not os.path.exists(EARLY_DATA_PATH):
        return None
    edf = pd.read_csv(EARLY_DATA_PATH)
    edf[DATE_COL] = pd.to_datetime(edf[DATE_COL], dayfirst=True, errors="coerce")
    edf = edf.dropna(subset=[DATE_COL, "EarlyClose"])
    edf = edf.sort_values(DATE_COL).reset_index(drop=True)
    return edf


def find_today_confirmation(
    df: pd.DataFrame,
    early_df: pd.DataFrame,
):
    """
    Find if the latest date in df that exists in early_df is a confirmation
    bar for any square.

    Returns dict with keys:
      - date
      - early_close
      - side ("long"/"short")
      - square_type
      - margin_neutral_pts
      - margin_neutral_pct
      - margin_flip_pts
      - margin_flip_pct
    or None.
    """
    # map early-close by date
    early_map = early_df.set_index(DATE_COL)["EarlyClose"]

    # find latest date that has early-close
    merged_dates = sorted(set(df[DATE_COL]) & set(early_df[DATE_COL]))
    if not merged_dates:
        return None

    today_date = merged_dates[-1]
    today_idx_list = df.index[df[DATE_COL] == today_date].tolist()
    if not today_idx_list:
        return None
    today_idx = today_idx_list[0]

    early_close = float(early_map[today_date])

    n = len(df)

    # we will scan swings in the past that can lead to confirmation today
    # A confirmation bar is at index 'today_idx' == sq_idx + 1
    best_signal = None

    for i in range(0, today_idx - 1):
        # from swing low: potential short
        if df.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(
                df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
            )
            if sq_idx is not None and sq_idx + 1 == today_idx:
                sq_high = df.loc[sq_idx, HIGH_COL]
                sq_low = df.loc[sq_idx, LOW_COL]

                # use early close instead of final close for preview
                # short confirmation: close < square_low
                if early_close < sq_low:
                    # margins (short)
                    margin_neutral_pts = sq_low - early_close
                    margin_flip_pts = sq_high - early_close
                    margin_neutral_pct = 100.0 * margin_neutral_pts / early_close
                    margin_flip_pct = 100.0 * margin_flip_pts / early_close
                    safe = margin_neutral_pct >= SAFE_MARGIN_THRESHOLD_PCT

                    best_signal = {
                        "date": today_date,
                        "early_close": early_close,
                        "side": "short",
                        "square_type": sq_type,
                        "margin_neutral_pts": margin_neutral_pts,
                        "margin_neutral_pct": margin_neutral_pct,
                        "margin_flip_pts": margin_flip_pts,
                        "margin_flip_pct": margin_flip_pct,
                        "safe": safe,
                    }
                    break  # prefer first found

        # from swing high: potential long
        if df.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(
                df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
            )
            if sq_idx is not None and sq_idx + 1 == today_idx:
                sq_high = df.loc[sq_idx, HIGH_COL]
                sq_low = df.loc[sq_idx, LOW_COL]

                # long confirmation: close > square_high
                if early_close > sq_high:
                    # margins (long)
                    margin_neutral_pts = early_close - sq_high
                    margin_flip_pts = early_close - sq_low
                    margin_neutral_pct = 100.0 * margin_neutral_pts / early_close
                    margin_flip_pct = 100.0 * margin_flip_pts / early_close
                    safe = margin_neutral_pct >= SAFE_MARGIN_THRESHOLD_PCT

                    best_signal = {
                        "date": today_date,
                        "early_close": early_close,
                        "side": "long",
                        "square_type": sq_type,
                        "margin_neutral_pts": margin_neutral_pts,
                        "margin_neutral_pct": margin_neutral_pct,
                        "margin_flip_pts": margin_flip_pts,
                        "margin_flip_pct": margin_flip_pct,
                        "safe": safe,
                    }
                    break  # prefer first found

    return best_signal


def render_today_html(signal) -> str:
    if signal is None:
        html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nifty – Gann Early Signal (Live)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 700px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
      color: #111827;
    }
    .card {
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
  </style>
</head>
<body>
  <h1>Nifty – Gann Early Signal (Live)</h1>
  <div class="card">
    <p>No valid Gann confirmation setup detected for the latest day, or no early-close data available.</p>
  </div>
</body>
</html>
"""
        return html

    date_str = signal["date"].strftime("%Y-%m-%d")
    ec = signal["early_close"]
    side = signal["side"]
    sq_type = signal["square_type"]
    mn_pts = signal["margin_neutral_pts"]
    mn_pct = signal["margin_neutral_pct"]
    mf_pts = signal["margin_flip_pts"]
    mf_pct = signal["margin_flip_pct"]
    safe = signal["safe"]

    safe_label = "YES" if safe else "NO"
    safe_color = "#16a34a" if safe else "#dc2626"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nifty – Gann Early Signal (Live)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 700px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
      color: #111827;
    }}
    .card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 14px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
    }}
    .safe-yes {{
      color: #16a34a;
      font-weight: 600;
    }}
    .safe-no {{
      color: #dc2626;
      font-weight: 600;
    }}
  </style>
</head>
<body>

  <h1>Nifty – Gann Early Signal (Live)</h1>

  <div class="card">
    <p>
      This page shows the latest Gann Price-Time / Price-Date confirmation based on
      early-close data (approx. 10–15 minutes before market close).
    </p>
  </div>

  <div class="card">
    <h2>Today's Early Signal</h2>
    <table>
      <tr><th>Date</th><td>{date_str}</td></tr>
      <tr><th>Early close</th><td>{ec:.2f}</td></tr>
      <tr><th>Side</th><td>{side.upper()}</td></tr>
      <tr><th>Square type</th><td>{sq_type}</td></tr>
      <tr><th>Margin neutral (pts)</th><td>{mn_pts:.2f}</td></tr>
      <tr><th>Margin neutral (%)</th><td>{mn_pct:.2f}%</td></tr>
      <tr><th>Margin flip (pts)</th><td>{mf_pts:.2f}</td></tr>
      <tr><th>Margin flip (%)</th><td>{mf_pct:.2f}%</td></tr>
      <tr>
        <th>Safe early entry?</th>
        <td><span class="{ 'safe-yes' if safe else 'safe-no' }">{safe_label}</span></td>
      </tr>
    </table>
    <p style="font-size:12px; color:#6b7280; margin-top:8px;">
      The 'safe early entry' flag compares margin neutral (%) with a threshold of {SAFE_MARGIN_THRESHOLD_PCT:.2f}%,
      roughly matching your observed last 10-minute volatility.
    </p>
  </div>

</body>
</html>
"""
    return html


def main():
    os.makedirs("docs", exist_ok=True)

    df = load_data()
    df = compute_atr(df)
    df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL, lookback_main=1, lookback_fractal=2)

    early_df = load_early_close()
    if early_df is None:
        html = render_today_html(None)
    else:
        signal = find_today_confirmation(df, early_df)
        html = render_today_html(signal)

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("Live early-signal page written to", OUT_HTML)


if __name__ == "__main__":
    main()
