import os
from datetime import datetime

import pandas as pd

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high

DATA_PATH = "data/nifty_daily.csv"
DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

OUT_HTML = "docs/today.html"
OUT_JSON = "data/today_signal.json"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


def build_signal_comment(signal: dict) -> str:
    if not signal["has_signal"]:
        return (
            "No fresh Gann Price-Time or Price-Date Square was detected near the most recent bars. "
            "When the system does not see a clean square, it simply stays flat and waits for the next balance "
            "between price and time/date."
        )

    dir_txt = "potential short (downside) setup" if "down" in signal["direction"] else "potential long (upside) setup"
    sq_map = {
        "time": "a Price–Time square (bars)",
        "date": "a Price–Date square (calendar days)",
        "both": "both Price–Time and Price–Date squares lining up",
    }
    sq_txt = sq_map.get(signal["square_type"], "a Gann squaring condition")

    return (
        f"The scanner has identified {sq_txt} near the latest bars, creating a {dir_txt}. "
        "This does not guarantee reversal, but historically these conditions often mark turning zones. "
        "Always combine this with broader market context and risk management."
    )


def main():
    os.makedirs("docs", exist_ok=True)
    df = load_data()
    df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL, lookback_main=1, lookback_fractal=2)

    n = len(df)
    last_idx = n - 1
    last_date = df.loc[last_idx, DATE_COL].strftime("%d-%m-%Y")

    signal = {
        "has_signal": False,
        "type": None,
        "square_type": None,
        "direction": None,
        "swing_index": None,
        "square_index": None,
        "date": last_date,
    }

    # Check most recent swing within last 40 bars
    for i in range(max(0, n - 40), n - 5):
        if df.loc[i, "swing_low"]:
            sq_idx, sq_type = find_square_from_swing_low(df, i, DATE_COL, CLOSE_COL)
            if sq_idx is not None and sq_idx >= n - 5:
                signal.update(
                    {
                        "has_signal": True,
                        "type": "swing_low",
                        "square_type": sq_type,
                        "direction": "down (short setup)",
                        "swing_index": int(i),
                        "square_index": int(sq_idx),
                    }
                )
        if df.loc[i, "swing_high"]:
            sq_idx, sq_type = find_square_from_swing_high(df, i, DATE_COL, CLOSE_COL)
            if sq_idx is not None and sq_idx >= n - 5:
                signal.update(
                    {
                        "has_signal": True,
                        "type": "swing_high",
                        "square_type": sq_type,
                        "direction": "up (long setup)",
                        "swing_index": int(i),
                        "square_index": int(sq_idx),
                    }
                )

    # Write JSON
    os.makedirs("data", exist_ok=True)
    pd.Series(signal).to_json(OUT_JSON, indent=2)

    comment = build_signal_comment(signal)

    # Write mini HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nifty – Today&apos;s Gann Square Scanner</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 700px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
      color: #111827;
      line-height: 1.5;
    }}
    .card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
      margin-top: 20px;
    }}
  </style>
</head>
<body>
  <h1>Nifty – Today&apos;s Gann Square Scanner</h1>
  <p>Last bar in data: <strong>{last_date}</strong></p>
"""

    if not signal["has_signal"]:
        html += f"""
  <div class="card">
    <p>{comment}</p>
  </div>
</body>
</html>
"""
    else:
        html += f"""
  <div class="card">
    <p><strong>Signal detected:</strong></p>
    <ul>
      <li>Type: {signal["type"]}</li>
      <li>Direction: {signal["direction"]}</li>
      <li>Square type: {signal["square_type"]}</li>
      <li>Swing index: {signal["swing_index"]}</li>
      <li>Square index: {signal["square_index"]}</li>
    </ul>
    <p>{comment}</p>
  </div>
</body>
</html>
"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("Today scanner written to", OUT_HTML)


if __name__ == "__main__":
    main()
