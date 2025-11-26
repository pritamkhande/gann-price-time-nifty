import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================

DATA_PATH = "data/nifty_daily.csv"

DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

LOOKBACK_SWING = 3        # swing detection window (bars before/after)
MAX_LOOKAHEAD = 120       # how many bars after swing to look for square
SLOPE_TOL = 0.15          # |DeltaP/DeltaT - 1| <= 0.15
SQUARE_NUMBERS = [25, 36, 49, 64, 81, 100, 121]
NUMBER_TOL = 2            # allowed +- around SQUARE_NUMBERS
ATR_PERIOD = 14
R_MULTIPLIER = 2.0        # target = 2R
RISK_PER_TRADE = 0.01     # 1% equity per trade

OUT_REPORT_HTML = "docs/index.html"
OUT_TRADES_CSV = "data/gann_nifty_trades.csv"
OUT_EQUITY_PNG = "docs/gann_equity_curve.png"
OUT_DD_PNG = "docs/gann_drawdown_curve.png"

os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)


# ==========================
# DATA LOADING
# ==========================

def load_data():
    """
    Load Nifty daily data. Expects columns:
    Date, Open, High, Low, Close, Volume

    Dates in your CSV are like 17-09-2007, so we use dayfirst=True.
    """
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df


# ==========================
# ATR
# ==========================

def compute_atr(df, period=ATR_PERIOD):
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


# ==========================
# SWING HIGHS / LOWS
# ==========================

def detect_swings(df, lookback=LOOKBACK_SWING):
    n = len(df)
    swing_low = np.zeros(n, dtype=bool)
    swing_high = np.zeros(n, dtype=bool)

    lows = df[LOW_COL].values
    highs = df[HIGH_COL].values

    for i in range(lookback, n - lookback):
        window_lows = lows[i - lookback: i + lookback + 1]
        window_highs = highs[i - lookback: i + lookback + 1]

        if lows[i] == window_lows.min():
            swing_low[i] = True
        if highs[i] == window_highs.max():
            swing_high[i] = True

    df["swing_low"] = swing_low
    df["swing_high"] = swing_high
    return df


# ==========================
# PRICE–TIME / PRICE–DATE SQUARES
# ==========================

def is_square_number(n: int) -> bool:
    for s in SQUARE_NUMBERS:
        if abs(n - s) <= NUMBER_TOL:
            return True
    return False


def classify_square(delta_points: float,
                    delta_bars: int,
                    delta_days: int) -> str | None:
    """
    Decide whether we have:
    - time-based square (bars)
    - date-based square (calendar days)
    - both
    Return "time", "date", "both" or None.
    """
    if delta_bars <= 0 or delta_days <= 0:
        return None

    slope_bars = delta_points / delta_bars
    slope_days = delta_points / delta_days

    time_ok = (abs(slope_bars - 1.0) <= SLOPE_TOL) and is_square_number(delta_bars)
    date_ok = (abs(slope_days - 1.0) <= SLOPE_TOL) and is_square_number(delta_days)

    if time_ok and date_ok:
        return "both"
    if time_ok:
        return "time"
    if date_ok:
        return "date"
    return None


def find_square_from_swing_low(df, i0):
    """
    From swing low at index i0, look forward for first up-move
    where price-time and/or price-date is squared.
    Return (index, square_type) or (None, None).
    """
    n = len(df)
    p0 = df.loc[i0, CLOSE_COL]
    d0 = df.loc[i0, DATE_COL]

    for t in range(i0 + 5, min(i0 + MAX_LOOKAHEAD, n)):
        delta_bars = t - i0
        d_t = df.loc[t, DATE_COL]
        delta_days = (d_t - d0).days

        delta_p = df.loc[t, CLOSE_COL] - p0
        if delta_p <= 0:
            continue  # need up move

        sq_type = classify_square(abs(delta_p), delta_bars, delta_days)
        if sq_type is not None:
            return t, sq_type

    return None, None


def find_square_from_swing_high(df, i0):
    """
    From swing high at index i0, look forward for first down-move
    where price-time and/or price-date is squared.
    Return (index, square_type) or (None, None).
    """
    n = len(df)
    p0 = df.loc[i0, CLOSE_COL]
    d0 = df.loc[i0, DATE_COL]

    for t in range(i0 + 5, min(i0 + MAX_LOOKAHEAD, n)):
        delta_bars = t - i0
        d_t = df.loc[t, DATE_COL]
        delta_days = (d_t - d0).days

        delta_p = df.loc[t, CLOSE_COL] - p0
        if delta_p >= 0:
            continue  # need down move

        sq_type = classify_square(abs(delta_p), delta_bars, delta_days)
        if sq_type is not None:
            return t, sq_type

    return None, None


# ==========================
# BACKTEST
# ==========================

def backtest(df):
    equity = 1.0
    in_trade = False
    position = None
    entry_idx = None
    entry_price = None
    stop_price = None
    tp_price = None
    entry_square_type = None

    trades = []

    n = len(df)
    i = 0

    while i < n - 2:
        if not in_trade:
            # short setup from swing low
            if df.loc[i, "swing_low"]:
                sq_idx, sq_type = find_square_from_swing_low(df, i)
                if sq_idx is not None and sq_idx < n - 1:
                    # bearish confirmation
                    if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:
                        in_trade = True
                        position = "short"
                        entry_idx = sq_idx + 1
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        entry_square_type = sq_type
                        sl = df.loc[sq_idx, HIGH_COL] + df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        tp_price = entry_price - R_MULTIPLIER * (sl - entry_price)
                        i = entry_idx
                        continue

            # long setup from swing high
            if df.loc[i, "swing_high"]:
                sq_idx, sq_type = find_square_from_swing_high(df, i)
                if sq_idx is not None and sq_idx < n - 1:
                    # bullish confirmation
                    if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:
                        in_trade = True
                        position = "long"
                        entry_idx = sq_idx + 1
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        entry_square_type = sq_type
                        sl = df.loc[sq_idx, LOW_COL] - df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        tp_price = entry_price + R_MULTIPLIER * (entry_price - sl)
                        i = entry_idx
                        continue

            i += 1
        else:
            # manage open trade
            high = df.loc[i, HIGH_COL]
            low = df.loc[i, LOW_COL]
            close = df.loc[i, CLOSE_COL]
            date = df.loc[i, DATE_COL]

            exit_reason = None
            exit_price = None

            if position == "long":
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
            else:  # short
                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"

            # last bar forced exit
            if i == n - 1 and exit_reason is None:
                exit_price = close
                exit_reason = "End"

            if exit_reason is not None:
                if position == "long":
                    risk = entry_price - stop_price
                    pnl = exit_price - entry_price
                else:
                    risk = stop_price - entry_price
                    pnl = entry_price - exit_price

                r_mult = pnl / risk if risk != 0 else 0.0

                trades.append({
                    "entry_date": df.loc[entry_idx, DATE_COL],
                    "exit_date": date,
                    "position": position,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "stop_price": float(stop_price),
                    "tp_price": float(tp_price),
                    "R": float(r_mult),
                    "pnl": float(pnl),
                    "exit_reason": exit_reason,
                    "square_type": entry_square_type,
                })

                risk_amount = equity * RISK_PER_TRADE
                equity += r_mult * risk_amount

                in_trade = False
                position = None
                entry_idx = None
                entry_price = None
                stop_price = None
                tp_price = None
                entry_square_type = None

            i += 1

    trades_df = pd.DataFrame(trades)

    # build equity curve over time
    df["equity"] = np.nan
    equity = 1.0
    trade_iter = iter(trades)
    current_trade = next(trade_iter, None)

    for idx in range(n):
        date = df.loc[idx, DATE_COL]
        while current_trade is not None and current_trade["exit_date"] <= date:
            r_mult = current_trade["R"]
            risk_amount = equity * RISK_PER_TRADE
            equity += r_mult * risk_amount
            current_trade = next(trade_iter, None)
        df.loc[idx, "equity"] = equity

    return trades_df, df


# ==========================
# METRICS AND PLOTS
# ==========================

def compute_metrics(trades_df, price_df):
    if trades_df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_R": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "start_date": None,
            "end_date": None,
            "years": 0.0,
            "n_time": 0,
            "n_date": 0,
            "n_both": 0,
        }

    n_trades = len(trades_df)
    wins = (trades_df["R"] > 0).sum()
    win_rate = 100.0 * wins / n_trades
    avg_R = trades_df["R"].mean()

    eq = price_df["equity"].dropna()
    start_eq = eq.iloc[0]
    end_eq = eq.iloc[-1]

    start_date = price_df[DATE_COL].iloc[0]
    end_date = price_df[DATE_COL].iloc[-1]
    years = (end_date - start_date).days / 365.25
    if years > 0 and start_eq > 0:
        cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    equity = eq.values
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    n_time = (trades_df["square_type"] == "time").sum()
    n_date = (trades_df["square_type"] == "date").sum()
    n_both = (trades_df["square_type"] == "both").sum()

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "cagr": cagr,
        "max_dd": max_dd,
        "start_date": start_date,
        "end_date": end_date,
        "years": years,
        "n_time": int(n_time),
        "n_date": int(n_date),
        "n_both": int(n_both),
    }


def make_plots(price_df):
    eq = price_df.dropna(subset=["equity"])
    if eq.empty:
        return

    dates = eq[DATE_COL]
    equity = eq["equity"].values

    # equity curve
    plt.figure(figsize=(9, 4))
    plt.plot(dates, equity)
    plt.title("Gann Price–Time / Price–Date Squaring – Equity Curve (Nifty)")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.tight_layout()
    plt.savefig(OUT_EQUITY_PNG)
    plt.close()

    # drawdown
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks

    plt.figure(figsize=(9, 3))
    plt.plot(dates, dd)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(OUT_DD_PNG)
    plt.close()


# ==========================
# HTML REPORT
# ==========================

def render_html(metrics, trades_df):
    start_str = metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] else "N/A"
    end_str = metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] else "N/A"
    years_str = f"{metrics['years']:.1f}" if metrics["years"] else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nifty – Gann Price-Time / Price-Date Squaring System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Mechanical Gann Price-Time and Price-Date Squaring backtest on Nifty daily data.">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
      color: #111827;
      line-height: 1.5;
    }}
    h1, h2, h3 {{
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
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric-box {{
      background: #f9fafb;
      border-radius: 10px;
      padding: 10px 12px;
      border: 1px solid #e5e7eb;
      font-size: 14px;
    }}
    .metric-value {{
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid #e5e7eb;
    }}
    .footer {{
      font-size: 12px;
      color: #6b7280;
      margin-top: 24px;
    }}
  </style>
</head>
<body>

  <h1>Nifty – Gann Price-Time / Price-Date Squaring System</h1>
  <p>
    Fully mechanical backtest of a Price-Time and Price-Date Squaring system inspired by W.D. Gann,
    applied to Nifty daily data from {start_str} to {end_str}.
  </p>

  <div class="card">
    <h2>Backtest Summary</h2>
    <div class="metrics-grid">
      <div class="metric-box">
        <div class="metric-value">{metrics["n_trades"]}</div>
        <div>Number of trades</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{metrics["win_rate"]:.1f}%</div>
        <div>Win rate</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{metrics["avg_R"]:.2f} R</div>
        <div>Average R per trade</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{metrics["cagr"]*100:.1f}%</div>
        <div>CAGR (normalized equity)</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{metrics["max_dd"]*100:.1f}%</div>
        <div>Maximum drawdown</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{years_str} yrs</div>
        <div>Test length</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Price-Time vs Price-Date Squares</h2>
    <table>
      <tr>
        <th>Square type</th>
        <th>Count</th>
      </tr>
      <tr>
        <td>Time only (trading bars)</td>
        <td>{metrics["n_time"]}</td>
      </tr>
      <tr>
        <td>Date only (calendar days)</td>
        <td>{metrics["n_date"]}</td>
      </tr>
      <tr>
        <td>Both time and date aligned</td>
        <td>{metrics["n_both"]}</td>
      </tr>
    </table>
    <p>
      Each trade is opened from a swing where the price move (points) is approximately equal to
      either the number of trading bars since that swing (Price–Time square),
      the number of calendar days since that swing (Price–Date square),
      or both.
    </p>
  </div>

  <div class="card">
    <h2>System Logic</h2>
    <h3>1. Swing points</h3>
    <ul>
      <li>Timeframe: daily Nifty OHLC data.</li>
      <li>Swing low: bar whose low is the lowest in a ± {LOOKBACK_SWING}-bar window.</li>
      <li>Swing high: bar whose high is the highest in a ± {LOOKBACK_SWING}-bar window.</li>
    </ul>

    <h3>2. Price-Time / Price-Date Squaring</h3>
    <ul>
      <li>From each swing, scan forward up to {MAX_LOOKAHEAD} bars.</li>
      <li>Let ΔP = |Close - swing close| in points.</li>
      <li>Let ΔBars = bars since swing, ΔDays = calendar days since swing.</li>
      <li>Price-Time square if:
        <ul>
          <li>|ΔP/ΔBars − 1| ≤ {SLOPE_TOL}, and</li>
          <li>ΔBars close to one of {SQUARE_NUMBERS} (±{NUMBER_TOL}).</li>
        </ul>
      </li>
      <li>Price-Date square if:
        <ul>
          <li>|ΔP/ΔDays − 1| ≤ {SLOPE_TOL}, and</li>
          <li>ΔDays close to one of {SQUARE_NUMBERS} (±{NUMBER_TOL}).</li>
        </ul>
      </li>
      <li>A setup is valid if either Time, Date, or both conditions are satisfied.</li>
    </ul>

    <h3>3. Entries and exits</h3>
    <ul>
      <li>Short: from swing low squared up, if next bar closes below its low, open short at next open.</li>
      <li>Long: from swing high squared down, if next bar closes above its high, open long at next open.</li>
      <li>Stop loss: square bar high/low ± ATR(14).</li>
      <li>Target: {R_MULTIPLIER} R (R = initial risk per share).</li>
      <li>Risk per trade: {RISK_PER_TRADE*100:.0f}% of equity. One position at a time.</li>
    </ul>
  </div>

  <div class="card">
    <h2>Equity Curve and Drawdown</h2>
    <p>Equity starts at 1.0 and changes based on R-multiples with 1% risk per trade.</p>
    <img src="gann_equity_curve.png" alt="Equity curve">
    <p>Drawdown relative to running equity peak:</p>
    <img src="gann_drawdown_curve.png" alt="Drawdown curve">
  </div>

  <div class="card">
    <h2>Sample Trades (first 10)</h2>
    <table>
      <tr>
        <th>#</th>
        <th>Entry date</th>
        <th>Exit date</th>
        <th>Side</th>
        <th>R</th>
        <th>Square type</th>
        <th>Exit reason</th>
      </tr>
"""
    sample = trades_df.head(10)
    for idx, row in sample.iterrows():
        html += f"""
      <tr>
        <td>{idx + 1}</td>
        <td>{row['entry_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['exit_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['position']}</td>
        <td>{row['R']:.2f}</td>
        <td>{row.get('square_type', '')}</td>
        <td>{row['exit_reason']}</td>
      </tr>
"""

    html += """
    </table>
  </div>

  <div class="footer">
    This is a research backtest. It ignores costs, slippage and execution constraints.
    It is not trading advice.
  </div>

</body>
</html>
"""
    return html


# ==========================
# MAIN
# ==========================

def main():
    df = load_data()
    df = compute_atr(df)
    df = detect_swings(df)

    trades_df, price_df = backtest(df)
    trades_df.to_csv(OUT_TRADES_CSV, index=False)

    metrics = compute_metrics(trades_df, price_df)
    make_plots(price_df)

    html = render_html(metrics, trades_df)
    with open(OUT_REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("Report written to", OUT_REPORT_HTML)
    print("Trades CSV:", OUT_TRADES_CSV)


if __name__ == "__main__":
    main()
