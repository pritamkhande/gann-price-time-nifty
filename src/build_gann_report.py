import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

# ---------- CONFIG ----------

DATA_PATH = "data/nifty_daily.csv"

DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

LOOKBACK_SWING = 3
MAX_LOOKAHEAD = 120
SLOPE_TOL = 0.15
SQUARE_DAYS = [25, 36, 49, 64, 81, 100, 121]
DAY_TOL = 2
ATR_PERIOD = 14
R_MULTIPLIER = 2.0

OUT_REPORT_HTML = "docs/index.html"
OUT_TRADES_CSV = "data/gann_nifty_trades.csv"
OUT_EQUITY_PNG = "assets/gann_equity_curve.png"
OUT_DD_PNG = "assets/gann_drawdown_curve.png"

os.makedirs("data", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# ---------- LOAD DATA ----------

def load_data():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOL_COL]]
    return df

# ---------- ATR ----------

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

# ---------- SWING POINTS ----------

def detect_swings(df, lookback=LOOKBACK_SWING):
    n = len(df)
    swing_low = np.zeros(n, dtype=bool)
    swing_high = np.zeros(n, dtype=bool)

    lows = df[LOW_COL].values
    highs = df[HIGH_COL].values

    for i in range(lookback, n - lookback):
        if lows[i] == lows[i - lookback : i + lookback + 1].min():
            swing_low[i] = True
        if highs[i] == highs[i - lookback : i + lookback + 1].max():
            swing_high[i] = True

    df["swing_low"] = swing_low
    df["swing_high"] = swing_high
    return df

# ---------- PRICE–TIME SQUARING ----------

def is_square_day(delta_t):
    for s in SQUARE_DAYS:
        if abs(delta_t - s) <= DAY_TOL:
            return True
    return False

def find_square_from_swing_low(df, i0):
    n = len(df)
    p0 = df.loc[i0, CLOSE_COL]
    for t in range(i0 + 5, min(i0 + MAX_LOOKAHEAD, n)):
        delta_t = t - i0
        delta_p = df.loc[t, CLOSE_COL] - p0
        if delta_p <= 0:
            continue
        slope = delta_p / delta_t
        if abs(slope - 1.0) <= SLOPE_TOL and is_square_day(delta_t):
            return t
    return None

def find_square_from_swing_high(df, i0):
    n = len(df)
    p0 = df.loc[i0, CLOSE_COL]
    for t in range(i0 + 5, min(i0 + MAX_LOOKAHEAD, n)):
        delta_t = t - i0
        delta_p = df.loc[t, CLOSE_COL] - p0
        if delta_p >= 0:
            continue
        slope = abs(delta_p) / delta_t
        if abs(slope - 1.0) <= SLOPE_TOL and is_square_day(delta_t):
            return t
    return None

# ---------- BACKTEST ----------

def backtest(df):
    equity = 1.0
    in_trade = False
    position = None
    entry_idx = None
    entry_price = None
    stop_price = None
    tp_price = None
    risk_per_trade = 0.01

    trades = []
    n = len(df)
    i = 0

    while i < n - 2:
        if not in_trade:
            # Short from swing low
            if df.loc[i, "swing_low"]:
                sq_idx = find_square_from_swing_low(df, i)
                if sq_idx is not None and sq_idx < n - 1:
                    if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:
                        in_trade = True
                        position = "short"
                        entry_idx = sq_idx + 1
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        sl = df.loc[sq_idx, HIGH_COL] + df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        tp_price = entry_price - R_MULTIPLIER * (sl - entry_price)
                        i = entry_idx
                        continue

            # Long from swing high
            if df.loc[i, "swing_high"]:
                sq_idx = find_square_from_swing_high(df, i)
                if sq_idx is not None and sq_idx < n - 1:
                    if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:
                        in_trade = True
                        position = "long"
                        entry_idx = sq_idx + 1
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        sl = df.loc[sq_idx, LOW_COL] - df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        tp_price = entry_price + R_MULTIPLIER * (entry_price - sl)
                        i = entry_idx
                        continue

            i += 1

        else:
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
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "R": r_mult,
                    "pnl": pnl,
                    "exit_reason": exit_reason
                })

                risk_amount = equity * risk_per_trade
                equity += r_mult * risk_amount

                in_trade = False
                position = None
                entry_idx = None
                entry_price = None
                stop_price = None
                tp_price = None

            i += 1

    trades_df = pd.DataFrame(trades)

    # Equity curve
    df["equity"] = np.nan
    equity = 1.0
    trade_iter = iter(trades)
    current_trade = next(trade_iter, None)

    for idx in range(n):
        date = df.loc[idx, DATE_COL]
        while current_trade is not None and current_trade["exit_date"] <= date:
            r_mult = current_trade["R"]
            risk_amount = equity * risk_per_trade
            equity += r_mult * risk_amount
            current_trade = next(trade_iter, None)
        df.loc[idx, "equity"] = equity

    return trades_df, df

# ---------- METRICS & PLOTS ----------

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
        }

    n_trades = len(trades_df)
    wins = (trades_df["R"] > 0).sum()
    win_rate = wins / n_trades * 100.0
    avg_R = trades_df["R"].mean()

    eq = price_df["equity"].dropna()
    start_eq = eq.iloc[0]
    end_eq = eq.iloc[-1]
    start_date = price_df[DATE_COL].iloc[0]
    end_date = price_df[DATE_COL].iloc[-1]
    years = (end_date - start_date).days / 365.25
    cagr = (end_eq / start_eq) ** (1 / years) - 1 if years > 0 else 0.0

    equity = eq.values
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_R": avg_R,
        "cagr": cagr,
        "max_dd": max_dd,
        "start_date": start_date,
        "end_date": end_date,
        "years": years,
    }

def make_plots(price_df):
    eq = price_df.dropna(subset=["equity"])
    dates = eq[DATE_COL]

    plt.figure(figsize=(10, 4))
    plt.plot(dates, eq["equity"])
    plt.title("Gann Price–Time Squaring – Equity Curve (Nifty)")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.tight_layout()
    plt.savefig(OUT_EQUITY_PNG)
    plt.close()

    equity = eq["equity"].values
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks

    plt.figure(figsize=(10, 3))
    plt.plot(dates, dd)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(OUT_DD_PNG)
    plt.close()

# ---------- HTML REPORT ----------

def render_html(metrics, trades_df):
    start_str = metrics["start_date"].strftime("%d %b %Y") if metrics["start_date"] else "N/A"
    end_str = metrics["end_date"].strftime("%d %b %Y") if metrics["end_date"] else "N/A"
    years_str = f"{metrics['years']:.1f}" if metrics["years"] else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nifty – Gann Price–Time Squaring System</title>
  <meta name="description" content="Mechanical Gann Price–Time Squaring backtest on Nifty daily data from {start_str} to {end_str}.">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 16px;
      line-height: 1.5;
      background: #f7f7f9;
      color: #111827;
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
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 14px;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
    }}
    .metric-big {{
      font-size: 18px;
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

  <h1>Nifty – Gann Price–Time Squaring System</h1>
  <p>
    This page shows a fully mechanical backtest of a <strong>Price–Time Squaring</strong> system inspired by W.D. Gann,
    applied to Nifty daily data from {start_str} to {end_str}.
  </p>

  <div class="card">
    <h2>Test Summary</h2>
    <div class="metrics-grid">
      <div class="metric-box">
        <div class="metric-big">{metrics["n_trades"]}</div>
        <div>Number of trades</div>
      </div>
      <div class="metric-box">
        <div class="metric-big">{metrics["win_rate"]:.1f}%</div>
        <div>Win rate</div>
      </div>
      <div class="metric-box">
        <div class="metric-big">{metrics["avg_R"]:.2f} R</div>
        <div>Average R per trade</div>
      </div>
      <div class="metric-box">
        <div class="metric-big">{metrics["cagr"]*100:.1f}%</div>
        <div>CAGR (normalized equity)</div>
      </div>
      <div class="metric-box">
        <div class="metric-big">{metrics["max_dd"]*100:.1f}%</div>
        <div>Max drawdown</div>
      </div>
      <div class="metric-box">
        <div class="metric-big">{years_str} yrs</div>
        <div>Test length</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>System Logic (Mechanical Rules)</h2>
    <h3>1. Swing Points</h3>
    <ul>
      <li>Use Nifty <strong>daily</strong> OHLC data.</li>
      <li>A <strong>swing low</strong> is a bar whose low is the lowest in a ±{LOOKBACK_SWING}-bar window.</li>
      <li>A <strong>swing high</strong> is a bar whose high is the highest in a ±{LOOKBACK_SWING}-bar window.</li>
    </ul>

    <h3>2. Price–Time Squaring</h3>
    <ul>
      <li>From each swing low/high, scan up to {MAX_LOOKAHEAD} bars ahead.</li>
      <li><strongSquared zone</strong> when:
        <ul>
          <li>Price move per bar: <code>slope = ΔP / ΔT</code> is within ±{SLOPE_TOL*100:.0f}% of 1 (points ≈ bars).</li>
          <li><code>ΔT</code> lies within ±{DAY_TOL} bars of one of: {SQUARE_DAYS}.</li>
        </ul>
      </li>
      <li>From swing low: squared zone treated as potential top → short setup.</li>
      <li>From swing high: squared zone treated as potential bottom → long setup.</li>
    </ul>

    <h3>3. Entries & Exits</h3>
    <ul>
      <li><strong>Short:</strong> After squared up-zone, if next bar closes below that bar’s low, open short at next day’s open.</li>
      <li><strong>Long:</strong> After squared down-zone, if next bar closes above that bar’s high, open long at next day’s open.</li>
      <li>Stop-loss: prior square bar high/low ± ATR14.</li>
      <li>Target: {R_MULTIPLIER}R (R = initial risk per share).</li>
      <li>Only one position at a time, risk per trade = 1% of current equity.</li>
    </ul>
  </div>

  <div class="card">
    <h2>Equity Curve & Drawdown</h2>
    <p>Equity starts at 1.0 and is adjusted based on realized R-multiples with 1% risk per trade.</p>
    <img src="../assets/gann_equity_curve.png" alt="Equity curve">
    <p>Drawdown (relative to equity peaks):</p>
    <img src="../assets/gann_drawdown_curve.png" alt="Drawdown curve">
  </div>

  <div class="card">
    <h2>Sample Trades</h2>
    <p>First few trades (for transparency):</p>
    <table>
      <tr>
        <th>#</th>
        <th>Entry date</th>
        <th>Exit date</th>
        <th>Side</th>
        <th>R</th>
        <th>Exit reason</th>
      </tr>
"""
    sample = trades_df.head(10)
    for idx, row in sample.iterrows():
        html += f"""
      <tr>
        <td>{idx+1}</td>
        <td>{row['entry_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['exit_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['position']}</td>
        <td>{row['R']:.2f}</td>
        <td>{row['exit_reason']}</td>
      </tr>
"""

    html += """
    </table>
  </div>

  <div class="footer">
    <p>
      This backtest ignores costs, slippage and real-world frictions. Educational use only, not trading advice.
    </p>
  </div>

</body>
</html>
"""
    return html

# ---------- MAIN ----------

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

if __name__ == "__main__":
    main()
