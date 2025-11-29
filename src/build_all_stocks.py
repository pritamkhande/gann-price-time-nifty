import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils_swing import detect_swings
from utils_gann import find_square_from_swing_low, find_square_from_swing_high
from utils_plot import make_equity_and_dd_plots, generate_trade_charts

# ==========================
# CONFIG
# ==========================

EOD_DIR = "EOD_Upstox"          # folder with 20MICRONS_EOD.csv etc.
EARLY_DIR = "Early_Data"        # optional per-symbol early close files

DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

ATR_PERIOD = 14
RISK_PER_TRADE = 0.02     # 2% risk
SLOPE_TOL = 0.25
MAX_LOOKAHEAD = 160

MASTER_INDEX_HTML = "docs/index.html"
TRADES_CSV_DIR = "data"   # per-symbol trades CSVs

os.makedirs("docs", exist_ok=True)
os.makedirs(TRADES_CSV_DIR, exist_ok=True)


# ==========================
# DATA LOADING
# ==========================

def load_symbol_data(path: str) -> pd.DataFrame:
    """
    Load one symbol CSV from EOD_Upstox.

    Expected columns:
      Symbol, Date, Open, High, Low, Close, Volume

    Date examples:
      2008-10-06 00:00:00+05:30

    We strip timezone and keep naive date for Gann calculations.
    """
    df = pd.read_csv(path)

    # Normalize column names / subset
    if "Symbol" in df.columns:
        # We don't need Symbol column except for sanity; keep rest
        pass

    df[DATE_COL] = pd.to_datetime(df["Date"], errors="coerce")
    # Remove timezone if present
    try:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    except TypeError:
        # already naive
        pass

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


def load_early_close_for_symbol(symbol: str) -> pd.DataFrame | None:
    """
    For each symbol, optional early close file:

      Early_Data/<SYMBOL>_early.csv

    Format:
      Date,EarlyClose
      2008-10-06,16.20
      2008-10-07,15.90
      ...

    Dates must match main EOD dates.
    """
    early_path = os.path.join(EARLY_DIR, f"{symbol}_early.csv")
    if not os.path.exists(early_path):
        return None

    edf = pd.read_csv(early_path)
    edf[DATE_COL] = pd.to_datetime(edf["Date"], errors="coerce")
    try:
        edf[DATE_COL] = edf[DATE_COL].dt.tz_localize(None)
    except TypeError:
        pass

    edf = edf.dropna(subset=[DATE_COL, "EarlyClose"])
    edf = edf.sort_values(DATE_COL).reset_index(drop=True)
    return edf


# ==========================
# HELPERS – POINT PROFITS
# ==========================

def calc_forward_point_profits(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    position: str,
    max_horizon: int = 4,
) -> list[float]:
    sign = 1.0 if position == "long" else -1.0
    pnls = []
    n = len(df)

    for k in range(0, max_horizon + 1):
        idx = entry_idx + k
        if idx >= n:
            pnls.append(np.nan)
        else:
            close_k = df.loc[idx, CLOSE_COL]
            pnl_pts = sign * (close_k - entry_price)
            pnls.append(float(pnl_pts))
    return pnls


def calc_tminus1_profit(
    df: pd.DataFrame,
    signal_idx: int | None,
    position: str,
) -> float:
    if signal_idx is None:
        return np.nan

    n = len(df)
    if signal_idx + 1 >= n:
        return np.nan

    sign = 1.0 if position == "long" else -1.0
    c_signal = df.loc[signal_idx, CLOSE_COL]
    c_next = df.loc[signal_idx + 1, CLOSE_COL]
    pnl_pts = sign * (c_next - c_signal)
    return float(pnl_pts)


# ==========================
# BACKTEST (SAME LOGIC AS NIFTY)
# ==========================

def backtest_symbol(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = 1.0
    in_trade = False
    position = None
    entry_idx = None
    entry_price = None
    stop_price = None
    initial_stop_price = None
    entry_square_type = None
    signal_idx = None
    signal_date = None

    trades = []

    n = len(df)
    i = 0

    while i < n - 2:
        if not in_trade:
            # SHORT setup from swing low
            if df.loc[i, "swing_low"]:
                from utils_gann import find_square_from_swing_low
                sq_idx, sq_type = find_square_from_swing_low(
                    df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
                )
                if sq_idx is not None and sq_idx < n - 1:
                    if df.loc[sq_idx + 1, CLOSE_COL] < df.loc[sq_idx, LOW_COL]:
                        in_trade = True
                        position = "short"
                        entry_idx = sq_idx + 1
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        entry_square_type = sq_type
                        sl = df.loc[sq_idx, HIGH_COL] + 2 * df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        initial_stop_price = sl
                        signal_idx = sq_idx
                        signal_date = df.loc[sq_idx, DATE_COL]
                        i = entry_idx
                        continue

            # LONG setup from swing high
            if df.loc[i, "swing_high"]:
                from utils_gann import find_square_from_swing_high
                sq_idx, sq_type = find_square_from_swing_high(
                    df, i, DATE_COL, CLOSE_COL, slope_tol=SLOPE_TOL, max_lookahead=MAX_LOOKAHEAD
                )
                if sq_idx is not None and sq_idx < n - 1:
                    if df.loc[sq_idx + 1, CLOSE_COL] > df.loc[sq_idx, HIGH_COL]:
                        in_trade = True
                        position = "long"
                        entry_idx = sq_idx + 1
                        entry_price = df.loc[entry_idx, OPEN_COL]
                        entry_square_type = sq_type
                        sl = df.loc[sq_idx, LOW_COL] - 2 * df.loc[sq_idx, "ATR"]
                        stop_price = sl
                        initial_stop_price = sl
                        signal_idx = sq_idx
                        signal_date = df.loc[sq_idx, DATE_COL]
                        i = entry_idx
                        continue

            i += 1

        else:
            # manage open trade with ATR trailing stop x3
            atr = df.loc[i, "ATR"]
            close = df.loc[i, CLOSE_COL]
            high = df.loc[i, HIGH_COL]
            low = df.loc[i, LOW_COL]
            date = df.loc[i, DATE_COL]

            if position == "long":
                trail = close - 3 * atr
                if trail > stop_price:
                    stop_price = trail
            else:  # short
                trail = close + 3 * atr
                if trail < stop_price:
                    stop_price = trail

            exit_reason = None
            exit_price = None

            if position == "long":
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
            else:
                if high >= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"

            if i == n - 1 and exit_reason is None:
                exit_price = close
                exit_reason = "End"

            if exit_reason is not None:
                if position == "long":
                    risk = entry_price - initial_stop_price
                    pnl = exit_price - entry_price
                else:
                    risk = initial_stop_price - entry_price
                    pnl = entry_price - exit_price

                r_mult = pnl / risk if risk != 0 else 0.0

                pts_Tm1 = calc_tminus1_profit(df, signal_idx, position)
                pts_T, pts_T1, pts_T2, pts_T3, pts_T4 = calc_forward_point_profits(
                    df, entry_idx, entry_price, position, max_horizon=4
                )

                trades.append(
                    {
                        "trade_no": len(trades) + 1,
                        "signal_index": signal_idx,
                        "signal_date": signal_date,
                        "entry_index": entry_idx,
                        "exit_index": i,
                        "entry_date": df.loc[entry_idx, DATE_COL],
                        "exit_date": date,
                        "position": position,
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "initial_stop_price": float(initial_stop_price),
                        "final_stop_price": float(stop_price),
                        "R": float(r_mult),
                        "pnl": float(pnl),
                        "exit_reason": exit_reason,
                        "square_type": entry_square_type,
                        "pts_Tm1": pts_Tm1,
                        "pts_T": pts_T,
                        "pts_T1": pts_T1,
                        "pts_T2": pts_T2,
                        "pts_T3": pts_T3,
                        "pts_T4": pts_T4,
                    }
                )

                risk_amount = equity * RISK_PER_TRADE
                equity += r_mult * risk_amount

                in_trade = False
                position = None
                entry_idx = None
                entry_price = None
                stop_price = None
                initial_stop_price = None
                entry_square_type = None
                signal_idx = None
                signal_date = None

            i += 1

    trades_df = pd.DataFrame(trades)

    # equity curve for symbol
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
# EARLY-CLOSE MARGINS
# ==========================

def attach_early_margins(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    early_df: pd.DataFrame,
) -> pd.DataFrame:
    early_map = early_df.set_index(DATE_COL)["EarlyClose"]

    early_closes = []
    m_neutral_pts = []
    m_neutral_pct = []
    m_flip_pts = []
    m_flip_pct = []

    for _, tr in trades_df.iterrows():
        entry_date = tr["entry_date"]
        pos = tr["position"]
        sig_idx = int(tr["signal_index"])

        ec = early_map.get(entry_date, np.nan)
        early_closes.append(ec)

        if pd.isna(ec):
            m_neutral_pts.append(np.nan)
            m_neutral_pct.append(np.nan)
            m_flip_pts.append(np.nan)
            m_flip_pct.append(np.nan)
            continue

        sq_high = price_df.loc[sig_idx, HIGH_COL]
        sq_low = price_df.loc[sig_idx, LOW_COL]

        if pos == "long":
            buf_neutral_pts = ec - sq_high
            buf_flip_pts = ec - sq_low
        else:
            buf_neutral_pts = sq_low - ec
            buf_flip_pts = sq_high - ec

        m_neutral_pts.append(float(buf_neutral_pts))
        m_flip_pts.append(float(buf_flip_pts))
        m_neutral_pct.append(100.0 * buf_neutral_pts / ec)
        m_flip_pct.append(100.0 * buf_flip_pts / ec)

    trades_df = trades_df.copy()
    trades_df["early_close"] = early_closes
    trades_df["margin_neutral_pts"] = m_neutral_pts
    trades_df["margin_neutral_pct"] = m_neutral_pct
    trades_df["margin_flip_pts"] = m_flip_pts
    trades_df["margin_flip_pct"] = m_flip_pct
    return trades_df


# ==========================
# METRICS & COMMENTARY
# ==========================

def compute_metrics(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
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


def build_system_commentary(symbol: str, metrics: dict, trades_df: pd.DataFrame) -> str:
    n = metrics["n_trades"]
    years = metrics["years"] or 0.0
    avg_R = metrics["avg_R"]
    win_rate = metrics["win_rate"]
    cagr = metrics["cagr"] * 100
    max_dd = metrics["max_dd"] * 100

    if years > 0:
        trades_per_year = n / years
    else:
        trades_per_year = 0.0

    if trades_df.empty:
        return f"No trades were generated for {symbol}. The current parameter set is too strict for this series."

    avg_hold = (trades_df["exit_index"] - trades_df["entry_index"]).mean()

    style = []
    if trades_per_year < 5:
        style.append("very selective, long-term system")
    elif trades_per_year < 15:
        style.append("moderately active swing system")
    else:
        style.append("active swing/position system")

    if max_dd < 5:
        style.append("with very conservative risk")
    elif max_dd < 12:
        style.append("with moderate risk")
    else:
        style.append("with aggressive risk")

    if cagr < 2:
        style.append("designed more for research than raw returns")
    elif cagr < 8:
        style.append("balanced between robustness and return")
    else:
        style.append("tilted towards maximising return")

    style_txt = ", ".join(style)

    return (
        f"For {symbol}, the system generated {n} trades over the full sample, averaging "
        f"about {trades_per_year:.1f} trades per year. The typical holding "
        f"period is around {avg_hold:.1f} bars. With a win rate of "
        f"{win_rate:.1f}% and an average outcome of {avg_R:.2f}R per trade, "
        f"the equity curve grows at roughly {cagr:.1f}% CAGR while suffering "
        f"a maximum drawdown of {max_dd:.1f}%. Overall, this behaves like a {style_txt}."
    )


# ==========================
# PER-SYMBOL HTML
# ==========================

def render_stock_html(symbol: str, metrics: dict, trades_df: pd.DataFrame, commentary: str) -> str:
    start_str = metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] else "N/A"
    end_str = metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] else "N/A"
    years_str = f"{metrics['years']:.1f}" if metrics["years"] else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{symbol} – Gann Squaring System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Mechanical Gann Price-Time and Price-Date Squaring backtest on {symbol} daily data.">
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
      font-size: 13px;
      table-layout: auto;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      white-space: nowrap;
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
    a.trade-link {{
      color: #2563eb;
      text-decoration: none;
    }}
    a.trade-link:hover {{
      text-decoration: underline;
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

  <h1>{symbol} – Gann Squaring System</h1>
  <p>
    Fully mechanical backtest of a Price-Time / Price-Date Squaring system inspired by W.D. Gann,
    applied to {symbol} daily data from {start_str} to {end_str}.
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
    <h2>System Behaviour Commentary</h2>
    <p>{commentary}</p>
  </div>

  <div class="card">
    <h2>Equity Curve and Drawdown</h2>
    <p>Equity starts at 1.0 and changes based on realized R-multiples with 2% risk per trade.</p>
    <img src="equity_curve.png" alt="Equity curve">
    <p>Drawdown relative to running equity peak:</p>
    <img src="drawdown_curve.png" alt="Drawdown curve">
  </div>

  <div class="card">
    <h2>All Trades (point profits + early-close margins)</h2>
    <table>
      <tr>
        <th>#</th>
        <th>Signal date</th>
        <th>Entry date</th>
        <th>Entry price</th>
        <th>Exit date</th>
        <th>Side</th>
        <th>R</th>
        <th>Square type</th>
        <th>Exit reason</th>
        <th>T(-1)</th>
        <th>T</th>
        <th>T+1</th>
        <th>T+2</th>
        <th>T+3</th>
        <th>T+4</th>
        <th>Early close</th>
        <th>Margin neutral (pts)</th>
        <th>Margin neutral (%)</th>
        <th>Margin flip (pts)</th>
        <th>Margin flip (%)</th>
        <th>Chart</th>
      </tr>
"""
    for _, row in trades_df.iterrows():
        trade_no = int(row["trade_no"])
        sig_date = row["signal_date"].strftime('%Y-%m-%d') if pd.notna(row["signal_date"]) else "NA"

        ec = row.get("early_close", np.nan)
        mn_pts = row.get("margin_neutral_pts", np.nan)
        mn_pct = row.get("margin_neutral_pct", np.nan)
        mf_pts = row.get("margin_flip_pts", np.nan)
        mf_pct = row.get("margin_flip_pct", np.nan)

        html += f"""
      <tr>
        <td>{trade_no}</td>
        <td>{sig_date}</td>
        <td>{row['entry_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['entry_price']:.2f}</td>
        <td>{row['exit_date'].strftime('%Y-%m-%d')}</td>
        <td>{row['position']}</td>
        <td>{row['R']:.2f}</td>
        <td>{row['square_type']}</td>
        <td>{row['exit_reason']}</td>
        <td>{row['pts_Tm1']:.2f}</td>
        <td>{row['pts_T']:.2f}</td>
        <td>{row['pts_T1']:.2f}</td>
        <td>{row['pts_T2']:.2f}</td>
        <td>{row['pts_T3']:.2f}</td>
        <td>{row['pts_T4']:.2f}</td>
        <td>{"" if pd.isna(ec) else f"{ec:.2f}"}</td>
        <td>{"" if pd.isna(mn_pts) else f"{mn_pts:.2f}"}</td>
        <td>{"" if pd.isna(mn_pct) else f"{mn_pct:.2f}%"}</td>
        <td>{"" if pd.isna(mf_pts) else f"{mf_pts:.2f}"}</td>
        <td>{"" if pd.isna(mf_pct) else f"{mf_pct:.2f}%"}</td>
        <td><a class="trade-link" href="trades/trade_{trade_no:03d}.html" target="_blank">View</a></td>
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
# MASTER INDEX HTML
# ==========================

def render_master_index(summaries: list[dict]) -> str:
    rows_html = ""
    for s in summaries:
        rows_html += f"""
      <tr>
        <td><a href="{s['link']}">{s['symbol']}</a></td>
        <td>{s['n_trades']}</td>
        <td>{s['win_rate']:.1f}%</td>
        <td>{s['avg_R']:.2f}</td>
        <td>{s['cagr']*100:.1f}%</td>
        <td>{s['max_dd']*100:.1f}%</td>
        <td>{s['years']:.1f}</td>
      </tr>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nifty Stocks – Gann Squaring System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Gann Price-Time / Price-Date Squaring backtest for all Nifty stocks.">
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 16px;
      background: #f7f7f9;
      color: #111827;
    }}
    h1 {{
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
      font-size: 13px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
    }}
    a {{
      color: #2563eb;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>

  <h1>Nifty Stocks – Gann Squaring System</h1>
  <div class="card">
    <p>
      Each stock below has a dedicated Gann Price-Time / Price-Date Squaring backtest report.
      Click on a symbol to see detailed trades, equity curve, and early-close margin statistics.
    </p>
  </div>

  <div class="card">
    <h2>Stock List</h2>
    <table>
      <tr>
        <th>Symbol</th>
        <th># Trades</th>
        <th>Win rate</th>
        <th>Avg R</th>
        <th>CAGR</th>
        <th>Max DD</th>
        <th>Years</th>
      </tr>
{rows_html}
    </table>
  </div>

</body>
</html>
"""
    return html


# ==========================
# MAIN LOOP: ALL FILES IN EOD_Upstox
# ==========================

def main():
    summaries = []

    if not os.path.isdir(EOD_DIR):
        print("EOD_Upstox folder not found.")
        return

    files = sorted(f for f in os.listdir(EOD_DIR) if f.endswith("_EOD.csv"))
    if not files:
        print("No *_EOD.csv files found in EOD_Upstox.")
        return

    for fname in files:
        symbol = fname.replace("_EOD.csv", "")
        path = os.path.join(EOD_DIR, fname)
        print(f"Processing {symbol} from {path} ...")

        try:
            df = load_symbol_data(path)
        except Exception as e:
            print(f"  Failed to load {symbol}: {e}")
            continue

        df = compute_atr(df)
        df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL,
                           lookback_main=1, lookback_fractal=2)

        trades_df, price_df = backtest_symbol(df)

        early_df = load_early_close_for_symbol(symbol)
        if early_df is not None and not trades_df.empty:
            trades_df = attach_early_margins(trades_df, price_df, early_df)

        # save trades CSV per symbol
        out_csv = os.path.join(TRADES_CSV_DIR, f"{symbol}_gann_trades.csv")
        trades_df.to_csv(out_csv, index=False)

        metrics = compute_metrics(trades_df, price_df)
        commentary = build_system_commentary(symbol, metrics, trades_df)

        # per-symbol output dirs
        sym_dir = os.path.join("docs", "stocks", symbol)
        os.makedirs(sym_dir, exist_ok=True)
        eq_png = os.path.join(sym_dir, "equity_curve.png")
        dd_png = os.path.join(sym_dir, "drawdown_curve.png")
        trades_dir = os.path.join(sym_dir, "trades")

        make_equity_and_dd_plots(price_df, DATE_COL, "equity", eq_png, dd_png)
        generate_trade_charts(price_df, trades_df, DATE_COL,
                              OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL,
                              out_dir=trades_dir)

        html = render_stock_html(symbol, metrics, trades_df, commentary)
        out_html = os.path.join(sym_dir, "index.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)

        summaries.append(
            {
                "symbol": symbol,
                "n_trades": metrics["n_trades"],
                "win_rate": metrics["win_rate"],
                "avg_R": metrics["avg_R"],
                "cagr": metrics["cagr"],
                "max_dd": metrics["max_dd"],
                "years": metrics["years"],
                "link": f"stocks/{symbol}/index.html",
            }
        )

    # Master index
    master_html = render_master_index(summaries)
    with open(MASTER_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(master_html)

    print("Master index written to", MASTER_INDEX_HTML)


if __name__ == "__main__":
    main()
