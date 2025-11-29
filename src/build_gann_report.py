import os
from datetime import datetime

import numpy as np
import pandas as pd

from utils_swing import detect_swings
from utils_plot import (
    make_equity_and_dd_plots,
    generate_trade_charts,
    make_signals_chart,
)

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/nifty_daily.csv"
EARLY_DATA_PATH = "Early_Data/nifty_early_close.csv"  # optional

DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

ATR_PERIOD = 14
RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_LOOKAHEAD = 160    # bars to scan forward from a swing

OUT_REPORT_HTML = "docs/index.html"
OUT_TRADES_CSV = "data/gann_nifty_trades.csv"
OUT_EQUITY_PNG = "docs/gann_equity_curve.png"
OUT_DD_PNG = "docs/gann_drawdown_curve.png"
OUT_SIGNALS_PNG = "docs/gann_signals_nifty.png"

os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)


# =====================================================
# DATA LOADING & INDICATORS
# =====================================================

def load_data() -> pd.DataFrame:
    """
    Load Nifty daily OHLCV.

    Expect csv: data/nifty_daily.csv with columns:
    Date, Open, High, Low, Close, AdjClose, Volume
    Date format: dd-mm-YYYY
    """
    df = pd.read_csv(DATA_PATH)

    # Parse dd-mm-YYYY safely
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")

    df = df.sort_values(DATE_COL).reset_index(drop=True)

    for c in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Classic ATR with simple moving average."""
    high = df[HIGH_COL]
    low = df[LOW_COL]
    close = df[CLOSE_COL]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()

    df = df.copy()
    df["ATR"] = atr
    return df


# =====================================================
# GANN SQUARE HELPERS (local implementation)
# =====================================================

def _is_near_square(n: float, tol: float = 0.25) -> bool:
    """Check if n is close to an integer square like 25, 36, 49, etc."""
    if not np.isfinite(n) or n <= 0:
        return False
    root = np.sqrt(n)
    nearest = round(root)
    if nearest <= 0:
        return False
    sq = nearest * nearest
    return abs(n - sq) <= tol * sq


def _is_near_extended_square(n: float, tol: float = 0.25) -> bool:
    """Check against classic / extended Gann square numbers."""
    if not np.isfinite(n) or n <= 0:
        return False
    important = np.array(
        [25, 36, 49, 64, 81, 100, 121, 50, 72, 98, 128], dtype=float
    )
    diff = np.abs(important - n)
    idx = diff.argmin()
    closest = important[idx]
    return diff[idx] <= tol * closest


def _classify_square(delta_p: float, delta_bars: int, delta_days: int) -> str:
    """Return 'time', 'date', 'both' or '' depending on which dimension squares."""
    flags = []
    if _is_near_extended_square(abs(delta_bars)):
        flags.append("time")
    if _is_near_extended_square(abs(delta_days)):
        flags.append("date")

    if len(flags) == 2:
        return "both"
    elif len(flags) == 1:
        return flags[0]
    else:
        return ""


def find_square_from_swing_low(
    df: pd.DataFrame,
    swing_idx: int,
    max_lookahead: int = MAX_LOOKAHEAD,
) -> tuple[bool, int | None]:
    """
    From a swing low at swing_idx, scan forward up to max_lookahead bars
    looking for an up-move where price/time/date differences are close to
    square numbers.

    Returns (is_square_found, square_index).
    """
    n = len(df)
    swing_price = float(df.loc[swing_idx, CLOSE_COL])
    swing_date = df.loc[swing_idx, DATE_COL]

    for j in range(swing_idx + 1, min(n, swing_idx + max_lookahead + 1)):
        close_j = float(df.loc[j, CLOSE_COL])
        dp = close_j - swing_price
        if dp <= 0:
            continue  # need up-move from swing low

        bars = j - swing_idx
        days = (df.loc[j, DATE_COL] - swing_date).days
        dp_abs = abs(dp)

        # Rough price-time/date square check
        if _is_near_extended_square(abs(bars)) or _is_near_extended_square(abs(days)):
            return True, j
        if _is_near_square(dp_abs):
            return True, j

    return False, None


def find_square_from_swing_high(
    df: pd.DataFrame,
    swing_idx: int,
    max_lookahead: int = MAX_LOOKAHEAD,
) -> tuple[bool, int | None]:
    """
    From a swing high at swing_idx, scan forward up to max_lookahead bars
    for a down-move that squares price/time/date.
    """
    n = len(df)
    swing_price = float(df.loc[swing_idx, CLOSE_COL])
    swing_date = df.loc[swing_idx, DATE_COL]

    for j in range(swing_idx + 1, min(n, swing_idx + max_lookahead + 1)):
        close_j = float(df.loc[j, CLOSE_COL])
        dp = close_j - swing_price
        if dp >= 0:
            continue  # need down-move from swing high

        bars = j - swing_idx
        days = (df.loc[j, DATE_COL] - swing_date).days
        dp_abs = abs(dp)

        if _is_near_extended_square(abs(bars)) or _is_near_extended_square(abs(days)):
            return True, j
        if _is_near_square(dp_abs):
            return True, j

    return False, None


# =====================================================
# BACKTEST HELPERS
# =====================================================

def _atr_at(df: pd.DataFrame, idx: int) -> float:
    v = df.loc[idx, "ATR"]
    return float(v) if np.isfinite(v) else 0.0


def _update_trailing_stop(position: str, stop: float, price: float, atr: float) -> float:
    """3×ATR trailing stop."""
    offset = 3.0 * atr
    if position == "long":
        new_stop = max(stop, price - offset)
    else:
        new_stop = min(stop, price + offset)
    return new_stop


def calc_forward_point_profits(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    position: str,
    max_horizon: int = 4,
) -> tuple[float, float, float, float, float]:
    """
    Point P&L at T (entry day close) and forward T+1...T+4 close, in points.
    """
    n = len(df)
    sign = 1.0 if position == "long" else -1.0

    def pts_at(offset: int) -> float:
        idx = entry_idx + offset
        if idx >= n:
            return np.nan
        close = df.loc[idx, CLOSE_COL]
        return float(sign * (close - entry_price))

    return (
        pts_at(0),
        pts_at(1),
        pts_at(2),
        pts_at(3),
        pts_at(4),
    )


def calc_tminus1_profit(
    df: pd.DataFrame,
    signal_idx: int,
    position: str,
) -> float:
    """
    Profit if you enter at the signal bar's close (T-1) and exit next bar close.
    """
    n = len(df)
    if signal_idx < 0 or signal_idx >= n:
        return np.nan
    if signal_idx + 1 >= n:
        return np.nan

    sign = 1.0 if position == "long" else -1.0
    c_signal = df.loc[signal_idx, CLOSE_COL]
    c_next = df.loc[signal_idx + 1, CLOSE_COL]
    pnl_pts = sign * (c_next - c_signal)
    return float(pnl_pts)


# =====================================================
# BACKTEST WITH TRAILING STOP
# =====================================================

def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    while i < n:
        row = df.iloc[i]
        date = row[DATE_COL]
        close = row[CLOSE_COL]
        atr_val = row["ATR"]

        # store equity as we go
        df.loc[i, "equity"] = equity

        if not in_trade:
            # ---- new trades ----
            if row.get("swing_low", False):
                is_sq, sq_idx = find_square_from_swing_low(df, i, MAX_LOOKAHEAD)
                if is_sq and sq_idx is not None:
                    swing_date = df.loc[i, DATE_COL]
                    sq_date = df.loc[sq_idx, DATE_COL]
                    delta_days = (sq_date - swing_date).days
                    delta_bars = sq_idx - i
                    delta_p = df.loc[sq_idx, CLOSE_COL] - df.loc[i, CLOSE_COL]
                    sq_type = _classify_square(delta_p, delta_bars, delta_days)
                    entry_square_type = sq_type if sq_type else "time"

                    # bearish confirmation: close below square low
                    if close < df.loc[sq_idx, LOW_COL] and i + 1 < n:
                        in_trade = True
                        position = "short"
                        entry_idx = i + 1
                        entry_price = float(df.loc[entry_idx, OPEN_COL])
                        signal_idx = i
                        signal_date = date

                        atr_here = _atr_at(df, entry_idx) or atr_val
                        if atr_here == 0:
                            risk_pts = 0.02 * entry_price
                        else:
                            risk_pts = 2.0 * atr_here
                        if np.isnan(risk_pts) or risk_pts <= 0:
                            risk_pts = 0.02 * entry_price

                        stop_price = entry_price + risk_pts
                        initial_stop_price = stop_price

            elif row.get("swing_high", False):
                is_sq, sq_idx = find_square_from_swing_high(df, i, MAX_LOOKAHEAD)
                if is_sq and sq_idx is not None:
                    swing_date = df.loc[i, DATE_COL]
                    sq_date = df.loc[sq_idx, DATE_COL]
                    delta_days = (sq_date - swing_date).days
                    delta_bars = sq_idx - i
                    delta_p = df.loc[sq_idx, CLOSE_COL] - df.loc[i, CLOSE_COL]
                    sq_type = _classify_square(delta_p, delta_bars, delta_days)
                    entry_square_type = sq_type if sq_type else "time"

                    # bullish confirmation: close above square high
                    if close > df.loc[sq_idx, HIGH_COL] and i + 1 < n:
                        in_trade = True
                        position = "long"
                        entry_idx = i + 1
                        entry_price = float(df.loc[entry_idx, OPEN_COL])
                        signal_idx = i
                        signal_date = date

                        atr_here = _atr_at(df, entry_idx) or atr_val
                        if atr_here == 0:
                            risk_pts = 0.02 * entry_price
                        else:
                            risk_pts = 2.0 * atr_here
                        if np.isnan(risk_pts) or risk_pts <= 0:
                            risk_pts = 0.02 * entry_price

                        stop_price = entry_price - risk_pts
                        initial_stop_price = stop_price

        else:
            # ---- manage open trade ----
            low = row[LOW_COL]
            high = row[HIGH_COL]

            atr_here = atr_val if np.isfinite(atr_val) and atr_val > 0 else _atr_at(df, i)
            if atr_here == 0:
                atr_here = abs(close - entry_price)

            stop_price = _update_trailing_stop(position, stop_price, close, atr_here)

            exit_reason = None
            exit_price = None

            if position == "long":
                if low <= stop_price:
                    exit_price = stop_price
                    exit_reason = "SL"
            else:  # short
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
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position": position,
                        "R": r_mult,
                        "square_type": entry_square_type,
                        "exit_reason": exit_reason,
                        "T(-1)": pts_Tm1,
                        "T": pts_T,
                        "T+1": pts_T1,
                        "T+2": pts_T2,
                        "T+3": pts_T3,
                        "T+4": pts_T4,
                    }
                )

                equity *= (1.0 + RISK_PER_TRADE * r_mult)

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
    return trades_df, df


# =====================================================
# EARLY-CLOSE MARGINS
# =====================================================

def load_early_close() -> pd.DataFrame | None:
    """
    Optional early-close file:

    Early_Data/nifty_early_close.csv with columns:
      Date, EarlyClose

    Date can be yyyy-mm-dd or dd-mm-yyyy; we normalise.
    """
    if not os.path.exists(EARLY_DATA_PATH):
        return None

    edf = pd.read_csv(EARLY_DATA_PATH)
    # try both
    try:
        edf[DATE_COL] = pd.to_datetime(edf[DATE_COL], dayfirst=True, errors="coerce")
    except Exception:
        edf[DATE_COL] = pd.to_datetime(edf[DATE_COL], errors="coerce")

    edf = edf.sort_values(DATE_COL).reset_index(drop=True)
    if "EarlyClose" not in edf.columns:
        raise ValueError("Early_Data CSV must have column 'EarlyClose'")
    edf = edf.dropna(subset=[DATE_COL, "EarlyClose"])
    return edf


def attach_early_margins(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    early_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach early-close and margin buffers to each trade."""
    if trades_df.empty:
        return trades_df

    early_map = dict(zip(early_df[DATE_COL].dt.normalize(), early_df["EarlyClose"]))

    early_closes = []
    m_neutral_pts = []
    m_neutral_pct = []
    m_flip_pts = []
    m_flip_pct = []

    for _, tr in trades_df.iterrows():
        sig_idx = int(tr["signal_index"])
        pos = tr["position"]
        entry_date = pd.to_datetime(tr["entry_date"]).normalize()

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
        else:  # short
            buf_neutral_pts = sq_low - ec
            buf_flip_pts = sq_high - ec

        m_neutral_pts.append(float(buf_neutral_pts))
        m_flip_pts.append(float(buf_flip_pts))
        m_neutral_pct.append(100.0 * buf_neutral_pts / ec)
        m_flip_pct.append(100.0 * buf_flip_pts / ec)

    trades_df = trades_df.copy()
    trades_df["Early close"] = early_closes
    trades_df["Margin neutral (pts)"] = m_neutral_pts
    trades_df["Margin neutral (%)"] = m_neutral_pct
    trades_df["Margin flip (pts)"] = m_flip_pts
    trades_df["Margin flip (%)"] = m_flip_pct

    return trades_df


# =====================================================
# METRICS & COMMENTARY
# =====================================================

def compute_metrics(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    metrics = {
        "n_trades": len(trades_df),
        "win_rate": 0.0,
        "avg_R": 0.0,
        "cagr": 0.0,
        "max_dd": 0.0,
        "years": 0.0,
        "start_date": None,
        "end_date": None,
    }

    if trades_df.empty:
        return metrics

    wins = trades_df[trades_df["R"] > 0]
    metrics["win_rate"] = 100.0 * len(wins) / len(trades_df)
    metrics["avg_R"] = trades_df["R"].mean()

    start_date = price_df[DATE_COL].min()
    end_date = price_df[DATE_COL].max()
    metrics["start_date"] = start_date
    metrics["end_date"] = end_date

    years = (end_date - start_date).days / 365.25
    metrics["years"] = years

    trade_equity = 1.0
    eq_series = []
    for r in trades_df["R"]:
        trade_equity *= (1.0 + RISK_PER_TRADE * r)
        eq_series.append(trade_equity)

    if years > 0 and trade_equity > 0:
        cagr = (trade_equity ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr = 0.0
    metrics["cagr"] = cagr

    eq_arr = np.array(eq_series)
    peaks = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peaks) / peaks
    metrics["max_dd"] = dd.min() * 100.0

    return metrics


def build_system_commentary(metrics: dict, trades_df: pd.DataFrame) -> str:
    n = metrics["n_trades"]
    if n == 0:
        return "No trades were generated over the sample – the system conditions were too strict."

    trades_per_year = n / metrics["years"] if metrics["years"] > 0 else 0.0
    avg_hold = trades_df["exit_index"] - trades_df["entry_index"]
    avg_hold = float(avg_hold.mean()) if len(avg_hold) else 0.0

    win_rate = metrics["win_rate"]
    avg_R = metrics["avg_R"]
    cagr = metrics["cagr"]
    max_dd = metrics["max_dd"]

    style = []
    if trades_per_year < 3:
        style.append("very selective, long-term system")
    elif trades_per_year < 10:
        style.append("moderately active swing system")
    else:
        style.append("active swing / position system")

    if max_dd > -5:
        style.append("very conservative drawdown profile")
    elif max_dd > -12:
        style.append("moderate drawdown profile")
    else:
        style.append("willing to tolerate deeper drawdowns")

    if cagr < 4:
        style.append("designed more for research than raw returns")
    elif cagr < 8:
        style.append("balanced between robustness and return")
    else:
        style.append("tilted towards maximising return")

    style_txt = ", ".join(style)

    return (
        f"The system generated {n} trades over the full sample, averaging "
        f"about {trades_per_year:.1f} trades per year. The typical holding "
        f"period is around {avg_hold:.1f} bars. With a win rate of "
        f"{win_rate:.1f}% and an average outcome of {avg_R:.2f}R per trade, "
        f"the equity curve grows at roughly {cagr:.1f}% CAGR while suffering "
        f"a maximum drawdown of {max_dd:.1f}%. Overall, this behaves like a {style_txt}."
    )


# =====================================================
# HTML REPORT
# =====================================================

def render_html(metrics: dict, trades_df: pd.DataFrame, commentary: str) -> str:
    start_str = metrics["start_date"].strftime("%d-%m-%Y") if metrics["start_date"] else "N/A"
    end_str = metrics["end_date"].strftime("%d-%m-%Y") if metrics["end_date"] else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Nifty – Gann Squaring System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --bg: #f9fafb;
      --card-bg: #ffffff;
      --border: #e5e7eb;
      --text-main: #111827;
      --text-muted: #6b7280;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --loss: #b91c1c;
      --gain: #15803d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text-main);
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 4px;
    }}
    h2 {{
      font-size: 20px;
      margin-bottom: 8px;
    }}
    p {{ margin: 4px 0; }}
    .subtitle {{
      color: var(--text-muted);
      margin-bottom: 24px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card-bg);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 16px 18px;
      box-shadow: 0 10px 15px -10px rgba(15, 23, 42, 0.12);
    }}
    .card h3 {{
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-muted);
      margin: 0 0 4px;
    }}
    .card .value {{
      font-size: 20px;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    thead tr {{
      background: #f3f4f6;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    tbody tr:nth-child(even) {{
      background: #f9fafb;
    }}
    .gain {{ color: var(--gain); }}
    .loss {{ color: var(--loss); }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 768px) {{
      .cards {{
        grid-template-columns: 1fr 1fr;
      }}
      th:nth-child(n+9),
      td:nth-child(n+9) {{
        display: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Nifty – Gann Squaring System</h1>
    <p class="subtitle">
      Fully mechanical backtest of a Price-Time / Price-Date Squaring system inspired by W.D. Gann,
      applied to Nifty daily data from {start_str} to {end_str}.
    </p>

    <div class="card" style="margin-bottom: 20px;">
      <h2>Backtest Summary</h2>
      <div class="cards">
        <div class="card">
          <h3>Number of trades</h3>
          <div class="value">{metrics['n_trades']}</div>
          <p class="small">{metrics['years']:.1f} yrs test length</p>
        </div>
        <div class="card">
          <h3>Win rate</h3>
          <div class="value">{metrics['win_rate']:.1f}%</div>
        </div>
        <div class="card">
          <h3>Average R per trade</h3>
          <div class="value">{metrics['avg_R']:.2f} R</div>
        </div>
        <div class="card">
          <h3>CAGR (normalized equity)</h3>
          <div class="value">{metrics['cagr']:.1f}%</div>
        </div>
        <div class="card">
          <h3>Maximum drawdown</h3>
          <div class="value">{metrics['max_dd']:.1f}%</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>System Behaviour Commentary</h2>
      <p>{commentary}</p>
    </div>

    <div class="card">
      <h2>Price chart with all signals</h2>
      <p>Close price with all long / short entries (triangles) and exits (x markers) for this Gann system.</p>
      <img src="gann_signals_nifty.png" alt="Signals on price chart">
    </div>

    <div class="card">
      <h2>Equity Curve and Drawdown</h2>
      <p>Equity starts at 1.0 and changes based on realized R-multiples with 2% risk per trade.</p>
      <img src="gann_equity_curve.png" alt="Equity curve">
      <p>Drawdown relative to running equity peak:</p>
      <img src="gann_drawdown_curve.png" alt="Drawdown curve">
    </div>

    <div class="card">
      <h2>System Logic</h2>
      <h3>1. Swing points</h3>
      <ul>
        <li>Timeframe: daily Nifty OHLC data.</li>
        <li>Swing highs and lows detected using both tight +/-1-bar pivots and Williams-style fractals.</li>
      </ul>

      <h3>2. Gann Price-Time / Price-Date Squares</h3>
      <ul>
        <li>From each swing, scan forward up to {MAX_LOOKAHEAD} bars.</li>
        <li>Let ΔP = |Close − swing Close| in points, ΔBars = bars, ΔDays = calendar days.</li>
        <li>We look for cases where ΔP, ΔBars and/or ΔDays are near classic / extended square numbers (25, 36, 49, 64, 81, 100, 121, 50, 72, 98, 128).</li>
        <li>These zones identify potential turning points where price and time/date are in balance.</li>
      </ul>

      <h3>3. Entries and exits</h3>
      <ul>
        <li>Short: from squared up-move after swing low, with bearish confirmation; entry next open.</li>
        <li>Long: from squared down-move after swing high, with bullish confirmation; entry next open.</li>
        <li>Initial SL: swing square bar high/low ± 2×ATR(14).</li>
        <li>Exit: ATR trailing stop (3×ATR) moves in favour of the trade; no fixed profit target.</li>
        <li>Risk per trade: 2% of equity. One position at a time.</li>
      </ul>
    </div>

    <div class="card">
      <h2>Completed Trades (point profits + early-close margins)</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Signal date</th>
              <th>Entry date</th>
              <th>Entry price</th>
              <th>Exit date</th>
              <th>Exit price</th>
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
          </thead>
          <tbody>
"""
    for _, row in trades_df.iterrows():
        r_class = "gain" if row["R"] > 0 else "loss" if row["R"] < 0 else ""
        html += f"""
            <tr>
              <td>{int(row['trade_no'])}</td>
              <td>{pd.to_datetime(row['signal_date']).strftime('%d-%m-%Y')}</td>
              <td>{pd.to_datetime(row['entry_date']).strftime('%d-%m-%Y')}</td>
              <td>{row['entry_price']:.2f}</td>
              <td>{pd.to_datetime(row['exit_date']).strftime('%d-%m-%Y')}</td>
              <td>{row['exit_price']:.2f}</td>
              <td>{row['position']}</td>
              <td class="{r_class}">{row['R']:.2f}</td>
              <td>{row['square_type']}</td>
              <td>{row['exit_reason']}</td>
              <td>{row['T(-1)']:.2f}</td>
              <td>{row['T']:.2f}</td>
              <td>{row['T+1']:.2f}</td>
              <td>{row['T+2']:.2f}</td>
              <td>{row['T+3']:.2f}</td>
              <td>{row['T+4']:.2f}</td>
              <td>{"" if pd.isna(row['Early close']) else f"{row['Early close']:.2f}"}</td>
              <td>{"" if pd.isna(row['Margin neutral (pts)']) else f"{row['Margin neutral (pts)']:.2f}"}</td>
              <td>{"" if pd.isna(row['Margin neutral (%)']) else f"{row['Margin neutral (%)']:.2f}"}</td>
              <td>{"" if pd.isna(row['Margin flip (pts)']) else f"{row['Margin flip (pts)']:.2f}"}</td>
              <td>{"" if pd.isna(row['Margin flip (%)']) else f"{row['Margin flip (%)']:.2f}"}</td>
              <td><a href="trades/trade_{int(row['trade_no']):03d}.html">View</a></td>
            </tr>
"""
    html += """
          </tbody>
        </table>
      </div>
    </div>

  </div>
</body>
</html>
"""
    return html


# =====================================================
# MAIN
# =====================================================

def main():
    df = load_data()
    df = compute_atr(df)

    # Swing detection from utils_swing.py
    df = detect_swings(
        df,
        low_col=LOW_COL,
        high_col=HIGH_COL,
        lookback_main=1,
        lookback_fractal=2,
    )

    trades_df, price_df = backtest(df)

    early_df = load_early_close()
    if early_df is not None:
        trades_df = attach_early_margins(trades_df, price_df, early_df)

    trades_df.to_csv(OUT_TRADES_CSV, index=False)

    metrics = compute_metrics(trades_df, price_df)
    commentary = build_system_commentary(metrics, trades_df)

    make_equity_and_dd_plots(price_df, DATE_COL, "equity", OUT_EQUITY_PNG, OUT_DD_PNG)
    make_signals_chart(price_df, trades_df, DATE_COL, CLOSE_COL, OUT_SIGNALS_PNG)
    generate_trade_charts(price_df, trades_df, DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL)

    html = render_html(metrics, trades_df, commentary)
    with open(OUT_REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("Report written to", OUT_REPORT_HTML)
    print("Trades CSV:", OUT_TRADES_CSV)


if __name__ == "__main__":
    main()
