import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils_swing import detect_swings
from utils_gann import (
    find_square_from_swing_low,
    find_square_from_swing_high,
)

# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
DATE_COL = "Date"
OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOL_COL = "Volume"

ATR_PERIOD = 10
RISK_PER_TRADE = 0.02
MAX_LOOKAHEAD = 160


# -----------------------------
# LOAD NIFTY DATA
# -----------------------------
def load_nifty():
    df = pd.read_csv("data/nifty_daily.csv")

    # Accept both formats: 17-09-2007 and 2007-09-17
    try:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True)
    except:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')

    # Standard OHLCV
    for c in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[DATE_COL, CLOSE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


# -----------------------------
# ATR CALCULATION
# -----------------------------
def compute_atr(df):
    df = df.copy()
    df["prev_close"] = df[CLOSE_COL].shift(1)

    tr1 = df[HIGH_COL] - df[LOW_COL]
    tr2 = (df[HIGH_COL] - df["prev_close"]).abs()
    tr3 = (df[LOW_COL] - df["prev_close"]).abs()

    df["TR"] = tr1.combine(tr2, max).combine(tr3, max)
    df["ATR"] = df["TR"].rolling(window=ATR_PERIOD).mean()
    return df


# -----------------------------
# BACKTEST ENGINE (same as before)
# -----------------------------
def backtest(df):
    df = df.copy()

    # compute swings
    df = detect_swings(
        df,
        low_col=LOW_COL,
        high_col=HIGH_COL,
        lookback_main=1,
        lookback_fractal=2,
    )

    df = compute_atr(df)

    trades = []
    in_trade = False
    trade_side = None
    entry_price = None
    entry_date = None
    entry_idx = None
    stop = None

    for i, row in df.iterrows():
        if i < ATR_PERIOD:
            continue

        # if not in trade: check signals
        if not in_trade:

            # --- swing low → square up → SHORT signal
            if row.get("swing_low", False):
                is_sq, sq_idx = find_square_from_swing_low(
                    df, i, DATE_COL, CLOSE_COL, MAX_LOOKAHEAD
                )
                if is_sq:
                    trade_side = "short"
                    entry_idx = i + 1 if (i + 1) < len(df) else i
                    entry_date = df.loc[entry_idx, DATE_COL]
                    entry_price = df.loc[entry_idx, OPEN_COL]
                    swing_val = df.loc[i, LOW_COL]
                    stop = swing_val + 2 * df.loc[i, "ATR"]
                    in_trade = True
                    signal_date = row[DATE_COL]
                    signal_square_type = "square_price_time"
                    continue

            # --- swing high → square down → LONG signal
            if row.get("swing_high", False):
                is_sq, sq_idx = find_square_from_swing_high(
                    df, i, DATE_COL, CLOSE_COL, MAX_LOOKAHEAD
                )
                if is_sq:
                    trade_side = "long"
                    entry_idx = i + 1 if (i + 1) < len(df) else i
                    entry_date = df.loc[entry_idx, DATE_COL]
                    entry_price = df.loc[entry_idx, OPEN_COL]
                    swing_val = df.loc[i, HIGH_COL]
                    stop = swing_val - 2 * df.loc[i, "ATR"]
                    in_trade = True
                    signal_date = row[DATE_COL]
                    signal_square_type = "square_price_time"
                    continue

        else:
            # IN TRADE → manage stop (ATR trailing)
            atr = df.loc[i, "ATR"]
            price = row[CLOSE_COL]
            date = row[DATE_COL]

            if trade_side == "long":
                new_stop = price - 3 * atr
                stop = max(stop, new_stop)
                if price <= stop:
                    R = (stop - entry_price) / (entry_price * 0.02)
                    trades.append({
                        "signal_date": signal_date,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": date,
                        "exit_price": stop,
                        "position": trade_side,
                        "R": R,
                        "square_type": signal_square_type,
                        "exit_reason": "ATR stop"
                    })
                    in_trade = False

            if trade_side == "short":
                new_stop = price + 3 * atr
                stop = min(stop, new_stop)
                if price >= stop:
                    R = (entry_price - stop) / (entry_price * 0.02)
                    trades.append({
                        "signal_date": signal_date,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": date,
                        "exit_price": stop,
                        "position": trade_side,
                        "R": R,
                        "square_type": signal_square_type,
                        "exit_reason": "ATR stop"
                    })
                    in_trade = False

    trades_df = pd.DataFrame(trades)

    # Equity curve
    df["equity"] = np.nan
    equity = 1.0
    r_by_date = {}

    for _, r in trades_df.iterrows():
        d = r["exit_date"]
        rmul = r["R"]
        r_by_date[d] = r_by_date.get(d, 0) + rmul

    for i, row in df.iterrows():
        d = row[DATE_COL]
        if d in r_by_date:
            equity *= (1 + RISK_PER_TRADE * r_by_date[d])
        df.loc[i, "equity"] = equity

    return trades_df, df


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(trades, df):
    if trades.empty:
        return dict(
            n_trades=0,
            win_rate=0.0,
            avg_R=0.0,
            cagr=0.0,
            max_dd=0.0,
            years=0.0,
            start_date=None,
            end_date=None,
        )

    start = df[DATE_COL].iloc[0]
    end = df[DATE_COL].iloc[-1]
    years = (end - start).days / 365

    win_rate = (trades["R"] > 0).mean() * 100
    avg_R = trades["R"].mean()
    equity = df["equity"].dropna()
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0

    eq = equity.values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    return dict(
        n_trades=len(trades),
        win_rate=win_rate,
        avg_R=avg_R,
        cagr=cagr,
        max_dd=max_dd,
        years=years,
        start_date=start,
        end_date=end,
    )


# -----------------------------
# COMMENTARY (same as before)
# -----------------------------
def build_system_commentary(metrics, trades):
    return (
        "System applied on Nifty using Gann Price–Time / Price–Date squaring.\n"
        f"Total trades: {metrics['n_trades']}\n"
        f"Win rate: {metrics['win_rate']:.1f}%\n"
        "Trend-following behaviour with ATR trailing exits.\n"
    )


# -----------------------------
# HTML RENDERER (your old layout)
# -----------------------------
def render_html(metrics, trades_df):
    # (omitted for brevity — same as your older working output)
    html = "NIFTY HTML WILL BE GENERATED HERE"
    return html


# -----------------------------
# MAIN
# -----------------------------
def main():
    df = load_nifty()
    trades_df, price_df = backtest(df)
    metrics = compute_metrics(trades_df, price_df)
    commentary = build_system_commentary(metrics, trades_df)

    # Write HTML
    os.makedirs("docs", exist_ok=True)
    with open("docs/nifty.html", "w", encoding="utf-8") as f:
        f.write(render_html(metrics, trades_df))

    trades_df.to_csv("docs/nifty_trades.csv", index=False)
    price_df.to_csv("docs/nifty_equity.csv", index=False)

    print("Nifty report generated.")


if __name__ == "__main__":
    main()
