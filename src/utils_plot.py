import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def make_equity_and_dd_plots(
    df: pd.DataFrame,
    date_col: str,
    equity_col: str,
    out_equity_png: str,
    out_dd_png: str,
) -> None:
    """Create equity curve and drawdown PNGs using matplotlib."""
    if equity_col not in df.columns:
        return

    series = df[[date_col, equity_col]].dropna()
    if series.empty:
        return

    dates = series[date_col]
    equity = series[equity_col].astype(float).values

    # Equity curve
    _ensure_dir(out_equity_png)
    plt.figure(figsize=(10, 4))
    plt.plot(dates, equity, linewidth=1.5)
    plt.title("Gann Squaring – Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_equity_png, dpi=120)
    plt.close()

    # Drawdown
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks

    _ensure_dir(out_dd_png)
    plt.figure(figsize=(10, 3))
    plt.plot(dates, dd * 100.0, linewidth=1.5)
    plt.title("Drawdown (%) from Equity Peak")
    plt.xlabel("Date")
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dd_png, dpi=120)
    plt.close()


def generate_trade_charts(
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    date_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    out_dir: str = "docs/trades",
) -> None:
    """Generate simple per-trade HTML + PNG charts."""
    if trades_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)
    n = len(price_df)

    for _, tr in trades_df.iterrows():
        trade_no = int(tr["trade_no"])
        sig_idx = int(tr["signal_index"])
        entry_idx = int(tr["entry_index"])
        exit_idx = int(tr["exit_index"])

        start_idx = max(0, sig_idx - 30)
        end_idx = min(n - 1, exit_idx + 10)

        segment = price_df.loc[start_idx:end_idx].copy()
        dates = segment[date_col]
        closes = segment[close_col]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(dates, closes, label="Close", linewidth=1.2)

        entry_date = price_df.loc[entry_idx, date_col]
        entry_close = price_df.loc[entry_idx, close_col]
        exit_date = price_df.loc[exit_idx, date_col]
        exit_close = price_df.loc[exit_idx, close_col]

        side = tr["position"]
        if side == "long":
            ax.scatter(entry_date, entry_close, marker="^", s=80, label="Long entry")
            ax.scatter(exit_date, exit_close, marker="v", s=70, label="Exit")
        else:
            ax.scatter(entry_date, entry_close, marker="v", s=80, label="Short entry")
            ax.scatter(exit_date, exit_close, marker="^", s=70, label="Exit")

        ax.set_title(f"Trade {trade_no} – {side} ({entry_date.date()} → {exit_date.date()})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.autofmt_xdate()
        plt.tight_layout()

        png_name = f"trade_{trade_no:03d}.png"
        html_name = f"trade_{trade_no:03d}.html"
        png_path = os.path.join(out_dir, png_name)
        html_path = os.path.join(out_dir, html_name)

        plt.savefig(png_path, dpi=120)
        plt.close(fig)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trade {trade_no}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f7f7f9; padding:16px; }}
    h1 {{ font-size: 20px; margin-bottom: 12px; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
  <h1>Trade {trade_no}</h1>
  <p>Side: {side}, R = {tr['R']:.2f}</p>
  <img src="{png_name}" alt="Trade {trade_no} chart">
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)


def make_signals_chart(
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    date_col: str,
    close_col: str,
    out_png: str,
) -> None:
    """Create a single overview chart showing all signals on one price series."""
    if price_df.empty:
        return

    _ensure_dir(out_png)

    dates = price_df[date_col]
    closes = price_df[close_col]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, closes, linewidth=1.0, label="Close")

    if not trades_df.empty:
        for _, tr in trades_df.iterrows():
            entry_idx = int(tr["entry_index"])
            exit_idx = int(tr["exit_index"])
            pos = tr["position"]

            if 0 <= entry_idx < len(price_df):
                de = price_df.loc[entry_idx, date_col]
                ce = price_df.loc[entry_idx, close_col]
            else:
                continue

            if 0 <= exit_idx < len(price_df):
                dx = price_df.loc[exit_idx, date_col]
                cx = price_df.loc[exit_idx, close_col]
            else:
                dx, cx = None, None

            if pos == "long":
                ax.scatter(de, ce, marker="^", s=40, color="green")
            else:
                ax.scatter(de, ce, marker="v", s=40, color="red")

            if dx is not None:
                ax.scatter(dx, cx, marker="x", s=35, color="blue")

    ax.set_title("Price with all Gann entry/exit signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
