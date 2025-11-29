# src/build_gann_stocks.py

import os
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Reuse core Nifty logic
from build_gann_report import (
    DATE_COL,
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    backtest,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_nifty_df() -> Tuple[str, pd.DataFrame]:
    """
    Load Nifty from data/nifty_daily.csv and give it symbol 'NIFTY'.
    """
    csv_path = Path("data") / "nifty_daily.csv"
    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "Date": DATE_COL,
            "Open": OPEN_COL,
            "High": HIGH_COL,
            "Low": LOW_COL,
            "Close": CLOSE_COL,
        }
    )
    df["Symbol"] = "NIFTY"
    return "NIFTY", df


def load_stock_df(csv_path: Path) -> Tuple[str, pd.DataFrame]:
    """
    Load one stock from EOD_Upstox/*.csv
    Expected columns: Symbol, Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(csv_path)

    # Date like 2008-10-06 00:00:00+05:30 -> keep only calendar date
    df[DATE_COL] = pd.to_datetime(df["Date"]).dt.date

    df = df.rename(
        columns={
            "Open": OPEN_COL,
            "High": HIGH_COL,
            "Low": LOW_COL,
            "Close": CLOSE_COL,
        }
    )

    symbol = str(df["Symbol"].iloc[0]).strip()
    df["Symbol"] = symbol

    return symbol, df[[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, "Symbol"]]


def compute_holding_point_profits(
    trades: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each completed trade, compute point P/L if we exit at:
      T(-1): signal day close
      T(0): entry day close
      T(+1..+4): close 1..4 bars after entry
    """
    price_series = price_df.set_index(DATE_COL)[CLOSE_COL].sort_index()

    offsets = [-1, 0, 1, 2, 3, 4]
    colnames = ["T(-1)", "T", "T+1", "T+2", "T+3", "T+4"]

    results: Dict[str, List[float]] = {c: [] for c in colnames}

    for _, row in trades.iterrows():
        entry_date = row["Entry date"]
        entry_px = row["Entry price"]
        side = row["Side"]

        if pd.isna(entry_date) or entry_date not in price_series.index:
            for c in colnames:
                results[c].append(np.nan)
            continue

        idx = price_series.index.get_loc(entry_date)

        for off, cname in zip(offsets, colnames):
            idx2 = idx + off
            if idx2 < 0 or idx2 >= len(price_series):
                results[cname].append(np.nan)
                continue

            close_px = price_series.iloc[idx2]
            pts = (close_px - entry_px) if side == "long" else (entry_px - close_px)
            results[cname].append(round(float(pts), 2))

    for cname in colnames:
        trades[cname] = results[cname]

    return trades


# ---------------------------------------------------------------------
# OHLC chart with all signals
# ---------------------------------------------------------------------


def plot_price_with_signals(
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    """
    Candlestick (OHLC) chart with all entry/exit markers.

    - Candle = daily OHLC
    - Up candle: Close >= Open
    - Down candle: Close < Open
    - Long entry: triangle up
    - Short entry: triangle down
    - Exit: X marker
    """
    ensure_dir(out_png.parent)

    df = price_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    dates = mdates.date2num(df[DATE_COL].to_list())
    opens = df[OPEN_COL].to_numpy()
    highs = df[HIGH_COL].to_numpy()
    lows = df[LOW_COL].to_numpy()
    closes = df[CLOSE_COL].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))

    # Draw candles
    width = 0.6  # in days
    for x, o, h, l, c in zip(dates, opens, highs, lows, closes):
        color = "green" if c >= o else "red"

        # High/low line
        ax.vlines(x, l, h, color=color, linewidth=0.7)

        # Body rectangle
        body_low = min(o, c)
        body_height = abs(c - o)
        if body_height == 0:
            body_height = 0.01  # tiny line if doji

        rect = Rectangle(
            (x - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.7,
        )
        ax.add_patch(rect)

    # Overlay signals
    for _, tr in trades_df.iterrows():
        ed = tr["Entry date"]
        ep = tr["Entry price"]
        side = tr.get("Side", "")

        if not pd.isna(ed):
            x = mdates.date2num(pd.to_datetime(ed))
            if side == "long":
                ax.scatter(x, ep, marker="^", s=35, color="blue")
            else:
                ax.scatter(x, ep, marker="v", s=35, color="black")

        xd = tr.get("Exit date")
        xp = tr.get("Exit price")
        if not pd.isna(xd):
            x2 = mdates.date2num(pd.to_datetime(xd))
            ax.scatter(x2, xp, marker="x", s=30, color="orange")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------


def trades_to_html_table(trades: pd.DataFrame) -> str:
    display_cols = [
        "#",
        "Signal date",
        "Entry date",
        "Entry price",
        "Exit date",
        "Exit price",
        "Side",
        "R",
        "Square type",
        "Exit reason",
        "T(-1)",
        "T",
        "T+1",
        "T+2",
        "T+3",
        "T+4",
        "Early close",
        "Margin neutral (pts)",
        "Margin neutral (%)",
        "Margin flip (pts)",
        "Margin flip (%)",
    ]

    df = trades.copy()
    df.insert(0, "#", range(1, len(df) + 1))
    existing_cols = [c for c in display_cols if c in df.columns]
    df = df[existing_cols]

    html = df.to_html(
        classes="table table-striped",
        index=False,
        border=0,
        justify="center",
        float_format=lambda x: f"{x:.2f}",
    )
    return html


def stock_summary_row(
    symbol: str, trades: pd.DataFrame, price_df: pd.DataFrame, rel_link: str
) -> Dict[str, str]:
    if trades.empty:
        last_signal = "—"
        last_side = "—"
        last_date = "—"
    else:
        last = trades.iloc[-1]
        last_signal = "Yes"
        last_side = last["Side"]
        last_date = last["Signal date"]

    n_trades = len(trades)
    win_rate = (trades["R"] > 0).mean() * 100 if n_trades > 0 else 0.0
    avg_r = trades["R"].mean() if n_trades > 0 else 0.0

    return {
        "Symbol": symbol,
        "Trades": str(n_trades),
        "WinRate": f"{win_rate:.1f}%",
        "AvgR": f"{avg_r:.2f}",
        "LastSignalDate": str(last_date),
        "LastSide": str(last_side),
        "HasSignal": last_signal,
        "Link": rel_link,
    }


def render_stock_page(
    symbol: str,
    trades: pd.DataFrame,
    price_df: pd.DataFrame,
    out_html: Path,
    chart_rel_path: str,
) -> None:
    html_trades = trades_to_html_table(trades)

    title = f"{symbol} – Gann Price–Time Squaring System"

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
    rel="stylesheet"
  >
</head>
<body class="bg-light">
  <div class="container my-4">
    <h1 class="mb-3">{title}</h1>
    <p>Fully mechanical backtest of the same Gann Price–Time / Price–Date Squaring system used for Nifty, applied to this stock.</p>

    <h3 class="mt-4 mb-3">Price chart with all signals (OHLC)</h3>
    <img src="{chart_rel_path}" class="img-fluid border" alt="Price with signals">

    <h3 class="mt-5 mb-3">Completed trades (with T(-1)…T+4 & margins)</h3>
    <div class="table-responsive">
      {html_trades}
    </div>

    <p class="mt-4">
      <a href="../index.html">&larr; Back to all stocks</a>
      &nbsp;|&nbsp;
      <a href="../../index.html">Back to Nifty system</a>
    </p>
  </div>
</body>
</html>
"""
    ensure_dir(out_html.parent)
    out_html.write_text(body, encoding="utf-8")


def render_master_stocks_index(rows: List[Dict[str, str]], out_html: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["Symbol"])

    tr_html = ""
    for r in rows_sorted:
        tr_html += f"""
        <tr>
          <td><a href="{r['Link']}">{r['Symbol']}</a></td>
          <td>{r['Trades']}</td>
          <td>{r['WinRate']}</td>
          <td>{r['AvgR']}</td>
          <td>{r['HasSignal']}</td>
          <td>{r['LastSide']}</td>
          <td>{r['LastSignalDate']}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gann Price–Time Squaring – All NSE Stocks</title>
  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
    rel="stylesheet"
  >
</head>
<body class="bg-light">
  <div class="container my-4">
    <h1 class="mb-3">Gann Price–Time Squaring – All NSE Stocks</h1>
    <p>Each row below is one NSE stock (including NIFTY). The system is identical to the Nifty backtest.</p>

    <div class="table-responsive mt-4">
      <table class="table table-striped table-hover align-middle">
        <thead>
          <tr>
            <th>Symbol</th>
            <th># trades</th>
            <th>Win rate</th>
            <th>Average R</th>
            <th>Signal now?</th>
            <th>Side</th>
            <th>Last signal date</th>
          </tr>
        </thead>
        <tbody>
          {tr_html}
        </tbody>
      </table>
    </div>

    <p class="mt-4">
      <a href="../index.html">Back to Nifty main page</a>
    </p>
  </div>
</body>
</html>
"""
    ensure_dir(out_html.parent)
    out_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def process_one_symbol(symbol: str, df: pd.DataFrame, out_root: Path) -> Dict[str, str]:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.date

    trades_df, price_df = backtest(df)
    trades_df = compute_holding_point_profits(trades_df, price_df)

    stock_dir = out_root / symbol
    chart_png = stock_dir / "price_signals.png"
    plot_price_with_signals(
        price_df, trades_df, chart_png, title=f"{symbol} – Price with Gann signals"
    )

    rel_chart = "price_signals.png"
    out_html = stock_dir / "index.html"
    render_stock_page(symbol, trades_df, price_df, out_html, rel_chart)

    rel_link = f"{symbol}/index.html"
    return stock_summary_row(symbol, trades_df, price_df, rel_link)


def main():
    docs_root = Path("docs")
    stocks_root = docs_root / "stocks"
    ensure_dir(stocks_root)

    all_rows: List[Dict[str, str]] = []

    # 1) NIFTY
    nifty_symbol, nifty_df = load_nifty_df()
    all_rows.append(process_one_symbol(nifty_symbol, nifty_df, stocks_root))

    # 2) All NSE stocks from EOD_Upstox
    eod_folder = Path("EOD_Upstox")
    for csv_path in sorted(eod_folder.glob("*_EOD.csv")):
        symbol, df = load_stock_df(csv_path)
        all_rows.append(process_one_symbol(symbol, df, stocks_root))

    render_master_stocks_index(all_rows, stocks_root / "index.html")


if __name__ == "__main__":
    main()
