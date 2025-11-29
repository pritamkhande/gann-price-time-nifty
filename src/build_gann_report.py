"""
build_gann_stocks.py

Multi-stock wrapper around the existing Gann backtest logic.

- Reads all *_EOD.csv files from EOD_Upstox (format: Symbol,Date,Open,High,Low,Close,Volume)
- Runs the same backtest as build_gann_report.backtest() for each stock
- Generates one HTML report per stock in docs/stocks/<SYMBOL>/index.html
- Creates simple price+signals chart and equity/drawdown charts (PNG) for each stock
- Builds docs/index.html listing all stocks and their latest signal.
"""

import os
import glob
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# Re-use the core logic from your existing single-instrument script
import build_gann_report as core

DATE_COL = core.DATE_COL
OPEN_COL = core.OPEN_COL
HIGH_COL = core.HIGH_COL
LOW_COL = core.LOW_COL
CLOSE_COL = core.CLOSE_COL

RISK_PER_TRADE = 0.02  # 2% per trade, same convention as in the main script


def load_upstox_eod(csv_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Load one EOD_Upstox CSV.

    Expected columns: Symbol, Date, Open, High, Low, Close, Volume
    Date format example: 2021-10-22 00:00:00+05:30
    """
    df = pd.read_csv(csv_path)
    if "Symbol" not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"{csv_path} is missing Symbol/Date columns")

    # Symbol is constant for the file; take first non-null
    symbol = str(df["Symbol"].dropna().iloc[0]).strip()

    # Normalize date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    # Ensure numeric OHLC
    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return symbol, df


def build_equity_and_drawdown(price_df: pd.DataFrame,
                              trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily equity curve and drawdown series given completed trades.

    Equity starts at 1.0 and is updated on each trade exit using R-multiples
    with fixed fractional risk (RISK_PER_TRADE).
    """
    if price_df.empty:
        return price_df.copy()

    price_df = price_df.copy()
    equity = 1.0
    equity_series = []

    # Map exit date -> total R on that date (in case of multiple exits)
    r_by_date: Dict[pd.Timestamp, float] = {}
    if not trades_df.empty:
        # Assume trades_df has columns "Exit date" and "R"
        for _, row in trades_df.iterrows():
            exit_date = pd.to_datetime(row["Exit date"])
            r_val = float(row["R"])
            r_by_date[exit_date] = r_by_date.get(exit_date, 0.0) + r_val

    for d in price_df[DATE_COL]:
        # Apply any R realised on this date
        r_today = r_by_date.get(pd.to_datetime(d), 0.0)
        if r_today != 0.0:
            equity *= (1.0 + RISK_PER_TRADE * r_today)
        equity_series.append(equity)

    price_df["equity"] = equity_series
    peak = price_df["equity"].cummax()
    price_df["drawdown_pct"] = (price_df["equity"] / peak - 1.0) * 100.0

    return price_df


def plot_price_with_signals(
    symbol: str,
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Simple price chart with long/short entries and exits marked.
    """
    if price_df.empty:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(price_df[DATE_COL], price_df[CLOSE_COL], linewidth=1.0)

    if not trades_df.empty:
        # Long entries: green up triangles; Short entries: red down triangles
        long_entries = trades_df[trades_df["Side"] == "long"]
        short_entries = trades_df[trades_df["Side"] == "short"]

        plt.plot(long_entries["Entry date"], long_entries["Entry price"], "^", markersize=5)
        plt.plot(short_entries["Entry date"], short_entries["Entry price"], "v", markersize=5)

        # Exits as 'x'
        plt.plot(trades_df["Exit date"], trades_df["Exit price"], "x", markersize=4)

    plt.title(f"{symbol} – Close price with Gann signals")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_equity_and_drawdown(
    symbol: str,
    price_df: pd.DataFrame,
    out_equity_path: str,
    out_dd_path: str,
) -> None:
    """
    Plot equity curve and drawdown series from price_df which already
    contains 'equity' and 'drawdown_pct' columns.
    """
    if price_df.empty or "equity" not in price_df.columns:
        return

    # Equity
    plt.figure(figsize=(10, 3))
    plt.plot(price_df[DATE_COL], price_df["equity"], linewidth=1.0)
    plt.title(f"{symbol} – Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.tight_layout()
    plt.savefig(out_equity_path, dpi=120)
    plt.close()

    # Drawdown
    plt.figure(figsize=(10, 3))
    plt.plot(price_df[DATE_COL], price_df["drawdown_pct"], linewidth=1.0)
    plt.title(f"{symbol} – Drawdown from peak")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.savefig(out_dd_path, dpi=120)
    plt.close()


def render_stock_html(
    symbol: str,
    metrics: Dict[str, float],
    commentary: str,
    trades_df: pd.DataFrame,
    price_png: str,
    equity_png: str,
    dd_png: str,
) -> str:
    """
    Render a per-stock HTML report.
    """
    # Relative paths (PNG files live in the same folder as HTML)
    price_img = os.path.basename(price_png)
    eq_img = os.path.basename(equity_png)
    dd_img = os.path.basename(dd_png)

    # Trades table rows
    rows_html: List[str] = []
    for i, row in trades_df.iterrows():
        rows_html.append(
            f"<tr>"
            f"<td>{i+1}</td>"
            f"<td>{row['Signal date'].date()}</td>"
            f"<td>{row['Entry date'].date()}</td>"
            f"<td>{row['Entry price']:.2f}</td>"
            f"<td>{row['Exit date'].date()}</td>"
            f"<td>{row['Exit price']:.2f}</td>"
            f"<td>{row['Side']}</td>"
            f"<td>{row['R']:.2f}</td>"
            f"</tr>"
        )

    trades_table_rows = "\n".join(rows_html) if rows_html else "<tr><td colspan='8'>No trades</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{symbol} – Gann Squaring System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #f4f5f7;
      color: #111827;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 4px;
    }}
    .subtitle {{
      color: #4b5563;
      font-size: 14px;
      margin-bottom: 24px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }}
    .card-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #6b7280;
      margin-bottom: 2px;
    }}
    .card-value {{
      font-size: 18px;
      font-weight: 600;
    }}
    .section {{
      background: white;
      border-radius: 12px;
      padding: 16px 18px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.08);
      margin-bottom: 20px;
    }}
    .section h2 {{
      font-size: 18px;
      margin-top: 0;
      margin-bottom: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #e5e7eb;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #f9fafb;
      color: #4b5563;
      font-weight: 500;
    }}
    tr:nth-child(even) td {{
      background: #f9fafb;
    }}
    img.chart {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 8px 0 0;
      border-radius: 6px;
      border: 1px solid #e5e7eb;
    }}
    .commentary {{
      font-size: 13px;
      line-height: 1.55;
      color: #374151;
      white-space: pre-line;
    }}
    a.back-link {{
      font-size: 13px;
      color: #2563eb;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="container">
    <a class="back-link" href="../index.html">← Back to all stocks</a>
    <h1>{symbol} – Gann Squaring System</h1>
    <div class="subtitle">
      Fully mechanical backtest of a Price-Time / Price-Date Squaring idea,
      applied to daily OHLC data for {symbol}.
    </div>

    <div class="cards">
      <div class="card">
        <div class="card-label">Number of trades</div>
        <div class="card-value">{metrics['num_trades']}</div>
      </div>
      <div class="card">
        <div class="card-label">Win rate</div>
        <div class="card-value">{metrics['win_rate']:.1f}%</div>
      </div>
      <div class="card">
        <div class="card-label">Average R per trade</div>
        <div class="card-value">{metrics['avg_R']:.2f} R</div>
      </div>
      <div class="card">
        <div class="card-label">CAGR (normalized equity)</div>
        <div class="card-value">{metrics['cagr']:.1f}%</div>
      </div>
      <div class="card">
        <div class="card-label">Maximum drawdown</div>
        <div class="card-value">{metrics['max_dd']:.1f}%</div>
      </div>
      <div class="card">
        <div class="card-label">Test length</div>
        <div class="card-value">{metrics['test_years']:.1f} yrs</div>
      </div>
    </div>

    <div class="section">
      <h2>Price chart with all signals</h2>
      <img class="chart" src="{price_img}" alt="Price with Gann signals" />
    </div>

    <div class="section">
      <h2>Equity curve and drawdown</h2>
      <img class="chart" src="{eq_img}" alt="Equity curve" />
      <img class="chart" src="{dd_img}" alt="Drawdown" />
    </div>

    <div class="section">
      <h2>System behaviour commentary</h2>
      <div class="commentary">{commentary}</div>
    </div>

    <div class="section">
      <h2>Completed trades</h2>
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
          </tr>
        </thead>
        <tbody>
          {trades_table_rows}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    return html


def build_index_html(rows: List[Dict[str, str]], out_path: str) -> None:
    """
    Build landing page listing all symbols with last signal and stats.
    """
    rows_html = []
    for row in rows:
        rows_html.append(
            f"<tr>"
            f"<td><a href=\"stocks/{row['symbol']}/index.html\">{row['symbol']}</a></td>"
            f"<td>{row['last_signal_side']}</td>"
            f"<td>{row['last_signal_date']}</td>"
            f"<td>{row['num_trades']}</td>"
            f"<td>{row['win_rate']:.1f}%</td>"
            f"<td>{row['cagr']:.1f}%</td>"
            f"<td>{row['max_dd']:.1f}%</td>"
            f"</tr>"
        )
    body_rows = "\n".join(rows_html) if rows_html else "<tr><td colspan='7'>No stocks processed</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gann Squaring System – All NSE Stocks</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #f4f5f7;
      color: #111827;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 8px;
    }}
    .subtitle {{
      color: #4b5563;
      font-size: 14px;
      margin-bottom: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e5e7eb;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #f9fafb;
      color: #4b5563;
      font-weight: 500;
    }}
    tr:nth-child(even) td {{
      background: #f9fafb;
    }}
    a {{
      color: #2563eb;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Gann Squaring System – All NSE Stocks</h1>
    <div class="subtitle">
      Each row shows the latest completed signal for that stock and summary
      stats of the full backtest. Click a symbol to see full details.
    </div>
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th>Last signal</th>
          <th>Last signal date</th>
          <th># trades</th>
          <th>Win rate</th>
          <th>CAGR</th>
          <th>Max drawdown</th>
        </tr>
      </thead>
      <tbody>
        {body_rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eod_dir = os.path.join(repo_root, "EOD_Upstox")
    docs_dir = os.path.join(repo_root, "docs")
    stocks_root = os.path.join(docs_dir, "stocks")

    os.makedirs(stocks_root, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(eod_dir, "*_EOD.csv")))
    index_rows: List[Dict[str, str]] = []

    for csv_path in csv_files:
        symbol, df = load_upstox_eod(csv_path)
        if df.empty:
            continue

        # Run core backtest logic from the existing module
        trades_df, price_df = core.backtest(df)

        # Basic metrics & commentary (same helpers as Nifty script)
        metrics = core.compute_metrics(trades_df, price_df)
        commentary = core.build_system_commentary(metrics, trades_df)

        # Build equity + drawdown and charts
        price_df_ed = build_equity_and_drawdown(price_df, trades_df)

        stock_dir = os.path.join(stocks_root, symbol)
        os.makedirs(stock_dir, exist_ok=True)

        price_png = os.path.join(stock_dir, "price_signals.png")
        equity_png = os.path.join(stock_dir, "equity.png")
        dd_png = os.path.join(stock_dir, "drawdown.png")

        plot_price_with_signals(symbol, price_df, trades_df, price_png)
        plot_equity_and_drawdown(symbol, price_df_ed, equity_png, dd_png)

        # Save trades CSV for this stock
        trades_csv_path = os.path.join(stock_dir, f"{symbol}_gann_trades.csv")
        trades_df.to_csv(trades_csv_path, index=False)

        # Per-stock HTML
        html = render_stock_html(
            symbol=symbol,
            metrics=metrics,
            commentary=commentary,
            trades_df=trades_df,
            price_png=price_png,
            equity_png=equity_png,
            dd_png=dd_png,
        )
        stock_html_path = os.path.join(stock_dir, "index.html")
        with open(stock_html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Row for index page (latest completed signal)
        if trades_df.empty:
            last_side = "—"
            last_date_str = "—"
        else:
            last = trades_df.iloc[-1]
            last_side = str(last["Side"])
            sd = last["Signal date"]
            last_date_str = sd.date() if hasattr(sd, "date") else sd

        index_rows.append(
            {
                "symbol": symbol,
                "last_signal_side": last_side,
                "last_signal_date": last_date_str,
                "num_trades": metrics["num_trades"],
                "win_rate": metrics["win_rate"],
                "cagr": metrics["cagr"],
                "max_dd": metrics["max_dd"],
            }
        )

    # Build landing page
    os.makedirs(docs_dir, exist_ok=True)
    index_html_path = os.path.join(docs_dir, "index.html")
    build_index_html(index_rows, index_html_path)


if __name__ == "__main__":
    main()
