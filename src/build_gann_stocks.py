import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from utils_swing import detect_swings
from build_gann_report import (
    DATE_COL,
    OPEN_COL,
    HIGH_COL,
    LOW_COL,
    CLOSE_COL,
    VOL_COL,
    ATR_PERIOD,
    RISK_PER_TRADE,
    MAX_LOOKAHEAD,
    compute_atr,
    backtest,
    compute_metrics,
    build_system_commentary,
)


# -----------------------------
# Load one Upstox EOD file
# -----------------------------

def load_upstox_eod(csv_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Expected columns (Upstox EOD):

        Symbol,Date,Open,High,Low,Close,Volume

    Example:
        20MICRONS,2008-10-06 00:00:00+05:30,40.0,40.0,15.8,16.82,23501730
    """
    df = pd.read_csv(csv_path)

    if "Symbol" in df.columns and not df["Symbol"].dropna().empty:
        symbol = str(df["Symbol"].dropna().iloc[0]).strip()
    else:
        symbol = os.path.basename(csv_path).replace("_EOD.csv", "")

    # Normalise column names
    rename = {
        "Date": DATE_COL,
        "Open": OPEN_COL,
        "High": HIGH_COL,
        "Low": LOW_COL,
        "Close": CLOSE_COL,
        "Volume": VOL_COL,
    }
    df = df.rename(columns=rename)

    # Parse Date (Upstox gives timezone – strip it)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    try:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
    except (AttributeError, TypeError):
        try:
            df[DATE_COL] = df[DATE_COL].dt.tz_convert(None)
        except (AttributeError, TypeError):
            pass

    # Numeric OHLCV
    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if VOL_COL in df.columns:
        df[VOL_COL] = pd.to_numeric(df[VOL_COL], errors="coerce")

    df = df.dropna(subset=[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return symbol, df


# -----------------------------
# Unified HTML for one stock
# -----------------------------

def render_stock_html(
    symbol: str,
    metrics: Dict,
    trades_df: pd.DataFrame,
    commentary: str,
) -> str:
    start = metrics.get("start_date")
    end = metrics.get("end_date")
    years = metrics.get("years", 0.0) or 0.0

    if start is not None:
        start_str = pd.to_datetime(start).strftime("%d-%m-%Y")
    else:
        start_str = "N/A"
    if end is not None:
        end_str = pd.to_datetime(end).strftime("%d-%m-%Y")
    else:
        end_str = "N/A"

    n_trades = metrics.get("n_trades", 0)
    win_rate = metrics.get("win_rate", 0.0) or 0.0
    avg_R = metrics.get("avg_R", 0.0) or 0.0
    cagr = metrics.get("cagr", 0.0) or 0.0      # already in %
    max_dd = metrics.get("max_dd", 0.0) or 0.0  # already in %

    # Build trades table rows
    rows_html: List[str] = []
    if trades_df.empty:
        rows_html.append(
            "<tr><td colspan='21' style='text-align:center;'>No trades for this instrument.</td></tr>"
        )
    else:
        for _, row in trades_df.iterrows():
            trade_no = int(row.get("trade_no", 0))

            sig_date = row.get("signal_date")
            if pd.notna(sig_date):
                sig_date_str = pd.to_datetime(sig_date).strftime("%Y-%m-%d")
            else:
                sig_date_str = "NA"

            entry_date = row.get("entry_date")
            entry_date_str = (
                pd.to_datetime(entry_date).strftime("%Y-%m-%d")
                if pd.notna(entry_date)
                else "NA"
            )

            exit_date = row.get("exit_date")
            exit_date_str = (
                pd.to_datetime(exit_date).strftime("%Y-%m-%d")
                if pd.notna(exit_date)
                else "NA"
            )

            def fmt(v, fmt_str="{:.2f}"):
                if v is None:
                    return ""
                if isinstance(v, float) and np.isnan(v):
                    return ""
                try:
                    return fmt_str.format(v)
                except Exception:
                    return str(v)

            ec = row.get("early_close", np.nan)
            mn_pts = row.get("margin_neutral_pts", np.nan)
            mn_pct = row.get("margin_neutral_pct", np.nan)
            mf_pts = row.get("margin_flip_pts", np.nan)
            mf_pct = row.get("margin_flip_pct", np.nan)

            rows_html.append(
                f"<tr>"
                f"<td>{trade_no}</td>"
                f"<td>{sig_date_str}</td>"
                f"<td>{entry_date_str}</td>"
                f"<td>{fmt(row.get('entry_price'))}</td>"
                f"<td>{exit_date_str}</td>"
                f"<td>{fmt(row.get('exit_price'))}</td>"
                f"<td>{row.get('position','')}</td>"
                f"<td>{fmt(row.get('R'))}</td>"
                f"<td>{row.get('square_type','')}</td>"
                f"<td>{row.get('exit_reason','')}</td>"
                f"<td>{fmt(row.get('pts_Tm1'))}</td>"
                f"<td>{fmt(row.get('pts_T'))}</td>"
                f"<td>{fmt(row.get('pts_T1'))}</td>"
                f"<td>{fmt(row.get('pts_T2'))}</td>"
                f"<td>{fmt(row.get('pts_T3'))}</td>"
                f"<td>{fmt(row.get('pts_T4'))}</td>"
                f"<td>{'' if np.isnan(ec) else fmt(ec)}</td>"
                f"<td>{'' if np.isnan(mn_pts) else fmt(mn_pts)}</td>"
                f"<td>{'' if np.isnan(mn_pct) else fmt(mn_pct) + '%'}</td>"
                f"<td>{'' if np.isnan(mf_pts) else fmt(mf_pts)}</td>"
                f"<td>{'' if np.isnan(mf_pct) else fmt(mf_pct) + '%'}</td>"
                f"</tr>"
            )

    trades_rows_str = "\n".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{symbol} – Gann Price–Time Squaring Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="Mechanical Gann Price–Time and Price–Date Squaring backtest on {symbol} daily data.">
  <style>
    :root {{
      --bg: #f3f4f6;
      --card-bg: #ffffff;
      --border: #e5e7eb;
      --text-main: #111827;
      --text-muted: #6b7280;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text-main);
    }}
    .container {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 20px 12px 40px;
    }}
    a.back-link {{
      display: inline-block;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--accent);
      text-decoration: none;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 4px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-bottom: 18px;
    }}
    .grid-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}
    .metric-card {{
      background: var(--card-bg);
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 2px rgba(15,23,42,0.06);
    }}
    .metric-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 2px;
    }}
    .metric-value {{
      font-size: 18px;
      font-weight: 600;
    }}
    .section {{
      background: var(--card-bg);
      border-radius: 12px;
      padding: 14px 16px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(15,23,42,0.08);
      margin-bottom: 16px;
    }}
    .section h2 {{
      font-size: 18px;
      margin: 0 0 6px;
    }}
    .section p {{
      font-size: 13px;
      margin: 4px 0;
    }}
    img.chart {{
      max-width: 100%;
      height: auto;
      display: block;
      margin-top: 6px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      margin-top: 4px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 4px 5px;
      white-space: nowrap;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: center;
    }}
    th:nth-child(2), td:nth-child(2),
    th:nth-child(3), td:nth-child(3),
    th:nth-child(5), td:nth-child(5) {{
      text-align: center;
    }}
    th {{
      background: #f9fafb;
      color: var(--text-muted);
      font-weight: 500;
    }}
    tr:nth-child(even) td {{
      background: #f9fafb;
    }}
    .footer {{
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 20px;
    }}
    @media (max-width: 768px) {{
      table {{ font-size: 10px; }}
      .grid-metrics {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <a href="../index.html" class="back-link">← All stocks</a>
    <h1>{symbol} – Gann Price–Time Squaring System</h1>
    <div class="subtitle">
      Fully mechanical backtest of a W.D. Gann-inspired Price–Time / Price–Date Squaring idea
      on daily data for <strong>{symbol}</strong>, from {start_str} to {end_str}.
    </div>

    <div class="grid-metrics">
      <div class="metric-card">
        <div class="metric-label">Number of trades</div>
        <div class="metric-value">{n_trades}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Win rate</div>
        <div class="metric-value">{win_rate:.1f}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Average R per trade</div>
        <div class="metric-value">{avg_R:.2f} R</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">CAGR (normalized equity)</div>
        <div class="metric-value">{cagr:.1f}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Max drawdown</div>
        <div class="metric-value">{max_dd:.1f}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Test length</div>
        <div class="metric-value">{years:.1f} yrs</div>
      </div>
    </div>

    <div class="section">
      <h2>System behaviour commentary</h2>
      <p>{commentary}</p>
    </div>

    <div class="section">
      <h2>Equity curve and drawdown</h2>
      <p>Equity starts at 1.0 and changes based on realized R-multiples with {RISK_PER_TRADE*100:.0f}% risk per trade.</p>
      <img src="gann_equity_curve.png" class="chart" alt="Equity curve">
      <p style="margin-top:8px;">Drawdown relative to running equity peak:</p>
      <img src="gann_drawdown_curve.png" class="chart" alt="Drawdown curve">
    </div>

    <div class="section">
      <h2>Completed trades</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Signal</th>
            <th>Entry</th>
            <th>Entry Px</th>
            <th>Exit</th>
            <th>Exit Px</th>
            <th>Side</th>
            <th>R</th>
            <th>Square</th>
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
          </tr>
        </thead>
        <tbody>
          {trades_rows_str}
        </tbody>
      </table>
    </div>

    <div class="footer">
      This is a research backtest. It ignores brokerage, slippage and execution constraints.
      It is not trading or investment advice.
    </div>
  </div>
</body>
</html>
"""
    return html


# -----------------------------
# Landing page (all stocks)
# -----------------------------

def render_index_html(rows: List[Dict]) -> str:
    body_rows: List[str] = []

    if not rows:
        body_rows.append(
            "<tr><td colspan='7' style='text-align:center;'>No stocks processed.</td></tr>"
        )
    else:
        for r in rows:
            sym = r["symbol"]
            last_side = r.get("last_signal_side", "—")
            last_date = r.get("last_signal_date", "—")
            body_rows.append(
                f"<tr>"
                f"<td><a href='stocks/{sym}/index.html'>{sym}</a></td>"
                f"<td>{last_side}</td>"
                f"<td>{last_date}</td>"
                f"<td>{int(r.get('num_trades', 0))}</td>"
                f"<td>{float(r.get('win_rate', 0.0)):.1f}%</td>"
                f"<td>{float(r.get('cagr', 0.0)):.1f}%</td>"
                f"<td>{float(r.get('max_dd', 0.0)):.1f}%</td>"
                f"</tr>"
            )

    rows_html = "\n".join(body_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gann Price–Time Squaring – All NSE Stocks</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --bg: #f3f4f6;
      --card-bg: #ffffff;
      --border: #e5e7eb;
      --text-main: #111827;
      --text-muted: #6b7280;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text-main);
    }}
    .container {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 20px 12px 40px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 6px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: var(--card-bg);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(15,23,42,0.08);
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      background: #f9fafb;
      color: var(--text-muted);
      font-weight: 500;
    }}
    tr:nth-child(even) td {{
      background: #f9fafb;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 768px) {{
      table {{ font-size: 11px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Gann Price–Time Squaring – All NSE Stocks</h1>
    <div class="subtitle">
      Each row shows the latest completed signal and key backtest stats.
      Click a symbol to open its full report.
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
          <th>Max DD</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    return html


# -----------------------------
# Main driver
# -----------------------------

def main() -> None:
    # Resolve paths relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eod_dir = os.path.join(repo_root, "EOD_Upstox")
    docs_dir = os.path.join(repo_root, "docs")
    stocks_root = os.path.join(docs_dir, "stocks")

    os.makedirs(stocks_root, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(eod_dir, "*_EOD.csv")))
    index_rows: List[Dict] = []

    if not csv_files:
        print("No *_EOD.csv files found in EOD_Upstox.")
        return

    for csv_path in csv_files:
        symbol, df = load_upstox_eod(csv_path)
        if df.empty:
            print(f"{symbol}: no valid rows, skipping.")
            continue

        # Same pipeline as Nifty: ATR + swings + backtest
        df = compute_atr(df)
        df = detect_swings(df, low_col=LOW_COL, high_col=HIGH_COL,
                           lookback_main=1, lookback_fractal=2)

        trades_df, price_df = backtest(df)
        metrics = compute_metrics(trades_df, price_df)
        commentary = build_system_commentary(metrics, trades_df)

        # Build equity & drawdown PNGs for this stock
        stock_dir = os.path.join(stocks_root, symbol)
        os.makedirs(stock_dir, exist_ok=True)
        equity_png = os.path.join(stock_dir, "gann_equity_curve.png")
        dd_png = os.path.join(stock_dir, "gann_drawdown_curve.png")

        # local import to avoid circular import at module import time
        from utils_plot import make_equity_and_dd_plots
        make_equity_and_dd_plots(price_df, DATE_COL, "equity", equity_png, dd_png)

        # Save trades CSV for this stock
        trades_csv_path = os.path.join(stock_dir, f"{symbol}_gann_trades.csv")
        trades_df.to_csv(trades_csv_path, index=False)

        # HTML report for this stock
        html = render_stock_html(symbol, metrics, trades_df, commentary)
        with open(os.path.join(stock_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html)

        # Data for landing page
        if trades_df.empty:
            last_side = "—"
            last_date_str = "—"
        else:
            last = trades_df.iloc[-1]
            last_side = last.get("position", "—")
            last_dt = last.get("signal_date")
            if pd.notna(last_dt):
                last_date_str = pd.to_datetime(last_dt).strftime("%d-%m-%Y")
            else:
                last_date_str = "—"

        index_rows.append(
            {
                "symbol": symbol,
                "last_signal_side": last_side,
                "last_signal_date": last_date_str,
                "num_trades": metrics.get("n_trades", 0),
                "win_rate": metrics.get("win_rate", 0.0),
                "cagr": metrics.get("cagr", 0.0),
                "max_dd": metrics.get("max_dd", 0.0),
            }
        )

        print(
            f"{symbol}: trades={metrics.get('n_trades', 0)}, "
            f"win_rate={metrics.get('win_rate', 0.0):.1f}%"
        )

    # Build global landing page
    os.makedirs(docs_dir, exist_ok=True)
    index_html = render_index_html(index_rows)
    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    print("Landing page written to docs/index.html")


if __name__ == "__main__":
    main()
