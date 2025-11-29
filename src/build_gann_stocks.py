import os
import glob
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_swing import detect_swings

# Reuse core logic and constants from your existing Nifty script
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


# ==========================
# DATA LOADING FOR UPSTOX EOD
# ==========================

def load_upstox_eod(path: str) -> pd.DataFrame:
    """
    Load a single Upstox EOD csv:

        Symbol,Date,Open,High,Low,Close,Volume
        20MICRONS,2008-10-06 00:00:00+05:30,40.0,40.0,15.8,16.82,23501730
        ...

    Convert Date to naive datetime, keep required OHLCV columns and sort.
    """
    df = pd.read_csv(path)

    # Normalise column names
    rename_map = {
        "Date": DATE_COL,
        "Open": OPEN_COL,
        "High": HIGH_COL,
        "Low": LOW_COL,
        "Close": CLOSE_COL,
        "Volume": VOL_COL,
    }
    df = df.rename(columns=rename_map)

    # Parse dates (format is YYYY-MM-DD ... so default dayfirst=False is fine)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Drop timezone if present
    if hasattr(df[DATE_COL].dt, "tz_localize"):
        try:
            df[DATE_COL] = df[DATE_COL].dt.tz_localize(None)
        except TypeError:
            # Already tz-naive
            try:
                df[DATE_COL] = df[DATE_COL].dt.tz_convert(None)
            except TypeError:
                pass

    # Ensure numeric types
    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if VOL_COL in df.columns:
        df[VOL_COL] = pd.to_numeric(df[VOL_COL], errors="coerce")

    # Clean and sort
    df = df.dropna(subset=[DATE_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


# ==========================
# PLOTTING – EQUITY & DRAWDOWN
# ==========================

def make_equity_and_dd_plots(
    price_df: pd.DataFrame,
    date_col: str,
    equity_col: str,
    out_equity_png: str,
    out_dd_png: str,
) -> None:
    """
    Simple Matplotlib version of the equity & drawdown plots,
    similar to the Nifty report.
    """
    dates = price_df[date_col]
    equity = price_df[equity_col]

    # --- Equity curve ---
    plt.figure(figsize=(10, 4))
    plt.plot(dates, equity)
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.title("Gann Squaring – Equity Curve")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_equity_png), exist_ok=True)
    plt.savefig(out_equity_png)
    plt.close()

    # --- Drawdown curve ---
    eq = equity.dropna().values
    if len(eq) > 0:
        peaks = np.maximum.accumulate(eq)
        dd = (eq - peaks) / peaks
        dd_dates = dates[equity.notna()]

        plt.figure(figsize=(10, 4))
        plt.plot(dd_dates, dd * 100.0)
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%) from equity peak")
        plt.title("Drawdown (% of equity peak)")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_dd_png), exist_ok=True)
        plt.savefig(out_dd_png)
        plt.close()


# ==========================
# HTML RENDER – SAME LAYOUT AS NIFTY, BUT GENERIC
# ==========================

def render_html_stock(
    instrument_name: str,
    metrics: dict,
    trades_df: pd.DataFrame,
    commentary: str,
) -> str:
    """
    Copy of your existing render_html, but with instrument_name parameter instead
    of hard-coded 'Nifty'. The 'Chart' column is kept, but for stocks we will
    *not* generate per-trade chart HTMLs, so the links are removed here.
    """
    start = metrics["start_date"]
    end = metrics["end_date"]
    if start is None or end is None:
        start_str = "NA"
        end_str = "NA"
        years_str = "0.0"
    else:
        start_str = start.strftime("%d-%m-%Y")
        end_str = end.strftime("%d-%m-%Y")
        years_str = f"{metrics['years']:.1f}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{instrument_name} – Gann Squaring System</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                   sans-serif;
      max-width: 1000px;
      margin: 0 auto;
      padding: 16px;
      background: #f3f4f6;
      color: #111827;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 4px;
    }}
    h2 {{
      font-size: 20px;
      margin-bottom: 8px;
    }}
    h3 {{
      font-size: 16px;
      margin-bottom: 4px;
    }}
    p {{
      line-height: 1.5;
      font-size: 14px;
    }}
    .card {{
      background: #ffffff;
      border-radius: 12px;
      padding: 16px 18px;
      margin-top: 16px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
      border: 1px solid #e5e7eb;
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
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 4px 6px;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
    }}
    td:first-child, th:first-child {{
      text-align: center;
    }}
    td:nth-child(2), th:nth-child(2),
    td:nth-child(3), th:nth-child(3),
    td:nth-child(5), th:nth-child(5) {{
      text-align: center;
    }}
    .footer {{
      font-size: 12px;
      color: #6b7280;
      margin-top: 24px;
    }}
  </style>
</head>
<body>

  <h1>{instrument_name} – Gann Squaring System</h1>
  <p>
    Fully mechanical backtest of a Price-Time / Price-Date Squaring system inspired by W.D. Gann,
    applied to {instrument_name} daily data from {start_str} to {end_str}.
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
    <img src="gann_equity_curve.png" alt="Equity curve">
    <p>Drawdown relative to running equity peak:</p>
    <img src="gann_drawdown_curve.png" alt="Drawdown curve">
  </div>

  <div class="card">
    <h2>System Logic</h2>

    <h3>1. Swing points</h3>
    <ul>
      <li>Timeframe: daily OHLC data.</li>
      <li>Swing highs and lows detected using both tight ±1-bar pivots and Williams-style fractals.</li>
    </ul>

    <h3>2. Gann Price-Time / Price-Date Squares</h3>
    <ul>
      <li>From each swing, scan forward up to {MAX_LOOKAHEAD} bars.</li>
      <li>Let ΔP = |Close − swing Close| in points, ΔBars = bars, ΔDays = calendar days.</li>
      <li>We look for cases where ΔP ≈ ΔBars and/or ΔP ≈ ΔDays and the count is near classic/extended square numbers (25, 36, 49, 64, 81, 100, 121, 50, 72, 98, 128).</li>
      <li>These zones identify potential turning points where price and time/date are in balance.</li>
    </ul>

    <h3>3. Entries and exits</h3>
    <ul>
      <li>Short: from squared up-move after swing low, with bearish confirmation; entry next open.</li>
      <li>Long: from squared down-move after swing high, with bullish confirmation; entry next open.</li>
      <li>Initial SL: swing square bar high/low ± 2×ATR({ATR_PERIOD}).</li>
      <li>Exit: ATR trailing stop (3×ATR) moves in favour of the trade; no fixed profit target.</li>
      <li>Risk per trade: {RISK_PER_TRADE*100:.0f}% of equity. One position at a time.</li>
    </ul>
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
      </tr>
"""

    for _, row in trades_df.iterrows():
        trade_no = int(row["trade_no"])
        sig_date = (
            row["signal_date"].strftime("%Y-%m-%d")
            if pd.notna(row["signal_date"])
            else "NA"
        )

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
# DRIVER – RUN FOR ALL STOCKS
# ==========================

def run_for_stock(symbol: str, csv_path: str) -> Tuple[str, dict]:
    """
    Run full pipeline for a single stock and write its HTML report and CSV.

    Returns (symbol, metrics) so you can optionally build a front-page index later.
    """
    print(f"Processing {symbol} from {csv_path} ...")

    df = load_upstox_eod(csv_path)
    if df.empty:
        print(f"  WARNING: no valid rows for {symbol}, skipping.")
        return symbol, {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_R": 0.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "years": 0.0,
        }

    df = compute_atr(df)
    df = detect_swings(
        df,
        low_col=LOW_COL,
        high_col=HIGH_COL,
        lookback_main=1,
        lookback_fractal=2,
    )

    trades_df, price_df = backtest(df)

    metrics = compute_metrics(trades_df, price_df)
    commentary = build_system_commentary(metrics, trades_df)

    out_dir = os.path.join("docs", "stocks", symbol)
    os.makedirs(out_dir, exist_ok=True)

    # Save trades CSV for this stock
    trades_csv_path = os.path.join(out_dir, f"{symbol}_gann_trades.csv")
    trades_df.to_csv(trades_csv_path, index=False)

    # Equity + drawdown charts (same filenames as Nifty page, but inside stock folder)
    equity_png = os.path.join(out_dir, "gann_equity_curve.png")
    dd_png = os.path.join(out_dir, "gann_drawdown_curve.png")
    make_equity_and_dd_plots(price_df, DATE_COL, "equity", equity_png, dd_png)

    # HTML report
    instr_name = symbol  # you can later map to "20MICRONS (NSE)" etc.
    html = render_html_stock(instr_name, metrics, trades_df, commentary)
    html_path = os.path.join(out_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Wrote {html_path}")
    return symbol, metrics


def main():
    # Process all Upstox EOD files
    csv_files = sorted(glob.glob(os.path.join("EOD_Upstox", "*_EOD.csv")))
    if not csv_files:
        print("No EOD_Upstox/*_EOD.csv files found.")
        return

    summary = []

    for path in csv_files:
        fname = os.path.basename(path)
        # e.g. 20MICRONS_EOD.csv -> 20MICRONS
        symbol = fname.replace("_EOD.csv", "")
        sym, metrics = run_for_stock(symbol, path)
        summary.append((sym, metrics))

    # (Optional) you can also build a docs/index.html with a list of all stocks
    # using `summary`, but your requirement here is per-stock reports identical
    # to the Nifty layout, which is now handled.
    print("Done. Per-stock Gann reports are under docs/stocks/<SYMBOL>/index.html")


if __name__ == "__main__":
    main()
