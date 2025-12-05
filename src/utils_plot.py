import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def make_equity_and_dd_plots(
    df: pd.DataFrame,
    date_col: str,
    equity_col: str,
    out_equity_png: str,
    out_dd_png: str,
) -> None:
    eq = df.dropna(subset=[equity_col])
    if eq.empty:
        return

    dates = eq[date_col]
    equity = eq[equity_col].values

    # equity curve
    plt.figure(figsize=(9, 4))
    plt.plot(dates, equity)
    plt.title("Gann Squaring – Equity Curve (Nifty)")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.tight_layout()
    plt.savefig(out_equity_png)
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
    plt.savefig(out_dd_png)
    plt.close()


def _build_trade_commentary(trade: pd.Series) -> str:
    side = trade["position"]
    r = trade["R"]
    sq_type = trade.get("square_type", "")
    exit_reason = trade["exit_reason"]
    duration_bars = int(trade["exit_index"]) - int(trade["entry_index"])
    dur_txt = f"{duration_bars} bars"

    if r > 1.5:
        perf = "strong winner"
    elif r > 0:
        perf = "modest winner"
    elif r > -0.5:
        perf = "small loss"
    else:
        perf = "larger loss"

    sq_txt = {
        "time": "Price–Time square (bars)",
        "date": "Price–Date square (calendar days)",
        "both": "Price–Time and Price–Date square in alignment",
        None: "Gann square",
        "": "Gann square",
    }.get(sq_type, "Gann square")

    return (
        f"This {side} trade was triggered from a {sq_txt} setup. "
        f"It stayed open for {dur_txt} and closed as a {perf} with {r:.2f}R. "
        f"The exit happened due to '{exit_reason}' (stop/forced exit logic)."
    )


def create_trade_chart(
    price_df: pd.DataFrame,
    trade: pd.Series,
    date_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    out_html_path: str,
    bars_before: int = 40,
    bars_after: int = 20,
) -> None:
    """
    Creates a Plotly candlestick chart around a single trade and writes a full
    HTML page with chart + commentary.
    """

    entry_idx = int(trade["entry_index"])
    exit_idx = int(trade["exit_index"])

    start_idx = max(0, entry_idx - bars_before)
    end_idx = min(len(price_df) - 1, exit_idx + bars_after)

    sub = price_df.iloc[start_idx : end_idx + 1].copy()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=sub[date_col],
                open=sub[open_col],
                high=sub[high_col],
                low=sub[low_col],
                close=sub[close_col],
                name="Nifty",
            )
        ]
    )

    # entry & exit markers
    entry_row = price_df.iloc[entry_idx]
    exit_row = price_df.iloc[exit_idx]

    fig.add_trace(
        go.Scatter(
            x=[entry_row[date_col]],
            y=[trade["entry_price"]],
            mode="markers+text",
            name="Entry",
            marker=dict(symbol="triangle-up", size=12, color="green"),
            text=["Entry"],
            textposition="top center",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[exit_row[date_col]],
            y=[trade["exit_price"]],
            mode="markers+text",
            name="Exit",
            marker=dict(symbol="triangle-down", size=12, color="red"),
            text=["Exit"],
            textposition="bottom center",
        )
    )

    # initial stop line
    fig.add_hline(
        y=trade["initial_stop_price"],
        line=dict(dash="dot", width=1),
        annotation_text="Initial SL",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title=f"Trade {int(trade['trade_no'])} – {trade['position']} ({trade['square_type']})",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        height=500,
    )

    # Build full HTML page with commentary
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    commentary = _build_trade_commentary(trade)
    entry_date = trade["entry_date"].strftime("%d-%m-%Y")
    exit_date = trade["exit_date"].strftime("%d-%m-%Y")
    r = trade["R"]

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trade {int(trade['trade_no'])} – Gann Squaring (Nifty)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
    .card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
  </style>
</head>
<body>
  <h1>Trade {int(trade['trade_no'])} – Gann Squaring (Nifty)</h1>
  <p><strong>{trade['position'].capitalize()}</strong> trade from {entry_date} to {exit_date}, result: {r:.2f}R.</p>
  <div class="card">
    <p>{commentary}</p>
  </div>
  <div class="card">
    {fig_html}
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)


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
    if trades_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)

    for _, row in trades_df.iterrows():
        trade_no = int(row["trade_no"])
        out_html = os.path.join(out_dir, f"trade_{trade_no:03d}.html")
        create_trade_chart(
            price_df,
            row,
            date_col,
            open_col,
            high_col,
            low_col,
            close_col,
            out_html,
        )
