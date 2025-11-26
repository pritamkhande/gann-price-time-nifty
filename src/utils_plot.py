import os
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def make_equity_and_dd_plots(df: pd.DataFrame,
                             date_col: str,
                             equity_col: str,
                             out_equity_png: str,
                             out_dd_png: str) -> None:
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


def create_trade_chart(price_df: pd.DataFrame,
                       trade: pd.Series,
                       date_col: str,
                       open_col: str,
                       high_col: str,
                       low_col: str,
                       close_col: str,
                       out_html_path: str,
                       bars_before: int = 40,
                       bars_after: int = 20) -> None:
    """
    Creates a Plotly candlestick chart around a single trade and writes it to HTML.
    Looks like a mini-TradingView chart with entry/exit markers and stop lines.
    """

    entry_idx = int(trade["entry_index"])
    exit_idx = int(trade["exit_index"])

    start_idx = max(0, entry_idx - bars_before)
    end_idx = min(len(price_df) - 1, exit_idx + bars_after)

    sub = price_df.iloc[start_idx:end_idx + 1].copy()

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

    # layout
    fig.update_layout(
        title=f"Trade {int(trade['trade_no'])} – {trade['position']} ({trade['square_type']})",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        height=500,
    )

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    fig.write_html(out_html_path, include_plotlyjs="cdn")


def generate_trade_charts(price_df: pd.DataFrame,
                          trades_df: pd.DataFrame,
                          date_col: str,
                          open_col: str,
                          high_col: str,
                          low_col: str,
                          close_col: str,
                          out_dir: str = "docs/trades") -> None:
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
