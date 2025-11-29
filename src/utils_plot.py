import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def make_equity_and_dd_plots(
    df: pd.DataFrame,
    date_col: str,
    equity_col: str,
    out_equity_png: str,
    out_dd_png: str,
) -> None:
    os.makedirs(os.path.dirname(out_equity_png), exist_ok=True)

    eq_series = df[[date_col, equity_col]].dropna()
    if eq_series.empty:
        return

    dates = eq_series[date_col].values
    equity = eq_series[equity_col].values

    # Equity curve
    plt.figure(figsize=(8, 4))
    plt.plot(dates, equity)
    plt.title("Equity Curve (normalized)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_equity_png, dpi=120)
    plt.close()

    # Drawdown curve
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks

    plt.figure(figsize=(8, 4))
    plt.plot(dates, dd)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
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
    if trades_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)

    for _, tr in trades_df.iterrows():
        trade_no = int(tr["trade_no"])
        sig_idx = int(tr["signal_index"])
        entry_idx = int(tr["entry_index"])
        exit_idx = int(tr["exit_index"])

        start_idx = max(0, sig_idx - 30)
        end_idx = min(len(price_df) - 1, exit_idx + 10)

        slice_df = price_df.loc[start_idx:end_idx].copy()
        slice_df = slice_df.reset_index(drop=True)

        # Map global indices to local within slice
        def to_local_idx(global_idx: int) -> Optional[int]:
            if global_idx < start_idx or global_idx > end_idx:
                return None
            return global_idx - start_idx

        local_sig = to_local_idx(sig_idx)
        local_entry = to_local_idx(entry_idx)
        local_exit = to_local_idx(exit_idx)

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=slice_df[date_col],
                    open=slice_df[open_col],
                    high=slice_df[high_col],
                    low=slice_df[low_col],
                    close=slice_df[close_col],
                    name="Price",
                )
            ]
        )

        # Markers
        if local_sig is not None:
            fig.add_trace(
                go.Scatter(
                    x=[slice_df.loc[local_sig, date_col]],
                    y=[slice_df.loc[local_sig, close_col]],
                    mode="markers+text",
                    text=["Square"],
                    textposition="top center",
                    name="Square",
                )
            )

        if local_entry is not None:
            fig.add_trace(
                go.Scatter(
                    x=[slice_df.loc[local_entry, date_col]],
                    y=[slice_df.loc[local_entry, close_col]],
                    mode="markers+text",
                    text=["Entry"],
                    textposition="bottom center",
                    name="Entry",
                )
            )

        if local_exit is not None:
            fig.add_trace(
                go.Scatter(
                    x=[slice_df.loc[local_exit, date_col]],
                    y=[slice_df.loc[local_exit, close_col]],
                    mode="markers+text",
                    text=["Exit"],
                    textposition="bottom center",
                    name="Exit",
                )
            )

        fig.update_layout(
            title=f"Trade #{trade_no}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
        )

        out_path = os.path.join(out_dir, f"trade_{trade_no:03d}.html")
        pio.write_html(fig, file=out_path, auto_open=False, include_plotlyjs="cdn")
