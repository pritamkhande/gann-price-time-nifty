# Gann Price–Time Squaring – Nifty Backtest

This repository contains a fully mechanical implementation of a **Price–Time Squaring** system inspired by W.D. Gann, applied to Nifty daily data (from 2000 onwards).

## How it works

- Uses Nifty daily OHLC data from `data/nifty_daily.csv`.
- Detects swing highs and lows.
- From each swing, looks for a **price–time squared** zone where:
  - Price change (points) ≈ time elapsed (bars), and
  - Time count is near a square number (25, 36, 49, 64, 81, 100, 121...).
- Enters long/short trades with ATR-based stops and 2R targets.
- Builds an equity curve and generates an HTML report in `docs/index.html`.

## Usage

1. Put your Nifty data (from 2000) into:

   `data/nifty_daily.csv`

   with at least columns:

   `Date,Open,High,Low,Close,Volume`.

2. Create a virtual environment (optional) and install dependencies:

   ```bash
   pip install -r requirements.txt
