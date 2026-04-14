# STATS 507 Final Project

## Project Goal

This project studies whether data from **Polymarket** can help predict the **next-day direction of Bitcoin (BTC)** beyond using BTC market data alone.

The workflow has three parts:

1. Build a merged dataset from BTC daily OHLCV data and Polymarket daily BTC direction markets.
2. Train and compare two Transformer models:
   - a **BTC-only** model
   - a **BTC + Polymarket** model
3. Backtest two trading strategies using the BTC + Polymarket model:
   - a **BTC strategy**: if the model predicts up, buy BTC at open and sell at close
   - a **Polymarket strategy**: buy the predicted side (`Up` or `Down`) at price `0.3`

The final backtest period is:

- **2025-04-14 to 2026-04-13**

## Repository Structure

- `btc_polymarket_analysis.py`
  - Main script for data collection, feature engineering, Transformer training, tuning choices, and backtesting.
- `data.ipynb`
  - Early notebook used during initial data exploration.
- `report_ieee.tex`
  - IEEE-format LaTeX source for the final project report.
- `report_ieee.pdf`
  - Compiled final report.
- `IEEEtran.cls`
  - Local IEEE template class file used for LaTeX compilation.
- `outputs/`
  - Generated datasets, predictions, tuning results, and backtest outputs.

## Data Sources

### BTC Data

- Source: **Yahoo Finance**
- Symbol: `BTC-USD`
- Frequency: daily
- Fields used: `Open`, `High`, `Low`, `Close`, `Volume`

### Polymarket Data

- Source: **Polymarket public APIs**
- Contract family used:
  - `Bitcoin Up or Down on [date]?`
- Data used:
  - event discovery through public search
  - historical trade data before market resolution

## Model Overview

Both forecasting models use a **TensorFlow Transformer encoder**.

### BTC-only model

- Input: rolling sequences of engineered BTC features
- Final tuned configuration:
  - sequence length = `14`
  - threshold = `0.55`

### BTC + Polymarket model

- Input: rolling sequences of BTC features plus a smaller tuned subset of Polymarket features
- Final tuned configuration:
  - sequence length = `14`
  - threshold = `0.50`

## How to Run

### 1. Install dependencies

This project was run in Python 3.11. The main packages are:

- `pandas`
- `numpy`
- `requests`
- `yfinance`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `pypdf`

You can install them with:

```bash
pip install pandas numpy requests yfinance scikit-learn tensorflow matplotlib pypdf
```

### 2. Run the main pipeline

From the repository root:

```bash
python btc_polymarket_analysis.py
```

This script will:

1. download or refresh BTC daily data
2. collect Polymarket daily BTC up/down events
3. download Polymarket historical trades
4. engineer BTC and Polymarket features
5. train the Transformer models
6. generate out-of-sample predictions
7. run the BTC and Polymarket backtests
8. write all outputs into the `outputs/` folder

### 3. Compile the report

If you have LaTeX installed locally:

```bash
pdflatex report_ieee.tex
pdflatex report_ieee.tex
```

Or upload `report_ieee.tex`, `IEEEtran.cls`, and the `outputs/strategy_equity_curves.png` figure to Overleaf.

## Output Files

The `outputs/` folder contains:

- `btc_yf_daily.csv`
  - cached BTC daily data from Yahoo Finance
- `updown_events.csv`
  - list of Polymarket daily BTC direction events used in the project
- `polymarket_event_features.csv`
  - engineered Polymarket trade-based features by date
- `merged_btc_polymarket_dataset.csv`
  - final merged modeling dataset
- `btc_only_test_predictions.csv`
  - one-year backtest predictions from the BTC-only Transformer
- `btc_polymarket_test_predictions.csv`
  - one-year backtest predictions from the BTC + Polymarket Transformer
- `strategy_backtest.csv`
  - daily backtest results for the BTC and Polymarket strategies
- `strategy_equity_curves.png`
  - equity curve figure used in the report
- `metrics_summary.json`
  - headline predictive and strategy metrics
- `project_summary.md`
  - short markdown summary of final results
- `transformer_tuning_results.csv`
  - earlier tuning sweep for the shorter backtest setup
- `transformer_tuning_results_1y.csv`
  - tuning sweep for the one-year backtest setup

## Final Headline Results

Using the one-year backtest:

- **BTC-only Transformer**
  - Accuracy: `0.512`
  - ROC-AUC: `0.508`

- **BTC + Polymarket Transformer**
  - Accuracy: `0.504`
  - F1: `0.560`
  - ROC-AUC: `0.486`

- **BTC strategy**
  - Total return: about `-0.5%`

- **Polymarket strategy**
  - Total PnL: `74.5`
  - Hit rate: `0.504`

## Main Takeaways

1. In this project, adding Polymarket data did **not** improve classification accuracy relative to the BTC-only model.
2. A likely reason is that the project only used trade-based public data and did **not** include richer historical order book features.
3. Even without higher accuracy, the binary payoff structure of Polymarket made the prediction strategy much more profitable than the spot BTC strategy under the assumed entry rule.

## Notes and Limitations

- The daily Polymarket BTC up/down market family only starts in **March 2025**, so the one-year backtest leaves a short pre-backtest training history.
- The Polymarket strategy assumes the predicted side can always be bought at **0.3**, which is a stylized rule for analysis rather than a full execution model.
- The BTC strategy does not include transaction fees, slippage, or borrowing costs for short positions.

## Contact

Project repository owner: **JasperChen-us**
