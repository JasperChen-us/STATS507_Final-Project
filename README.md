# STATS 507 Final Project

This repository contains the code, generated outputs, and final IEEE-style report for a project on using Bitcoin market data and Polymarket prediction market data to forecast next-day BTC direction.

## Main Files

- `btc_polymarket_analysis.py`: end-to-end data collection, feature engineering, Transformer training, tuning choices, and backtesting.
- `data.ipynb`: initial notebook used during early data work.
- `report_ieee.tex`: LaTeX source for the final report.
- `report_ieee.pdf`: compiled final report.

## Outputs

The `outputs/` folder contains the processed datasets, prediction files, tuning summaries, and strategy backtest artifacts used in the report.

## Notes

- BTC daily OHLCV data is downloaded from Yahoo Finance.
- Polymarket features are built from daily `Bitcoin Up or Down on [date]?` contracts.
- The final backtest period is from `2025-04-14` to `2026-04-13`.
