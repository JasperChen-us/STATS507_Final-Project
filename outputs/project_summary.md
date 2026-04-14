# BTC + Polymarket Project Summary

Sample period: 2025-03-13 to 2026-04-13
Observations: 396
Model: TensorFlow Transformer encoder
Backtest period: 2025-04-14 to 2026-04-13
BTC-only config: sequence length=14, threshold=0.55
BTC+Polymarket config: sequence length=14, threshold=0.50

## Classification Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| BTC-only Transformer | 0.512 | 0.515 | 0.374 | 0.433 | 0.508 |
| BTC+Polymarket Transformer | 0.504 | 0.502 | 0.632 | 0.560 | 0.486 |

## Strategy Metrics (BTC + Polymarket Transformer)

- BTC strategy total return: -0.005
- BTC strategy average daily return: 0.00018
- BTC strategy Sharpe ratio: 0.145
- Polymarket strategy total PnL: 74.500
- Polymarket strategy average daily PnL: 0.20411
- Polymarket strategy hit rate: 0.504

## Notes

- The BTC-only Transformer uses a 14-day rolling sequence; the BTC+Polymarket Transformer uses a 14-day rolling sequence with a smaller Polymarket feature subset selected by tuning.
- BTC labels use next-day close direction from daily Yahoo Finance data.
- Polymarket features come from pre-resolution trades in the `Bitcoin Up or Down on [date]?` markets.
- The Polymarket strategy now buys the predicted side (Up or Down) at price 0.3 each day, earning +0.7 if correct and -0.3 if wrong.