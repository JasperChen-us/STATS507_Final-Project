from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from zoneinfo import ZoneInfo


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
DATA_TRADES_URL = "https://data-api.polymarket.com/trades"
PUBLIC_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
EASTERN_TZ = ZoneInfo("America/New_York")
SEED = 507


@dataclass
class EventRecord:
    event_id: int
    title: str
    target_date: pd.Timestamp
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    condition_id: str
    market_id: int
    up_asset: str
    down_asset: str


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_btc_data(start: str = "2025-01-01", end: str = "2026-04-15") -> pd.DataFrame:
    cache_path = OUTPUT_DIR / "btc_yf_daily.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        if cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=1):
            return cached

    btc = pd.DataFrame()
    for attempt in range(5):
        btc = yf.download(
            "BTC-USD",
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if not btc.empty:
            break
        wait_seconds = 5 * (attempt + 1)
        print(f"yfinance rate-limited, retrying in {wait_seconds} seconds...")
        time.sleep(wait_seconds)

    if btc.empty:
        print("Falling back to Yahoo Finance chart endpoint...")
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())
        resp = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD",
            params={
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "includePrePost": "false",
                "events": "div,splits",
            },
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        payload = resp.json()
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        btc = pd.DataFrame(
            {
                "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None),
                "Open": quote["open"],
                "High": quote["high"],
                "Low": quote["low"],
                "Close": quote["close"],
                "Volume": quote["volume"],
            }
        ).dropna()
        btc = btc.set_index("date")

    if btc.empty:
        raise RuntimeError("Failed to download BTC-USD data from Yahoo Finance.")

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc = btc.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    btc = btc[["open", "high", "low", "close", "volume"]].copy()
    btc.index.name = "date"
    btc.reset_index().to_csv(cache_path, index=False)
    return btc


def parse_target_date(title: str, end_date: str) -> pd.Timestamp:
    end_ts = pd.Timestamp(end_date).tz_convert(EASTERN_TZ).tz_localize(None)
    year = end_ts.year
    month_day_match = re.search(r"on ([A-Za-z]+ \d{1,2})", title)
    if month_day_match:
        return pd.to_datetime(f"{month_day_match.group(1)} {year}")
    raise ValueError(f"Could not parse target date from title: {title}")


def fetch_updown_events(
    start_date: str = "2025-03-13",
    end_date: str = "2026-04-13",
) -> list[EventRecord]:
    cache_path = OUTPUT_DIR / "updown_events.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["target_date", "start_date", "end_date"])
        return [
            EventRecord(
                event_id=int(row.event_id),
                title=row.title,
                target_date=pd.Timestamp(row.target_date),
                start_date=pd.Timestamp(row.start_date),
                end_date=pd.Timestamp(row.end_date),
                condition_id=row.condition_id,
                market_id=int(row.market_id),
                up_asset=row.up_asset,
                down_asset=row.down_asset,
            )
            for row in cached.itertuples(index=False)
        ]

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    current = start_ts
    events: list[EventRecord] = []
    seen_ids: set[int] = set()

    while current <= end_ts:
        query = f"Bitcoin Up or Down on {current.strftime('%B')} {current.day}"
        resp = requests.get(PUBLIC_SEARCH_URL, params={"q": query}, timeout=60)
        resp.raise_for_status()
        matches = resp.json().get("events", [])

        for event in matches:
            title = event.get("title") or ""
            if not title.startswith("Bitcoin Up or Down on "):
                continue
            if not event.get("markets"):
                continue
            target = parse_target_date(title, event["endDate"])
            if target != current:
                continue
            if int(event["id"]) in seen_ids:
                continue

            market = event["markets"][0]
            outcomes = json.loads(market["outcomes"])
            token_ids = json.loads(market["clobTokenIds"])
            outcome_to_token = dict(zip(outcomes, token_ids))
            if "Up" not in outcome_to_token or "Down" not in outcome_to_token:
                continue

            events.append(
                EventRecord(
                    event_id=int(event["id"]),
                    title=title,
                    target_date=target,
                    start_date=pd.Timestamp(event["startDate"]).tz_convert(EASTERN_TZ).tz_localize(None),
                    end_date=pd.Timestamp(event["endDate"]).tz_convert(EASTERN_TZ).tz_localize(None),
                    condition_id=market["conditionId"],
                    market_id=int(market["id"]),
                    up_asset=outcome_to_token["Up"],
                    down_asset=outcome_to_token["Down"],
                )
            )
            seen_ids.add(int(event["id"]))
            break

        current += pd.Timedelta(days=1)
        time.sleep(0.02)

    events = sorted(events, key=lambda x: x.target_date)
    pd.DataFrame(
        [
            {
                "event_id": e.event_id,
                "title": e.title,
                "target_date": e.target_date,
                "start_date": e.start_date,
                "end_date": e.end_date,
                "condition_id": e.condition_id,
                "market_id": e.market_id,
                "up_asset": e.up_asset,
                "down_asset": e.down_asset,
            }
            for e in events
        ]
    ).to_csv(cache_path, index=False)
    return events


def fetch_event_trades(event_id: int) -> pd.DataFrame:
    all_rows: list[dict] = []
    offset = 0
    limit = 10_000

    while True:
        resp = requests.get(
            DATA_TRADES_URL,
            params={"eventId": event_id, "limit": limit, "offset": offset, "takerOnly": "false"},
            timeout=60,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        offset += limit
        time.sleep(0.05)

    trades = pd.DataFrame(all_rows)
    if trades.empty:
        return trades

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="s", utc=True).dt.tz_convert(EASTERN_TZ)
    trades["price"] = trades["price"].astype(float)
    trades["size"] = trades["size"].astype(float)
    trades["notional"] = trades["price"] * trades["size"]
    return trades.sort_values("timestamp")


def polymarket_features(events: Iterable[EventRecord]) -> pd.DataFrame:
    cache_path = OUTPUT_DIR / "polymarket_event_features.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"]).sort_values("date")
        existing_dates = set(pd.to_datetime(cached["date"]).dt.normalize())
        needed = [e for e in events if pd.Timestamp(e.target_date).normalize() not in existing_dates]
        if not needed:
            return cached
        feature_rows = cached.to_dict("records")
    else:
        feature_rows = []

    start_idx = len(feature_rows)
    for idx, event in enumerate(needed if cache_path.exists() else events, start=1):
        trades = fetch_event_trades(event.event_id)
        cutoff = pd.Timestamp(event.target_date).tz_localize(EASTERN_TZ)

        if trades.empty:
            feature_rows.append(
                {
                    "date": event.target_date,
                    "pm_up_last": np.nan,
                    "pm_up_first": np.nan,
                    "pm_up_mean": np.nan,
                    "pm_up_std": np.nan,
                    "pm_up_change": np.nan,
                    "pm_up_min": np.nan,
                    "pm_up_max": np.nan,
                    "pm_up_trade_count": 0,
                    "pm_up_notional": 0.0,
                    "pm_total_trade_count": 0,
                    "pm_total_notional": 0.0,
                    "pm_market_age_hours": (cutoff.tz_localize(None) - event.start_date).total_seconds() / 3600.0,
                }
            )
            continue

        pre_cutoff = trades.loc[trades["timestamp"] < cutoff].copy()
        pre_cutoff["pm_implied_up"] = np.where(
            pre_cutoff["asset"] == event.up_asset,
            pre_cutoff["price"],
            np.where(pre_cutoff["asset"] == event.down_asset, 1 - pre_cutoff["price"], np.nan),
        )
        implied = pre_cutoff.dropna(subset=["pm_implied_up"]).copy()

        row = {
            "date": event.target_date,
            "pm_total_trade_count": int(len(pre_cutoff)),
            "pm_total_notional": float(pre_cutoff["notional"].sum()),
            "pm_market_age_hours": (cutoff.tz_localize(None) - event.start_date).total_seconds() / 3600.0,
        }

        if implied.empty:
            row.update(
                {
                    "pm_up_last": np.nan,
                    "pm_up_first": np.nan,
                    "pm_up_mean": np.nan,
                    "pm_up_std": np.nan,
                    "pm_up_change": np.nan,
                    "pm_up_min": np.nan,
                    "pm_up_max": np.nan,
                    "pm_up_trade_count": 0,
                    "pm_up_notional": 0.0,
                }
            )
        else:
            row.update(
                {
                    "pm_up_last": float(implied["pm_implied_up"].iloc[-1]),
                    "pm_up_first": float(implied["pm_implied_up"].iloc[0]),
                    "pm_up_mean": float(implied["pm_implied_up"].mean()),
                    "pm_up_std": float(implied["pm_implied_up"].std(ddof=0)),
                    "pm_up_change": float(implied["pm_implied_up"].iloc[-1] - implied["pm_implied_up"].iloc[0]),
                    "pm_up_min": float(implied["pm_implied_up"].min()),
                    "pm_up_max": float(implied["pm_implied_up"].max()),
                    "pm_up_trade_count": int(len(implied)),
                    "pm_up_notional": float(implied["notional"].sum()),
                }
            )

        feature_rows.append(row)
        if idx % 25 == 0:
            print(f"Fetched Polymarket trades for {start_idx + idx} total events...")

    pm = pd.DataFrame(feature_rows).sort_values("date")
    pm["date"] = pd.to_datetime(pm["date"])
    pm.to_csv(cache_path, index=False)
    return pm


def engineer_btc_features(btc: pd.DataFrame) -> pd.DataFrame:
    df = btc.copy()
    returns = df["close"].pct_change()
    df["ret_1d"] = returns
    df["ret_2d"] = df["close"].pct_change(2)
    df["ret_3d"] = df["close"].pct_change(3)
    df["ret_5d"] = df["close"].pct_change(5)
    df["open_close_ret"] = df["close"] / df["open"] - 1
    df["high_low_range"] = df["high"] / df["low"] - 1
    df["close_to_high"] = df["close"] / df["high"] - 1
    df["close_to_low"] = df["close"] / df["low"] - 1
    df["volume_chg_1d"] = df["volume"].pct_change()
    df["vol_5d"] = returns.rolling(5).std()
    df["vol_10d"] = returns.rolling(10).std()
    df["mom_5d"] = df["close"] / df["close"].shift(5) - 1
    df["mom_10d"] = df["close"] / df["close"].shift(10) - 1
    df["sma_5_ratio"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["sma_10_ratio"] = df["close"] / df["close"].rolling(10).mean() - 1
    df["sma_20_ratio"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df


def build_dataset() -> pd.DataFrame:
    merged_path = OUTPUT_DIR / "merged_btc_polymarket_dataset.csv"
    if merged_path.exists():
        cached = pd.read_csv(merged_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if cached["date"].max() >= pd.Timestamp("2026-04-13"):
            return cached

    print("Downloading BTC data...")
    btc = get_btc_data()
    btc_features = engineer_btc_features(btc).reset_index()

    print("Collecting Polymarket up/down events...")
    events = fetch_updown_events(start_date="2025-03-13", end_date="2026-04-13")
    pm = polymarket_features(events)

    merged = btc_features.merge(pm, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged.to_csv(merged_path, index=False)
    return merged


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df["target_up"].to_numpy(dtype=np.float32)
    dates = df["date"].to_numpy()

    xs: list[np.ndarray] = []
    ys: list[float] = []
    ds: list[np.datetime64] = []
    for idx in range(sequence_length - 1, len(df)):
        xs.append(values[idx - sequence_length + 1 : idx + 1])
        ys.append(targets[idx])
        ds.append(dates[idx])

    return np.stack(xs), np.array(ys, dtype=np.float32), np.array(ds)


def transformer_block(x: tf.Tensor, num_heads: int, key_dim: int, ff_dim: int, dropout: float) -> tf.Tensor:
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
    )(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(x.shape[-1])(ff)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_transformer_model(sequence_length: int, num_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, num_features))
    x = tf.keras.layers.Dense(32)(inputs)
    x = transformer_block(x, num_heads=4, key_dim=8, ff_dim=64, dropout=0.1)
    x = transformer_block(x, num_heads=4, key_dim=8, ff_dim=64, dropout=0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def fit_transformer(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = 10,
    train_ratio: float = 0.7,
    threshold: float = 0.5,
    backtest_days: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    features = work[feature_cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if backtest_days is not None:
        cutoff_date = work["date"].max() - pd.Timedelta(days=backtest_days - 1)
        split_idx = int(work.index[work["date"] >= cutoff_date][0])
    else:
        split_idx = math.floor(len(work) * train_ratio)
        cutoff_date = work["date"].iloc[split_idx]
    train_features = features.iloc[:split_idx]
    test_features = features.iloc[split_idx:]

    imputer.fit(train_features)
    scaler.fit(imputer.transform(train_features))

    transformed = scaler.transform(imputer.transform(features))
    transformed_df = pd.DataFrame(transformed, columns=feature_cols, index=work.index)
    seq_df = pd.concat([work[["date", "target_up", "open", "close"]], transformed_df], axis=1)

    X, y, dates = make_sequences(seq_df, feature_cols, sequence_length)
    sample_dates = pd.to_datetime(dates)
    train_mask = sample_dates < cutoff_date
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_dates = sample_dates[test_mask]

    val_size = max(10, int(len(X_train) * 0.2))
    val_size = min(val_size, max(1, len(X_train) - 20))
    train_end = len(X_train) - val_size
    X_fit, y_fit = X_train[:train_end], y_train[:train_end]
    X_val, y_val = X_train[train_end:], y_train[train_end:]

    set_seed()
    model = build_transformer_model(sequence_length=sequence_length, num_features=len(feature_cols))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
        )
    ]
    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        verbose=0,
        callbacks=callbacks,
    )

    pred_prob = model.predict(X_test, verbose=0).ravel()
    pred_label = (pred_prob >= threshold).astype(int)

    aligned_test = work.set_index("date").loc[test_dates].reset_index()
    if "date" not in aligned_test.columns:
        aligned_test = aligned_test.rename(columns={aligned_test.columns[0]: "date"})
    aligned_test["pred_prob"] = pred_prob
    aligned_test["pred_label"] = pred_label

    metrics = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "sequence_length": int(sequence_length),
        "threshold": float(threshold),
        "backtest_start": str(pd.Timestamp(test_dates.min()).date()),
        "backtest_end": str(pd.Timestamp(test_dates.max()).date()),
        "accuracy": float(accuracy_score(y_test, pred_label)),
        "precision": float(precision_score(y_test, pred_label, zero_division=0)),
        "recall": float(recall_score(y_test, pred_label, zero_division=0)),
        "f1": float(f1_score(y_test, pred_label, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, pred_prob)),
    }
    return aligned_test, metrics


def run_strategies(test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = test_df.copy()
    df["btc_signal"] = df["pred_label"]
    df["btc_daily_return"] = np.where(df["btc_signal"] == 1, df["close"] / df["open"] - 1, 0.0)
    df["btc_equity"] = (1 + df["btc_daily_return"]).cumprod()

    # Buy the predicted side every day at price 0.3.
    # Correct prediction pays 1.0 at expiry, so net PnL is +0.7.
    # Incorrect prediction expires worthless, so net PnL is -0.3.
    df["pm_pred_side"] = np.where(df["pred_label"] == 1, "up", "down")
    df["pm_correct"] = (df["pred_label"] == df["target_up"]).astype(int)
    df["pm_pnl"] = np.where(df["pm_correct"] == 1, 0.7, -0.3)
    df["pm_equity"] = 1.0 + df["pm_pnl"].cumsum()

    summary = {
        "btc_strategy_total_return": float(df["btc_equity"].iloc[-1] - 1),
        "btc_strategy_avg_daily_return": float(df["btc_daily_return"].mean()),
        "btc_strategy_sharpe": float(
            np.sqrt(252) * df["btc_daily_return"].mean() / df["btc_daily_return"].std(ddof=0)
        )
        if df["btc_daily_return"].std(ddof=0) > 0
        else np.nan,
        "pm_strategy_total_pnl": float(df["pm_pnl"].sum()),
        "pm_strategy_avg_daily_pnl": float(df["pm_pnl"].mean()),
        "pm_strategy_hit_rate": float(df["pm_correct"].mean()),
    }
    return df, summary


def plot_equity_curves(strategy_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(strategy_df["date"], strategy_df["btc_equity"], label="BTC strategy")
    plt.plot(strategy_df["date"], strategy_df["pm_equity"], label="Polymarket strategy")
    plt.title("Strategy Equity Curves (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_equity_curves.png", dpi=200)
    plt.close()


def main() -> None:
    set_seed()
    merged = build_dataset()

    btc_only_features = [
        "ret_1d",
        "ret_2d",
        "ret_3d",
        "ret_5d",
        "open_close_ret",
        "high_low_range",
        "close_to_high",
        "close_to_low",
        "volume_chg_1d",
        "vol_5d",
        "vol_10d",
        "mom_5d",
        "mom_10d",
        "sma_5_ratio",
        "sma_10_ratio",
        "sma_20_ratio",
    ]
    joint_features = btc_only_features + [
        "pm_up_last",
        "pm_up_mean",
        "pm_up_change",
        "pm_up_trade_count",
        "pm_total_notional",
    ]

    btc_config = {"sequence_length": 14, "threshold": 0.55, "backtest_days": 365}
    joint_config = {"sequence_length": 14, "threshold": 0.50, "backtest_days": 365}

    print("Training BTC-only Transformer...")
    btc_test, btc_metrics = fit_transformer(merged, btc_only_features, **btc_config)

    print("Training BTC + Polymarket Transformer...")
    joint_test, joint_metrics = fit_transformer(merged, joint_features, **joint_config)

    strategy_df, strategy_metrics = run_strategies(joint_test)
    plot_equity_curves(strategy_df)

    btc_test.to_csv(OUTPUT_DIR / "btc_only_test_predictions.csv", index=False)
    joint_test.to_csv(OUTPUT_DIR / "btc_polymarket_test_predictions.csv", index=False)
    strategy_df.to_csv(OUTPUT_DIR / "strategy_backtest.csv", index=False)

    results = {
        "sample": {
            "total_rows": int(len(merged)),
            "start_date": str(merged["date"].min().date()),
            "end_date": str(merged["date"].max().date()),
        },
        "model_type": "TensorFlow Transformer encoder",
        "btc_only_model": btc_metrics,
        "btc_plus_polymarket_model": joint_metrics,
        "strategies": strategy_metrics,
    }

    with open(OUTPUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_lines = [
        "# BTC + Polymarket Project Summary",
        "",
        f"Sample period: {results['sample']['start_date']} to {results['sample']['end_date']}",
        f"Observations: {results['sample']['total_rows']}",
        f"Model: {results['model_type']}",
        f"Backtest period: {joint_metrics['backtest_start']} to {joint_metrics['backtest_end']}",
        f"BTC-only config: sequence length={btc_metrics['sequence_length']}, threshold={btc_metrics['threshold']:.2f}",
        f"BTC+Polymarket config: sequence length={joint_metrics['sequence_length']}, threshold={joint_metrics['threshold']:.2f}",
        "",
        "## Classification Metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| BTC-only Transformer | {btc_metrics['accuracy']:.3f} | {btc_metrics['precision']:.3f} | "
            f"{btc_metrics['recall']:.3f} | {btc_metrics['f1']:.3f} | {btc_metrics['roc_auc']:.3f} |"
        ),
        (
            f"| BTC+Polymarket Transformer | {joint_metrics['accuracy']:.3f} | {joint_metrics['precision']:.3f} | "
            f"{joint_metrics['recall']:.3f} | {joint_metrics['f1']:.3f} | {joint_metrics['roc_auc']:.3f} |"
        ),
        "",
        "## Strategy Metrics (BTC + Polymarket Transformer)",
        "",
        f"- BTC strategy total return: {strategy_metrics['btc_strategy_total_return']:.3f}",
        f"- BTC strategy average daily return: {strategy_metrics['btc_strategy_avg_daily_return']:.5f}",
        f"- BTC strategy Sharpe ratio: {strategy_metrics['btc_strategy_sharpe']:.3f}",
        f"- Polymarket strategy total PnL: {strategy_metrics['pm_strategy_total_pnl']:.3f}",
        f"- Polymarket strategy average daily PnL: {strategy_metrics['pm_strategy_avg_daily_pnl']:.5f}",
        f"- Polymarket strategy hit rate: {strategy_metrics['pm_strategy_hit_rate']:.3f}",
        "",
        "## Notes",
        "",
        "- The BTC-only Transformer uses a 14-day rolling sequence; the BTC+Polymarket Transformer uses a 14-day rolling sequence with a smaller Polymarket feature subset selected by tuning.",
        "- BTC labels use next-day close direction from daily Yahoo Finance data.",
        "- Polymarket features come from pre-resolution trades in the `Bitcoin Up or Down on [date]?` markets.",
        "- The Polymarket strategy now buys the predicted side (Up or Down) at price 0.3 each day, earning +0.7 if correct and -0.3 if wrong.",
    ]
    (OUTPUT_DIR / "project_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Finished. Outputs saved in:", OUTPUT_DIR)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
