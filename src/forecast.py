"""
forecast.py
-----------
Trains an XGBoost regressor to forecast daily total sales, using simple
calendar + lag features (a good, explainable baseline before reaching for
Prophet/ARIMA). Logs the run (params, metrics, model artifact) to MLflow
so you build the experiment-tracking habit from week 1.

Usage:
    python src/forecast.py
"""

import os
import sqlite3
import warnings

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bi_assistant.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "forecast_model.json")
MLRUNS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "mlruns.db")
FORECAST_HORIZON_DAYS = 30


def load_daily_sales(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT order_date, SUM(sales) AS total_sales FROM sales GROUP BY order_date ORDER BY order_date",
            conn,
        )
    finally:
        conn.close()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.set_index("order_date").asfreq("D").fillna(0.0).reset_index()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["order_date"].dt.dayofweek
    df["month"] = df["order_date"].dt.month
    df["day"] = df["order_date"].dt.day
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = df["total_sales"].shift(lag)
    df["rolling_mean_7"] = df["total_sales"].shift(1).rolling(7).mean()
    df["rolling_mean_28"] = df["total_sales"].shift(1).rolling(28).mean()
    return df.dropna().reset_index(drop=True)


FEATURE_COLS = [
    "dayofweek", "month", "day", "is_weekend", "is_holiday_season",
    "lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
]


def train_and_log(df_features: pd.DataFrame) -> tuple[XGBRegressor, dict]:
    split_idx = int(len(df_features) * 0.85)
    train, test = df_features.iloc[:split_idx], df_features.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLS], train["total_sales"]
    X_test, y_test = test[FEATURE_COLS], test["total_sales"]

    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9)
    model = XGBRegressor(**params, random_state=42)

    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DB_PATH}")
    mlflow.set_experiment("bi-ai-assistant-forecast")

    with mlflow.start_run(run_name="xgboost_daily_sales_forecast"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test.clip(lower=1), np.clip(preds, 1, None))

        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.xgboost.log_model(model, name="model")

        metrics = {"mae": mae, "mape": mape, "n_train": len(train), "n_test": len(test)}
        print(f"Trained. MAE={mae:.2f}  MAPE={mape:.2%}")

    model.save_model(MODEL_PATH)
    return model, metrics


def forecast_future(model: XGBRegressor, df_features: pd.DataFrame, horizon: int = FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """Iteratively forecast forward by re-deriving lag/rolling features each step."""
    history = df_features[["order_date", "total_sales"]].copy()
    future_rows = []

    for _ in range(horizon):
        next_date = history["order_date"].iloc[-1] + pd.Timedelta(days=1)
        tmp = pd.concat(
            [history, pd.DataFrame({"order_date": [next_date], "total_sales": [np.nan]})],
            ignore_index=True,
        )
        feat = add_features(tmp[["order_date", "total_sales"]].fillna(method="ffill"))
        # Rebuild features off known history only (last row uses lag/rolling of true history)
        row = tmp.iloc[[-1]].copy()
        row["dayofweek"] = next_date.dayofweek
        row["month"] = next_date.month
        row["day"] = next_date.day
        row["is_weekend"] = int(next_date.dayofweek >= 5)
        row["is_holiday_season"] = int(next_date.month in (11, 12))
        row["lag_1"] = history["total_sales"].iloc[-1]
        row["lag_7"] = history["total_sales"].iloc[-7] if len(history) >= 7 else history["total_sales"].mean()
        row["lag_14"] = history["total_sales"].iloc[-14] if len(history) >= 14 else history["total_sales"].mean()
        row["lag_28"] = history["total_sales"].iloc[-28] if len(history) >= 28 else history["total_sales"].mean()
        row["rolling_mean_7"] = history["total_sales"].tail(7).mean()
        row["rolling_mean_28"] = history["total_sales"].tail(28).mean()

        pred = float(model.predict(row[FEATURE_COLS])[0])
        future_rows.append({"order_date": next_date, "total_sales": pred})
        history = pd.concat([history, pd.DataFrame([{"order_date": next_date, "total_sales": pred}])], ignore_index=True)

    return pd.DataFrame(future_rows)


def run_pipeline() -> pd.DataFrame:
    daily = load_daily_sales()
    features = add_features(daily)
    model, metrics = train_and_log(features)
    future = forecast_future(model, features)
    return daily, future, metrics


if __name__ == "__main__":
    daily, future, metrics = run_pipeline()
    print(f"\nNext {FORECAST_HORIZON_DAYS} days forecast (head):")
    print(future.head())
