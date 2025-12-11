"""
00_baseline_xgboost.py

功能：
- 读取 merged_features.csv
- 只使用价格相关特征训练 XGBoost
- 按时间划分 train/val/test（与 LSTM 对齐）
- 评估 RMSE / MAE
- 保存：
    - 模型：outputs/xgb_price_only.json
    - 指标：outputs/metrics.csv（追加）
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()
    data_path = project_root / "data" / "features" / "merged_features.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path = outputs_dir / "xgb_price_only.json"
    metrics_path = outputs_dir / "metrics.csv"

    print(f"📥 Reading merged features from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ---- Feature columns (same as LSTM price-only) ----
    price_feature_cols = [
        c for c in df.columns
        if c not in ["date", "ticker", "sentiment_score", "target_return_1d"]
    ]

    target = df["target_return_1d"]

    X = df[price_feature_cols].values
    y = target.values

    # ---- Time-based split ----
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, X_val, X_test = (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
    )
    y_train, y_val, y_test = (
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )

    # ---- Train XGBoost ----
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ---- Evaluate ----
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✅ XGBoost Test RMSE={rmse:.6f}, MAE={mae:.6f}")

    # ---- Save model ----
    model.save_model(model_path)
    print(f"💾 模型已保存到 {model_path}")

    # ---- Append metrics.csv ----
    row = {
        "model_name": "xgb_price_only",
        "use_sentiment": 0,
        "rmse": rmse,
        "mae": mae,
    }

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([row])

    metrics_df.to_csv(metrics_path, index=False)
    print(f"📈 指标已写入 {metrics_path}")
    print(metrics_df.tail())


if __name__ == "__main__":
    main()
