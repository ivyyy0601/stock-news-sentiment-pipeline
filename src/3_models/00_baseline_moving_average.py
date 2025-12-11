"""
00_baseline_moving_average.py

Purpose:
- Read data/features/merged_features.csv
- Compute MA7 / MA14 baselines
- Prediction: next-day forecast = current MA (shift by 1)
- Split train / val / test by time proportion
- Evaluate RMSE / MAE
- Save metrics to outputs/metrics.csv (append rows)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()
    data_path = project_root / "data" / "features" / "merged_features.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = outputs_dir / "metrics.csv"

    print(f"Reading merged features from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ---- baseline: Moving Average ----
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma14"] = df["close"].rolling(14).mean()

    df["ma7_pred"] = df["ma7"].shift(1)
    df["ma14_pred"] = df["ma14"].shift(1)

    # target
    target_col = "target_return_1d"
    y = df[target_col].values

    # train/val/test split (same as your models)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    def evaluate(pred_col):
        y_pred = df[pred_col].values

        # same slicing as NN models
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        p_train, p_val, p_test = (
            y_pred[:train_end],
            y_pred[train_end:val_end],
            y_pred[val_end:],
        )

        rmse = np.sqrt(mean_squared_error(y_test, p_test))
        mae = mean_absolute_error(y_test, p_test)
        return rmse, mae

    # Evaluate MA7
    rmse7, mae7 = evaluate("ma7_pred")
    print(f"MA7 → RMSE={rmse7:.6f}, MAE={mae7:.6f}")

    # Evaluate MA14
    rmse14, mae14 = evaluate("ma14_pred")
    print(f"MA14 → RMSE={rmse14:.6f}, MAE={mae14:.6f}")

    # ---- Append to metrics.csv ----
    rows = [
        {"model_name": "ma7_baseline", "use_sentiment": 0, "rmse": rmse7, "mae": mae7},
        {"model_name": "ma14_baseline", "use_sentiment": 0, "rmse": rmse14, "mae": mae14},
    ]

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)
    else:
        metrics_df = pd.DataFrame(rows)

    metrics_df.to_csv(metrics_path, index=False)
    print(f"Baseline metrics written → {metrics_path}")
    print(metrics_df.tail())


if __name__ == "__main__":
    main()
