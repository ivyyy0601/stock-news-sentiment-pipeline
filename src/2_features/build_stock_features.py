"""
build_stock_features.py

Purpose:
- Read data/clean/stocks_clean.csv
- Sort by ticker + date
- Compute returns, lagged returns, rolling mean/volatility features
- Save to data/features/stock_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()
    clean_path = project_root / "data" / "clean" / "stocks_clean.csv"
    feat_dir = project_root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    output_path = feat_dir / "stock_features.csv"

    print(f"Reading cleaned stocks from {clean_path}")
    df = pd.read_csv(clean_path, parse_dates=["date"])

    # 1. Sort by ticker + date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 2. Per-ticker compute 1-day returns
    df["return_1d"] = (
        df.groupby("ticker")["adj_close"]
        .pct_change()
        .astype(float)
    )

    # 3. Lagged return features (1 / 3 / 7 days)
    for lag in [1, 3, 7]:
        df[f"return_lag_{lag}"] = (
            df.groupby("ticker")["return_1d"].shift(lag)
        )

    # 4. Rolling mean & std (5-day window)
    window = 5
    df[f"roll_mean_{window}"] = (
        df.groupby("ticker")["return_1d"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f"roll_std_{window}"] = (
        df.groupby("ticker")["return_1d"]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    # 5. Include a price-level feature (e.g., log_price)
    df["log_price"] = np.log(df["adj_close"])

    # 6. Drop rows with NaNs introduced by lag/rolling
    df = df.dropna().reset_index(drop=True)

    # 7. Save features
    df.to_csv(output_path, index=False)
    print(f"Saved stock features to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
