"""
00_prepare_data.py

Purpose:
- Merge stock features with news sentiment features
- Create target variable: next-day return (target_return_1d)
- Output merged_features.csv used by all 6 models
"""

from pathlib import Path

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()

    stock_feat_path = project_root / "data" / "features" / "stock_features.csv"
    news_feat_path = project_root / "data" / "features" / "news_features.csv"
    output_path = project_root / "data" / "features" / "merged_features.csv"

    print(f"Reading stock features from {stock_feat_path}")
    stock = pd.read_csv(stock_feat_path, parse_dates=["date"])

    print(f"Reading news features from {news_feat_path}")
    news = pd.read_csv(news_feat_path, parse_dates=["date"])

    # 1. Merge: left-join sentiment features onto stock features (stock as primary)
    df = stock.merge(
        news,
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_news"),
    )

    # 2. Fill missing sentiment columns with 0 (no news → neutral sentiment)
    sentiment_cols = [
        "sentiment_mean",
        "sentiment_max",
        "sentiment_min",
        "sentiment_count",
        "sentiment_index",
    ]
    for col in sentiment_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 3. Create target variable: predict next day's return_1d
    # First, sort by ticker + date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Current return_1d is today's return; we predict tomorrow's:
    df["target_return_1d"] = (
        df.groupby("ticker")["return_1d"].shift(-1)
    )

    # Last row per ticker has no tomorrow → target NaN; drop those
    df = df.dropna(subset=["target_return_1d"]).reset_index(drop=True)

    # 4. Save merged table
    df.to_csv(output_path, index=False)
    print(f"Saved merged features to {output_path}")
    print(df.head())
    print("\nColumns preview:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
