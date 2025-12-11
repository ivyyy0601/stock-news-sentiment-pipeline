"""
build_news_features.py

Purpose:
- Read data/clean/news_clean.csv
- Aggregate overall_sentiment_score by (date, ticker)
- Generate daily sentiment features
- Save to data/features/news_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()
    clean_path = project_root / "data" / "clean" / "news_clean.csv"
    feat_dir = project_root / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    output_path = feat_dir / "news_features.csv"

    print(f"📥 Reading cleaned news from {clean_path}")
    df = pd.read_csv(clean_path, parse_dates=["time_published"])

    # Ensure a date column exists; clean step already adds it, but double-check
    if "date" not in df.columns:
        df["date"] = df["time_published"].dt.date

    # Convert to datetime64[ns] for easier merging
    df["date"] = pd.to_datetime(df["date"])

    # Keep only needed columns (reduce memory)
    df = df[["date", "ticker", "overall_sentiment_score"]].copy()

    # 1. Aggregate by date + ticker
    agg = (
        df.groupby(["date", "ticker"])["overall_sentiment_score"]
        .agg(["mean", "max", "min", "count"])
        .reset_index()
    )

    agg = agg.rename(
        columns={
            "mean": "sentiment_mean",
            "max": "sentiment_max",
            "min": "sentiment_min",
            "count": "sentiment_count",
        }
    )

    # 2. Simple sentiment intensity index: mean * log(1 + count)
    agg["sentiment_index"] = (
        agg["sentiment_mean"] * np.log1p(agg["sentiment_count"])
    )

    # 3. Save features
    agg.to_csv(output_path, index=False)
    print(f"Saved news features to {output_path}")
    print(agg.head())


if __name__ == "__main__":
    main()
