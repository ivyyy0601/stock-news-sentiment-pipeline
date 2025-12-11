"""
build_news_features.py

功能：
- 读取 data/clean/news_clean.csv
- 按 (date, ticker) 聚合 overall_sentiment_score
- 生成日级情绪特征
- 保存到 data/features/news_features.csv
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

    # 确保有 date 列；你之前的 clean 里已经加过 date，这里再保险一下
    if "date" not in df.columns:
        df["date"] = df["time_published"].dt.date

    # 转回 datetime64[ns] 方便 merge
    df["date"] = pd.to_datetime(df["date"])

    # 只保留我们需要的列（减少内存）
    df = df[["date", "ticker", "overall_sentiment_score"]].copy()

    # 1. 按 date + ticker 聚合
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

    # 2. 设计一个简单的情绪强度指数：均值 * log(1 + 新闻条数)
    agg["sentiment_index"] = (
        agg["sentiment_mean"] * np.log1p(agg["sentiment_count"])
    )

    # 3. 保存
    agg.to_csv(output_path, index=False)
    print(f"✅ Saved news features to {output_path}")
    print(agg.head())


if __name__ == "__main__":
    main()
