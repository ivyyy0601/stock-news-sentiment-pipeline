"""
00_prepare_data.py

功能：
- 合并股票特征 和 新闻情绪特征
- 生成目标变量：下一日收益率 target_return_1d
- 输出一个总表 merged_features.csv，后续 6 个模型都会用到
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

    print(f"📥 Reading stock features from {stock_feat_path}")
    stock = pd.read_csv(stock_feat_path, parse_dates=["date"])

    print(f"📥 Reading news features from {news_feat_path}")
    news = pd.read_csv(news_feat_path, parse_dates=["date"])

    # 1. 合并：以股票为主表，左连接新闻情绪
    df = stock.merge(
        news,
        on=["date", "ticker"],
        how="left",
        suffixes=("", "_news"),
    )

    # 2. 对情绪相关列缺失值填 0（当天没新闻 → 情绪中性）
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

    # 3. 生成目标变量：预测“下一天的 return_1d”
    # 先按 ticker + date 排好序
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 当前 return_1d 是“今天的收益”，我们预测明天：
    df["target_return_1d"] = (
        df.groupby("ticker")["return_1d"].shift(-1)
    )

    # 最后一行（每个 ticker 的最后一天）没有明天 → target NaN，删掉
    df = df.dropna(subset=["target_return_1d"]).reset_index(drop=True)

    # 4. 保存总表
    df.to_csv(output_path, index=False)
    print(f"✅ Saved merged features to {output_path}")
    print(df.head())
    print("\n📐 Columns preview:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
