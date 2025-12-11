"""
build_stock_features.py

功能：
- 读取 data/clean/stocks_clean.csv
- 按 ticker + date 排序
- 计算收益率、滞后收益、滚动均值/波动率等特征
- 保存到 data/features/stock_features.csv
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

    print(f"📥 Reading cleaned stocks from {clean_path}")
    df = pd.read_csv(clean_path, parse_dates=["date"])

    # 1. 按 ticker + date 排序
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 2. 按 ticker 分组计算 1 日收益率
    df["return_1d"] = (
        df.groupby("ticker")["adj_close"]
        .pct_change()
        .astype(float)
    )

    # 3. 生成几个滞后收益特征（1 / 3 / 7 天）
    for lag in [1, 3, 7]:
        df[f"return_lag_{lag}"] = (
            df.groupby("ticker")["return_1d"].shift(lag)
        )

    # 4. 生成滚动均值 & 标准差（5 日窗口）
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

    # 5. 可以再保留一个“价位级别”作为特征（比如 log_price）
    df["log_price"] = np.log(df["adj_close"])

    # 6. 去掉前期因为 lag / rolling 产生的大量 NaN 行
    df = df.dropna().reset_index(drop=True)

    # 7. 保存
    df.to_csv(output_path, index=False)
    print(f"✅ Saved stock features to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
