"""
clean_stocks.py

功能：
- 读取 data/raw/stocks.csv（yfinance 多 ticker 导出的宽表）
- 解析两层表头（价格类型 / ticker）
- 转换成长表：date, ticker, open, high, low, close, adj_close, volume
- 去重、按日期排序
- 保存到 data/clean/stocks_clean.csv
"""

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    # TRY/ 作为项目根目录
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()
    raw_path = project_root / "data" / "raw" / "stocks.csv"
    clean_dir = project_root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_path = clean_dir / "stocks_clean.csv"

    # 1. 读取带两层表头的 CSV
    #   header=[0,1] 表示前两行都是表头
    #   index_col=0 把日期那一列当成索引
    print(f"📥 Reading raw stocks from {raw_path}")
    df_raw = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True)

    # 2. 把多层列索引改成标准形式
    # 第一层：价格类型 (Adj Close / Close / High / Low / Open / Volume)
    # 第二层：ticker (AAPL / GOOG)
    # stack(level=1) 把 ticker 这一层“拉下来”，变成长表
    df_long = df_raw.stack(level=1).reset_index()

    # 现在列大概是：["Date", "level_1", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
    # 我们把 level_1 重命名为 ticker
    df_long = df_long.rename(
        columns={
            df_long.columns[0]: "date",    # 原来的索引列，通常叫 "Date"
            "level_1": "ticker",
            "Adj Close": "adj_close",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Open": "open",
            "Volume": "volume",
        }
    )

    # 3. 只保留需要的列，并排序
    keep_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    df_long = df_long[keep_cols].copy()

    # 去掉明显的缺失（比如没价格的行）
    df_long = df_long.dropna(subset=["close"]).reset_index(drop=True)

    # 排序
    df_long = df_long.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 4. 保存
    df_long.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned stocks to {output_path}")
    print(df_long.head())


if __name__ == "__main__":
    main()
