"""
clean_news.py

功能：
- 读取 data/raw/news.csv（Alpha Vantage NEWS_SENTIMENT 抓下来的新闻）
- 解析时间字段 time_published
- 添加 date 列（按天对齐股价用）
- 简单清洗（去掉缺少关键信息的行）
- 保存到 data/clean/news_clean.csv
"""

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()
    raw_path = project_root / "data" / "raw" / "news.csv"
    clean_dir = project_root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_path = clean_dir / "news_clean.csv"

    print(f"📥 Reading raw news from {raw_path}")
    df = pd.read_csv(raw_path)

    # 1. 解析时间
    # news.csv 里 time_published 已经是类似 "2023-01-03 12:05:00" 这种格式
    # 如果你看到还是 "20230103T120500"，可以换成对应的 format
    df["time_published"] = pd.to_datetime(
        df["time_published"], errors="coerce"
    )

    # 2. 添加按天的 date 列（和股票对齐）
    df["date"] = df["time_published"].dt.date

    # 3. 去掉没有日期或没有 ticker 的行
    df = df.dropna(subset=["time_published", "ticker"]).reset_index(drop=True)

    # 4. 可以简单去掉 title、summary 都空的行
    df = df[~(df["title"].isna() & df["summary"].isna())].reset_index(drop=True)

    # 5. 按 ticker + date + time 排序，方便后续聚合
    df = df.sort_values(["ticker", "date", "time_published"]).reset_index(drop=True)

    # 6. 保存
    df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned news to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
