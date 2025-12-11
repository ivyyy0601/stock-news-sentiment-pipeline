"""
clean_news.py

Purpose:
- Read data/raw/news.csv (news fetched via Alpha Vantage NEWS_SENTIMENT)
- Parse time_published field
- Add a daily date column (to align with stock prices)
- Basic cleaning (drop rows missing key information)
- Save to data/clean/news_clean.csv
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

    print(f"Reading raw news from {raw_path}")
    df = pd.read_csv(raw_path)

    # 1. Parse timestamps
    # In news.csv, time_published is typically like "2023-01-03 12:05:00".
    # If it's like "20230103T120500", adjust the format accordingly.
    df["time_published"] = pd.to_datetime(
        df["time_published"], errors="coerce"
    )

    # 2. Add a daily date column (align with stock data)
    df["date"] = df["time_published"].dt.date

    # 3. Drop rows missing timestamp or ticker
    df = df.dropna(subset=["time_published", "ticker"]).reset_index(drop=True)

    # 4. Optionally drop rows where both title and summary are empty
    df = df[~(df["title"].isna() & df["summary"].isna())].reset_index(drop=True)

    # 5. Sort by ticker + date + time for easier aggregation later
    df = df.sort_values(["ticker", "date", "time_published"]).reset_index(drop=True)

    # 6. Save cleaned file
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned news to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
