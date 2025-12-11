"""
make_dataset_news.py — Fetch multi-year news using Alpha Vantage NEWS_SENTIMENT

Purpose:
- For multiple tickers (e.g., AAPL, GOOG), pull news from a given start date forward along the timeline
- Automatically advance time_from until near the current time or the API returns no more data
- Save to data/raw/news.csv for subsequent cleaning, FinBERT sentiment analysis, and feature engineering
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


def get_project_root() -> Path:
    # Assume this file is at <project>/src/0_data/make_dataset_news.py
    # parents[2] resolves to the project root
    return Path(__file__).resolve().parents[2]


def fetch_articles_for_ticker(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    api_key: str,
    limit: int = 1000,
    sleep_sec: int = 12,
) -> pd.DataFrame:
    """
    Use Alpha Vantage NEWS_SENTIMENT API to fetch news from start_dt up to end_dt.
    - ticker: e.g., "AAPL", "GOOG"
    - start_dt: start time (datetime)
    - end_dt: end time (datetime)
    - limit: max items per request (1–1000)
    - sleep_sec: delay between requests to avoid rate limits

    Returns:
    - A DataFrame with multiple news records, including:
        ticker, time_published, title, summary, overall_sentiment_score, overall_sentiment_label
    """
    url = "https://www.alphavantage.co/query"
    all_records = []

    current_start = start_dt

    while True:
        # If already past end time, stop
        if current_start >= end_dt:
            print(f"[{ticker}] Reached end_dt {end_dt}, stop fetching.")
            break

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": current_start.strftime("%Y%m%dT%H%M"),
            "sort": "EARLIEST",  # Return earliest first and move forward
            "limit": limit,
            "apikey": api_key,
        }

        print(f"[{ticker}] Requesting from {params['time_from']} ...")
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[{ticker}] Request failed: {e}")
            break

        # Rate limit or other informational messages
        if "Information" in data:
            print(f"[{ticker}] Information from API: {data['Information']}")
            break
        if "Note" in data:
            print(f"[{ticker}] Note from API (rate limit?): {data['Note']}")
            break

        # If items == '0', there is no more news to fetch
        if data.get("items") == "0":
            print(f"[{ticker}] No more articles to extract (items=0).")
            break

        feed = data.get("feed")
        if not feed:
            print(f"[{ticker}] No 'feed' field in response, raw data: {data}")
            break

        # Parse the batch and remember the time of the last article
        last_time = None

        for item in feed:
            time_published = item.get("time_published")  # e.g., "20230103T120500"
            title = item.get("title")
            summary = item.get("summary")
            overall_score = item.get("overall_sentiment_score")
            overall_label = item.get("overall_sentiment_label")

            all_records.append(
                {
                    "ticker": ticker,
                    "time_published": time_published,
                    "title": title,
                    "summary": summary,
                    "overall_sentiment_score": overall_score,
                    "overall_sentiment_label": overall_label,
                }
            )

            # Convert to datetime for the next start point
            if time_published:
                try:
                    last_time = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                except ValueError:
                    # If format is unexpected, skip without updating last_time
                    pass

        if last_time is None:
            print(f"[{ticker}] Could not parse any valid time_published, stop.")
            break

        print(f"[{ticker}] Got {len(feed)} articles, last_time = {last_time}")

        # If last_time is at/after end_dt, we can stop
        if last_time >= end_dt:
            print(f"[{ticker}] last_time >= end_dt, stop fetching.")
            break

        # Advance next start to last_time + 1 minute to avoid duplicates
        current_start = last_time + timedelta(minutes=1)

        # Wait to avoid hitting API rate limits
        print(f"[{ticker}] Sleeping {sleep_sec} seconds before next request ...")
        time.sleep(sleep_sec)

    if not all_records:
        print(f"[{ticker}] No records fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Convert time_published to datetime
    df["time_published"] = pd.to_datetime(
        df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce"
    )
    # Drop rows with failed datetime parsing
    df = df.dropna(subset=["time_published"]).reset_index(drop=True)
    return df


def main():
    # 1. Read API Key
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Environment variable ALPHAVANTAGE_API_KEY not set. Please run:\n"
            'export ALPHAVANTAGE_API_KEY="your_api_key"'
        )

    # 2. Set time range: from 2023-01-01 to now (adjust as needed)
    start_dt = datetime(2023, 1, 1, 0, 0)
    end_dt = datetime.now()
    print(f"Fetching news from {start_dt} to {end_dt}")

    # 3. Configure tickers to fetch; start with single stocks, e.g., AAPL, GOOG
    tickers = ["AAPL", "GOOG"]

    all_dfs = []

    for ticker in tickers:
        df_ticker = fetch_articles_for_ticker(
            ticker=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
            api_key=api_key,
            limit=1000,      # up to 1000 per request
            sleep_sec=12,    # avoid rate limit; increase if needed
        )
        if not df_ticker.empty:
            all_dfs.append(df_ticker)

    if not all_dfs:
        raise ValueError("No news fetched. Check API key or quota limits.")

    news_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 4. Save to data/raw/news.csv
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / "news.csv"
    news_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(news_df)} news rows to {output_path}")


if __name__ == "__main__":
    main()
