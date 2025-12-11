"""
make_dataset_news.py  — 使用 Alpha Vantage NEWS_SENTIMENT 拉多年的新闻

功能：
- 对多个 ticker（如 AAPL, GOOG）从指定起始日期开始，沿时间轴不断向后抓新闻
- 自动更新 time_from，直到接近当前时间或 API 不再返回数据
- 保存为 data/raw/news.csv，后续用于清洗、FinBERT 情绪分析、特征工程

注意：
- 需要环境变量 ALPHAVANTAGE_API_KEY
  在终端中先执行：
    export ALPHAVANTAGE_API_KEY="你的_api_key"
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


def get_project_root() -> Path:
    # 假设当前文件在 TRY/src/0_data/make_dataset_news.py
    # parents[2] 就是项目根目录 TRY/
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
    使用 Alpha Vantage NEWS_SENTIMENT 接口，沿时间轴从 start_dt 拉到 end_dt 附近。
    - ticker: 例如 "AAPL", "GOOG"
    - start_dt: 起始时间（datetime）
    - end_dt: 结束时间（datetime）
    - limit: 每次请求最多返回多少条（1-1000）
    - sleep_sec: 为了避免频率限制，两次请求之间的等待秒数

    返回：
    - 包含多行新闻记录的 DataFrame，每行包括：
      ticker, time_published, title, summary, overall_sentiment_score, overall_sentiment_label
    """
    url = "https://www.alphavantage.co/query"
    all_records = []

    current_start = start_dt

    while True:
        # 如果已经超过结束时间，就退出
        if current_start >= end_dt:
            print(f"[{ticker}] Reached end_dt {end_dt}, stop fetching.")
            break

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": current_start.strftime("%Y%m%dT%H%M"),
            "sort": "EARLIEST",  # 从最早的新闻往后给
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

        # 频率限制或其它提示信息
        if "Information" in data:
            print(f"[{ticker}] Information from API: {data['Information']}")
            break
        if "Note" in data:
            print(f"[{ticker}] Note from API (rate limit?): {data['Note']}")
            break

        # 如果 items == '0'，说明没有更多新闻可拉
        if data.get("items") == "0":
            print(f"[{ticker}] No more articles to extract (items=0).")
            break

        feed = data.get("feed")
        if not feed:
            print(f"[{ticker}] No 'feed' field in response, raw data: {data}")
            break

        # 解析这批新闻，并记录最后一条新闻的时间
        last_time = None

        for item in feed:
            time_published = item.get("time_published")  # 格式类似 '20240507T022200'
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

            # 转成 datetime，用于下一轮的起点
            if time_published:
                try:
                    last_time = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                except ValueError:
                    # 万一格式怪，就跳过，不更新 last_time
                    pass

        if last_time is None:
            print(f"[{ticker}] Could not parse any valid time_published, stop.")
            break

        print(f"[{ticker}] Got {len(feed)} articles, last_time = {last_time}")

        # 如果 last_time 已经接近或超过 end_dt，就可以停止了
        if last_time >= end_dt:
            print(f"[{ticker}] last_time >= end_dt, stop fetching.")
            break

        # 更新下一轮的起点：last_time + 1 分钟
        # 避免重复抓到同一条新闻
        current_start = last_time + timedelta(minutes=1)

        # 适当等待，避免触发 API 频率限制
        print(f"[{ticker}] Sleeping {sleep_sec} seconds before next request ...")
        time.sleep(sleep_sec)

    if not all_records:
        print(f"[{ticker}] No records fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # 把 time_published 转成 datetime
    df["time_published"] = pd.to_datetime(
        df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce"
    )
    # 丢掉解析失败的
    df = df.dropna(subset=["time_published"]).reset_index(drop=True)
    return df


def main():
    # 1. 读取 API Key
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "环境变量 ALPHAVANTAGE_API_KEY 未设置，请先在终端中执行：\n"
            'export ALPHAVANTAGE_API_KEY="你的_api_key"'
        )

    # 2. 设置时间范围：从 2023-01-01 拉到“现在”
    #    你可以根据需要调整起始时间
    start_dt = datetime(2023, 1, 1, 0, 0)
    end_dt = datetime.now()
    print(f"Fetching news from {start_dt} to {end_dt}")

    # 3. 配置要拉哪些标的的新闻
    #    你之前 stock 那边有 AAPL / 指数，可以先从个股开始，比如 AAPL, GOOG
    tickers = ["AAPL", "GOOG"]

    all_dfs = []

    for ticker in tickers:
        df_ticker = fetch_articles_for_ticker(
            ticker=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
            api_key=api_key,
            limit=1000,      # 每次最多 1000 条
            sleep_sec=12,    # 避免触发 rate limit，必要时可以再调大
        )
        if not df_ticker.empty:
            all_dfs.append(df_ticker)

    if not all_dfs:
        raise ValueError("没有成功获取到任何新闻，请检查 API Key 或配额限制。")

    news_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 4. 保存到 data/raw/news.csv
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / "news.csv"
    news_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(news_df)} news rows to {output_path}")


if __name__ == "__main__":
    main()
