# src/0_data/make_dataset_stocks.py

"""
功能：
- 从 Yahoo Finance 拉取一个或多个股票 / 指数的历史行情数据
- 保存到 data/raw/stocks.csv，供后续清洗和特征工程使用
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


def get_project_root() -> Path:
    """
    返回项目根目录路径：
    假设当前文件在 TRY/src/0_data/make_dataset_stocks.py
    parents[0] = src/0_data
    parents[1] = src
    parents[2] = TRY  ← 我们要的 project root
    """
    return Path(__file__).resolve().parents[2]


def fetch_stock_history(tickers, start_date, end_date, interval="1d") -> pd.DataFrame:
    """
    从 Yahoo Finance 下载多个 ticker 的历史数据，并合并成一个 DataFrame。
    每行包含：date, Ticker, open, high, low, close, adj_close, volume
    """
    all_data = []

    for ticker in tickers:
        print(f"Downloading {ticker} ...")
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            print(f"Warning: no data for {ticker}")
            continue

        # index 中的日期变成一列
        data = data.reset_index()
        data["Ticker"] = ticker
        all_data.append(data)

    if not all_data:
        raise ValueError("No data downloaded for any ticker.")

    df = pd.concat(all_data, axis=0, ignore_index=True)

    # 统一列名
    df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # 按 ticker + date 排序
    df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
    return df


def main():
    # 1. 配置部分：你可以根据需要修改
    tickers = ["AAPL", "GOOG"]  
    start_date = "2023-01-01"
    end_date   = "2025-12-09"
    interval = "1d"  # 日频数据；如果以后要调成 1h / 30m 也可以

    # 2. 下载数据
    df = fetch_stock_history(tickers, start_date, end_date, interval=interval)

    # 3. 保存到 data/raw/stocks.csv
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / "stocks.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
