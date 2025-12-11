# src/0_data/make_dataset_stocks.py

"""
Purpose:
- Fetch historical price data for one or more stocks/indices from Yahoo Finance
- Save to data/raw/stocks.csv for later cleaning and feature engineering
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


def get_project_root() -> Path:
    """
    Return project root path.
    """
    return Path(__file__).resolve().parents[2]


def fetch_stock_history(tickers, start_date, end_date, interval="1d") -> pd.DataFrame:
    """
    Download historical data for multiple tickers from Yahoo Finance and combine into a DataFrame.
    Each row contains: date, Ticker, open, high, low, close, adj_close, volume
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

        # index is DatetimeIndex; convert to column
        data = data.reset_index()
        data["Ticker"] = ticker
        all_data.append(data)

    if not all_data:
        raise ValueError("No data downloaded for any ticker.")

    df = pd.concat(all_data, axis=0, ignore_index=True)

    # Normalize column names
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

    # Sort by ticker + date
    df = df.sort_values(["Ticker", "date"]).reset_index(drop=True)
    return df


def main():
    # 1. Configuration: adjust as needed
    tickers = ["AAPL", "GOOG"]  
    start_date = "2023-01-01"
    end_date   = "2025-12-09"
    interval = "1d"  # daily frequency; can switch to 1h/30m later

    # 2. Download data
    df = fetch_stock_history(tickers, start_date, end_date, interval=interval)

    # 3. Save to data/raw/stocks.csv
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / "stocks.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
