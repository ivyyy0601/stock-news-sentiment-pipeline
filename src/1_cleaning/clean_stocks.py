"""
clean_stocks.py

Purpose:
- Read data/raw/stocks.csv (wide-format export from yfinance with multiple tickers)
- Parse two-level headers (price type / ticker)
- Convert to long format: date, ticker, open, high, low, close, adj_close, volume
- Drop duplicates and sort by date
- Save to data/clean/stocks_clean.csv
"""

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    # Project root directory
    return Path(__file__).resolve().parents[2]


def main():
    project_root = get_project_root()
    raw_path = project_root / "data" / "raw" / "stocks.csv"
    clean_dir = project_root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_path = clean_dir / "stocks_clean.csv"

    # 1. Read CSV with two-level headers
    #   header=[0,1] means first two rows are header
    #   index_col=0 uses the date column as index
    print(f"Reading raw stocks from {raw_path}")
    df_raw = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True)

    # 2. Normalize multi-index columns
    # Level 0: price type (Adj Close / Close / High / Low / Open / Volume)
    # Level 1: ticker (AAPL / GOOG)
    # stack(level=1) pivots the ticker level to rows (long format)
    df_long = df_raw.stack(level=1).reset_index()

    # Now columns are like: ["Date", "level_1", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
    # Rename level_1 to ticker and standardize names
    df_long = df_long.rename(
        columns={
            df_long.columns[0]: "date",    # Date
            "level_1": "ticker",
            "Adj Close": "adj_close",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Open": "open",
            "Volume": "volume",
        }
    )

    # 3. Keep necessary columns and sort
    keep_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    df_long = df_long[keep_cols].copy()

    # Drop obvious missing values (e.g., rows without close price)
    df_long = df_long.dropna(subset=["close"]).reset_index(drop=True)

    # Sort
    df_long = df_long.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 4. Save cleaned file
    df_long.to_csv(output_path, index=False)
    print(f"Saved cleaned stocks to {output_path}")
    print(df_long.head())


if __name__ == "__main__":
    main()
