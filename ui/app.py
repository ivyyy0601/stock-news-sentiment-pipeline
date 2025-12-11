from pathlib import Path
import os
import streamlit as st

from components import (
    get_project_dir,
    inject_css,
    header,
    sidebar_menu,
    sidebar_filters,
    filter_features,
    load_metrics,
    load_features,
    metrics_leaderboard,
    feature_timeseries,
    comparison_plots,
    model_artifacts,
)


PROJECT_DIR = get_project_dir()
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
PICS_DIR = PROJECT_DIR / "pics"


def main():
    st.set_page_config(page_title="AI Trader – Stock Price Prediction", page_icon="📈", layout="wide")
    inject_css(Path(__file__).resolve().parent / "styles.css")

    # Animated title with requested palette and branding
    header(
        "AI Trader for Stock Price Prediction",
        "Explore cleaned data, engineered features, and model results",
    )

    view = sidebar_menu()

    if view == "Data":
        # Data view: show cleaned datasets with filters
        st.subheader("Cleaned Data")
        news_path = DATA_DIR / "clean" / "news_clean.csv"
        stocks_path = DATA_DIR / "clean" / "stocks_clean.csv"
        tabs = st.tabs(["News", "Stocks", "Features"])
        # News
        with tabs[0]:
            if news_path.exists():
                import pandas as pd
                news_df = pd.read_csv(news_path)
                st.dataframe(news_df, use_container_width=True)
            else:
                st.info("news_clean.csv not found.")
        # Stocks
        with tabs[1]:
            if stocks_path.exists():
                import pandas as pd
                stocks_df = pd.read_csv(stocks_path)
                st.dataframe(stocks_df, use_container_width=True)
            else:
                st.info("stocks_clean.csv not found.")
        # Features with filters
        with tabs[2]:
            features_df = load_features(DATA_DIR)
            tickers, date_range = sidebar_filters(features_df)
            filtered_df = filter_features(features_df, tickers, date_range)
            st.dataframe(filtered_df, use_container_width=True)
            feature_timeseries(filtered_df)

    elif view == "Results":
        # Results view: show leaderboard and plots
        metrics_df = load_metrics(OUTPUTS_DIR)
        metrics_leaderboard(metrics_df)
        comparison_plots(PICS_DIR, OUTPUTS_DIR)
        # Model artifacts hidden per request


if __name__ == "__main__":
    main()
