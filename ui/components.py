from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import streamlit as st
import plotly.express as px


def get_project_dir() -> Path:
    return Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()


def inject_css(css_path: Path):
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def header(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div class="section-header fadein">
            <div class="section-accent"></div>
            <div>
                <div class="typing-title">{title}</div>
                {f'<div class="subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_metrics(outputs_dir: Path) -> pd.DataFrame:
    metrics_path = outputs_dir / "metrics.csv"
    if metrics_path.exists():
        try:
            return pd.read_csv(metrics_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_features(data_dir: Path) -> pd.DataFrame:
    merged_path = data_dir / "features" / "merged_features.csv"
    if merged_path.exists():
        try:
            return pd.read_csv(merged_path)
        except Exception:
            pass
    # fallback to separate features
    stock_f = data_dir / "features" / "stock_features.csv"
    news_f = data_dir / "features" / "news_features.csv"
    parts = []
    if stock_f.exists():
        parts.append(pd.read_csv(stock_f))
    if news_f.exists():
        parts.append(pd.read_csv(news_f))
    if parts:
        try:
            return parts[0].merge(parts[1], how="inner") if len(parts) == 2 else parts[0]
        except Exception:
            return parts[0]
    return pd.DataFrame()


def sidebar_menu() -> str:
    st.sidebar.title("AI Trader")
    choice = st.sidebar.radio("View", ["Data", "Results"], index=0)
    return choice


def sidebar_filters(features_df: pd.DataFrame):
    st.sidebar.header("Filters")
    tickers = sorted(set(features_df.get("ticker", []))) if not features_df.empty else []
    selected_tickers = st.sidebar.multiselect("Tickers", tickers, default=tickers[:1] if tickers else [])
    date_col = "date"
    min_date = max_date = None
    if date_col in features_df.columns:
        try:
            features_df[date_col] = pd.to_datetime(features_df[date_col])
            min_date = features_df[date_col].min()
            max_date = features_df[date_col].max()
        except Exception:
            pass
    date_range = None
    if min_date is not None and max_date is not None:
        date_range = st.sidebar.slider(
            "Date Range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        )
    return selected_tickers, date_range


def filter_features(df: pd.DataFrame, tickers, date_range):
    if df.empty:
        return df
    out = df.copy()
    if tickers and "ticker" in out.columns:
        out = out[out["ticker"].isin(tickers)]
    if date_range and "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[(out["date"] >= date_range[0]) & (out["date"] <= date_range[1])]
    return out


def metrics_leaderboard(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        st.info("No metrics.csv found in outputs/ yet.")
        return
    header("Model Leaderboard", "Compare performance across models")
    metric_cols = [c for c in ["rmse", "mae", "mape", "accuracy"] if c in metrics_df.columns]
    sort_metric = st.selectbox("Sort by", options=metric_cols or metrics_df.columns.tolist(), index=0)
    ascending = sort_metric.lower() in ["accuracy"]
    ranked = metrics_df.sort_values(by=sort_metric, ascending=ascending)
    st.dataframe(ranked, use_container_width=True)
    if "model" in ranked.columns and sort_metric in ranked.columns:
        fig = px.bar(ranked, x="model", y=sort_metric, color="model", title=f"Model {sort_metric.upper()} Comparison")
        fig.update_layout(transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)


def feature_timeseries(features_df: pd.DataFrame):
    if features_df.empty:
        return
    header("Feature Trends", "Explore engineered features over time")
    numeric_cols = features_df.select_dtypes(include="number").columns.tolist()
    exclude = {"target", "label"}
    options = [c for c in numeric_cols if c not in exclude]
    if not options:
        return
    y_col = st.selectbox("Feature to plot", options=options, index=0)
    if "date" in features_df.columns:
        dfp = features_df.copy()
        dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
        fig = px.line(
            dfp.sort_values("date"),
            x="date",
            y=y_col,
            color=dfp.get("ticker") if "ticker" in dfp.columns else None,
            title=f"{y_col} over time",
        )
        fig.update_layout(transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)


def comparison_plots(pics_dir: Path, outputs_dir: Path):
    header("Comparison Plots", "Visual summaries of model results")
    # Show only pics1 and pics2 from pics/ as requested
    targets = [pics_dir / "pics1.png", pics_dir / "pics2.png"]
    existing = [p for p in targets if p.exists()]
    if not existing:
        st.info("pics1.png or pics2.png not found under pics/.")
        return
    cols = st.columns([1, 1])
    for i, img in enumerate(existing):
        with cols[i % 2]:
            st.image(str(img), caption=img.name, width=800)


def model_artifacts(outputs_dir: Path):
    header("Model Artifacts", "Download trained models")
    artifacts = []
    if outputs_dir.exists():
        artifacts = sorted([p for p in outputs_dir.glob("*.*") if p.suffix in (".h5", ".json")])
    if not artifacts:
        st.info("No model artifacts found in outputs/ yet.")
        return
    for art in artifacts:
        with open(art, "rb") as f:
            st.download_button(label=f"Download {art.name}", data=f, file_name=art.name)
