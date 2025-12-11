from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator

# ======== 项目路径 & Python 路径 ========
# Prefer env var PROJECT_DIR; fallback to repository root relative to this file.
# airflow_home/dags/<this_file> → project root is two levels up.
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
PYTHON_BIN = sys.executable  # 当前 venv 里的 python


def run_script(script_rel_path: str) -> None:
    """
    在 PROJECT_DIR 下，用当前虚拟环境的 python 运行一个脚本。
    script_rel_path 例如: 'src/0_data/make_dataset_news.py'
    """
    script_path = PROJECT_DIR / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("=" * 80)
    print(f"▶ Running script: {script_path}")
    print("=" * 80)

    # 传递当前环境变量到子进程；不硬编码 API key，改为读取环境变量
    env = os.environ.copy()
    env["ALPHAVANTAGE_API_KEY"] = "ZDOU31JUM67TI5K0"  # <<< 这里换成你的真实 key

    result = subprocess.run(
        [PYTHON_BIN, str(script_path)],
        cwd=PROJECT_DIR,   # 在项目根目录下执行，方便脚本用相对路径
        check=True,
        env=env,
    )

    print(f"✅ Finished {script_rel_path}, return code = {result.returncode}")


# ======== 默认参数 ========
default_args = {
    "owner": "ivy",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# ======== DAG 定义 ========
with DAG(
    dag_id="stock_news_sentiment_pipeline",
    description="End-to-end pipeline: data -> cleaning -> features -> models -> analysis",
    default_args=default_args,
    start_date=datetime(2025, 12, 1),
    schedule=None,      # 手动触发；以后要定时可以改成 '0 8 * * *' 之类
    catchup=False,
) as dag:

    # 0️⃣ 原始数据构建
    make_dataset_stocks = PythonOperator(
        task_id="make_dataset_stocks",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/0_data/make_dataset_stocks.py"},
    )

    make_dataset_news = PythonOperator(
        task_id="make_dataset_news",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/0_data/make_dataset_news.py"},
    )

    # 1️⃣ 清洗
    clean_news = PythonOperator(
        task_id="clean_news",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/1_cleaning/clean_news.py"},
    )

    clean_stocks = PythonOperator(
        task_id="clean_stocks",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/1_cleaning/clean_stocks.py"},
    )

    # 2️⃣ 特征工程
    build_news_features = PythonOperator(
        task_id="build_news_features",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/2_features/build_news_features.py"},
    )

    build_stock_features = PythonOperator(
        task_id="build_stock_features",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/2_features/build_stock_features.py"},
    )

    # 2.5️⃣ 准备最终训练用的数据
    prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/3_models/00_prepare_data.py"},
    )

    # 3️⃣ 六个模型（逻辑上并行；默认 SequentialExecutor 会一个个跑）
    lstm_price_only = PythonOperator(
        task_id="lstm_price_only",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/3_models/01_lstm_price_only.py"},
    )

    gru_price_only = PythonOperator(
        task_id="gru_price_only",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/3_models/02_gru_price_only.py"},
    )

    att_cnn_lstm_price_only = PythonOperator(
        task_id="att_cnn_lstm_price_only",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/3_models/03_att_cnn_lstm_price_only.py"},
    )

    lstm_with_sentiment = PythonOperator(
        task_id="lstm_with_sentiment",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/3_models/04_lstm_with_sentiment.py"},
    )

    gru_with_sentiment = PythonOperator(
        task_id="gru_with_sentiment",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/3_models/05_gru_with_sentiment.py"},
    )

    att_cnn_lstm_with_sentiment = PythonOperator(
        task_id="att_cnn_lstm_with_sentiment",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/3_models/06_att_cnn_lstm_with_sentiment.py"},
    )

    model_tasks = [
        lstm_price_only,
        gru_price_only,
        att_cnn_lstm_price_only,
        lstm_with_sentiment,
        gru_with_sentiment,
        att_cnn_lstm_with_sentiment,
    ]

    # 4️⃣ 对比和画图
    compare_models = PythonOperator(
        task_id="compare_models",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/4_analysis/compare_models.py"},
    )

    plot_results = PythonOperator(
        task_id="plot_results",
        python_callable=run_script,
        op_kwargs={"script_rel_path": "src/4_analysis/plot_results.py"},
    )

    # ======== 依赖关系 ========

    # 股票数据链：0_data -> cleaning -> features -> prepare
    make_dataset_stocks >> clean_stocks >> build_stock_features >> prepare_data

    # 新闻数据链：0_data -> cleaning -> features -> prepare
    make_dataset_news >> clean_news >> build_news_features >> prepare_data

    # 准备好训练数据后，跑 6 个模型
    prepare_data >> model_tasks

    # 6 个模型跑完，再做对比和画图
    model_tasks >> compare_models >> plot_results
