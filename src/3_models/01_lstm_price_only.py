"""
01_lstm_price_only.py

功能：
- 从 merged_features.csv 读取数据
- 只使用“价格相关特征”作为输入 X（不含情绪）
- 做时间序列切片（LSTM 输入：样本数 × lookback × 特征数）
- 划分 train / val / test
- 训练 LSTM 模型，评估 RMSE
- 保存：
    - 模型：outputs/lstm_price_only.h5
    - 指标：outputs/metrics.csv（追加一行）
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# ====== 基础工具 ======

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def create_sequences(X, y, lookback: int):
    """
    把按时间排序好的 2D 特征 + 1D 目标，切成 LSTM 需要的 3D 序列。
    X: (N, num_features)
    y: (N,)
    return:
        X_seq: (N-lookback, lookback, num_features)
        y_seq: (N-lookback,)
    """
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


# ====== 主流程 ======

def main():
    project_root = get_project_root()
    data_path = project_root / "data" / "features" / "merged_features.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path = outputs_dir / "lstm_price_only.h5"
    metrics_path = outputs_dir / "metrics.csv"

    print(f"📥 Reading merged features from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 1. 选择特征列（只用价格 / 技术相关，不含 sentiment）
    # 你可以根据自己在 stock_features 里生成的列名调整一下
    price_feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "return_1d",
        "return_lag_1",
        "return_lag_3",
        "return_lag_7",
        "roll_mean_5",
        "roll_std_5",
        "log_price",
    ]

    # 确保这些列都在 df 里
    price_feature_cols = [c for c in price_feature_cols if c in df.columns]
    print("✅ 使用的 Price 特征列：", price_feature_cols)

    # 目标变量
    target_col = "target_return_1d"

    # 2. 按日期排序（已经含有多个 ticker，一起按时间排）
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # 3. 取出特征和目标
    X_all = df[price_feature_cols].values.astype(float)
    y_all = df[target_col].values.astype(float)

    # 4. Train / Val / Test 按时间比例切分（例如 70% / 15% / 15%）
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, y_train_raw = X_all[:train_end], y_all[:train_end]
    X_val_raw, y_val_raw = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test_raw, y_test_raw = X_all[val_end:], y_all[val_end:]

    print(f"📊 样本数：train={len(X_train_raw)}, val={len(X_val_raw)}, test={len(X_test_raw)}")

    # 5. 对特征标准化（只在 train 上 fit）
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 6. 构造时间序列切片
    lookback = 20  # 用过去 20 天的数据预测下一天

    X_train_seq, y_train = create_sequences(X_train_scaled, y_train_raw, lookback)
    X_val_seq, y_val = create_sequences(X_val_scaled, y_val_raw, lookback)
    X_test_seq, y_test = create_sequences(X_test_scaled, y_test_raw, lookback)

    print("📐 LSTM 输入维度：", X_train_seq.shape)

    # 7. 定义 LSTM 模型（价格-only）
    num_features = X_train_seq.shape[-1]

    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, num_features)),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),  # 回归
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

    # 8. 训练
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_seq,
        y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val_seq, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 9. 在 test 集上评估 RMSE / MAE
    y_pred = model.predict(X_test_seq).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✅ Test RMSE = {rmse:.6f}, MAE = {mae:.6f}")

    # 10. 保存模型
    model.save(model_path)
    print(f"💾 模型已保存到 {model_path}")

    # 11. 记录 metrics.csv（追加模式）
    row = {
        "model_name": "lstm_price_only",
        "use_sentiment": 0,
        "rmse": rmse,
        "mae": mae,
    }

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
    else:
        metrics_df = pd.DataFrame([row])

    metrics_df.to_csv(metrics_path, index=False)
    print(f"📈 指标已写入 {metrics_path}")
    print(metrics_df.tail())


if __name__ == "__main__":
    main()
