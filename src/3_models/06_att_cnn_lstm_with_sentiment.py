"""
06_att_cnn_lstm_with_sentiment.py

功能：
- 使用 价格特征 + 情绪特征
- 模型：Conv1D → LSTM → Attention → Dense
- 训练并把结果追加写入 outputs/metrics.csv
- 保存模型为 outputs/att_cnn_lstm_with_sentiment.h5
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K # 👈 添加了 Keras 后端导入


def get_project_root() -> Path:
    # 这个脚本在 src/3_models/ 下面，往上两层就是项目根目录
    return Path(__file__).resolve().parents[2]


def create_sequences(X, y, lookback: int):
    """把二维特征变成 (samples, timesteps, features) 的序列数据"""
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i : i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


def attention_block(inputs):
    """
    简单 Attention 块：
    inputs: (batch, timesteps, features)
    """
    score = layers.Dense(1, activation="tanh")(inputs)   # (batch, T, 1)
    weights = layers.Softmax(axis=1)(score)              # (batch, T, 1)
    context = layers.Multiply()([inputs, weights])       # (batch, T, F)
    # 🌟 修复：使用 Lambda 层封装 K.sum 来代替 tf.reduce_sum
    context = layers.Lambda(lambda x: K.sum(x, axis=1))(context) # (batch, F)
    return context


def main():
    project_root = get_project_root()
    data_path = project_root / "data" / "features" / "merged_features.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 路径已更新
    model_path = outputs_dir / "att_cnn_lstm_with_sentiment.h5"
    metrics_path = outputs_dir / "metrics.csv"

    print(f"📥 Reading merged features from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 价格特征
    price_cols = [
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
    price_cols = [c for c in price_cols if c in df.columns]

    # 情绪特征 (已包含)
    sentiment_cols = [
        "sentiment_mean",
        "sentiment_max",
        "sentiment_min",
        "sentiment_count",
        "sentiment_index",
    ]
    sentiment_cols = [c for c in sentiment_cols if c in df.columns]

    feature_cols = price_cols + sentiment_cols
    target_col = "target_return_1d"

    print("✅ 使用的特征列（Price + Sentiment）：", feature_cols)

    # 按时间排序，防止乱序
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    X_all = df[feature_cols].values.astype(float)
    y_all = df[target_col].values.astype(float)

    # 时间切分：70% 训练，15% 验证，15% 测试
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, y_train_raw = X_all[:train_end], y_all[:train_end]
    X_val_raw, y_val_raw = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test_raw, y_test_raw = X_all[val_end:], y_all[val_end:]

    print(f"📊 样本数：train={len(X_train_raw)}, val={len(X_val_raw)}, test={len(X_test_raw)}")

    # 标准化（按训练集 fit）
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 生成序列
    lookback = 20
    X_train_seq, y_train = create_sequences(X_train_scaled, y_train_raw, lookback)
    X_val_seq, y_val = create_sequences(X_val_scaled, y_val_raw, lookback)
    X_test_seq, y_test = create_sequences(X_test_scaled, y_test_raw, lookback)

    num_features = X_train_seq.shape[-1]

    # 🟣 CNN + LSTM + Attention + Sentiment
    inputs = layers.Input(shape=(lookback, num_features))
    x = layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = attention_block(x)                 # (batch, features)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

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

    # 测试集评估
    y_pred = model.predict(X_test_seq).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✅ Att-CNN-LSTM + Sentiment Test RMSE = {rmse:.6f}, MAE = {mae:.6f}")

    # 保存模型
    model.save(model_path)
    print(f"💾 模型已保存到 {model_path}")

    # 记录指标
    row = {
        "model_name": "att_cnn_lstm_with_sentiment",
        "use_sentiment": 1, # 👈 已更新为 1
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