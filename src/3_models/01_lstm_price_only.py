"""
01_lstm_price_only.py

Purpose:
- Read merged_features.csv
- Use price-related features only (no sentiment) as input X
- Build time-series sequences (LSTM input: samples × lookback × features)
- Split train / val / test
- Train LSTM and evaluate RMSE/MAE
- Save:
    - Model: outputs/lstm_price_only.h5
    - Metrics: outputs/metrics.csv (append a row)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# ====== Utilities ======

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def create_sequences(X, y, lookback: int):
    """
    Slice time-ordered 2D features + 1D targets into 3D sequences required by LSTM.
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


# ====== Main flow ======

def main():
    project_root = get_project_root()
    data_path = project_root / "data" / "features" / "merged_features.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path = outputs_dir / "lstm_price_only.h5"
    metrics_path = outputs_dir / "metrics.csv"

    print(f"📥 Reading merged features from {data_path}")
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 1. Select feature columns (price/technical only, no sentiment)
    # Adjust based on columns generated in stock_features if needed
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

    # Ensure these columns exist in df
    price_feature_cols = [c for c in price_feature_cols if c in df.columns]
    print("Using price feature columns:", price_feature_cols)

    # Target variable
    target_col = "target_return_1d"

    # 2. Sort by date (multiple tickers included; keep temporal order)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # 3. Extract features and target
    X_all = df[price_feature_cols].values.astype(float)
    y_all = df[target_col].values.astype(float)

    # 4. Train/Val/Test split by time proportion (e.g., 70%/15%/15%)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, y_train_raw = X_all[:train_end], y_all[:train_end]
    X_val_raw, y_val_raw = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test_raw, y_test_raw = X_all[val_end:], y_all[val_end:]

    print(f"Samples: train={len(X_train_raw)}, val={len(X_val_raw)}, test={len(X_test_raw)}")

    # 5. Standardize features (fit on train only)
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 6. Build time-series slices
    lookback = 20  # use past 20 days to predict next day

    X_train_seq, y_train = create_sequences(X_train_scaled, y_train_raw, lookback)
    X_val_seq, y_val = create_sequences(X_val_scaled, y_val_raw, lookback)
    X_test_seq, y_test = create_sequences(X_test_scaled, y_test_raw, lookback)

    print("LSTM input shape:", X_train_seq.shape)

    # 7. Define LSTM model (price-only)
    num_features = X_train_seq.shape[-1]

    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, num_features)),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="linear"),  # regression
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )

    model.summary()

    # 8. Train
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

    # 9. Evaluate RMSE / MAE on test set
    y_pred = model.predict(X_test_seq).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test RMSE = {rmse:.6f}, MAE = {mae:.6f}")

    # 10. Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    # 11. Log metrics.csv (append mode)
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
    print(f"Metrics written to {metrics_path}")
    print(metrics_df.tail())


if __name__ == "__main__":
    main()
