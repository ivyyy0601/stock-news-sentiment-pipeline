import pandas as pd

metrics_path = "outputs/metrics.csv"
df = pd.read_csv(metrics_path)

# === 📌 All Model Performance  ===
print("\n=== 📌 All Model Performance ===")
print(df)

# === Ranked by RMSE (Lower is Better)===
print("\n=== 🏆 Ranked by RMSE (Lower is Better) ===")
print(df.sort_values("rmse"))

# === 🏆 Ranked by MAE (Lower is Better) ===
print("\n=== 🏆 Ranked by MAE (Lower is Better) ===")
print(df.sort_values("mae"))

# === 🧠 Do sentiment features improve prediction? ===
print("\n=== 🧠 Did Sentiment Features Improve Prediction? ===")
mean_no_sent = df[df.use_sentiment == 0][["rmse", "mae"]].mean()
mean_with_sent = df[df.use_sentiment == 1][["rmse", "mae"]].mean()

print("🔹 No Sentiment Features:", mean_no_sent)
print("🔹 With Sentiment Features:", mean_with_sent)