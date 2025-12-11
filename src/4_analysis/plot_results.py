import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/metrics.csv")

# --- RMSE plot ---
plt.figure(figsize=(10,5))
plt.bar(df.model_name, df.rmse)
plt.xticks(rotation=45)
plt.title("RMSE Comparison of Models")
plt.tight_layout()
plt.savefig("outputs/rmse_comparison.png")
plt.close()

# --- MAE plot ---
plt.figure(figsize=(10,5))
plt.bar(df.model_name, df.mae, color="orange")
plt.xticks(rotation=45)
plt.title("MAE Comparison of Models")
plt.tight_layout()
plt.savefig("outputs/mae_comparison.png")
plt.close()

# --- Sentiment vs No Sentiment ---
groups = df.groupby("use_sentiment")[["rmse", "mae"]].mean()

groups.plot(kind="bar", figsize=(8,5), title="Sentiment Feature Impact")
plt.xticks([0,1], ["No Sentiment", "With Sentiment"], rotation=0)
plt.tight_layout()
plt.savefig("outputs/sentiment_impact.png")
plt.close()
