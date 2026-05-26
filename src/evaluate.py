import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("reports/plots", exist_ok=True)

# ==========================================
# Load scores and pick best model
# ==========================================
with open("artifacts/metrics/model_scores.json") as f:
    scores = json.load(f)

best = max(scores, key=lambda k: scores[k]["r2"])

with open("artifacts/metrics/best_model.json", "w") as f:
    json.dump({
        "best_model": best,
        "r2":   scores[best]["r2"],
        "rmse": scores[best]["rmse"]
    }, f, indent=2)

print("=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)
for model_name, metrics in scores.items():
    marker = " <-- BEST" if model_name == best else ""
    print(f"  {model_name.upper():6s}  |  R2: {metrics['r2']:.4f}  |  RMSE: {metrics['rmse']:.4f}{marker}")
print("=" * 50)

# ==========================================
# Load data for all three models
# ==========================================
X_test_flat = np.load("artifacts/data/x_test.npy")
y_test_flat = np.load("artifacts/data/y_test.npy").ravel()

X_test_seq  = np.load("artifacts/data/x_test_seq.npy")
y_test_seq  = np.load("artifacts/data/y_test_seq.npy").ravel()

# ==========================================
# Load all models
# ==========================================
rf   = pickle.load(open("artifacts/models/rf_model.pkl", "rb"))
mlp  = pickle.load(open("artifacts/models/mlp_model.pkl", "rb"))
lstm = load_model("artifacts/models/lstm_model.keras")

rf_preds   = rf.predict(X_test_flat)
mlp_preds  = mlp.predict(X_test_flat)
lstm_preds = lstm.predict(X_test_seq).ravel()

# ==========================================
# Plot helper
# ==========================================
def plot_actual_vs_predicted(y_true, y_pred, model_name, r2, rmse, filename):
    """
    Two-panel plot:
      Left  — scatter of actual vs predicted with perfect-fit line
      Right — line plot of first 200 samples for visual comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model_name} — Actual vs Predicted\nR² = {r2:.4f}  |  RMSE = {rmse:.4f}",
        fontsize=13, fontweight="bold"
    )

    # Scatter
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue", label="Predictions")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual (normalised)")
    ax.set_ylabel("Predicted (normalised)")
    ax.set_title("Scatter: Actual vs Predicted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Time-series line (first 200 samples)
    ax2 = axes[1]
    n = min(200, len(y_true))
    ax2.plot(y_true[:n], label="Actual",    color="steelblue", linewidth=1.2)
    ax2.plot(y_pred[:n], label="Predicted", color="orange",    linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Power consumption (normalised)")
    ax2.set_title(f"First {n} Samples")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

# ==========================================
# Individual plots for all 3 models
# ==========================================
print("\nGenerating Actual vs Predicted plots...")

plot_actual_vs_predicted(
    y_test_flat, rf_preds,
    model_name="Random Forest",
    r2=scores["rf"]["r2"], rmse=scores["rf"]["rmse"],
    filename="reports/plots/rf_actual_vs_predicted.png"
)

plot_actual_vs_predicted(
    y_test_flat, mlp_preds,
    model_name="MLP",
    r2=scores["mlp"]["r2"], rmse=scores["mlp"]["rmse"],
    filename="reports/plots/mlp_actual_vs_predicted.png"
)

plot_actual_vs_predicted(
    y_test_seq, lstm_preds,
    model_name="LSTM",
    r2=scores["lstm"]["r2"], rmse=scores["lstm"]["rmse"],
    filename="reports/plots/lstm_actual_vs_predicted.png"
)

# ==========================================
# Combined 3-panel comparison
# ==========================================
print("  Generating combined comparison plot...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison — Actual vs Predicted (Scatter)", fontsize=13, fontweight="bold")

model_data = [
    ("Random Forest", y_test_flat, rf_preds,   "steelblue", "rf"),
    ("MLP",           y_test_flat, mlp_preds,  "seagreen",  "mlp"),
    ("LSTM",          y_test_seq,  lstm_preds, "darkorange","lstm"),
]

for ax, (name, y_true, y_pred, color, key) in zip(axes, model_data):
    r2   = scores[key]["r2"]
    rmse = scores[key]["rmse"]
    ax.scatter(y_true, y_pred, alpha=0.3, s=6, color=color)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_title(f"{name}\nR²={r2:.4f}  RMSE={rmse:.4f}", fontsize=10)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/plots/all_models_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: reports/plots/all_models_comparison.png")

print("=" * 50)
print(f"All plots saved to reports/plots/")
print(f"Best model: {best.upper()}")
