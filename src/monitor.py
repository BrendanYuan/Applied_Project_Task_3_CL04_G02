import os
import json
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error
from scipy import stats
from tensorflow.keras.models import load_model
import pickle

os.makedirs("reports", exist_ok=True)

# ==========================================
# Config
# ==========================================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

monitor_cfg = params["monitor"]

# ==========================================
# Load best model info
# ==========================================
with open("artifacts/metrics/best_model.json") as f:
    best_model_info = json.load(f)

best_model    = best_model_info["best_model"]
baseline_r2   = best_model_info["r2"]
baseline_rmse = best_model_info["rmse"]

# ==========================================
# Load correct data based on model type
# ==========================================
if best_model == "lstm":
    X_train = np.load("artifacts/data/x_train_seq.npy")
    X_test  = np.load("artifacts/data/x_test_seq.npy")
    y_test  = np.load("artifacts/data/y_test_seq.npy")
    model   = load_model("artifacts/models/lstm_model.keras")
else:
    X_train = np.load("artifacts/data/x_train.npy")
    X_test  = np.load("artifacts/data/x_test.npy")
    y_test  = np.load("artifacts/data/y_test.npy")
    if best_model == "mlp":
        model = pickle.load(open("artifacts/models/mlp_model.pkl", "rb"))
    else:
        model = pickle.load(open("artifacts/models/rf_model.pkl", "rb"))

# ==========================================
# Predictions
# ==========================================
train_preds = model.predict(X_train).ravel()
test_preds  = model.predict(X_test).ravel()
y_test_flat = y_test.ravel()

# ==========================================
# KS Test — distribution drift
# ==========================================
ks_stat, ks_p   = stats.ks_2samp(train_preds, test_preds)
ks_threshold    = monitor_cfg["ks_p_threshold"]
drift_detected  = bool(ks_p < ks_threshold)

if drift_detected:
    ks_interpretation = (
        f"DRIFT DETECTED — prediction distribution has shifted significantly "
        f"(KS stat={ks_stat:.4f}, p={ks_p:.4f} < threshold {ks_threshold}). "
        f"The model may be responding differently to test data than training data."
    )
else:
    ks_interpretation = (
        f"No drift detected — prediction distributions are consistent "
        f"(KS stat={ks_stat:.4f}, p={ks_p:.4f} >= threshold {ks_threshold})."
    )

# ==========================================
# RMSE degradation check
# ==========================================
rmse        = float(np.sqrt(mean_squared_error(y_test_flat, test_preds)))
rmse_factor = monitor_cfg["rmse_drift_factor"]
rmse_limit  = baseline_rmse * rmse_factor
performance_degraded = bool(rmse > rmse_limit)
rmse_pct_change = ((rmse - baseline_rmse) / baseline_rmse) * 100

if performance_degraded:
    rmse_interpretation = (
        f"PERFORMANCE DEGRADED — current RMSE ({rmse:.4f}) exceeds "
        f"{rmse_factor}x the baseline RMSE ({baseline_rmse:.4f}), "
        f"limit was {rmse_limit:.4f}. "
        f"Change from baseline: {rmse_pct_change:+.1f}%."
    )
else:
    rmse_interpretation = (
        f"Performance is stable — current RMSE ({rmse:.4f}) is within "
        f"{rmse_factor}x of baseline RMSE ({baseline_rmse:.4f}), "
        f"limit is {rmse_limit:.4f}. "
        f"Change from baseline: {rmse_pct_change:+.1f}%."
    )

# ==========================================
# Overall health status
# ==========================================
if drift_detected and performance_degraded:
    overall_status = "CRITICAL — drift detected AND performance degraded. Retraining recommended."
elif drift_detected:
    overall_status = "WARNING — distribution drift detected but performance is still acceptable."
elif performance_degraded:
    overall_status = "WARNING — performance has degraded but no distribution drift detected."
else:
    overall_status = "HEALTHY — no drift detected and performance is stable."

# ==========================================
# Build interpretable report
# ==========================================
report = {
    "model": best_model,
    "overall_status": overall_status,
    "drift_detection": {
        "ks_statistic":   float(ks_stat),
        "ks_p_value":     float(ks_p),
        "ks_threshold":   ks_threshold,
        "drift_detected": drift_detected,
        "interpretation": ks_interpretation
    },
    "performance": {
        "current_rmse":                  rmse,
        "baseline_rmse":                 baseline_rmse,
        "rmse_limit":                    float(rmse_limit),
        "rmse_drift_factor":             rmse_factor,
        "rmse_pct_change_from_baseline": round(rmse_pct_change, 2),
        "performance_degraded":          performance_degraded,
        "interpretation":                rmse_interpretation
    },
    "baseline_info": {
        "baseline_r2":   baseline_r2,
        "baseline_rmse": baseline_rmse
    }
}

with open("reports/monitoring_report.json", "w") as f:
    json.dump(report, f, indent=2)

# ==========================================
print("=" * 50)
print("MONITORING REPORT")
print("=" * 50)
print(f"Model:          {best_model.upper()}")
print(f"Status:         {overall_status}")
print(f"KS Statistic:   {ks_stat:.4f}  |  p-value: {ks_p:.4f}  |  Drift: {drift_detected}")
print(f"RMSE:           {rmse:.4f}  |  Baseline: {baseline_rmse:.4f}  |  Degraded: {performance_degraded}")
print("=" * 50)
print("Report saved to reports/monitoring_report.json")
