import os
import json
import numpy as np
import yaml
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("artifacts/data", exist_ok=True)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# ==========================================
# Load flat split — RF and MLP
# ==========================================

X_train_flat = np.load("artifacts/data/x_train_flat.npy")
X_test_flat  = np.load("artifacts/data/x_test_flat.npy")
y_train_flat = np.load("artifacts/data/y_train_flat.npy")
y_test_flat  = np.load("artifacts/data/y_test_flat.npy")

# ==========================================
# Load sequential split — LSTM raw arrays
# create_sequences is applied below
# ==========================================

X_train_seq_raw = np.load("artifacts/data/x_train_seq.npy")
X_test_seq_raw  = np.load("artifacts/data/x_test_seq.npy")
y_train_seq_raw = np.load("artifacts/data/y_train_seq.npy")
y_test_seq_raw  = np.load("artifacts/data/y_test_seq.npy")

results = {}

# ==========================================
# Random Forest Regressor
# ==========================================
print("=" * 50)
print("Training: Random Forest")
print("=" * 50)

rf_p = params["model"]["random_forest"]

rf = RandomForestRegressor(
    n_estimators=rf_p["n_estimators"],
    max_depth=rf_p["max_depth"],
    min_samples_split=rf_p["min_samples_split"],
    min_samples_leaf=rf_p["min_samples_leaf"],
    random_state=params["data"]["random_seed"]
)

rf.fit(X_train_flat, y_train_flat.ravel())
rf_preds = rf.predict(X_test_flat)

results["rf"] = {
    "r2":   float(r2_score(y_test_flat, rf_preds)),
    "rmse": float(np.sqrt(mean_squared_error(y_test_flat, rf_preds)))
}

pickle.dump(rf, open("artifacts/models/rf_model.pkl", "wb"))
print(f"  R2: {results['rf']['r2']:.4f}  |  RMSE: {results['rf']['rmse']:.4f}")

# ==========================================
# MLP Regressor
# ==========================================
print("=" * 50)
print("Training: MLP")
print("=" * 50)

mlp_p = params["model"]["mlp"]

mlp = MLPRegressor(
    hidden_layer_sizes=tuple(mlp_p["hidden_layer_sizes"]),
    max_iter=mlp_p["max_iter"],
    random_state=params["data"]["random_seed"]
)

mlp.fit(X_train_flat, y_train_flat.ravel())
mlp_preds = mlp.predict(X_test_flat)

results["mlp"] = {
    "r2":   float(r2_score(y_test_flat, mlp_preds)),
    "rmse": float(np.sqrt(mean_squared_error(y_test_flat, mlp_preds)))
}

pickle.dump(mlp, open("artifacts/models/mlp_model.pkl", "wb"))
print(f"  R2: {results['mlp']['r2']:.4f}  |  RMSE: {results['mlp']['rmse']:.4f}")

# ==========================================
# LSTM — sequences
# ==========================================
print("=" * 50)
print("Training: LSTM")
print("=" * 50)

lstm_p  = params["model"]["lstm"]
cb_p    = params["callbacks"]
SEQ_LEN = lstm_p["sequence_length"]

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

X_train_lstm, y_train_lstm = create_sequences(X_train_seq_raw, y_train_seq_raw, SEQ_LEN)
X_test_lstm,  y_test_lstm  = create_sequences(X_test_seq_raw,  y_test_seq_raw,  SEQ_LEN)

# Save final LSTM sequences for evaluate.py and monitor.py
np.save("artifacts/data/x_train_lstm.npy", X_train_lstm)
np.save("artifacts/data/x_test_lstm.npy",  X_test_lstm)
np.save("artifacts/data/y_train_lstm.npy", y_train_lstm)
np.save("artifacts/data/y_test_lstm.npy",  y_test_lstm)

# ---- Improved LSTM architecture ----
# BatchNormalization stabilises training between LSTM blocks
# EarlyStopping restores best weights to avoid overfitting
# ReduceLROnPlateau backs off LR when val_loss stalls
# validation_split so the model monitors generalisation during training
# All patience/factor values read from params.yaml callbacks section

lstm_model = Sequential([
    LSTM(lstm_p["units_1"], return_sequences=True,
         input_shape=(SEQ_LEN, X_train_seq_raw.shape[1])),
    Dropout(lstm_p["dropout_1"]),
    BatchNormalization(),

    LSTM(lstm_p["units_2"]),
    Dropout(lstm_p["dropout_2"]),
    BatchNormalization(),

    Dense(lstm_p["dense_units"], activation="relu"),
    Dense(1)
])

lstm_model.compile(
    optimizer=Adam(params["model"]["learning_rate"]),
    loss="mse"
)

lstm_model.summary()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=cb_p["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=cb_p["reduce_lr_factor"],
        patience=cb_p["reduce_lr_patience"],
        min_lr=cb_p["reduce_lr_min_lr"],
        verbose=1
    )
]

history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=params["model"]["epochs"],
    batch_size=params["model"]["batch_size"],
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

lstm_preds = lstm_model.predict(X_test_lstm)

results["lstm"] = {
    "r2":   float(r2_score(y_test_lstm, lstm_preds)),
    "rmse": float(np.sqrt(mean_squared_error(y_test_lstm, lstm_preds)))
}

lstm_model.save("artifacts/models/lstm_model.keras")

# Save training history for reference
with open("artifacts/training_history.json", "w") as f:
    json.dump({
        "loss":     history.history["loss"],
        "val_loss": history.history["val_loss"]
    }, f, indent=2)

print(f"  R2: {results['lstm']['r2']:.4f}  |  RMSE: {results['lstm']['rmse']:.4f}")

# ==========================================
# Save all metrics
# ==========================================
with open("artifacts/metrics/model_scores.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 50)
print("ALL MODEL RESULTS")
print("=" * 50)
for name, m in results.items():
    print(f"  {name.upper():6s}  |  R2: {m['r2']:.4f}  |  RMSE: {m['rmse']:.4f}")
print("=" * 50)
