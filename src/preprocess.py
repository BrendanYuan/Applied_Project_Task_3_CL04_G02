import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

os.makedirs("artifacts/data", exist_ok=True)

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Path to raw dataset
DATA_PATH = "data/raw/power_consumption_tetuan_city.csv"

# Load data function
def load_data():
    return pd.read_csv(DATA_PATH)

# Preprocess data function
def preprocess(df):
    df = df.dropna()
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    df["hour_of_day"] = df["DateTime"].dt.hour
    df["day_of_week"] = df["DateTime"].dt.dayofweek
    df["month"] = df["DateTime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    features = [
        "Temperature", "Humidity", "Wind Speed",
        "general diffuse flows", "diffuse flows",
        "hour_of_day", "day_of_week", "month", "is_weekend"
    ]

    X = df[features].values

    y = df["Zone 1 Power Consumption"].values.reshape(-1, 1)

    # ==========================================
    # Split 1 — RF and MLP (shuffled, 70/30)
    # shuffle=True ensures train and test are drawn
    # from the same overall distribution. These models
    # treat each row independently so temporal order
    # does not matter.
    # ==========================================

    X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
        X, y,
        test_size=0.30,
        random_state=params["data"]["random_seed"],
        shuffle=True
    )

    scaler_X_flat = StandardScaler()
    scaler_y_flat = StandardScaler()

    X_train_flat = scaler_X_flat.fit_transform(X_train_flat)
    X_test_flat  = scaler_X_flat.transform(X_test_flat)
    y_train_flat = scaler_y_flat.fit_transform(y_train_flat)
    y_test_flat  = scaler_y_flat.transform(y_test_flat)

    np.save("artifacts/data/x_train_flat.npy", X_train_flat)
    np.save("artifacts/data/x_test_flat.npy",  X_test_flat)
    np.save("artifacts/data/y_train_flat.npy", y_train_flat)
    np.save("artifacts/data/y_test_flat.npy",  y_test_flat)

    with open("artifacts/scaler_X_flat.pkl", "wb") as f:
        pickle.dump(scaler_X_flat, f)
    with open("artifacts/scaler_y_flat.pkl", "wb") as f:
        pickle.dump(scaler_y_flat, f)

    print("Flat split (RF/MLP) saved.")
    print(f"  X_train_flat : {X_train_flat.shape}")
    print(f"  X_test_flat  : {X_test_flat.shape}")

    # ==========================================
    # Split 2 — LSTM (temporal order preserved, 70/30)
    # shuffle=False keeps chronological order so that
    # sequences built by create_sequences in model.py
    # reflect real temporal dependencies. The 70/30
    # ratio still gives the model a large enough training
    # window to cover multiple seasons.
    # ==========================================

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X, y,
        test_size=0.10,
        random_state=params["data"]["random_seed"],
        shuffle=False
    )
 
    scaler_X_seq = StandardScaler()
    scaler_y_seq = StandardScaler()
 
    X_train_seq = scaler_X_seq.fit_transform(X_train_seq)
    X_test_seq  = scaler_X_seq.transform(X_test_seq)
    y_train_seq = scaler_y_seq.fit_transform(y_train_seq)
    y_test_seq  = scaler_y_seq.transform(y_test_seq)
 
    np.save("artifacts/data/x_train_seq.npy", X_train_seq)
    np.save("artifacts/data/x_test_seq.npy",  X_test_seq)
    np.save("artifacts/data/y_train_seq.npy", y_train_seq)
    np.save("artifacts/data/y_test_seq.npy",  y_test_seq)
 
    with open("artifacts/scaler_X_seq.pkl", "wb") as f:
        pickle.dump(scaler_X_seq, f)
    with open("artifacts/scaler_y_seq.pkl", "wb") as f:
        pickle.dump(scaler_y_seq, f)
 
    print("Sequential split (LSTM) saved.")
    print(f"  X_train_seq  : {X_train_seq.shape}")
    print(f"  X_test_seq   : {X_test_seq.shape}")

df = load_data()
preprocess(df)