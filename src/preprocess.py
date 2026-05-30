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

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["data"]["test_size"],
    random_state=params["data"]["random_seed"],
    shuffle=False
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    np.save("artifacts/data/x_train.npy", X_train)
    np.save("artifacts/data/x_test.npy", X_test)
    np.save("artifacts/data/y_train.npy", y_train)
    np.save("artifacts/data/y_test.npy", y_test)

    with open("artifacts/scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("artifacts/scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

df = load_data()
preprocess(df)