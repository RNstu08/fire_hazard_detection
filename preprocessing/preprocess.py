import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from feature_engineering import add_rolling_features, add_derivative_features, compute_fire_risk_score

DATA_DIR = "../data"
RAW_FILE = "simulated_sensor_data.csv"

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """
    Fill missing values using linear interpolation, fallback to forward-fill.
    """
    df.interpolate(method='linear', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def smooth_data(df):
    """
    Smooth sensor signals using rolling average.
    """
    df = add_rolling_features(df, window_size=5)
    return df

def feature_engineering(df):
    """
    Create derivatives and fire risk score features.
    """
    df = add_derivative_features(df)
    df = compute_fire_risk_score(df)
    return df

def normalize_features(df):
    """
    Normalize features using StandardScaler (except label column).
    """
    features = df.drop(columns=["fire_label"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df["fire_label"] = df["fire_label"].values  # add back label
    
    return scaled_df, scaler

def split_train_test(df, train_frac=0.7):
    """
    Chronologically split data into train/test.
    """
    split_index = int(len(df) * train_frac)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df

def save_preprocessed_data(train_df, test_df):
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_DIR, "train_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_preprocessed.csv"), index=False)
    print("Preprocessed train/test datasets saved!")

if __name__ == "__main__":
    # Load
    df = load_data(os.path.join(DATA_DIR, RAW_FILE))
    print(f"Loaded data: {df.shape}")

    # Preprocess
    df = handle_missing_values(df)
    df = smooth_data(df)
    df = feature_engineering(df)
    df, scaler = normalize_features(df)

    # Split
    train_df, test_df = split_train_test(df)

    # Save
    save_preprocessed_data(train_df, test_df)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
