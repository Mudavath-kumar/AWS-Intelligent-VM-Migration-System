"""
preprocess.py - Dataset Preprocessing
========================================
Loads raw simulation CSV data, cleans it, normalizes features,
creates labels, and splits into train/test sets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_and_preprocess(csv_path="data/vm_metrics.csv", test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    1. Load CSV data
    2. Clean missing values
    3. Normalize features
    4. Create binary labels (overloaded: 1 = overload, 0 = normal)
    5. Split into train/test sets
    
    Args:
        csv_path (str): Path to the raw CSV data file.
        test_size (float): Fraction of data for testing (default 20%).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    # ---- Step 1: Load Data ----
    print(f"\n[PREPROCESS] Loading data from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    print(f"  Raw dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ---- Step 2: Clean Missing Values ----
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_after = df.isnull().sum().sum()
    print(f"  Missing values removed: {missing_before - missing_after}")
    print(f"  Clean dataset shape: {df.shape}")

    # ---- Step 3: Select Features ----
    feature_columns = ["cpu", "ram", "network", "total_host_cpu", "total_host_ram"]
    label_column = "overloaded"

    X = df[feature_columns].values
    y = df[label_column].values.astype(int)

    print(f"  Features: {feature_columns}")
    print(f"  Label distribution: Normal={np.sum(y == 0)}, Overloaded={np.sum(y == 1)}")

    # ---- Step 4: Normalize Features ----
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Features normalized using MinMaxScaler (range 0-1)")

    # Save the scaler for later use
    os.makedirs("model", exist_ok=True)
    scaler_path = "model/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to '{scaler_path}'")

    # ---- Step 5: Train/Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")
    print(f"[PREPROCESS] Preprocessing complete!\n")

    return X_train, X_test, y_train, y_test, scaler, feature_columns


def preprocess_realtime(metrics_dict, scaler, feature_columns):
    """
    Preprocess a single real-time metric reading for prediction.
    
    Args:
        metrics_dict (dict): Dictionary with feature values.
        scaler (MinMaxScaler): Fitted scaler from training.
        feature_columns (list): List of feature column names.
        
    Returns:
        np.ndarray: Scaled features ready for model prediction (shape: 1 x n_features).
    """
    values = [metrics_dict[col] for col in feature_columns]
    values_array = np.array(values).reshape(1, -1)
    return scaler.transform(values_array)
