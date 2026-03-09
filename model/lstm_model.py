"""
lstm_model.py - LSTM Time-Series Overload Prediction
======================================================
Trains an LSTM neural network on sequential VM metrics to predict
host overload. Compares with Random Forest on the same test set.
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)


def _build_sequences(X, y, seq_length):
    """
    Convert flat feature matrix into overlapping sequences for LSTM.

    Args:
        X (np.ndarray): Shape (n_samples, n_features).
        y (np.ndarray): Shape (n_samples,).
        seq_length (int): Number of timesteps per sequence.

    Returns:
        X_seq (np.ndarray): Shape (n_sequences, seq_length, n_features).
        y_seq (np.ndarray): Shape (n_sequences,).
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(csv_path=None, model_path=None):
    """
    Full LSTM training pipeline.

    1. Load & preprocess data
    2. Build sequences
    3. Create LSTM model
    4. Train and evaluate
    5. Save model

    Returns:
        tuple: (model, accuracy, history)
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        logger.warning("TensorFlow not installed. Install with: pip install tensorflow")
        logger.warning("Skipping LSTM training.")
        return None, 0.0, None

    from model.preprocess import load_and_preprocess

    csv_path = csv_path or cfg("simulation.output_csv", "data/vm_metrics.csv")
    model_path = model_path or cfg("model.lstm.model_path", "model/lstm_model.h5")
    seq_length = cfg("model.lstm.sequence_length", 10)
    epochs = cfg("model.lstm.epochs", 50)
    batch_size = cfg("model.lstm.batch_size", 32)
    hidden_units = cfg("model.lstm.hidden_units", 64)
    lr = cfg("model.lstm.learning_rate", 0.001)
    dropout_rate = cfg("model.lstm.dropout", 0.2)

    # ---- Load data ----
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess(csv_path)

    # ---- Build sequences ----
    X_train_seq, y_train_seq = _build_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = _build_sequences(X_test, y_test, seq_length)

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        logger.error("Not enough data to build LSTM sequences. Increase simulation ticks.")
        return None, 0.0, None

    n_features = X_train_seq.shape[2]

    logger.info("=" * 60)
    logger.info("TRAINING LSTM MODEL")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Training sequences: {X_train_seq.shape[0]}")
    logger.info(f"  Test sequences: {X_test_seq.shape[0]}")
    logger.info("=" * 60)

    # ---- Build LSTM ----
    model = Sequential([
        LSTM(hidden_units, input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(hidden_units // 2),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)

    # ---- Train ----
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # ---- Evaluate ----
    y_pred_prob = model.predict(X_test_seq).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test_seq, y_pred)
    logger.info(f"\n  LSTM Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    logger.info("\n  LSTM Classification Report:")
    logger.info(classification_report(y_test_seq, y_pred,
                                      target_names=["Normal", "Overloaded"]))

    # ---- Plot training history ----
    _plot_training_history(history)

    # ---- Save model ----
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"  LSTM model saved to '{model_path}'")

    return model, accuracy, history


def _plot_training_history(history):
    """Plot LSTM training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Training History", fontsize=14, fontweight="bold")

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = "model/lstm_training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  LSTM training history saved to '{path}'")


def compare_rf_vs_lstm(rf_accuracy, lstm_accuracy):
    """Plot accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Random Forest", "LSTM"]
    accuracies = [rf_accuracy, lstm_accuracy]
    colors = ["#2ecc71", "#3498db"]

    bars = ax.bar(models, accuracies, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Model Comparison: Random Forest vs LSTM",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.4f}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = "model/rf_vs_lstm_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Model comparison chart saved to '{path}'")
