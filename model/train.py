"""
train.py - Machine Learning Model Training (Enhanced)
=======================================================
Trains a Random Forest Classifier with optional:
  - GridSearchCV hyperparameter tuning
  - k-fold cross-validation
  - Feature importance visualization
  - PR-AUC and ROC-AUC curves
Saves the trained model as a pickle file.
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from model.preprocess import load_and_preprocess
from config import get as cfg
from logger import setup_logger

logger = setup_logger(__name__)


def train_model(csv_path=None, model_path=None):
    """
    Full ML training pipeline with optional hyperparameter tuning.

    Returns:
        tuple: (model, accuracy, feature_names)
    """
    csv_path = csv_path or cfg("simulation.output_csv", "data/vm_metrics.csv")
    model_path = model_path or cfg("model.model_path", "model/trained_model.pkl")

    # ---- Step 1: Preprocess ----
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(csv_path)
    
    # ---- Dataset Validation: Check for both classes ----
    unique_train = set(y_train)
    unique_test = set(y_test)
    unique_all = unique_train.union(unique_test)
    
    logger.info(f"  Dataset class distribution:")
    logger.info(f"    Training:  {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"    Testing:   {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    if len(unique_all) < 2:
        error_msg = (
            "\n" + "=" * 60 + "\n"
            "  ERROR: Dataset contains only one class!\n"
            "=" * 60 + "\n"
            f"  Found classes: {unique_all}\n"
            "  ML classification requires both classes (0 and 1).\n\n"
            "  SOLUTION: Increase simulation load to generate overloads:\n"
            "    1. Increase base_cpu_range in config.yaml\n"
            "    2. Reduce host cpu_capacity\n"
            "    3. Increase burst probability\n"
            "    4. Re-run simulation (Step 1)\n"
            "=" * 60
        )
        logger.error(error_msg)
        print(error_msg)
        raise ValueError("Dataset must contain both classes (0 and 1). Increase simulation load.")

    # ---- Step 2: Train (with optional GridSearchCV) ----
    grid_enabled = cfg("model.random_forest.grid_search.enabled", False)

    if grid_enabled:
        model = _train_with_grid_search(X_train, y_train)
    else:
        rf_cfg = cfg("model.random_forest", {})
        model = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", 10),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            random_state=cfg("random_seed", 42),
            n_jobs=rf_cfg.get("n_jobs", -1),
        )
        model.fit(X_train, y_train)

    logger.info("Model training complete!")

    # ---- Step 3: Cross-Validation ----
    cv_enabled = cfg("model.cross_validation.enabled", False)
    if cv_enabled:
        cv_folds = cfg("model.cross_validation.cv_folds", 5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
        logger.info(f"\n  {cv_folds}-Fold Cross-Validation:")
        logger.info(f"    Scores: {np.round(cv_scores, 4)}")
        logger.info(f"    Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # ---- Step 4: Evaluate ----
    y_pred = model.predict(X_test)
    
    # Safe probability extraction (handles single-class edge case)
    proba = model.predict_proba(X_test)
    if proba.shape[1] > 1:
        y_pred_proba = proba[:, 1]  # Probability of class 1 (overloaded)
    else:
        y_pred_proba = proba[:, 0]  # Fallback if only one class
        logger.warning("  Warning: Model only predicts one class. Check dataset balance.")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Advanced metrics (with safe handling for single class)
    try:
        if len(set(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
        else:
            logger.warning("  Warning: Test set has only one class. ROC-AUC/PR-AUC set to 0.")
            roc_auc = 0.0
            pr_auc = 0.0
    except ValueError as e:
        logger.warning(f"  Warning: Could not compute AUC metrics: {e}")
        roc_auc = 0.0
        pr_auc = 0.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  MODEL PERFORMANCE METRICS")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"  PR-AUC:    {pr_auc:.4f}")
    logger.info(f"{'=' * 60}")

    logger.info("\n  Classification Report:")
    logger.info(classification_report(y_test, y_pred,
                                      target_names=["Normal", "Overloaded"]))

    # ---- Step 5: Plots ----
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_path)
    plot_feature_importance(model, feature_names)
    plot_roc_pr_curves(y_test, y_pred_proba, roc_auc, pr_auc)

    # ---- Step 6: Save Model ----
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"  Model saved to '{model_path}'")

    return model, accuracy, feature_names


def _train_with_grid_search(X_train, y_train):
    """Run GridSearchCV for hyperparameter tuning."""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING (GridSearchCV)")
    logger.info("=" * 60)

    gs_cfg = cfg("model.random_forest.grid_search", {})
    param_grid = gs_cfg.get("param_grid", {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    })
    cv_folds = gs_cfg.get("cv_folds", 5)

    base_model = RandomForestClassifier(
        random_state=cfg("random_seed", 42),
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"\n  Best Parameters: {grid_search.best_params_}")
    logger.info(f"  Best F1 Score:   {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def plot_confusion_matrix(cm, model_path="model/trained_model.pkl"):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Overloaded"],
        yticklabels=["Normal", "Overloaded"],
        linewidths=0.5, linecolor="gray",
    )
    plt.title("Confusion Matrix - Random Forest Classifier",
              fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(model_path), "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"  Confusion matrix saved to '{plot_path}'")


def plot_feature_importance(model, feature_names):
    """Plot and save feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))

    bars = ax.barh(
        [feature_names[i] for i in indices],
        [importances[i] for i in indices],
        color=colors,
        edgecolor="white",
    )
    ax.set_title("Feature Importance - Random Forest",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")

    for bar, imp in zip(bars, [importances[i] for i in indices]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    path = "model/feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Feature importance plot saved to '{path}'")

    # Print to console
    logger.info("  Feature Importance:")
    for i in indices:
        bar_str = "#" * int(importances[i] * 40)
        logger.info(f"    {feature_names[i]:20s} {importances[i]:.4f} {bar_str}")


def plot_roc_pr_curves(y_test, y_pred_proba, roc_auc, pr_auc):
    """Plot ROC and Precision-Recall curves side by side."""
    # Skip plotting if only one class in test set
    if len(set(y_test)) < 2:
        logger.warning("  Skipping ROC/PR curves: Test set has only one class.")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation Curves", fontsize=14, fontweight="bold")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, color="#2ecc71", linewidth=2, label=f"ROC (AUC={roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    ax2.plot(rec, prec, color="#3498db", linewidth=2, label=f"PR (AUC={pr_auc:.4f})")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    path = "model/roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ROC/PR curves saved to '{path}'")


def load_trained_model(model_path=None):
    """Load a previously saved trained model."""
    model_path = model_path or cfg("model.model_path", "model/trained_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'. Train the model first.")
    model = joblib.load(model_path)
    logger.info(f"Loaded trained model from '{model_path}'")
    return model
