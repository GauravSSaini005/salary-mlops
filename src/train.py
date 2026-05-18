"""
train.py
--------
Trains a Random Forest model to predict annual salary.
Saves versioned model + metrics.json.
R2 threshold = 0.80 — if model drops below this,
the CT pipeline will automatically retrain.
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set up logging so every step prints a readable message with its level to the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Build all paths relative to this script's location so they work on any machine.
# __file__ is src/train.py, so ".." steps up to the project root.
ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")  # where train.csv and test.csv live
MODELS_DIR    = os.path.join(ROOT_DIR, "models")              # where trained model files are saved
METRICS_PATH  = os.path.join(ROOT_DIR, "metrics.json")        # output file that records all metrics

# The minimum R2 score the model must achieve to be considered acceptable.
# If the trained model falls below this, the CT pipeline will trigger a retrain.
R2_THRESHOLD  = 0.80


def load_processed_data():
    """
    Reads the train and test CSV files produced by preprocess.py and returns
    them as four separate variables: features and labels for each split.

    The feature column list must match exactly what preprocess.py saved,
    including the two derived columns (ExperienceRatio and CareerScore),
    otherwise the model will receive the wrong inputs and scores will be wrong.
    """
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    # These seven columns are the inputs the model is trained on.
    # The order here must stay consistent with preprocess.py, evaluate.py, and app.py.
    feature_cols = [
        "Age", "YearsExperience", "EducationLevel",
        "JobRole", "Gender", "ExperienceRatio", "CareerScore"
    ]

    # Return features and labels for both splits as four separate variables
    return (
        train_df[feature_cols], test_df[feature_cols],
        train_df["Salary"],     test_df["Salary"]
    )


def build_pipeline():
    """
    Creates a two-step sklearn Pipeline that first scales the features and
    then trains a Random Forest regressor.

    Wrapping both steps in a Pipeline means scaling is always applied
    consistently — both during training and when predict() is called later —
    so there is no risk of forgetting to scale new inputs.

    RandomForestRegressor settings:
    - n_estimators=100: builds 100 decision trees and averages their predictions
    - random_state=42: makes the result reproducible across runs
    - max_depth=10: limits how deep each tree can grow to reduce overfitting
    """
    return Pipeline([
        ("scaler",    StandardScaler()),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )),
    ])


def evaluate(model, X_test, y_test):
    """
    Runs the trained model on the test set and returns a dictionary of
    four performance metrics. All values are rounded to keep metrics.json readable.

    MAE  — average prediction error in pounds (lower is better)
    MSE  — mean squared error, penalises large mistakes more heavily
    RMSE — square root of MSE, expressed in the same unit as salary (pounds)
    R2   — proportion of salary variance explained by the model (1.0 is perfect)
    """
    y_pred = model.predict(X_test)
    return {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
        "mse":  round(float(mean_squared_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }


def get_next_version():
    """
    Scans the models folder and works out what the next version number should be.
    If no versioned files exist yet, the first version is v1.
    Otherwise it finds the highest existing version number and adds one.

    This gives every trained model a unique, traceable filename
    (e.g. model_v1.joblib, model_v2.joblib) so older versions are never overwritten.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Find all files that follow the model_vN.joblib naming pattern
    existing = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("model_v") and f.endswith(".joblib")
    ]

    if not existing:
        return "v1"

    # Extract the integer from each filename (e.g. "model_v3.joblib" -> 3)
    # then return the next number in the sequence
    numbers = [int(f.split("_v")[1].split(".")[0]) for f in existing]
    return f"v{max(numbers) + 1}"


def save_model(model, version):
    """
    Saves the trained model in two places:
    1. A versioned file (e.g. model_v3.joblib) so we can roll back if needed
    2. model_stable.joblib, which is the file the API and evaluate.py always load

    Writing both files means the history is preserved while the rest of the
    codebase always has a single consistent filename to point at.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    versioned = os.path.join(MODELS_DIR, f"model_{version}.joblib")
    stable    = os.path.join(MODELS_DIR, "model_stable.joblib")

    joblib.dump(model, versioned)  # save the permanent versioned copy
    joblib.dump(model, stable)     # overwrite the stable pointer with this version

    log.info(f"Saved versioned model: {versioned}")
    log.info(f"Stable model updated: {stable}")


def main(force_version=None):
    """
    Runs the full training pipeline in order:
    load data -> build pipeline -> fit model -> evaluate -> save -> write metrics.

    force_version lets the caller pin the version string (e.g. "v5") instead
    of auto-incrementing. This is useful when re-running a specific training job.
    """
    log.info("=== Salary Model Training Started ===")

    # Load the preprocessed train and test splits from disk
    X_train, X_test, y_train, y_test = load_processed_data()
    log.info(f"Training on {len(X_train)} samples")

    # Build the scaler + regressor pipeline and fit it on the training data
    model = build_pipeline()
    model.fit(X_train, y_train)
    log.info("Model fitted")

    # Score the model on the held-out test set to measure real-world performance
    metrics = evaluate(model, X_test, y_test)
    log.info(f"Metrics: {metrics}")

    # Determine the version string, then save both the versioned and stable files
    version = force_version or get_next_version()
    save_model(model, version)

    # Add metadata to the metrics dictionary before writing it to disk.
    # These extra fields let the dashboard and CT pipeline read version and
    # threshold status without having to re-run the model.
    metrics["version"]          = version
    metrics["timestamp"]        = datetime.now().isoformat()
    metrics["r2_threshold"]     = R2_THRESHOLD
    metrics["passed_threshold"] = bool(metrics["r2"] >= R2_THRESHOLD)

    # Write the complete metrics dictionary to metrics.json
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved: {METRICS_PATH}")

    # Log a clear pass or fail message so the GitHub Actions log is easy to read
    if metrics["r2"] >= R2_THRESHOLD:
        log.info(f"R2 {metrics['r2']} >= {R2_THRESHOLD} — model accepted")
    else:
        log.warning(f"R2 {metrics['r2']} < {R2_THRESHOLD} — below threshold")

    return metrics


if __name__ == "__main__":
    # Accept an optional --version argument so a specific version string can be forced
    # from the command line or a GitHub Actions workflow step.
    # If not provided, the version is auto-incremented by get_next_version().
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    main(force_version=args.version)