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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
METRICS_PATH  = os.path.join(ROOT_DIR, "metrics.json")
R2_THRESHOLD  = 0.80


def load_processed_data():
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    feature_cols = [
        "Age", "YearsExperience", "EducationLevel",
        "JobRole", "Gender", "ExperienceRatio", "CareerScore"
    ]
    return (
        train_df[feature_cols], test_df[feature_cols],
        train_df["Salary"],     test_df["Salary"]
    )


def build_pipeline():
    return Pipeline([
        ("scaler",    StandardScaler()),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )),
    ])


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
        "mse":  round(float(mean_squared_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }


def get_next_version():
    os.makedirs(MODELS_DIR, exist_ok=True)
    existing = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("model_v") and f.endswith(".joblib")
    ]
    if not existing:
        return "v1"
    numbers = [int(f.split("_v")[1].split(".")[0]) for f in existing]
    return f"v{max(numbers) + 1}"


def save_model(model, version):
    os.makedirs(MODELS_DIR, exist_ok=True)
    versioned = os.path.join(MODELS_DIR, f"model_{version}.joblib")
    stable    = os.path.join(MODELS_DIR, "model_stable.joblib")
    joblib.dump(model, versioned)
    joblib.dump(model, stable)
    log.info(f"Saved → {versioned}")
    log.info(f"Stable pointer → {stable}")


def main(force_version=None):
    log.info("=== Salary Model Training Started ===")
    X_train, X_test, y_train, y_test = load_processed_data()
    log.info(f"Training on {len(X_train)} samples")

    model = build_pipeline()
    model.fit(X_train, y_train)
    log.info("Model fitted ✓")

    metrics = evaluate(model, X_test, y_test)
    log.info(f"Metrics: {metrics}")

    version = force_version or get_next_version()
    save_model(model, version)

    metrics["version"]          = version
    metrics["timestamp"]        = datetime.now().isoformat()
    metrics["r2_threshold"]     = R2_THRESHOLD
    metrics["passed_threshold"] = bool(metrics["r2"] >= R2_THRESHOLD)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {METRICS_PATH}")

    if metrics["r2"] >= R2_THRESHOLD:
        log.info(f"✅ R² {metrics['r2']} >= {R2_THRESHOLD} — model accepted")
    else:
        log.warning(f"⚠️  R² {metrics['r2']} < {R2_THRESHOLD} — below threshold")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    main(force_version=args.version)