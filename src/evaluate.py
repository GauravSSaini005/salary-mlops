"""
evaluate.py
-----------
Loads the model, evaluates on test set.
Exits with code 1 if R2 below threshold.
This non-zero exit makes GitHub Actions mark
the step as FAILED and triggers rollback.
"""

import os
import sys
import json
import logging
import shutil
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
METRICS_PATH  = os.path.join(ROOT_DIR, "metrics.json")
R2_THRESHOLD  = 0.80


def load_test_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    feature_cols = [
        "Age", "YearsExperience", "EducationLevel",
        "JobRole", "Gender", "ExperienceRatio", "CareerScore"
    ]
    return df[feature_cols], df["Salary"]


def trigger_rollback():
    versions = sorted(
        [f for f in os.listdir(MODELS_DIR)
         if f.startswith("model_v") and f.endswith(".joblib")],
        reverse=True,
    )
    if len(versions) < 2:
        log.error("No previous version to roll back to!")
        return False
    previous = os.path.join(MODELS_DIR, versions[1])
    shutil.copy2(previous, os.path.join(MODELS_DIR, "model_stable.joblib"))
    log.info(f"🔄 Rolled back to {versions[1]}")
    return True


def main(model_path, auto_rollback=False):
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}")
        sys.exit(1)

    model          = joblib.load(model_path)
    X_test, y_test = load_test_data()
    y_pred         = model.predict(X_test)

    r2  = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    log.info(f"R²  : {round(r2, 4)}  (threshold={R2_THRESHOLD})")
    log.info(f"MAE : £{round(mae, 2)}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            existing = json.load(f)
        existing.update({
            "eval_r2":     round(r2, 4),
            "eval_mae":    round(mae, 2),
            "eval_passed": bool(r2 >= R2_THRESHOLD),
        })
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    if r2 < R2_THRESHOLD:
        log.warning("❌ Model below threshold — retraining needed!")
        if auto_rollback:
            trigger_rollback()
        sys.exit(1)

    log.info("✅ Model passed threshold check")
    sys.exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "model_stable.joblib")
    )
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    main(args.model, auto_rollback=args.rollback)