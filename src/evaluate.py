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

# Set up logging so every message prints with its level (INFO, WARNING, ERROR)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Build all file paths relative to this script's location so they work
# on any machine regardless of where the repository is cloned.
# os.path.join(__file__, "..") steps one level up from src/ to the project root.
ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")  # where test.csv lives
MODELS_DIR    = os.path.join(ROOT_DIR, "models")              # where model files are saved
METRICS_PATH  = os.path.join(ROOT_DIR, "metrics.json")        # written by train.py, updated here

# The minimum acceptable R2 score. If the model scores below this,
# evaluation fails and GitHub Actions marks the step as failed.
R2_THRESHOLD  = 0.80


def load_test_data():
    """
    Reads only the held-out test split from disk and returns the feature matrix
    and true salary labels as two separate variables.

    We load only test.csv here because evaluate.py never needs to see the
    training data — it only scores the model against unseen examples.
    The feature column list must match exactly what preprocess.py saved,
    otherwise the model receives unexpected input and scores will be meaningless.
    """
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    # These seven columns are the inputs the model was trained on.
    # The last two (ExperienceRatio and CareerScore) were engineered in preprocess.py.
    feature_cols = [
        "Age", "YearsExperience", "EducationLevel",
        "JobRole", "Gender", "ExperienceRatio", "CareerScore"
    ]

    # Return features and true labels as separate variables ready for scoring
    return df[feature_cols], df["Salary"]


def trigger_rollback():
    """
    Attempts to restore the previous model version as the stable model.
    This is called when auto_rollback is enabled and the current model
    fails the threshold check.

    It sorts all versioned model files in reverse order (newest first),
    then copies the second one (the previous version) over model_stable.joblib.
    Returns True if rollback succeeded, False if there was no previous version.
    """
    # Collect all versioned model files and sort them newest-first
    versions = sorted(
        [f for f in os.listdir(MODELS_DIR)
         if f.startswith("model_v") and f.endswith(".joblib")],
        reverse=True,
    )

    # We need at least two versions: the current one and one to roll back to
    if len(versions) < 2:
        log.error("No previous version to roll back to!")
        return False

    # versions[0] is the current (failing) model, versions[1] is the previous one
    previous = os.path.join(MODELS_DIR, versions[1])
    shutil.copy2(previous, os.path.join(MODELS_DIR, "model_stable.joblib"))
    log.info(f"Rolled back to {versions[1]}")
    return True


def main(model_path, auto_rollback=False):
    """
    Main evaluation function. Loads the model, scores it on the test set,
    updates metrics.json with the evaluation results, and exits with code 0
    (pass) or code 1 (fail) so GitHub Actions knows whether to continue.
    """

    # Make sure the model file actually exists before trying to load it
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}")
        sys.exit(1)

    # Load the saved model from disk
    model          = joblib.load(model_path)

    # Load the test features and true salary labels
    X_test, y_test = load_test_data()

    # Run the model on the test features to get predicted salaries
    y_pred         = model.predict(X_test)

    # Calculate the two key metrics we care about:
    # R2 measures how well the model explains variance in salary (1.0 is perfect)
    # MAE measures the average prediction error in pounds
    r2  = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    log.info(f"R2  : {round(r2, 4)}  (threshold={R2_THRESHOLD})")
    log.info(f"MAE : GBP {round(mae, 2)}")

    # If metrics.json already exists from the training step, update it in place
    # rather than overwriting it, so we preserve the version and training metadata.
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            existing = json.load(f)

        # Add the evaluation results alongside the existing training metrics
        existing.update({
            "eval_r2":     round(r2, 4),
            "eval_mae":    round(mae, 2),
            "eval_passed": bool(r2 >= R2_THRESHOLD),  # True or False for easy reading
        })
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    # If the model did not meet the threshold, decide what to do next.
    # sys.exit(1) causes GitHub Actions to mark this step as failed,
    # which blocks any downstream deploy jobs from running.
    if r2 < R2_THRESHOLD:
        log.warning("Model below threshold — retraining needed!")
        if auto_rollback:
            # Optionally restore the previous model so the service keeps running
            trigger_rollback()
        sys.exit(1)

    # R2 met the threshold — log success and exit cleanly so the pipeline continues
    log.info("Model passed threshold check")
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments so this script can be called from GitHub Actions
    # or the terminal with different model paths and options.
    parser = argparse.ArgumentParser()

    # --model lets you point at a specific model file; defaults to model_stable.joblib
    parser.add_argument(
        "--model",
        default=os.path.join(MODELS_DIR, "model_stable.joblib")
    )

    # --rollback enables automatic fallback to the previous model if this one fails
    parser.add_argument("--rollback", action="store_true")

    args = parser.parse_args()
    main(args.model, auto_rollback=args.rollback)