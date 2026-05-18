"""
app.py
------
Flask REST API serving Salary predictions.

Endpoints:
  GET  /health       → is the API alive and model loaded?
  GET  /model/info   → current model version and metrics
  POST /predict      → predict salary from inputs
"""

import os
import json
import logging
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(ROOT_DIR, "models")
METRICS_PATH = os.path.join(ROOT_DIR, "metrics.json")

_model = None


def load_model():
    global _model
    stable = os.path.join(MODELS_DIR, "model_stable.joblib")
    if os.path.exists(stable):
        log.info(f"Loading model from {stable}")
        _model = joblib.load(stable)
        return _model
    # Fallback: pick any versioned model
    candidates = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("model_v") and f.endswith(".joblib")
    ])
    if candidates:
        path = os.path.join(MODELS_DIR, candidates[-1])
        log.info(f"Fallback model: {path}")
        _model = joblib.load(path)
        return _model
    raise RuntimeError("No model found. Run train.py first.")


def get_model():
    global _model
    if _model is None:
        load_model()
    return _model


def salary_band(salary):
    """Return a human-readable salary band."""
    if salary < 25000:
        return "Entry Level"
    elif salary < 40000:
        return "Junior"
    elif salary < 60000:
        return "Mid Level"
    elif salary < 85000:
        return "Senior"
    else:
        return "Lead / Manager"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — used by Docker and Kubernetes probes."""
    try:
        get_model()
        model_ok = True
    except Exception:
        model_ok = False

    status = "healthy" if model_ok else "degraded"
    return jsonify({
        "status":       status,
        "timestamp":    datetime.now().isoformat(),
        "model_loaded": model_ok,
    }), 200 if model_ok else 503


@app.route("/model/info", methods=["GET"])
def model_info():
    """Returns current model version and performance metrics."""
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return jsonify({
        "version": metrics.get("version", "unknown"),
        "metrics": metrics,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict annual salary.

    Expected JSON body:
    {
        "age": 30,
        "years_experience": 5,
        "education_level": 2,
        "job_role": 3,
        "gender": 0
    }

    education_level: 0=High School, 1=Bachelor, 2=Master/PhD
    job_role:        0=Junior, 1=Mid, 2=Senior, 3=Lead, 4=Manager
    gender:          0=Male, 1=Female
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    # Check all required fields are present
    required = ["age", "years_experience", "education_level", "job_role", "gender"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Parse and validate inputs
    try:
        age               = float(data["age"])
        years_experience  = float(data["years_experience"])
        education_level   = int(data["education_level"])
        job_role          = int(data["job_role"])
        gender            = int(data["gender"])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid values: {e}"}), 400

    # Sanity checks
    if not (16 < age < 80):
        return jsonify({"error": "Age must be between 16 and 80"}), 422
    if not (0 <= years_experience < 50):
        return jsonify({"error": "YearsExperience must be 0-50"}), 422
    if education_level not in [0, 1, 2]:
        return jsonify({"error": "EducationLevel must be 0, 1, or 2"}), 422
    if job_role not in [0, 1, 2, 3, 4]:
        return jsonify({"error": "JobRole must be 0, 1, 2, 3, or 4"}), 422
    if gender not in [0, 1]:
        return jsonify({"error": "Gender must be 0 or 1"}), 422

    # Engineer features — must match preprocess.py exactly
    experience_ratio = years_experience / age
    career_score     = education_level * 2 + job_role

    features = np.array([[
        age, years_experience, education_level,
        job_role, gender, experience_ratio, career_score
    ]])

    try:
        salary_pred = float(get_model().predict(features)[0])
    except Exception as e:
        log.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    return jsonify({
        "predicted_salary": round(salary_pred, 2),
        "salary_band":      salary_band(salary_pred),
        "currency":         "GBP",
        "version":          metrics.get("version", "unknown"),
        "inputs": {
            "age":               age,
            "years_experience":  years_experience,
            "education_level":   education_level,
            "job_role":          job_role,
            "gender":            gender,
        },
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        load_model()
        log.info("Model pre-loaded ✓")
    except RuntimeError as e:
        log.warning(str(e))

    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)