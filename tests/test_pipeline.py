"""
tests/test_pipeline.py
----------------------
Unit tests for the Salary MLOps pipeline.
Run with: pytest tests/ -v
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Clean sample dataframe matching real dataset structure."""
    return pd.DataFrame({
        "Age":             [22, 30, 35, 40, 50],
        "YearsExperience": [0,   5,  8, 12, 18],
        "EducationLevel":  [0,   1,  2,  2,  2],
        "JobRole":         [0,   2,  3,  4,  4],
        "Gender":          [0,   1,  0,  1,  0],
        "Salary":          [22000, 43000, 59000, 76000, 104000],
    })


@pytest.fixture
def dirty_df():
    """Dataframe with bad rows that should be removed by clean_data."""
    return pd.DataFrame({
        "Age":             [25,  -5,  200,  30],
        "YearsExperience": [3,    2,   10,   5],
        "EducationLevel":  [1,    1,    2,   9],
        "JobRole":         [1,    2,    3,   1],
        "Gender":          [0,    1,    0,   0],
        "Salary":          [32000, 30000, 50000, 40000],
    })


# ── TEST 1: clean_data removes invalid rows ───────────────────────────────────

def test_clean_data_removes_invalid_rows(dirty_df):
    """
    UNIT TEST 1
    clean_data() must remove rows with impossible Age
    and invalid EducationLevel values.
    """
    from preprocess import clean_data

    cleaned = clean_data(dirty_df)

    assert len(cleaned) >= 1
    assert (cleaned["Age"] > 16).all()
    assert (cleaned["Age"] < 80).all()
    assert cleaned["EducationLevel"].isin([0, 1, 2]).all()
    assert not (cleaned["Age"] == -5).any()
    assert not (cleaned["Age"] == 200).any()


# ── TEST 2: engineer_features adds correct columns ────────────────────────────

def test_engineer_features_adds_columns(sample_df):
    """
    UNIT TEST 2
    engineer_features() must add ExperienceRatio and CareerScore
    with correct calculated values.
    """
    from preprocess import engineer_features

    result = engineer_features(sample_df)

    assert "ExperienceRatio" in result.columns
    assert "CareerScore" in result.columns

    expected_ratio = sample_df.iloc[0]["YearsExperience"] / sample_df.iloc[0]["Age"]
    assert abs(result.iloc[0]["ExperienceRatio"] - expected_ratio) < 1e-6

    expected_score = sample_df.iloc[0]["EducationLevel"] * 2 + sample_df.iloc[0]["JobRole"]
    assert result.iloc[0]["CareerScore"] == expected_score


# ── TEST 3: model trains and produces valid predictions ───────────────────────

def test_model_trains_and_predicts(sample_df):
    """
    UNIT TEST 3
    build_pipeline() must train successfully and produce
    salary predictions in a realistic range.
    """
    from preprocess import engineer_features
    from train import build_pipeline

    df = engineer_features(sample_df)
    feature_cols = [
        "Age", "YearsExperience", "EducationLevel",
        "JobRole", "Gender", "ExperienceRatio", "CareerScore"
    ]
    X = df[feature_cols]
    y = df["Salary"]

    model = build_pipeline()
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert all(isinstance(p, (float, np.floating)) for p in preds)
    assert all(10000 < p < 500000 for p in preds)


# ── TEST 4: /health endpoint returns 200 ─────────────────────────────────────

def test_health_endpoint(tmp_path):
    """
    UNIT TEST 4
    /health must return HTTP 200 when model is loaded.
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=5, random_state=42))
    ])
    X = np.array([
        [22, 0,  0, 0, 0, 0.0,  0],
        [30, 5,  1, 2, 1, 0.16, 4],
        [40, 12, 2, 3, 0, 0.3,  7],
    ])
    y = np.array([22000, 43000, 76000])
    model.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "model_stable.joblib")

    import app as flask_app
    flask_app._model       = None
    flask_app.MODELS_DIR   = str(models_dir)
    flask_app.METRICS_PATH = str(tmp_path / "metrics.json")

    client = flask_app.app.test_client()
    resp   = client.get("/health")

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


# ── TEST 5: /predict returns salary and band ──────────────────────────────────

def test_predict_endpoint(tmp_path):
    """
    UNIT TEST 5
    /predict must return predicted_salary and salary_band
    for valid input.
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=5, random_state=42))
    ])
    X = np.array([
        [22, 0,  0, 0, 0, 0.0,  0],
        [30, 5,  1, 2, 1, 0.16, 4],
        [40, 12, 2, 3, 0, 0.3,  7],
    ])
    y = np.array([22000, 43000, 76000])
    model.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "model_stable.joblib")

    import app as flask_app
    flask_app._model       = None
    flask_app.MODELS_DIR   = str(models_dir)
    flask_app.METRICS_PATH = str(tmp_path / "metrics.json")

    client = flask_app.app.test_client()
    resp   = client.post("/predict", json={
        "age":              30,
        "years_experience": 5,
        "education_level":  2,
        "job_role":         3,
        "gender":           0
    })

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert "predicted_salary" in body
    assert "salary_band"      in body
    assert isinstance(body["predicted_salary"], float)
    assert body["salary_band"] in [
        "Entry Level", "Junior", "Mid Level", "Senior", "Lead / Manager"
    ]


# ── TEST 6: /predict returns 400 on missing field ────────────────────────────

def test_predict_missing_field(tmp_path):
    """
    UNIT TEST 6
    /predict must return HTTP 400 when a required
    field is missing from the request.
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=5, random_state=42))
    ])
    X = np.array([[22, 0, 0, 0, 0, 0.0, 0]])
    y = np.array([22000])
    model.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "model_stable.joblib")

    import app as flask_app
    flask_app._model       = None
    flask_app.MODELS_DIR   = str(models_dir)
    flask_app.METRICS_PATH = str(tmp_path / "metrics.json")

    client = flask_app.app.test_client()

    # Send request with missing 'gender' field
    resp = client.post("/predict", json={
        "age":              30,
        "years_experience": 5,
        "education_level":  2,
        "job_role":         3
    })

    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert "error" in body