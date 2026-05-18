"""
tests/test_pipeline.py
Unit tests for the Salary MLOps pipeline with Kaggle dataset.
Run with: pytest tests/ -v
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Add the project root and src/ folder to Python's module search path so that
# imports like "from preprocess import ..." and "import app" work correctly
# regardless of where pytest is run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Fixtures
# Fixtures are reusable pieces of test data that pytest injects automatically
# into any test function that lists them as a parameter.
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """
    A small clean dataframe that mirrors the structure of the real Kaggle CSV.
    Uses the original Kaggle column names and raw text values (not encoded numbers)
    because this data is passed through clean_data() and encode_columns() in tests
    that need to verify those functions work correctly.
    """
    return pd.DataFrame({
        "Age":                 [22, 30, 35, 40, 50],
        "Gender":              ["Male", "Female", "Male", "Female", "Male"],
        "Education Level":     ["Bachelor's", "Master's", "PhD", "Master's", "PhD"],
        "Job Title":           ["Software Engineer", "Data Analyst",
                                "Senior Manager", "Product Manager", "Director"],
        "Years of Experience": [0, 5, 8, 12, 18],
        "Salary":              [50000, 65000, 120000, 100000, 180000],
    })


@pytest.fixture
def dirty_df():
    """
    A dataframe that deliberately contains bad rows to verify that
    clean_data() removes them correctly. Bad rows include a negative age
    of -5 and an impossible age of 200, both of which must be filtered out.
    The one good row (age 25) must survive the cleaning step.
    """
    return pd.DataFrame({
        "Age":                 [25,  -5,  200,  30],
        "Gender":              ["Male", "Female", "Male", "Female"],
        "Education Level":     ["Bachelor's", "Master's", "PhD", "Bachelor's"],
        "Job Title":           ["Engineer", "Analyst", "Manager", "Sales"],
        "Years of Experience": [3, 2, 10, 5],
        "Salary":              [50000, 45000, 90000, 55000],
    })


# ---------------------------------------------------------------------------
# TEST 1: clean_data removes invalid rows
# ---------------------------------------------------------------------------

def test_clean_data_removes_invalid_rows(dirty_df):
    """
    UNIT TEST 1
    clean_data() must remove rows with impossible Age values.

    We pass in the dirty_df fixture which has two bad age rows (-5 and 200)
    and two valid rows. After cleaning, only rows with ages between 17 and 79
    should remain, and the specific bad values must be gone.
    """
    from preprocess import clean_data

    cleaned = clean_data(dirty_df)

    # At least one valid row must survive the cleaning step
    assert len(cleaned) >= 1

    # Every remaining row must have an age within the acceptable range
    assert (cleaned["Age"] > 16).all()
    assert (cleaned["Age"] < 80).all()

    # Confirm the specific bad rows were removed and are not present
    assert not (cleaned["Age"] == -5).any()
    assert not (cleaned["Age"] == 200).any()


# ---------------------------------------------------------------------------
# TEST 2: encode_columns adds correctly encoded columns
# ---------------------------------------------------------------------------

def test_encode_columns_adds_correct_values(sample_df):
    """
    UNIT TEST 2
    encode_columns() must correctly encode Gender, EducationLevel,
    and JobRole columns from plain text into numeric codes.

    We use a temporary directory for the job_mapping.txt file that
    encode_columns() writes, so the test does not leave files on disk.
    We override preprocess.PROCESSED_DIR to point at that temp folder.
    """
    from preprocess import clean_data, encode_columns
    import tempfile

    # Redirect the job mapping output to a throwaway temp folder
    with tempfile.TemporaryDirectory() as tmp:
        import preprocess
        preprocess.PROCESSED_DIR = tmp

        df     = clean_data(sample_df)
        result = encode_columns(df)

    # All three encoded columns must be present in the output
    assert "Gender"         in result.columns
    assert "EducationLevel" in result.columns
    assert "JobRole"        in result.columns

    # Check Gender encoding: Male=0, Female=1
    assert result.iloc[0]["Gender"] == 0   # first row is Male
    assert result.iloc[1]["Gender"] == 1   # second row is Female

    # Check Education Level encoding: Bachelor's=0, Master's=1, PhD=2
    assert result.iloc[0]["EducationLevel"] == 0  # Bachelor's
    assert result.iloc[1]["EducationLevel"] == 1  # Master's
    assert result.iloc[2]["EducationLevel"] == 2  # PhD


# ---------------------------------------------------------------------------
# TEST 3: model trains and produces valid predictions
# ---------------------------------------------------------------------------

def test_model_trains_and_predicts(sample_df):
    """
    UNIT TEST 3
    build_pipeline() must train successfully and produce salary predictions
    in a realistic range.

    We pass in pre-encoded numpy arrays directly rather than going through
    the full preprocessing step, so this test focuses purely on whether
    the model pipeline can fit and predict without errors.
    """
    from train import build_pipeline

    # Pre-encoded feature rows matching the seven expected columns
    X = np.array([
        [22, 0,  0, 2, 0, 0.0,  4],
        [30, 5,  1, 1, 1, 0.16, 3],
        [40, 12, 2, 3, 0, 0.3,  7],
    ])
    y = np.array([50000, 65000, 120000])

    # Build and fit the pipeline on this small sample
    model = build_pipeline()
    model.fit(X, y)
    preds = model.predict(X)

    # The number of predictions must match the number of input rows
    assert len(preds) == len(y)

    # Every prediction must be a floating point number
    assert all(isinstance(p, (float, np.floating)) for p in preds)

    # Every predicted salary must fall within a realistic salary range.
    # The upper limit is 1,000,000 to match the Kaggle dataset salary ceiling.
    assert all(10000 < p < 1000000 for p in preds)


# ---------------------------------------------------------------------------
# TEST 4: /health endpoint returns HTTP 200
# ---------------------------------------------------------------------------

def test_health_endpoint(tmp_path):
    """
    UNIT TEST 4
    /health must return HTTP 200 when a model is loaded.

    We create a minimal trained model, save it to a temporary directory,
    then point the Flask app at that directory before making the request.
    This avoids any dependency on real model files that may not exist in CI.
    tmp_path is a built-in pytest fixture that provides a fresh temporary
    folder that is deleted automatically after the test finishes.
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Build and train a tiny model — enough for the pipeline to fit without
    # errors, not intended to produce accurate salary predictions
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=5, random_state=42))
    ])
    X = np.array([
        [22, 0,  0, 2, 0, 0.0,  4],
        [30, 5,  1, 1, 1, 0.16, 3],
        [40, 12, 2, 3, 0, 0.3,  7],
    ])
    y = np.array([50000, 65000, 120000])
    model.fit(X, y)

    # Save the trained model into a temporary models/ folder
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "model_stable.joblib")

    # Point the Flask app at the temporary folder so it loads our test model
    # instead of looking for a real model on disk. Reset _model to None so
    # the app re-loads it fresh from the temporary path we just set.
    import app as flask_app
    flask_app._model       = None
    flask_app.MODELS_DIR   = str(models_dir)
    flask_app.METRICS_PATH = str(tmp_path / "metrics.json")

    # Create a Flask test client and send a GET request to /health
    client = flask_app.app.test_client()
    resp   = client.get("/health")

    # The response must be HTTP 200 with a healthy status and model_loaded=True
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# TEST 5: /predict returns a salary and salary band for valid input
# ---------------------------------------------------------------------------

def test_predict_endpoint(tmp_path):
    """
    UNIT TEST 5
    /predict must return predicted_salary and salary_band for valid input
    using plain text gender and education level values (not numeric codes),
    which is the format the updated Kaggle-based API now expects.
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Build and train the same minimal model used in the health endpoint test
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=5, random_state=42))
    ])
    X = np.array([
        [22, 0,  0, 2, 0, 0.0,  4],
        [30, 5,  1, 1, 1, 0.16, 3],
        [40, 12, 2, 3, 0, 0.3,  7],
    ])
    y = np.array([50000, 65000, 120000])
    model.fit(X, y)

    # Save the model and redirect the Flask app to the temporary folder
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "model_stable.joblib")

    import app as flask_app
    flask_app._model       = None
    flask_app.MODELS_DIR   = str(models_dir)
    flask_app.METRICS_PATH = str(tmp_path / "metrics.json")

    # Send a POST request with a complete and valid JSON body.
    # Note that gender and education_level are plain text strings,
    # matching the updated API format for the Kaggle dataset.
    client = flask_app.app.test_client()
    resp   = client.post("/predict", json={
        "age":              30,
        "years_experience": 5,
        "education_level":  "Master's",
        "job_role":         3,
        "gender":           "Male"
    })

    # The response must be HTTP 200 with both required fields present
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert "predicted_salary" in body
    assert "salary_band"      in body

    # predicted_salary must be a float, not an integer or string
    assert isinstance(body["predicted_salary"], float)

    # salary_band must be one of the five defined band labels
    assert body["salary_band"] in [
        "Entry Level", "Junior", "Mid Level", "Senior", "Lead / Manager"
    ]


# ---------------------------------------------------------------------------
# TEST 6: /predict returns HTTP 400 when a required field is missing
# ---------------------------------------------------------------------------

def test_predict_missing_field(tmp_path):
    """
    UNIT TEST 6
    /predict must return HTTP 400 when a required field is missing.

    We send a request body that is missing the 'gender' field and verify
    that the API rejects it with a 400 Bad Request and includes an 'error'
    key in the response so the caller knows what went wrong.
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Build a minimal model — the model will never actually be called because
    # the missing field validation happens before any prediction is attempted
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=5, random_state=42))
    ])
    X = np.array([[22, 0, 0, 2, 0, 0.0, 4]])
    y = np.array([50000])
    model.fit(X, y)

    # Save the model and redirect the Flask app to the temporary folder
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(model, models_dir / "model_stable.joblib")

    import app as flask_app
    flask_app._model       = None
    flask_app.MODELS_DIR   = str(models_dir)
    flask_app.METRICS_PATH = str(tmp_path / "metrics.json")

    client = flask_app.app.test_client()

    # Send a request that is missing the required 'gender' field
    resp = client.post("/predict", json={
        "age":              30,
        "years_experience": 5,
        "education_level":  "Master's",
        "job_role":         3
        # 'gender' is intentionally omitted to trigger the 400 response
    })

    # The API must reject this with HTTP 400 and include an error message
    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert "error" in body