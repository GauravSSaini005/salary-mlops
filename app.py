"""
app.py
------
Flask REST API serving Salary predictions.

This file is the main entry point for the web API. It loads a trained
machine learning model and exposes three endpoints so other applications
or a browser can ask for predictions.

Endpoints:
  GET  /health       -> checks whether the API is alive and the model is loaded
  GET  /model/info   -> returns the current model version and its performance metrics
  POST /predict      -> accepts person details and returns a predicted annual salary
"""

# Standard library imports for file paths, JSON reading, logging, and timestamps
import os
import json
import logging
from datetime import datetime

# joblib is used to load the saved model file from disk
# numpy is used to build the feature array that the model expects
import joblib
import numpy as np

# Flask is the web framework. We use it to create the app, read incoming
# requests, and send back JSON responses.
from flask import Flask, request, jsonify

# Set up logging so that every important action prints a readable message
# to the terminal, showing whether it is INFO or WARNING level.
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Create the Flask application instance
app = Flask(__name__)

# Build file paths relative to wherever this script lives on disk.
# This avoids hardcoding paths that would break on a different machine.
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(ROOT_DIR, "models")       # folder that holds saved model files
METRICS_PATH = os.path.join(ROOT_DIR, "metrics.json") # JSON file written by train.py after training

# This module-level variable holds the loaded model in memory.
# It starts as None and gets filled the first time a request comes in.
_model = None


def load_model():
    """
    Tries to load the best available model from the models folder.

    First preference is 'model_stable.joblib', which is the model that
    passed all quality checks and was promoted to production.
    If that file does not exist, we fall back to the most recently
    versioned model file (e.g. model_v3.joblib) as a safety net.
    If no model file exists at all, we raise an error so the problem
    is obvious rather than silently returning wrong results.
    """
    global _model

    # Try the stable production model first
    stable = os.path.join(MODELS_DIR, "model_stable.joblib")
    if os.path.exists(stable):
        log.info(f"Loading model from {stable}")
        _model = joblib.load(stable)
        return _model

    # Stable model not found, so look for any versioned model file.
    # sorted() gives us them in alphabetical order, so [-1] picks the latest version.
    candidates = sorted([
        f for f in os.listdir(MODELS_DIR)
        if f.startswith("model_v") and f.endswith(".joblib")
    ])
    if candidates:
        path = os.path.join(MODELS_DIR, candidates[-1])
        log.info(f"Fallback model: {path}")
        _model = joblib.load(path)
        return _model

    # No model found at all. The user must run train.py before starting the API.
    raise RuntimeError("No model found. Run train.py first.")


def get_model():
    """
    Returns the loaded model, loading it from disk first if it has not
    been loaded yet. This lazy-loading pattern means we only pay the
    disk-read cost once, on the very first request.
    """
    global _model
    if _model is None:
        load_model()
    return _model


def salary_band(salary):
    """
    Converts a raw predicted salary number into a human-readable band label.
    These bands match the reference table shown on the dashboard page.
    """
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


# ---------------------------------------------------------------------------
# Routes
# Each function below handles one URL. Flask maps the URL and HTTP method
# to the correct function automatically via the @app.route decorator.
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint used by Docker and Kubernetes to know whether
    this container is ready to serve traffic.

    Returns HTTP 200 if the model is loaded and ready.
    Returns HTTP 503 (Service Unavailable) if the model failed to load,
    so the orchestrator knows to restart or replace this instance.
    """
    # Try to get the model. If it raises, we mark the service as degraded.
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
    """
    Returns the current model version and all performance metrics that
    were saved when the model was trained. If the metrics file does not
    exist yet (e.g. before the first training run), we return an empty dict.
    """
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
    Main prediction endpoint. Accepts a JSON body describing a person,
    runs the trained model, and returns the predicted annual salary.

    Expected JSON body:
    {
        "age": 30,
        "years_experience": 5,
        "education_level": "Master's",
        "job_role": 3,
        "gender": "Male"
    }

    education_level: "Bachelor's", "Master's", or "PhD"
    gender:          "Male" or "Female"
    job_role:        integer (0-N based on job title encoding in job_mapping.txt)
    """

    # Read the JSON body from the incoming request.
    # force=True means we try to parse JSON even if the content-type header is missing.
    # silent=True means we get None instead of an exception if parsing fails.
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    # Make sure every required field was included in the request.
    # If anything is missing, tell the caller exactly which fields are absent.
    required = ["age", "years_experience", "education_level", "job_role", "gender"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Parse the three numeric fields first. age and years_experience are floats
    # because the model was trained on floats. job_role is an integer category code.
    try:
        age              = float(data["age"])
        years_experience = float(data["years_experience"])
        job_role         = int(data["job_role"])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid values: {e}"}), 400

    # Encode Gender from a plain English string to the numeric code the model expects.
    # We keep the original string in gender_input so we can echo it back in the response.
    gender_map   = {"Male": 0, "Female": 1}
    gender_input = data["gender"]
    if gender_input not in gender_map:
        return jsonify({"error": "Gender must be 'Male' or 'Female'"}), 422
    gender = gender_map[gender_input]

    # Encode Education Level from a plain English string to the numeric code.
    # We keep the original string in edu_input so we can echo it back in the response.
    education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
    edu_input     = data["education_level"]
    if edu_input not in education_map:
        return jsonify({"error": "education_level must be Bachelor's, Master's, or PhD"}), 422
    education_level = education_map[edu_input]

    # Reject values that are outside the realistic range the model was trained on.
    # Predictions for values outside this range would be unreliable.
    if not (16 < age < 80):
        return jsonify({"error": "Age must be between 16 and 80"}), 422
    if not (0 <= years_experience < 50):
        return jsonify({"error": "years_experience must be 0-50"}), 422

    # Build the two derived features using the exact same formulas as preprocess.py.
    # These must match perfectly — if the formula changes in one place it must
    # change in the other, otherwise live predictions will differ from training.
    experience_ratio = years_experience / age   # how much of their life has been working
    career_score     = education_level * 2 + job_role  # combined seniority signal

    # Assemble all seven features into a 2D numpy array with one row.
    # The model's predict() method always expects a 2D input (rows x columns).
    features = np.array([[
        age, years_experience, education_level,
        job_role, gender, experience_ratio, career_score
    ]])

    # Run the model and extract the single predicted salary value.
    # If something unexpected goes wrong inside the model, we catch it and
    # return a 500 error rather than crashing the whole server.
    try:
        salary_pred = float(get_model().predict(features)[0])
    except Exception as e:
        log.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    # Read the metrics file so we can include the model version in the response.
    # This helps the caller know which model version produced this prediction.
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    # Return the predicted salary along with the salary band label, the currency,
    # the model version, and an echo of the inputs using the original readable
    # strings (e.g. "Master's" and "Male") rather than the encoded numbers,
    # so the caller can easily confirm what was actually used.
    return jsonify({
        "predicted_salary": round(salary_pred, 2),
        "salary_band":      salary_band(salary_pred),
        "currency":         "GBP",
        "version":          metrics.get("version", "unknown"),
        "inputs": {
            "age":               age,
            "years_experience":  years_experience,
            "education_level":   edu_input,    # echoed as readable string, not numeric code
            "job_role":          job_role,
            "gender":            gender_input, # echoed as readable string, not numeric code
        },
    })


# ---------------------------------------------------------------------------
# Dashboard (Tabular HTML Form)
# This route serves a browser-friendly page so users can test the API
# without needing a separate tool like Postman or curl.
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def dashboard():
    """
    Serves the main HTML dashboard page. The page shows three things:
    1. A model status table with current metrics pulled from metrics.json.
    2. A salary band reference table so users know what each band means.
    3. An interactive form where users can enter details and get a prediction.

    Unlike the previous numeric version, the Education Level and Gender
    dropdowns now show plain English options (e.g. "Master's", "Male")
    because the API accepts and encodes them server-side.
    """

    # Load the latest metrics from disk so the dashboard always shows
    # up-to-date numbers without needing to restart the server.
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    # Build the full HTML page as a Python f-string so we can embed
    # the live metric values directly into the table cells.
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Salary MLOps Dashboard</title>
        <style>
            /* Overall page layout - centred, light grey background */
            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            /* Dark navy header bar */
            h1 {{
                text-align: center;
                padding: 20px;
                background: #2c3e50;
                color: white;
                border-radius: 8px;
            }}
            h2 {{
                color: #2c3e50;
                margin-top: 30px;
            }}
            /* Card-style tables with a subtle shadow */
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}
            th {{
                background: #2c3e50;
                color: white;
                padding: 12px 15px;
                text-align: left;
            }}
            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }}
            tr:hover {{ background: #f9f9f9; }}
            /* Green text for passing metrics, red for failing ones */
            .good {{ color: green; font-weight: bold; }}
            .bad  {{ color: red;   font-weight: bold; }}
            /* White card wrapper around the prediction form */
            .form-group {{
                margin-bottom: 15px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            input, select {{
                width: 100%;
                padding: 10px;
                margin-top: 5px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }}
            /* Full-width submit button */
            button {{
                background: #2c3e50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                margin-top: 10px;
            }}
            button:hover {{ background: #34495e; }}
            /* Result section is hidden until a prediction comes back */
            #result {{
                margin-top: 20px;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: none;
            }}
            /* Green header row for the result table to distinguish it visually */
            .result-table th {{ background: #27ae60; }}
            /* Small pill-shaped label used for the model version */
            .badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            }}
            .badge-blue {{ background: #cce5ff; color: #004085; }}
        </style>
    </head>
    <body>

        <h1>Salary Prediction MLOps Pipeline</h1>

        <!-- Model Status Table: shows live metrics read from metrics.json -->
        <h2>Model Status</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Model Version</td>
                <td><span class="badge badge-blue">{metrics.get("version", "unknown")}</span></td>
                <td><span class="good">Active</span></td>
            </tr>
            <tr>
                <td>R2 Score (Accuracy)</td>
                <td>{metrics.get("r2", "N/A")}</td>
                <td><span class="good">{metrics.get("r2", 0)} >= threshold {metrics.get("r2_threshold", 0.8)}</span></td>
            </tr>
            <tr>
                <td>Mean Absolute Error</td>
                <td>GBP {metrics.get("mae", "N/A")}</td>
                <td><span class="good">Acceptable</span></td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>GBP {metrics.get("rmse", "N/A")}</td>
                <td><span class="good">Acceptable</span></td>
            </tr>
            <tr>
                <!-- Threshold check row changes colour depending on whether the model passed -->
                <td>Threshold Check</td>
                <td>R2 >= {metrics.get("r2_threshold", 0.8)}</td>
                <td><span class="{'good' if metrics.get('passed_threshold') else 'bad'}">
                    {'PASSED' if metrics.get('passed_threshold') else 'FAILED - Retraining triggered'}
                </span></td>
            </tr>
            <tr>
                <td>Last Trained</td>
                <td colspan="2">{metrics.get("timestamp", "N/A")}</td>
            </tr>
        </table>

        <!-- Salary Band Reference Table: helps users understand the prediction output -->
        <h2>Salary Band Reference</h2>
        <table>
            <tr>
                <th>Band</th>
                <th>Salary Range</th>
                <th>Typical Role</th>
            </tr>
            <tr><td>Entry Level</td><td>Below GBP 25,000</td><td>Graduate / Intern</td></tr>
            <tr><td>Junior</td><td>GBP 25,000 to 40,000</td><td>Junior Developer / Analyst</td></tr>
            <tr><td>Mid Level</td><td>GBP 40,000 to 60,000</td><td>Software Engineer / Data Analyst</td></tr>
            <tr><td>Senior</td><td>GBP 60,000 to 85,000</td><td>Senior Engineer / Tech Lead</td></tr>
            <tr><td>Lead / Manager</td><td>Above GBP 85,000</td><td>Engineering Manager / Director</td></tr>
        </table>

        <!-- Prediction Form: user fills in their details and clicks the button.
             Education Level and Gender use plain English option values because
             the API now accepts and encodes them server-side. -->
        <h2>Predict Salary</h2>
        <div class="form-group">
            <table style="box-shadow:none; margin-bottom:0;">
                <tr>
                    <th>Field</th>
                    <th>Input</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>Age</td>
                    <td><input type="number" id="age" value="30" min="17" max="79"></td>
                    <td>Person's age in years</td>
                </tr>
                <tr>
                    <td>Years Experience</td>
                    <td><input type="number" id="exp" value="5" min="0" max="49"></td>
                    <td>Total years of work experience</td>
                </tr>
                <tr>
                    <td>Education Level</td>
                    <td>
                        <!-- Values are plain text strings — the API encodes them to numbers -->
                        <select id="edu">
                            <option value="Bachelor's">Bachelor's</option>
                            <option value="Master's" selected>Master's</option>
                            <option value="PhD">PhD</option>
                        </select>
                    </td>
                    <td>Highest qualification</td>
                </tr>
                <tr>
                    <td>Job Role</td>
                    <td>
                        <!-- Values are integer codes from the job_mapping.txt file -->
                        <select id="role">
                            <option value="0">0 - Junior</option>
                            <option value="1">1 - Mid Level</option>
                            <option value="2">2 - Senior</option>
                            <option value="3" selected>3 - Lead</option>
                            <option value="4">4 - Manager</option>
                        </select>
                    </td>
                    <td>Current job level</td>
                </tr>
                <tr>
                    <td>Gender</td>
                    <td>
                        <!-- Values are plain text strings — the API encodes them to numbers -->
                        <select id="gender">
                            <option value="Male" selected>Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </td>
                    <td>Gender</td>
                </tr>
            </table>
            <button onclick="predict()">Predict Salary</button>
        </div>

        <!-- Result Table: hidden by default, shown after a successful prediction -->
        <div id="result">
            <h2>Prediction Result</h2>
            <table class="result-table">
                <tr>
                    <th>Field</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Predicted Salary</td>
                    <!-- Large green text to make the salary figure stand out -->
                    <td id="res-salary" style="font-size:1.4em; font-weight:bold; color:#27ae60;"></td>
                </tr>
                <tr>
                    <td>Salary Band</td>
                    <td id="res-band"></td>
                </tr>
                <tr>
                    <td>Model Version</td>
                    <td id="res-version"></td>
                </tr>
                <tr>
                    <td>Age</td>
                    <td id="res-age"></td>
                </tr>
                <tr>
                    <td>Years Experience</td>
                    <td id="res-exp"></td>
                </tr>
                <tr>
                    <td>Education Level</td>
                    <td id="res-edu"></td>
                </tr>
                <tr>
                    <td>Job Role</td>
                    <td id="res-role"></td>
                </tr>
                <tr>
                    <td>Gender</td>
                    <td id="res-gender"></td>
                </tr>
            </table>
        </div>

        <script>
            // Lookup array so we can show a readable role label next to the numeric code
            const roleLabels = ["Junior", "Mid Level", "Senior", "Lead", "Manager"];

            async function predict() {{
                // Read every form field and build the JSON body for the API call.
                // Education level and gender are sent as plain text strings exactly
                // as the user selected them — the API handles encoding server-side.
                const body = {{
                    age:              parseInt(document.getElementById("age").value),
                    years_experience: parseInt(document.getElementById("exp").value),
                    education_level:  document.getElementById("edu").value,
                    job_role:         parseInt(document.getElementById("role").value),
                    gender:           document.getElementById("gender").value,
                }};

                try {{
                    // Send the form data to the /predict endpoint and wait for the response
                    const resp = await fetch("/predict", {{
                        method:  "POST",
                        headers: {{"Content-Type": "application/json"}},
                        body:    JSON.stringify(body),
                    }});
                    const data = await resp.json();

                    // If the API returned an error field, show it and stop here
                    if (data.error) {{
                        alert("Error: " + data.error);
                        return;
                    }}

                    // Fill each result table cell with the value returned by the API.
                    // toLocaleString() adds comma separators to the salary number (e.g. 85,000).
                    document.getElementById("res-salary").textContent =
                        "GBP " + data.predicted_salary.toLocaleString();
                    document.getElementById("res-band").textContent =
                        data.salary_band;
                    document.getElementById("res-version").textContent =
                        data.version;
                    document.getElementById("res-age").textContent =
                        data.inputs.age + " years";
                    document.getElementById("res-exp").textContent =
                        data.inputs.years_experience + " years";
                    // Education level and gender come back as readable strings from the API
                    document.getElementById("res-edu").textContent =
                        data.inputs.education_level;
                    // Show the readable role label; fall back to the raw code if out of range
                    document.getElementById("res-role").textContent =
                        roleLabels[data.inputs.job_role] || data.inputs.job_role;
                    document.getElementById("res-gender").textContent =
                        data.inputs.gender;

                    // Make the result section visible now that we have data to show
                    document.getElementById("result").style.display = "block";
                }} catch(e) {{
                    // If the fetch or JSON parsing fails, show a simple alert
                    alert("Prediction failed: " + e);
                }}
            }}
        </script>

    </body>
    </html>
    """
    return html


# ---------------------------------------------------------------------------
# Entry point
# This block only runs when the file is executed directly (python app.py).
# It does not run when Flask is started via gunicorn or another WSGI server.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Try to load the model at startup so the first real request is fast.
    # If no model exists yet, we log a warning and continue anyway --
    # the health endpoint will report degraded until a model is trained.
    try:
        load_model()
        log.info("Model pre-loaded successfully")
    except RuntimeError as e:
        log.warning(str(e))

    # Read the port from an environment variable so it can be changed
    # without editing the code (useful in Docker or cloud deployments).
    port  = int(os.getenv("PORT", 5000))

    # Enable Flask's debug mode only when running locally in development.
    # Debug mode should never be on in production because it exposes internals.
    debug = os.getenv("FLASK_ENV", "production") == "development"

    app.run(host="0.0.0.0", port=port, debug=debug)