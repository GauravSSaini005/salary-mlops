"""
validate_data.py
----------------
Validates an uploaded dataset before it is merged into
the main training data. Exits with code 1 if validation
fails so the CI pipeline stops immediately.
Usage:
  python src/validate_data.py --data data/updated_salary_data.csv
"""

import os
import sys
import argparse
import logging

import pandas as pd

# Set up logging so every check prints a clear pass or fail message to the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# The columns that must be present in any uploaded CSV before we accept it.
# These match the raw Kaggle column names — encoding happens later in preprocess.py.
REQUIRED_COLUMNS = {
    "Age", "Gender", "Education Level",
    "Job Title", "Years of Experience", "Salary"
}

# The only accepted values for Gender and Education Level.
# Any row containing a value outside these sets is flagged as invalid.
VALID_GENDERS    = {"Male", "Female"}
VALID_EDUCATIONS = {"Bachelor's", "Master's", "PhD"}


def validate(path: str) -> bool:
    """
    Runs a series of checks against the CSV at the given path and returns
    True if every check passes, or False as soon as any check fails.

    The checks run in order from cheapest to most specific:
    file exists -> readable CSV -> required columns -> no empty rows ->
    valid category values -> numeric ranges.

    Returning False early means we stop at the first problem and log a clear
    error message so the person uploading the data knows exactly what to fix.
    """
    log.info(f"Validating {path}")

    # Confirm the file actually exists on disk before trying to open it
    if not os.path.exists(path):
        log.error(f"File not found: {path}")
        return False

    # Try to read the CSV. If the file is corrupt or not a valid CSV,
    # we catch the exception and report it rather than crashing.
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.error(f"Could not read CSV: {e}")
        return False

    # Check that every required column is present in the file.
    # We do this before any row-level checks so the error message is clear.
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        log.error(f"Missing columns: {missing}")
        return False
    log.info(f"Columns OK: {list(df.columns)}")

    # Drop rows that have a missing value in any required column, then check
    # that at least some rows remain. A file with all nulls is useless for training.
    before = len(df)
    df     = df.dropna(subset=list(REQUIRED_COLUMNS))
    after  = len(df)
    if after == 0:
        log.error("No valid rows found after dropping nulls")
        return False
    log.info(f"Rows: {before} total, {after} valid")

    # Check that every Gender value is one of the two accepted strings.
    # Any unexpected value (e.g. a typo or a third category) would break encoding.
    invalid_gender = df[~df["Gender"].isin(VALID_GENDERS)]
    if len(invalid_gender) > 0:
        log.error(f"Invalid Gender values: {invalid_gender['Gender'].unique()}")
        log.error(f"Valid values are: {VALID_GENDERS}")
        return False
    log.info("Gender values OK")

    # Check that every Education Level value is one of the three accepted strings.
    # Any other value would fail the encoding map in preprocess.py and produce NaN.
    invalid_edu = df[~df["Education Level"].isin(VALID_EDUCATIONS)]
    if len(invalid_edu) > 0:
        log.error(f"Invalid Education Level values: {invalid_edu['Education Level'].unique()}")
        log.error(f"Valid values are: {VALID_EDUCATIONS}")
        return False
    log.info("Education Level values OK")

    # Check that all ages fall within the realistic working range.
    # Values at or below 16 or at or above 80 are outside what the model was trained on.
    invalid_age = df[(df["Age"] <= 16) | (df["Age"] >= 80)]
    if len(invalid_age) > 0:
        log.error(f"Invalid Age values found: {invalid_age['Age'].tolist()}")
        return False
    log.info("Age range OK")

    # Check that all salaries fall within a plausible range.
    # Values at or below 10,000 or at or above 1,000,000 are likely data entry errors.
    invalid_salary = df[(df["Salary"] <= 10000) | (df["Salary"] >= 1000000)]
    if len(invalid_salary) > 0:
        log.error(f"Invalid Salary values found: {invalid_salary['Salary'].tolist()}")
        return False
    log.info("Salary range OK")

    # Check that years of experience is non-negative and below 50.
    # Negative values and implausibly high values would distort the ExperienceRatio feature.
    invalid_exp = df[
        (df["Years of Experience"] < 0) | (df["Years of Experience"] >= 50)
    ]
    if len(invalid_exp) > 0:
        log.error(f"Invalid Years of Experience: {invalid_exp['Years of Experience'].tolist()}")
        return False
    log.info("Years of Experience OK")

    # All checks passed — log a summary so the CI log is easy to read
    log.info(f"Validation PASSED — {after} valid rows ready for training")
    return True


def main(path: str):
    """
    Calls validate() and exits with the appropriate code so GitHub Actions
    knows whether to continue the pipeline or stop and report a failure.

    Exit code 0 means the dataset is clean and safe to merge.
    Exit code 1 means something is wrong and the dataset must be fixed first.
    """
    passed = validate(path)

    if not passed:
        log.error("Validation FAILED — fix the dataset before merging")
        sys.exit(1)

    # Validation passed — exit cleanly so the next pipeline step can run
    log.info("Validation complete")
    sys.exit(0)


if __name__ == "__main__":
    # Accept the path to the new dataset as a required command line argument.
    # required=True means the script will print a usage error if --data is omitted.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the new dataset CSV to validate"
    )
    args = parser.parse_args()
    main(args.data)