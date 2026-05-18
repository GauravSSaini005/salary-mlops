"""
validate_data.py
----------------
Validates an uploaded dataset before merging.
Exits with code 1 if validation fails.

Usage:
  python src/validate_data.py --data data/updated_salary_data.csv
"""

import os
import sys
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Age", "Gender", "Education Level",
    "Job Title", "Years of Experience", "Salary"
}
VALID_GENDERS    = {"Male", "Female"}
VALID_EDUCATIONS = {"Bachelor's", "Master's", "PhD"}


def validate(path):
    log.info(f"Validating {path}")

    if not os.path.exists(path):
        log.error(f"File not found: {path}")
        return False

    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.error(f"Could not read CSV: {e}")
        return False

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        log.error(f"Missing columns: {missing}")
        return False
    log.info(f"Columns OK: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    after = len(df)
    if after == 0:
        log.error("No valid rows after dropping nulls")
        return False
    log.info(f"Rows: {before} total, {after} valid")

    invalid_gender = df[~df["Gender"].isin(VALID_GENDERS)]
    if len(invalid_gender) > 0:
        log.error(f"Invalid Gender values: {invalid_gender['Gender'].unique()}")
        return False
    log.info("Gender values OK")

    invalid_edu = df[~df["Education Level"].isin(VALID_EDUCATIONS)]
    if len(invalid_edu) > 0:
        log.error(f"Invalid Education Level: {invalid_edu['Education Level'].unique()}")
        return False
    log.info("Education Level values OK")

    invalid_age = df[(df["Age"] <= 16) | (df["Age"] >= 80)]
    if len(invalid_age) > 0:
        log.error(f"Invalid Age values: {invalid_age['Age'].tolist()}")
        return False
    log.info("Age range OK")

    invalid_salary = df[(df["Salary"] <= 10000) | (df["Salary"] >= 1000000)]
    if len(invalid_salary) > 0:
        log.error(f"Invalid Salary values: {invalid_salary['Salary'].tolist()}")
        return False
    log.info("Salary range OK")

    invalid_exp = df[(df["Years of Experience"] < 0) | (df["Years of Experience"] >= 50)]
    if len(invalid_exp) > 0:
        log.error(f"Invalid Years of Experience: {invalid_exp['Years of Experience'].tolist()}")
        return False
    log.info("Years of Experience OK")

    log.info(f"Validation PASSED — {after} valid rows ready for merging")
    return True


def main(path):
    passed = validate(path)
    if not passed:
        log.error("Validation FAILED — fix the dataset before merging")
        sys.exit(1)
    log.info("Validation complete ✓")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    main(args.data)