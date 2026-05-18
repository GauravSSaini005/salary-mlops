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
        log.error(