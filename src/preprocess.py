"""
preprocess.py
-------------
Loads Kaggle salary CSV, encodes text columns,
engineers features, saves train/test splits.
Column encoding:
  Gender:          Male=0, Female=1
  Education Level: Bachelor's=0, Master's=1, PhD=2
  Job Title:       encoded to integer category
"""

import os
import argparse
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging so every step prints a readable message with its level to the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# The columns that must exist in the raw Kaggle CSV before we do anything else.
# These are the original Kaggle column names — renaming and encoding happen later.
# If any of these are missing we raise an error immediately so the problem is obvious.
REQUIRED_COLUMNS = {
    "Age", "Gender", "Education Level",
    "Job Title", "Years of Experience", "Salary"
}

# Build paths relative to this script's location so they work on any machine.
# __file__ is src/preprocess.py, so ".." steps up to the project root.
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # where train.csv and test.csv will be saved

# Encoding maps defined at module level so they can be imported by app.py as well.
# This ensures live predictions use the exact same mapping as training.
GENDER_MAP    = {"Male": 0, "Female": 1}
EDUCATION_MAP = {"Bachelor's": 0, "Master's": 1, "PhD": 2}


def load_data(path):
    """
    Reads the raw Kaggle CSV from disk and checks that all required columns
    are present before any processing begins. Raises a ValueError immediately
    if any column is missing so the error is obvious rather than surfacing
    later as a cryptic KeyError deep in the pipeline.
    """
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # Find any required columns that are absent from the loaded file
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    log.info(f"Loaded {len(df)} rows")
    return df


def clean_data(df):
    """
    Removes rows that would produce unreliable training examples.
    Each filter targets a specific kind of bad data:
    - Missing values in any required column
    - Ages outside the realistic working range (17 to 79)
    - Negative or implausibly high years of experience
    - Salaries that are clearly erroneous (too low or astronomically high)

    Note: we filter on the original Kaggle column names here because
    encoding has not happened yet at this stage.
    """
    before = len(df)

    # Drop any row that has a missing value in one of the required columns
    df = df.dropna(subset=list(REQUIRED_COLUMNS))

    # Keep only rows with a realistic working age
    df = df[(df["Age"] > 16) & (df["Age"] < 80)]

    # Keep only rows where years of experience is non-negative and below 50
    df = df[(df["Years of Experience"] >= 0) & (df["Years of Experience"] < 50)]

    # Keep only rows where the salary is within a plausible range.
    # The upper limit is 1,000,000 here (higher than the numeric dataset)
    # to accommodate the wider salary range in the real Kaggle data.
    df = df[(df["Salary"] > 10000) & (df["Salary"] < 1000000)]

    log.info(f"Cleaned: {before} -> {len(df)} rows")

    # reset_index(drop=True) gives the cleaned dataframe a fresh 0-based index
    # so row numbers stay consistent for any code that accesses rows by position
    return df.reset_index(drop=True)


def encode_columns(df):
    """
    Converts the three raw Kaggle text columns into the numeric codes the
    model expects, and renames columns to match the rest of the codebase.

    Gender and Education Level use fixed maps defined at the top of this file.
    Job Title uses pandas category codes, which assigns a unique integer to
    each distinct job title alphabetically. The mapping is saved to a text
    file in data/processed/ so it can be inspected or reused by app.py.
    """
    # Work on a copy so we never modify the dataframe that was passed in
    df = df.copy()

    # Encode Gender using the fixed map: Male=0, Female=1
    df["Gender"] = df["Gender"].map(GENDER_MAP)

    # Encode Education Level using the fixed map: Bachelor's=0, Master's=1, PhD=2
    df["EducationLevel"] = df["Education Level"].map(EDUCATION_MAP)

    # Encode Job Title by converting it to a pandas category and using the
    # auto-generated integer codes. This handles any number of distinct job titles
    # without needing a hardcoded list, but the mapping must be saved so we can
    # decode predictions back to readable titles later if needed.
    df["JobRole"] = df["Job Title"].astype("category").cat.codes

    # Save the job title mapping to a text file so anyone can look up which
    # integer code corresponds to which job title.
    job_mapping  = dict(enumerate(df["Job Title"].astype("category").cat.categories))
    mapping_path = os.path.join(PROCESSED_DIR, "job_mapping.txt")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(mapping_path, "w") as f:
        for code, title in job_mapping.items():
            f.write(f"{code}: {title}\n")
    log.info(f"Job title mapping saved: {mapping_path}")

    # Rename the Kaggle column to the shorter name used everywhere else in the code
    df["YearsExperience"] = df["Years of Experience"]

    log.info("Columns encoded: Gender, EducationLevel, JobRole, YearsExperience")
    return df


def engineer_features(df):
    """
    Creates two new columns derived from the encoded data.
    These derived features give the model extra signal that the raw columns
    alone do not provide. The same formulas must be used in app.py when
    building features for a live prediction, otherwise the model receives
    different input at inference time than it saw during training.
    """
    # Work on a copy so we never modify the dataframe that was passed in
    df = df.copy()

    # ExperienceRatio: how much of a person's life has been spent working.
    # A 25-year-old with 5 years experience has a higher ratio than a
    # 40-year-old with the same 5 years, which the model can use as a signal.
    df["ExperienceRatio"] = df["YearsExperience"] / df["Age"]

    # CareerScore: combines education level and job role into one seniority number.
    # Education is weighted by 2 because it tends to have a stronger influence
    # on salary than the job role code alone.
    df["CareerScore"] = df["EducationLevel"] * 2 + df["JobRole"]

    log.info("Features engineered: ExperienceRatio, CareerScore")
    return df


def split_and_save(df):
    """
    Splits the cleaned and encoded data into a training set (80%) and a test
    set (20%), then saves each as a CSV file in data/processed/.

    random_state=42 makes the split reproducible — the same rows will always
    end up in train and test, which keeps evaluation results comparable across runs.
    """
    # Create the processed output folder if it does not already exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # These seven columns are the inputs the model will be trained on.
    # The order here must stay consistent with train.py and evaluate.py.
    feature_cols = [
        "Age", "YearsExperience", "EducationLevel",
        "JobRole", "Gender", "ExperienceRatio", "CareerScore"
    ]
    target_col = "Salary"

    # Separate features from the target label before splitting
    X = df[feature_cols]
    y = df[target_col]

    # Split into 80% train and 20% test with a fixed random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Recombine features and labels into a single dataframe for each split
    # so each CSV file is self-contained and easy to load in train.py and evaluate.py.
    # .values is used on y to avoid index mismatch when assigning back to the dataframe.
    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)

    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    log.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")


def main(raw_path):
    """
    Runs the full preprocessing pipeline in order:
    load -> clean -> encode columns -> engineer features -> split and save.

    Encoding must happen before feature engineering because ExperienceRatio
    and CareerScore depend on the numeric YearsExperience and EducationLevel
    columns that encoding produces.
    """
    df = load_data(raw_path)
    df = clean_data(df)
    df = encode_columns(df)
    df = engineer_features(df)
    split_and_save(df)
    log.info("Preprocessing complete")


if __name__ == "__main__":
    # Accept the path to the raw Kaggle CSV as a command line argument so this
    # script can be called from GitHub Actions or the terminal with different files.
    # If no --data argument is given, it defaults to data/salary_data.csv.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(DATA_DIR, "salary_data.csv")
    )
    args = parser.parse_args()
    main(args.data)