"""
preprocess.py
-------------
Loads raw salary CSV, validates it, cleans bad rows,
engineers new features, and saves train/test splits.
"""

import os
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Age", "YearsExperience", "EducationLevel", "JobRole", "Salary"}
DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_data(path):
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    log.info(f"Loaded {len(df)} rows")
    return df


def clean_data(df):
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df[(df["Age"] > 16) & (df["Age"] < 80)]
    df = df[(df["YearsExperience"] >= 0) & (df["YearsExperience"] < 50)]
    df = df[(df["Salary"] > 10000) & (df["Salary"] < 500000)]
    df = df[df["EducationLevel"].isin([0, 1, 2])]
    df = df[df["JobRole"].isin([0, 1, 2, 3, 4])]
    log.info(f"Cleaned: {before} → {len(df)} rows")
    return df.reset_index(drop=True)


def engineer_features(df):
    df = df.copy()
    # How much experience relative to age
    df["ExperienceRatio"] = df["YearsExperience"] / df["Age"]
    # Combined seniority score from education and job role
    df["CareerScore"] = df["EducationLevel"] * 2 + df["JobRole"]
    log.info("Features engineered: ExperienceRatio, CareerScore")
    return df


def split_and_save(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feature_cols = [
        "Age", "YearsExperience", "EducationLevel",
        "JobRole", "Gender", "ExperienceRatio", "CareerScore"
    ]
    target_col = "Salary"
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)

    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    log.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")


def main(raw_path):
    df = load_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    split_and_save(df)
    log.info("Preprocessing complete ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(DATA_DIR, "salary_data.csv")
    )
    args = parser.parse_args()
    main(args.data)