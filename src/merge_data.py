"""
merge_data.py
-------------
Merges updated_salary_data.csv into salary_data.csv,
removes duplicates, and deletes the update file.

Usage:
  python src/merge_data.py
"""

import os
import sys
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR         = os.path.join(os.path.dirname(__file__), "..")
MAIN_DATA_PATH   = os.path.join(ROOT_DIR, "data", "salary_data.csv")
UPDATE_DATA_PATH = os.path.join(ROOT_DIR, "data", "updated_salary_data.csv")


def merge():
    # Load main dataset
    if not os.path.exists(MAIN_DATA_PATH):
        log.error(f"Main dataset not found: {MAIN_DATA_PATH}")
        sys.exit(1)

    main_df = pd.read_csv(MAIN_DATA_PATH)
    log.info(f"Main dataset: {len(main_df)} rows")

    # Load update dataset
    if not os.path.exists(UPDATE_DATA_PATH):
        log.error(f"Update dataset not found: {UPDATE_DATA_PATH}")
        sys.exit(1)

    update_df = pd.read_csv(UPDATE_DATA_PATH)
    log.info(f"Update dataset: {len(update_df)} rows")

    # Merge datasets
    merged_df = pd.concat([main_df, update_df], ignore_index=True)
    log.info(f"After merge: {len(merged_df)} rows")

    # Remove exact duplicates
    before = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    after = len(merged_df)
    log.info(f"After deduplication: {after} rows ({before - after} duplicates removed)")

    # Save merged dataset back to main file
    merged_df.to_csv(MAIN_DATA_PATH, index=False)
    log.info(f"Merged dataset saved to {MAIN_DATA_PATH}")

    # Clear the update file instead of deleting it
    # This keeps the file in the repo but removes all data rows
    # so it is ready for the next batch of new data
    with open(UPDATE_DATA_PATH, "w") as f:
        f.write("Age,Gender,Education Level,Job Title,Years of Experience,Salary\n")
    log.info(f"Cleared update file (kept headers): {UPDATE_DATA_PATH}")

    return after


if __name__ == "__main__":
    total_rows = merge()
    log.info(f"Merge complete — {total_rows} total rows in training dataset")