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

# Set up logging so every step prints a readable message with its level to the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Build paths relative to this script's location so they work on any machine.
# __file__ is src/merge_data.py, so ".." steps up to the project root.
ROOT_DIR         = os.path.join(os.path.dirname(__file__), "..")
MAIN_DATA_PATH   = os.path.join(ROOT_DIR, "data", "salary_data.csv")        # the permanent training dataset
UPDATE_DATA_PATH = os.path.join(ROOT_DIR, "data", "updated_salary_data.csv") # the new rows to merge in


def merge():
    """
    Loads both CSV files, concatenates them, removes exact duplicate rows,
    saves the result back over the main dataset, and deletes the update file.

    Removing the update file at the end prevents it from being merged again
    on the next run, which would silently inflate the dataset with duplicates.

    Returns the total number of rows in the merged dataset so the caller
    can log a meaningful summary message.
    """

    # Load the main training dataset. If it does not exist, we cannot merge
    # anything into it, so we exit immediately with a clear error message.
    if not os.path.exists(MAIN_DATA_PATH):
        log.error(f"Main dataset not found: {MAIN_DATA_PATH}")
        sys.exit(1)
    main_df = pd.read_csv(MAIN_DATA_PATH)
    log.info(f"Main dataset: {len(main_df)} rows")

    # Load the update dataset containing the new rows to add.
    # This file must have been validated by validate_data.py before this script runs.
    if not os.path.exists(UPDATE_DATA_PATH):
        log.error(f"Update dataset not found: {UPDATE_DATA_PATH}")
        sys.exit(1)
    update_df = pd.read_csv(UPDATE_DATA_PATH)
    log.info(f"Update dataset: {len(update_df)} rows")

    # Stack the two dataframes on top of each other into one combined dataframe.
    # ignore_index=True gives the result a clean 0-based index rather than
    # preserving the original row numbers from each file.
    merged_df = pd.concat([main_df, update_df], ignore_index=True)
    log.info(f"After merge: {len(merged_df)} rows")

    # Remove any rows that are completely identical across every column.
    # This handles the case where the update file contains rows that already
    # exist in the main dataset, which would otherwise skew the model.
    before    = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    after     = len(merged_df)
    log.info(f"After deduplication: {after} rows ({before - after} duplicates removed)")

    # Overwrite the main dataset file with the merged and deduplicated result.
    # index=False prevents pandas from writing the row numbers as an extra column.
    merged_df.to_csv(MAIN_DATA_PATH, index=False)
    log.info(f"Merged dataset saved: {MAIN_DATA_PATH}")

    # Delete the update file now that its rows have been absorbed into the main dataset.
    # Leaving it in place would cause it to be merged again on the next pipeline run.
    os.remove(UPDATE_DATA_PATH)
    log.info(f"Deleted update file: {UPDATE_DATA_PATH}")

    return after


if __name__ == "__main__":
    # Run the merge and print a final summary line showing the new total row count
    total_rows = merge()
    log.info(f"Merge complete — {total_rows} total rows in training dataset")