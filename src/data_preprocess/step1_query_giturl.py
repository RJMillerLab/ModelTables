"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-11
Description: Load the GitHub README cache data from a parquet file,
store it in an in-memory SQLite database, and allow SQL query.
Usage:
    (get abnormal README.md file) find data/downloaded_github_readmes -type f -exec stat -f "%z %N" {} \; | sort -nr | head -n 10
    python -m src.data_preprocess.step1_query_giturl load --query 	/Users/doradong/Repo/CitationLake/data/downloaded_github_readmes/595398559da5e7d4bea35b73dda5abe9.md
"""

import argparse
import os
import sys
import pandas as pd
import sqlite3
from src.utils import load_config

# Fixed configuration and file paths
config = load_config('config.yaml')
PROCESSED_DIR = os.path.join(config.get('base_path'), 'processed')
PARQUET_FILE = os.path.join(PROCESSED_DIR, "github_readme_cache.parquet")

def load_and_query(query_value):
    """
    Load the parquet file into a DataFrame, write it into an in-memory SQLite database,
    and execute an SQL query to find the raw_url corresponding to the given downloaded_path.
    """
    try:
        df = pd.read_parquet(PARQUET_FILE)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    # Create an in-memory SQLite database and load the DataFrame into a table.
    try:
        conn = sqlite3.connect(":memory:")
        df.to_sql("cache_table", conn, if_exists="replace", index=False)
    except Exception as e:
        print(f"Error loading DataFrame into SQLite: {e}")
        sys.exit(1)

    try:
        cursor = conn.cursor()
        # Query for the record matching the downloaded markdown filename.
        cursor.execute("SELECT raw_url FROM cache_table WHERE downloaded_path = ?;", (query_value,))
        result = cursor.fetchone()
        conn.close()
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        sys.exit(1)
    
    if result:
        print(f"GitHub URL for '{query_value}': {result[0]}")
    else:
        print(f"Query '{query_value}' not found in the database.")

def main():
    parser = argparse.ArgumentParser(
        description="Load GitHub README cache data from parquet and query using SQL."
    )
    parser.add_argument("mode", choices=["load"],
                        help="Mode of operation: currently only 'load' is supported")
    parser.add_argument("--query", help="Downloaded markdown filename to query (e.g., README.md)", default=None)
    args = parser.parse_args()

    if args.mode == "load":
        if not args.query:
            print("Please provide a query using the --query parameter.")
            sys.exit(1)
        load_and_query(args.query)

if __name__ == "__main__":
    main()
