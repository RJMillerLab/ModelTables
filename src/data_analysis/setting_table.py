"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-03
Description: Get statistics of tables in CSV files from different resources
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

tqdm.pandas()

def get_statistics_table(df, csv_columns):
    benchmarks = []
    aggregate_valid_paths = set()
    all_num_tables = 0
    all_num_cols = 0
    all_total_rows = 0

    for benchmark_name, cols in csv_columns.items():
        if benchmark_name != 'scilake-all':
            valid_paths = []
            for col in cols:
                if col in df.columns:
                    valid_paths.extend([p for paths in df[col].dropna() for p in paths if isinstance(p, str) and os.path.exists(p)])

            aggregate_valid_paths.update(valid_paths)

            num_tables = len(valid_paths)
            num_cols = 0
            total_rows = 0

            for csv_file in tqdm(valid_paths, desc=f"Processing {benchmark_name}"):
                try:
                    csv_df = pd.read_csv(csv_file)
                    num_cols += csv_df.shape[1]
                    total_rows += csv_df.shape[0]
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")

            avg_rows = total_rows / num_tables if num_tables else 0

            benchmarks.append({
                "Benchmark": benchmark_name,
                "# Tables": num_tables,
                "# Cols": num_cols,
                "Avg # Rows": int(avg_rows),
                "Size (GB)": "nan"
            })

            all_num_tables += num_tables
            all_num_cols += num_cols
            all_total_rows += total_rows

    avg_rows_all = all_total_rows / all_num_tables if all_num_tables else 0

    benchmarks.append({
        "Benchmark": "scilake-all",
        "# Tables": all_num_tables,
        "# Cols": all_num_cols,
        "Avg # Rows": int(avg_rows_all),
        "Size (GB)": "nan"
    })

    benchmark_data = { # borrowed from starmie
        "Benchmark": ["SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large", "WDC"],
        "# Tables": [550, 1530, 5043, 11090, 50000000],
        "# Cols": [6322, 14810, 54923, 123477, 250000000],
        "Avg # Rows": [6921, 4466, 1915, 7675, 14],
        "Size (GB)": [0.45, 1, 1.5, 11, 500]
    }

    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df = pd.concat([benchmark_df, pd.DataFrame(benchmarks)], ignore_index=True)
    return benchmark_df

def main():
    df = pd.read_parquet("data/processed/modelcard_step4.parquet")

    csv_columns = {
        'scilake-hugging': ['hugging_table_list'],
        'scilake-github': ['github_table_list'],
        'scilake-html': ['html_table_list_mapped'],
        'scilake-llm': ['llm_table_list_mapped'],
        'scilake-all': ['hugging_table_list', 'github_table_list', 'html_table_list_mapped', 'llm_table_list_mapped']
    }

    print("⚠️ Step 1: Filtering valid CSV paths...")

    benchmark_df = get_statistics_table(df, csv_columns)

    os.makedirs("data/statistics", exist_ok=True)
    benchmark_df.to_parquet("data/statistics/benchmark_results.parquet", index=False)

    print("✅ Statistics saved successfully.")

if __name__ == "__main__":
    main()
