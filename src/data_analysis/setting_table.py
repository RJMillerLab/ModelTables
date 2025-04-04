"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-04
Description: Get statistics of tables in CSV files from different resources
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os

def process_csv_file(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if df.shape[1] == 0:
            return {"path": csv_file, "status": "zero_col"}, None
        if df.shape[0] == 0:
            return {"path": csv_file, "status": "zero_row"}, None
        if df.shape[0] == 1:
            return {"path": csv_file, "status": "one_row"}, None
        return {"path": csv_file, "rows": df.shape[0], "cols": df.shape[1], "status": "valid"}, None
    except Exception as e:
        return None, f"Error reading {csv_file}: {e}"

def get_statistics_table(df, csv_columns, n_jobs=8): ########
    benchmarks = []
    aggregate_valid_paths = set()
    all_num_tables = 0
    all_num_cols = 0
    all_total_rows = 0

    for benchmark_name, cols in csv_columns.items():
        if benchmark_name != 'scilake-all':  # for scilake-all, we re-use the count for previous resources
            valid_paths = []
            for col in cols:
                if col in df.columns:
                    valid_paths.extend([p for paths in df[col].dropna() for p in paths if isinstance(p, str) and os.path.exists(p)])

            aggregate_valid_paths.update(valid_paths)

            results = Parallel(n_jobs=n_jobs)(delayed(process_csv_file)(p) for p in tqdm(valid_paths, desc=f"Processing {benchmark_name}")) ########

            valid_file_list = []
            one_row_list = [] ########
            zero_row_list = [] ########
            zero_col_list = [] ########

            num_tables = 0
            num_cols = 0
            total_rows = 0

            for res, err in results:
                if err:
                    print(err)
                elif res:
                    status = res.get("status")
                    if status == "valid":
                        valid_file_list.append(res["path"])
                        num_tables += 1
                        num_cols += res["cols"]
                        total_rows += res["rows"]
                    elif status == "one_row":
                        one_row_list.append(res["path"])
                        print(f"Warning: {res['path']} has only 1 row, skipping.")
                    elif status == "zero_row":
                        zero_row_list.append(res["path"])
                        print(f"Warning: {res['path']} has 0 rows, skipping.")
                    elif status == "zero_col":
                        zero_col_list.append(res["path"])
                        print(f"Warning: {res['path']} has 0 columns, skipping.")

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

            base_name = benchmark_name.split('-')[-1]
            with open(f"data/analysis/valid_file_list_{base_name}.txt", "w") as f:
                for path in valid_file_list:
                    f.write(path + "\n")
            with open(f"data/analysis/one_row_list_{base_name}.txt", "w") as f:
                for path in one_row_list:
                    f.write(path + "\n")
            with open(f"data/analysis/zero_row_list_{base_name}.txt", "w") as f:
                for path in zero_row_list:
                    f.write(path + "\n")
            with open(f"data/analysis/zero_col_list_{base_name}.txt", "w") as f:
                for path in zero_col_list:
                    f.write(path + "\n")

    avg_rows_all = all_total_rows / all_num_tables if all_num_tables else 0

    benchmarks.append({
        "Benchmark": "scilake-all",
        "# Tables": all_num_tables,
        "# Cols": all_num_cols,
        "Avg # Rows": int(avg_rows_all),
        "Size (GB)": "nan"
    })

    benchmark_data = {  # borrowed from starmie
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

    os.makedirs("data/analysis", exist_ok=True)

    benchmark_df = get_statistics_table(df, csv_columns, n_jobs=8) ########

    benchmark_df.to_parquet("data/analysis/exp_setting_tab1.parquet", index=False)

    print("✅ Statistics saved successfully.")

if __name__ == "__main__":
    main()
