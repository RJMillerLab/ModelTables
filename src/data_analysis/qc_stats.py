"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-05
Description: Get statistics of tables in CSV files from different resources (optimized with joblib)
             with additional model-level quality control. For each benchmark resource, two rows are generated:
             one with deduped (weighted) statistics and one (labeled "(sym)") with raw statistics computed by processing
             each CSV file instance (so if a file appears twice, its rows and columns are counted twice).
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os, ast

INPUT_FILE = "data/processed/modelcard_step3_dedup.parquet"
BENCHMARK_FILE = "data/analysis/benchmark_results.parquet"
OUTPUT_ANALYSIS_DIR = "data/analysis"

def get_valid_paths_from_series(series):
    valid_paths = []
    for item in series.dropna():
        if isinstance(item, (list, tuple, np.ndarray)):
            lst = item
        else:
            raise ValueError
        valid_paths.extend([p for p in lst if isinstance(p, str) and os.path.exists(p)])
    return valid_paths

def process_csv_file(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return {"path": csv_file, "rows": df.shape[0], "cols": df.shape[1], "status": "valid"}, None
    except Exception as e:
        return None, f"Error reading {csv_file}: {e}"

def get_statistics_table(df, csv_columns, n_jobs=8):
    benchmarks = []
    aggregate_valid_paths = set()
    all_num_tables = 0
    all_num_cols = 0
    all_total_rows = 0
    dedup_all_num_tables = 0
    dedup_all_num_cols = 0
    dedup_all_total_rows = 0

    for benchmark_name, cols in csv_columns.items():
        if benchmark_name != 'scilake-all':  # for scilake-all, we re-use the count for previous resources
            # Collect raw valid paths (with duplicates)
            raw_valid_paths = []
            for col in cols:
                raw_valid_paths.extend(get_valid_paths_from_series(df[col]))
            #raw_valid_count = len(raw_valid_paths)

            # Create frequency dictionary for raw valid paths (for deduped stats)
            freq = {}
            for p in raw_valid_paths:
                freq[p] = freq.get(p, 0) + 1

            # Deduplicate the valid paths, There exist duplicate paths only across rows
            valid_paths = list(set(raw_valid_paths))
            #dedup_valid_count = len(valid_paths)
            aggregate_valid_paths.update(valid_paths)

            # Process deduped valid paths (weighted by frequency)
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_csv_file)(p) 
                for p in tqdm(valid_paths, desc=f"Processing {benchmark_name}")
            )

            valid_file_list = []

            num_tables = 0
            num_cols = 0
            total_rows = 0
            dedup_num_tables = 0
            dedup_num_cols = 0
            dedup_total_rows = 0

            for res, err in results:
                if err:
                    print(err)
                elif res:
                    status = res.get("status")
                    if status == "valid":
                        count = freq.get(res["path"], 1)
                        valid_file_list.append(res["path"])
                        num_tables += count
                        dedup_num_tables += 1
                        num_cols += count * res["cols"]
                        dedup_num_cols += res["cols"]
                        total_rows += count * res["rows"]
                        dedup_total_rows += res["rows"]
                    else:
                        print(f"Invalid file: {res['path']}")

            avg_rows = total_rows / num_tables if num_tables else 0
            dedup_avg_rows = dedup_total_rows / dedup_num_tables if dedup_num_tables else 0

            dedup_stats = {
                "Benchmark": benchmark_name,
                "# Tables": dedup_num_tables,
                "# Cols": dedup_num_cols,
                "Avg # Rows": int(dedup_avg_rows),
                "Size (GB)": np.nan
            }
            raw_stats = {
                "Benchmark": benchmark_name + " (sym)",
                "# Tables": num_tables,
                "# Cols": num_cols,
                "Avg # Rows": int(avg_rows),
                "Size (GB)": np.nan
            }
            benchmarks.append(dedup_stats)
            benchmarks.append(raw_stats)

            all_num_tables += num_tables
            all_num_cols += num_cols
            all_total_rows += total_rows
            dedup_all_num_tables += dedup_num_tables
            dedup_all_num_cols += dedup_num_cols
            dedup_all_total_rows += dedup_total_rows

            base_name = benchmark_name.split('-')[-1]
            with open(os.path.join(OUTPUT_ANALYSIS_DIR, f"valid_file_list_{base_name}.txt"), "w") as f:
                for path in valid_file_list:
                    f.write(path + "\n")

    avg_rows_all = all_total_rows / all_num_tables if all_num_tables else 0
    dedup_avg_rows_all = dedup_all_total_rows / dedup_all_num_tables if dedup_all_num_tables else 0

    benchmarks.append({
        "Benchmark": "scilake-all",
        "# Tables": dedup_all_num_tables,
        "# Cols": dedup_all_num_cols,
        "Avg # Rows": int(dedup_avg_rows_all),
        "Size (GB)": np.nan
    })
    benchmarks.append({
        "Benchmark": "scilake-all (sym)",
        "# Tables": all_num_tables,
        "# Cols": all_num_cols,
        "Avg # Rows": int(avg_rows_all),
        "Size (GB)": np.nan
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
    df = pd.read_parquet(INPUT_FILE)

    csv_columns = {
        'scilake-hugging': ['hugging_table_list_dedup'],
        'scilake-github': ['github_table_list_dedup'],
        'scilake-html': ['html_table_list_mapped_dedup'],
        'scilake-llm': ['llm_table_list_mapped_dedup'],
        'scilake-all': ['hugging_table_list_dedup', 'github_table_list_dedup',
                         'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup']
    }

    print("⚠️ Step 1: Filtering valid CSV paths...")

    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

    benchmark_df = get_statistics_table(df, csv_columns, n_jobs=-1)

    benchmark_df.to_parquet(BENCHMARK_FILE, index=False)

    print("✅ Statistics saved successfully.")

if __name__ == "__main__":
    main()
