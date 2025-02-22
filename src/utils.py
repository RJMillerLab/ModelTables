import pandas as pd
import dask.dataframe as dd
import os, re
from tqdm import tqdm
import numpy as np

def clean_title(title):
    """Removes unnecessary BibTeX characters like {} and trims spaces."""
    if title:
        return re.sub(r"[{}]", "", title).strip()
    return title

def load_data(file_path, columns=None):
    """Load data from a Parquet file."""
    # I tried using Dask for parallel loading, but it was slower than Pandas for this dataset
    #df = dd.read_parquet(file_path) # load in parallel
    df = pd.read_parquet(file_path, columns=columns)
    print(f"Loaded {len(df)} rows.")
    return df

def get_statistics_table(unique_by_markdown, key_csv_path = "csv_path"):
    assert key_csv_path in unique_by_markdown.columns, f"Key column {key_csv_path} not found in DataFrame."
    valid_csv_df = unique_by_markdown[unique_by_markdown[key_csv_path].apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)]
    num_tables = len(valid_csv_df)
    num_cols = 0
    total_rows = 0
    for csv_file in tqdm(valid_csv_df[key_csv_path]):
        try:
            df = pd.read_csv(csv_file)
            num_cols += df.shape[1]  # Number of columns in the CSV
            total_rows += df.shape[0]  # Number of rows in the CSV
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    avg_rows = total_rows / num_tables if num_tables > 0 else 0
    new_row = {
        "Benchmark": "SciLake",
        "# Tables": num_tables,
        "# Cols": num_cols,
        "Avg # Rows": int(avg_rows),
        "Size (GB)": "nan"
    }
    # get from starmie paper
    benchmark_data = {
        "Benchmark": ["SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large", "WDC"],
        "# Tables": [550, 1530, 5043, 11090, 50000000],
        "# Cols": [6322, 14810, 54923, 123477, 250000000],
        "Avg # Rows": [6921, 4466, 1915, 7675, 14],
        "Size (GB)": [0.45, 1, 1.5, 11, 500]
    }
    benchmark_df = pd.DataFrame(benchmark_data)
    benchmark_df = pd.concat([benchmark_df, pd.DataFrame([new_row])], ignore_index=True)
    return benchmark_df


def extract_title_from_parsed(parsed_list):
    if isinstance(parsed_list, (list, tuple, np.ndarray)):  
        title_list = []
        for entry in parsed_list:
            if isinstance(entry, dict) and "title" in entry and entry["title"]:
                title_list.append(entry["title"].replace('{', '').replace('}', ''))
    return title_list

def save_analysis_results(df, returnResults, file_name="retrieval_results.csv"):
    # returnResults is like {query_csv: [retrieved_csvs], ...}, here we ignore the parent_path
    assert "csv_path" in df.columns, "csv_path column is required"
    assert "parsed_bibtex_tuple_list" in df.columns, "parsed_bibtex_tuple_list column is required"
    assert "modelId" in df.columns, "modelId column is required"
    df = df.dropna(subset=['csv_path']).copy()
    df['csv_path'] = df['csv_path'].apply(lambda x: x.split('/')[-1]) # only match in the file name !!!
    all_rows = []
    for sample_idx, (query_csv, retrieved_csvs) in enumerate(returnResults.items(), start=1):
        sample_name = f"Sample {sample_idx}"
        block_csvs = [query_csv] + retrieved_csvs
        matched_rows = df[df['csv_path'].isin(block_csvs)]
        if not matched_rows.empty:
            matched_rows = matched_rows.assign(Sample=sample_name)
            matched_rows.loc[matched_rows['csv_path'] == query_csv, 'Type'] = "Query"
            matched_rows.loc[matched_rows['csv_path'] != query_csv, 'Type'] = "Retrieved"
            matched_rows['title'] = matched_rows['parsed_bibtex_tuple_list'].apply(lambda x: extract_title_from_parsed(x) if isinstance(x, (list, tuple, np.ndarray)) else "Unknown title")
            all_rows.extend(matched_rows.to_dict(orient='records'))
    final_df = pd.DataFrame(all_rows, columns=['Sample', 'Type', 'modelId','title', 'parsed_bibtex_tuple_list', 'csv_path'])
    final_df.to_csv(file_name, index=False)
    return final_df