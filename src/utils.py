import pandas as pd
import dask.dataframe as dd
import os, re, json
from tqdm import tqdm
import numpy as np
import yaml
import duckdb
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_combined_data(data_type, file_path="~/Repo/CitationLake/data/raw", columns=[]):
    assert data_type in ["modelcard", "datasetcard"], "data_type must be 'modelcard' or 'datasetcard'"
    if data_type == "modelcard":
        file_names = [f"train-0000{i}-of-00004.parquet" for i in range(4)]
    elif data_type == "datasetcard":
        #file_names = [f"train-0000{i}-of-00003.parquet" for i in range(3)]
        file_names = [f"train-0000{i}-of-00002.parquet" for i in range(2)]
    if columns:
        dfs = [pd.read_parquet(os.path.join(file_path, file), columns=columns) for file in file_names]
    else:
        dfs = [pd.read_parquet(os.path.join(file_path, file)) for file in file_names]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def get_statistics_card(df):
    """
    Get statistics for the card data.
    -- Usage --
    stats = get_statistics_card(df)
    print(json.dumps(stats, indent=4))
    """
    assert "card" in df.columns, "card column is required"
    total_models = len(df)
    non_empty_model_cards = df['card'].notna().sum()
    created_at_dates = pd.to_datetime(df['createdAt'], errors='coerce')
    start_date = created_at_dates.min()
    end_date = created_at_dates.max()
    last_modiifed_dates = pd.to_datetime(df['last_modified'], errors='coerce')
    modified_early_date = last_modiifed_dates.min()
    modified_end_date = last_modiifed_dates.max()
    stats = {
        "Total Models": int(total_models),
        "Models with Non-Empty Model Card": int(non_empty_model_cards),
        "Start Date (createdAt)": str(start_date.isoformat()),
        "End Date (createdAt)": str(end_date.isoformat()),
        "Last Modified Early Date": str(modified_early_date.isoformat()),
        "Last Modified Last Date": str(modified_end_date.isoformat()),
    }
    return stats

def clean_title(title):
    """Removes unnecessary BibTeX characters like {} and trims spaces."""
    if title:
        return re.sub(r"[{}]", "", title).strip()
    return title

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


def safe_json_dumps(x):
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    elif isinstance(x, (list, tuple)):
        return json.dumps(x)
    else:
        return x

def load_table_from_duckdb(table_name, db_path="modellake_all.db"):
    """
    Load a table from DuckDB database as a pandas DataFrame.
    
    Args:
        table_name (str): Name of the table to load
        db_path (str): Path to the DuckDB database file
    
    Returns:
        pd.DataFrame: Loaded data from the table
    """
    print(f"📊 Loading table '{table_name}' from DuckDB: {db_path}")
    
    # Connect to DuckDB
    conn = duckdb.connect(db_path)
    
    try:
        # Check if table exists
        table_exists = conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()[0]
        
        if table_exists == 0:
            raise ValueError(f"Table '{table_name}' not found in database {db_path}")
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"   Found {row_count:,} rows in table '{table_name}'")
        
        # Load data using DuckDB's pandas integration
        df = conn.execute(f"SELECT * FROM {table_name}").df()
        
        print(f"✅ Successfully loaded {len(df):,} rows from DuckDB")
        return df
        
    finally:
        # Close connection
        conn.close()


def load_table_from_sqlite(table_name, db_path="modellake_all.db", parquet_path=None):
    """
    Load a table from SQLite database as a pandas DataFrame.
    If the table doesn't exist, it will import from parquet file to SQLite first.
    
    Args:
        table_name (str): Name of the table to load
        db_path (str): Path to the SQLite database file
        parquet_path (str, optional): Path to parquet file to import if table doesn't exist
    
    Returns:
        pd.DataFrame: Loaded data from the table
    """
    print(f"📊 Loading table '{table_name}' from SQLite: {db_path}")
    
    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    
    try:
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            if parquet_path is None:
                raise ValueError(f"Table '{table_name}' not found in database {db_path} and no parquet_path provided")
            
            print(f"   Table '{table_name}' not found. Importing from parquet: {parquet_path}")
            
            # Read parquet file
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_path)
            print(f"   Loaded {len(df):,} rows from parquet file")
            
            # Save to SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"   Imported {len(df):,} rows to SQLite table '{table_name}'")
            
        else:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"   Found {row_count:,} rows in table '{table_name}'")
            
            # Load data
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        print(f"✅ Successfully loaded {len(df):,} rows from SQLite")
        return df
        
    finally:
        # Close connection
        conn.close()



