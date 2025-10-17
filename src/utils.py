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

def find_v2_csv_path(original_path):
    """Find v2 version of CSV file if it exists, otherwise return original path."""
    # Check if file exists
    if not os.path.exists(original_path):
        return original_path
    
    # Get directory and filename
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    
    # Look for v2 directory
    v2_dir = dir_path.replace('deduped_hugging_csvs', 'deduped_hugging_csvs_v2')
    v2_dir = v2_dir.replace('deduped_github_csvs', 'deduped_github_csvs_v2')
    v2_dir = v2_dir.replace('tables_output', 'tables_output_v2')
    
    # Check if v2 directory exists
    if not os.path.exists(v2_dir):
        return original_path
    
    # Look for v2 file
    v2_path = os.path.join(v2_dir, filename)
    if os.path.exists(v2_path):
        return v2_path
    
    return original_path

def get_statistics_table(unique_by_markdown, key_csv_path = "csv_path", use_v2=False):
    assert key_csv_path in unique_by_markdown.columns, f"Key column {key_csv_path} not found in DataFrame."
    valid_csv_df = unique_by_markdown[unique_by_markdown[key_csv_path].apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)]
    num_tables = len(valid_csv_df)
    num_cols = 0
    total_rows = 0
    for csv_file in tqdm(valid_csv_df[key_csv_path]):
        try:
            # Use v2 version if requested and available
            actual_csv_file = find_v2_csv_path(csv_file) if use_v2 else csv_file
            df = pd.read_csv(actual_csv_file)
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




# ------------------------
# Optimized Parquet writer
# ------------------------
def _choose_int_type_for_range(min_value: int, max_value: int):
    """Return the smallest Arrow signed integer type that can represent [min, max]."""
    if min_value >= -128 and max_value <= 127:
        return pa.int8()
    if min_value >= -32768 and max_value <= 32767:
        return pa.int16()
    if min_value >= -2147483648 and max_value <= 2147483647:
        return pa.int32()
    return pa.int64()

def _safe_convert_to_int(series, target_type):
    """Safely convert series to target integer type, falling back to int64 if needed."""
    try:
        # Try converting to target type
        if target_type == pa.int8():
            return pa.array(series.astype("Int8"))
        elif target_type == pa.int16():
            return pa.array(series.astype("Int16"))
        elif target_type == pa.int32():
            return pa.array(series.astype("Int32"))
        else:
            return pa.array(series.astype("Int64"))
    except (ValueError, OverflowError, TypeError):
        # If conversion fails, fall back to int64
        return pa.array(series.astype("Int64"))


def _infer_list_element_type(values):
    """Infer Arrow element type for a list column without loss, supporting strings and integers.
    Falls back to string if mixed or unknown.
    """
    has_null = False
    min_v, max_v = None, None
    all_int = True
    all_str = True
    for lst in values:
        if lst is None:
            has_null = True
            continue
        # Support numpy arrays too
        if isinstance(lst, np.ndarray):
            lst = lst.tolist()
        if not isinstance(lst, (list, tuple)):
            # non-list present; give up and treat as string list
            return pa.string()
        for x in lst:
            if x is None:
                has_null = True
                continue
            sx = str(x)
            try:
                ix = int(x)
                # Check if it fits in int64 range
                if ix < -9223372036854775808 or ix > 9223372036854775807:
                    all_int = False
                    break
            except Exception:
                all_int = False
            else:
                if min_v is None:
                    min_v = ix
                    max_v = ix
                else:
                    if ix < min_v:
                        min_v = ix
                    if ix > max_v:
                        max_v = ix
            if not isinstance(x, (str, bytes)):
                all_str = False
            # If we already know it's mixed, break early
            if not all_int and not all_str:
                break
        if not all_int and not all_str:
            break
    if all_str and not all_int:
        return pa.string()
    if all_int and min_v is not None and max_v is not None:
        # Only try to downcast if values fit in int32 range
        if min_v >= -2147483648 and max_v <= 2147483647:
            return _choose_int_type_for_range(min_v, max_v)
        else:
            # Large integers, use int64
            return pa.int64()
    # default: string
    return pa.string()


def _is_simple_string_list_column(series):
    """Check if column is a simple string list column (np.array and elements are strings, not complex structures like dicts/structs)"""
    if series.dtype != 'object':
        return False
    
    # Check first few samples
    for val in series.head(10):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if isinstance(val, np.ndarray):
            # Empty array 
            if len(val) == 0:
                continue
            # Check if array elements are simple strings (not dict, list, tuple, etc.)
            for x in val[:5]:
                if not isinstance(x, str) or isinstance(x, (dict, list, tuple, np.ndarray)):
                    return False
        elif isinstance(val, (list, tuple)):
            # Empty list 
            if len(val) == 0:
                continue
            # Check if list elements are simple strings (not dicts/structs)
            for x in val[:5]:  # Only check first 5 elements
                if not isinstance(x, str) or isinstance(x, (dict, list, tuple, np.ndarray)):
                    return False
        else:
            return False
    return True

def _is_struct_list_column(series):
    """Check if column contains lists of structs/dictionaries (should NOT be compressed)"""
    if series.dtype != 'object':
        return False
    
    # Check first few samples
    for val in series.head(10):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if isinstance(val, np.ndarray):
            # Empty array 
            if len(val) == 0:
                continue
            # Check if array elements are dictionaries/structs
            for x in val[:5]:
                if isinstance(x, dict):
                    return True
        elif isinstance(val, (list, tuple)):
            # Empty list 
            if len(val) == 0:
                continue
            # Check if list elements are dictionaries/structs
            for x in val[:5]:  # Only check first 5 elements
                if isinstance(x, dict):
                    return True
    return False

def save_parquet_optimized(
    df: pd.DataFrame,
    output_path: str,
    *,
    compression: str = "zstd",
    compression_level: int = 6,
    compress_list_cols: tuple = None,
    downcast_integers: bool = True,
    downcast_floats_to_fp32: bool = False,
    use_dictionary: bool = True,
):
    """Save a DataFrame to Parquet with compact Arrow types and ZSTD compression.

    Args:
        compress_list_cols: columns to compress
    """
    if compress_list_cols is None:
        compress_list_cols = []
        for col in df.columns:
            # Only compress simple string lists, NOT struct lists
            if _is_simple_string_list_column(df[col]) and not _is_struct_list_column(df[col]):
                compress_list_cols.append(col)
        print(f"Auto detected columns to compress: {compress_list_cols}")
    else:
        print(f"Manually specified columns to compress: {compress_list_cols}")

    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []

    for col in df.columns:
        s = df[col]

        # Check if need to compress to Arrow List[string]
        if col in compress_list_cols and _is_simple_string_list_column(s) and not _is_struct_list_column(s):
            print(f"  Compress column {col}: np.array -> Arrow List[string]")
            # Process list column: convert to Arrow List[string]
            values = []
            for v in s.tolist():
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    values.append([])
                    continue
                seq = v.tolist() if isinstance(v, np.ndarray) else list(v)
                values.append([str(x) for x in seq])

            # Use Arrow List[string]
            list_type = pa.list_(pa.string())
            arrays.append(pa.array(values, type=list_type))
            fields.append(pa.field(col, list_type))
            continue

        # Other columns keep original type - convert to Arrow preserving original structure
        print(f"  Keep column {col} unchanged: {s.dtype}")
        
        # Convert pandas Series to Arrow Array preserving the original data structure
        try:
            # Use pyarrow to convert the entire DataFrame column to preserve complex types
            # This is more reliable than converting individual series
            temp_df = pd.DataFrame({col: s})
            temp_table = pa.Table.from_pandas(temp_df, preserve_index=False)
            arrow_array = temp_table[col]
            arrays.append(arrow_array)
            fields.append(pa.field(col, arrow_array.type))
        except Exception as e:
            print(f"    Warning: Could not preserve original type for {col}, falling back to string: {e}")
            # Fallback to string if conversion fails
            arrays.append(pa.array(s.astype(str)))
            fields.append(pa.field(col, pa.string()))

    schema = pa.schema(fields)
    table = pa.Table.from_arrays(arrays, schema=schema)

    pq.write_table(
        table,
        output_path,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=use_dictionary,
    )

    return output_path


def to_parquet(df: pd.DataFrame, output_path: str, **kwargs):
    """
    Alias for save_parquet_optimized to maintain compatibility with pandas .to_parquet() interface.
    
    Args:
        df: pandas DataFrame to save
        output_path: path where to save the parquet file
        **kwargs: additional arguments passed to save_parquet_optimized
    
    Returns:
        str: output_path
    """
    return save_parquet_optimized(df, output_path, **kwargs)


def sanitize_table_separators(obj):
    """
    Remove ALL separator rows (cells made only of '-', ':', spaces).
    - If obj is a markdown string table: remove all separator rows.
    - If obj is a pandas DataFrame: remove all separator rows.
    
    CSV files don't need markdown-style separators, so we remove them all.
    
    Handles various separator patterns:
    - :----: (with any number of dashes)
    - ---- (just dashes)
    - :---: (with colons)
    - Pure spaces
    - Mixed : - and spaces
    """
    import pandas as pd
    import re

    # More comprehensive regex to handle various separator patterns
    # Matches: :----:, ----, :---:, mixed : - spaces, but NOT pure spaces or empty
    cell_sep_re = re.compile(r"^\s*:?[-:]+:?\s*$")

    # Case 1: Markdown string table
    if isinstance(obj, str):
        table = obj
        if '|' not in table:
            return table
        lines = table.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            
            # Skip completely empty lines
            if not stripped:
                continue
                
            if not (stripped.startswith('|') and stripped.endswith('|')):
                cleaned_lines.append(line)
                continue
            inner = stripped[1:-1]
            cells = [c.strip() for c in inner.split('|')]
            # Filter out empty cells
            non_empty_cells = [c for c in cells if c != '']
            
            # Skip rows with only empty cells
            if not non_empty_cells:
                continue
                
            # Remove ALL separator rows (CSV doesn't need them)
            if all(cell_sep_re.match(c) is not None for c in non_empty_cells):
                continue  # Skip this separator row
            else:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    # Case 2: DataFrame (arxiv HTML parsed tables)
    if isinstance(obj, pd.DataFrame):
        df = obj
        if df.empty:
            return df
        keep_mask = []
        for _, row in df.iterrows():
            # Convert all values to strings and strip whitespace
            values = [str(v).strip() for v in row.tolist()]
            # Filter out empty strings
            non_empty = [v for v in values if v != '']
            
            # Remove completely empty rows (no useful information)
            if len(non_empty) == 0:
                keep_mask.append(False)
                continue
                
            # Remove ALL separator rows (CSV doesn't need them)
            is_sep_row = all(cell_sep_re.match(v) is not None for v in non_empty)
            if is_sep_row:
                keep_mask.append(False)  # Remove separator row
            else:
                keep_mask.append(True)   # Keep data row
        return df.loc[keep_mask].reset_index(drop=True)

    # Default: return unchanged
    return obj


def clean_dataframe_for_analysis(df, drop_empty_rows=True, drop_empty_cols=True, preserve_empty_cells=True):
    """
    清理DataFrame用于分析，可选择性地删除空行/列，同时保留空cell用于视觉布局。
    
    Args:
        df: pandas DataFrame
        drop_empty_rows: 是否删除完全空行
        drop_empty_cols: 是否删除完全空列  
        preserve_empty_cells: 是否保留空cell（空字符串），False会转换为NA
    
    Returns:
        清理后的DataFrame
    """
    if df is None or df.empty:
        return df
    
    result_df = df.copy()
    
    # 处理空cell
    if not preserve_empty_cells:
        result_df = result_df.replace(r'^\s*$', pd.NA, regex=True)
    
    # 删除空行
    if drop_empty_rows:
        if preserve_empty_cells:
            # 保留空cell时，使用更精确的空行检测（strip后为空字符串的行）
            mask = result_df.apply(lambda row: all(str(val).strip() == '' for val in row), axis=1)
            result_df = result_df[~mask]
        else:
            # 转换为NA后，使用pandas的dropna
            result_df = result_df.dropna(axis=0, how='all')
    
    # 删除空列
    if drop_empty_cols:
        if preserve_empty_cells:
            # 保留空cell时，使用更精确的空列检测（strip后为空字符串的列）
            mask = result_df.apply(lambda col: all(str(val).strip() == '' for val in col), axis=0)
            result_df = result_df.loc[:, ~mask]
        else:
            # 转换为NA后，使用pandas的dropna
            result_df = result_df.dropna(axis=1, how='all')
    
    # 如果保留空cell，将只包含空格的cell标准化为空字符串
    if preserve_empty_cells:
        result_df = result_df.map(lambda x: '' if str(x).strip() == '' else x)
    
    return result_df.reset_index(drop=True)


def read_csv_for_analysis(csv_path, **kwargs):
    """
    读取CSV文件并清理用于分析。
    默认删除空行和空列，但保留空cell。
    """
    # 确保不把空字符串转换为NaN
    if 'keep_default_na' not in kwargs:
        kwargs['keep_default_na'] = False
    df = pd.read_csv(csv_path, **kwargs)
    return clean_dataframe_for_analysis(df, drop_empty_rows=True, drop_empty_cols=True, preserve_empty_cells=True)


def read_csv_for_display(csv_path, **kwargs):
    """
    读取CSV文件用于显示，保留所有视觉布局。
    不删除空行/列，保留空cell。
    """
    # 确保不把空字符串转换为NaN
    if 'keep_default_na' not in kwargs:
        kwargs['keep_default_na'] = False
    df = pd.read_csv(csv_path, **kwargs)
    return clean_dataframe_for_analysis(df, drop_empty_rows=False, drop_empty_cols=False, preserve_empty_cells=True)

