#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-09-27
Last Updated: 2025-09-28
Description: Load all .csv/.sv/.tsv files from a directory into a single database.
Each file becomes a table named after the file's basename (sanitized).
Supports parallel processing and direct zip file processing for faster loading of large datasets.

Usage examples:
  # Process individual CSV files with parallel processing (default)
  python src/data_analysis/load_sv_to_db.py --engine sqlite --db-path deduped_github_csvs_v2.sqlite \
    --input-dir data/processed/deduped_github_csvs_v2 --workers 8 

  # Process zip files directly (MUCH faster for large datasets)
  python src/data_analysis/load_sv_to_db.py --engine sqlite --db-path deduped_github_csvs_v2.sqlite \
    --input-dir data/processed/deduped_github_csvs_v2 --use-zip

  # Process zip files with DuckDB (recommended for large datasets)
  python src/data_analysis/load_sv_to_db.py --engine duckdb --db-path deduped_github_csvs_v2.duckdb \
    --input-dir data/processed/deduped_github_csvs_v2 --use-zip

  # Specify number of workers for individual files
  python src/data_analysis/load_sv_to_db.py --engine duckdb --workers 8 \
    --input-dir data/processed/deduped_hugging_csvs_v2

  # Disable parallel processing (sequential)
  python src/data_analysis/load_sv_to_db.py --engine sqlite --no-parallel \
    --input-dir data/processed/deduped_hugging_csvs_v2

  # Force specific separator
  python src/data_analysis/load_sv_to_db.py --engine duckdb --sep ',' \
    --input-dir data/processed/deduped_hugging_csvs_v2
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import List, Tuple, Optional, Iterator
import io
import pandas as pd  # top-level import for type hints and potential fallback
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil
import zipfile
from pathlib import Path


def read_csv_with_fallback(path: str, sep: str | None):
    """Try reading CSV with multiple strategies and return a DataFrame.

    Strategies:
    - If file is empty, return empty DataFrame.
    - If sep is provided: try read_csv with low_memory=False.
    - If sep is None: try auto-detect with engine='python' (no low_memory),
      then fall back to common separators and encodings.
    """
    # Quick empty file check
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
    except OSError:
        # if file is unreadable, let pandas handle it later
        pass

    # If user forced a separator, trust it first
    if sep is not None:
        return pd.read_csv(path, sep=sep, low_memory=False)

    # Try auto-detect first using python engine (no low_memory)
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as e_auto:
        last_exc = e_auto

    # If auto-detect fails (e.g. Could not determine delimiter), try common separators
    common_seps = [",", "\t", ";", "|"]
    encodings = ["utf-8", "latin1"]
    for enc in encodings:
        for s in common_seps:
            try:
                return pd.read_csv(path, sep=s, encoding=enc, low_memory=False)
            except Exception as e:
                last_exc = e

    # As a last resort, try reading as a single-column by keeping the full line
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = [l.rstrip("\n") for l in fh]
        return pd.DataFrame({"line": lines})
    except Exception:
        pass

    # If everything failed, re-raise the last exception for logging
    raise last_exc


def sanitize_table_name(name: str) -> str:
    # Replace non-word characters with underscores, ensure not starting with digit
    n = re.sub(r"\W+", "_", name)
    if re.match(r"^\d", n):
        n = "t_" + n
    return n


def find_files(input_dir: str, exts: List[str]) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(set(files))
    return files


def find_zip_files(input_dir: str) -> List[str]:
    """Find all zip files in the input directory."""
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    return sorted(zip_files)


def process_zip_file(zip_path: str, sep: str | None) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Process a zip file and yield (table_name, dataframe) tuples.
    
    This function streams through the zip file without extracting it to disk.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Get list of CSV files in the zip
            csv_files = [f for f in zip_file.namelist() 
                        if f.lower().endswith(('.csv', '.tsv', '.sv')) and not f.startswith('__MACOSX')]
            
            for csv_file in csv_files:
                try:
                    # Read CSV directly from zip
                    with zip_file.open(csv_file) as f:
                        # Convert to text stream
                        text_stream = io.TextIOWrapper(f, encoding='utf-8', errors='replace')
                        
                        # Read CSV
                        df = read_csv_with_fallback_from_stream(text_stream, sep, csv_file)
                        
                        if not df.empty:
                            # Generate table name from zip file and CSV file
                            zip_base = os.path.splitext(os.path.basename(zip_path))[0]
                            csv_base = os.path.splitext(os.path.basename(csv_file))[0]
                            table_name = sanitize_table_name(f"{zip_base}_{csv_base}")
                            
                            yield (table_name, df)
                            
                except Exception as e:
                    print(f"  Failed to process {csv_file} in {zip_path}: {e}", file=sys.stderr)
                    continue
                    
    except Exception as e:
        print(f"Failed to open zip file {zip_path}: {e}", file=sys.stderr)


def read_csv_with_fallback_from_stream(stream, sep: str | None, filename: str = "unknown"):
    """Read CSV from a text stream with fallback strategies."""
    try:
        # Reset stream position
        stream.seek(0)
        
        # If user forced a separator, trust it first
        if sep is not None:
            return pd.read_csv(stream, sep=sep, low_memory=False)
        
        # Try auto-detect first using python engine
        stream.seek(0)
        try:
            return pd.read_csv(stream, sep=None, engine="python")
        except Exception as e_auto:
            last_exc = e_auto
        
        # If auto-detect fails, try common separators
        common_seps = [",", "\t", ";", "|"]
        for s in common_seps:
            try:
                stream.seek(0)
                return pd.read_csv(stream, sep=s, low_memory=False)
            except Exception as e:
                last_exc = e
        
        # As a last resort, try reading as a single-column
        try:
            stream.seek(0)
            lines = [line.rstrip("\n") for line in stream]
            return pd.DataFrame({"line": lines})
        except Exception:
            pass
        
        # If everything failed, re-raise the last exception
        raise last_exc
        
    except Exception as e:
        print(f"  Failed to read {filename}: {e}", file=sys.stderr)
        return pd.DataFrame()


def process_single_file(file_info: Tuple[str, int, str | None]) -> Tuple[str, str, pd.DataFrame | None]:
    """Process a single file and return (table_name, file_path, dataframe).
    
    Args:
        file_info: Tuple of (file_path, file_index, separator)
    
    Returns:
        Tuple of (table_name, file_path, dataframe or None if failed)
    """
    file_path, file_index, sep = file_info
    base = os.path.splitext(os.path.basename(file_path))[0]
    table_name = sanitize_table_name(base)
    
    try:
        df = read_csv_with_fallback(file_path, sep)
        return (table_name, file_path, df)
    except Exception as e:
        print(f"  Failed to read {file_path}: {e}", file=sys.stderr)
        return (table_name, file_path, None)


def load_zip_to_duckdb(db_path: str, zip_files: List[str], sep: str | None) -> int:
    """Load CSV files from zip files directly into DuckDB."""
    try:
        import duckdb
        import pandas as pd
    except Exception as e:
        print("DuckDB path chosen but required packages are missing:", e, file=sys.stderr)
        return 2

    conn = duckdb.connect(db_path)

    # optional progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(zip_files), desc="Processing zip files")
    except Exception:
        progress_bar = None

    total_tables = 0

    for i, zip_path in enumerate(zip_files, 1):
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            _tqdm.write(f"[{i}/{len(zip_files)}] Processing {zip_path}")
        except Exception:
            print(f"[{i}/{len(zip_files)}] Processing {zip_path}")
        
        zip_tables = 0
        for table_name, df in process_zip_file(zip_path, sep):
            try:
                tmp_name = f"__tmp_df_{total_tables}"
                # Register pandas dataframe and create/replace table
                conn.register(tmp_name, df)
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM {tmp_name}')
                # unregistering is safe but duckdb will overwrite the name on next register
                try:
                    conn.unregister(tmp_name)
                except Exception:
                    # older duckdb versions may not have unregister
                    pass
                
                zip_tables += 1
                total_tables += 1
                
            except Exception as e:
                print(f"  Failed to write table {table_name}: {e}", file=sys.stderr)
                continue
        
        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix(tables=total_tables)
        
        print(f"  Created {zip_tables} tables from {zip_path}")

    if progress_bar:
        progress_bar.close()

    conn.close()
    print(f"Successfully created {total_tables} tables from {len(zip_files)} zip files")
    return 0


def load_zip_to_sqlite(db_path: str, zip_files: List[str], sep: str | None) -> int:
    """Load CSV files from zip files directly into SQLite."""
    try:
        import pandas as pd
        import sqlite3
    except Exception as e:
        print("SQLite path chosen but required packages are missing:", e, file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # optional progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(zip_files), desc="Processing zip files")
    except Exception:
        progress_bar = None

    total_tables = 0
    
    for i, zip_path in enumerate(zip_files, 1):
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            _tqdm.write(f"[{i}/{len(zip_files)}] Processing {zip_path}")
        except Exception:
            print(f"[{i}/{len(zip_files)}] Processing {zip_path}")
        
        zip_tables = 0
        for table_name, df in process_zip_file(zip_path, sep):
            try:
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                zip_tables += 1
                total_tables += 1
                
        except Exception as e:
                print(f"  Failed to write table {table_name}: {e}", file=sys.stderr)
            continue
        
        if progress_bar:
            progress_bar.update(1)
            progress_bar.set_postfix(tables=total_tables)
        
        print(f"  Created {zip_tables} tables from {zip_path}")

    if progress_bar:
        progress_bar.close()

    conn.close()
    print(f"Successfully created {total_tables} tables from {len(zip_files)} zip files")
    return 0


def load_to_duckdb(db_path: str, files: List[str], sep: str | None, max_workers: int = None) -> int:
    try:
        import duckdb
        import pandas as pd
    except Exception as e:
        print("DuckDB path chosen but required packages are missing:", e, file=sys.stderr)
        return 2

    if max_workers is None:
        max_workers = min(cpu_count(), len(files))

    print(f"Using {max_workers} workers for parallel processing...")

    # Process files in parallel
    file_infos = [(f, i, sep) for i, f in enumerate(files, 1)]

    # optional progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(files), desc="Processing files")
    except Exception:
        progress_bar = None

    processed_data = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_info): file_info[0] 
            for file_info in file_infos
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                table_name, _, df = future.result()
                if df is not None:
                    processed_data.append((table_name, file_path, df))
                if progress_bar:
                    progress_bar.update(1)
            except Exception as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)
                if progress_bar:
                    progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    print(f"Successfully processed {len(processed_data)} files. Writing to DuckDB...")

    # Now write to DuckDB sequentially (DuckDB doesn't handle concurrent writes well)
    conn = duckdb.connect(db_path)
    
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(processed_data, 1), total=len(processed_data), desc="Writing to DuckDB")
    except Exception:
        iterator = enumerate(processed_data, 1)

    for i, (table_name, file_path, df) in iterator:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            _tqdm.write(f"[{i}/{len(processed_data)}] Writing {file_path} -> table: {table_name}")
        except Exception:
            print(f"[{i}/{len(processed_data)}] Writing {file_path} -> table: {table_name}")

        tmp_name = f"__tmp_df_{i}"
        # Register pandas dataframe and create/replace table
        conn.register(tmp_name, df)
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM {tmp_name}')
        # unregistering is safe but duckdb will overwrite the name on next register
        try:
            conn.unregister(tmp_name)
        except Exception:
            # older duckdb versions may not have unregister
            pass

    conn.close()
    return 0


def load_to_sqlite(db_path: str, files: List[str], sep: str | None, max_workers: int = None) -> int:
    try:
        import pandas as pd
        import sqlite3
    except Exception as e:
        print("SQLite path chosen but required packages are missing:", e, file=sys.stderr)
        return 2

    if max_workers is None:
        max_workers = min(cpu_count(), len(files))

    print(f"Using {max_workers} workers for parallel processing...")

    # Process files in parallel
    file_infos = [(f, i, sep) for i, f in enumerate(files, 1)]
    
    # optional progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(files), desc="Processing files")
    except Exception:
        progress_bar = None

    processed_data = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_info): file_info[0] 
            for file_info in file_infos
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                table_name, _, df = future.result()
                if df is not None:
                    processed_data.append((table_name, file_path, df))
                if progress_bar:
                    progress_bar.update(1)
            except Exception as e:
                print(f"Error processing {file_path}: {e}", file=sys.stderr)
                if progress_bar:
                    progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    print(f"Successfully processed {len(processed_data)} files. Writing to SQLite...")

    # Now write to SQLite sequentially (SQLite doesn't handle concurrent writes well)
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(processed_data, 1), total=len(processed_data), desc="Writing to SQLite")
    except Exception:
        iterator = enumerate(processed_data, 1)

    for i, (table_name, file_path, df) in iterator:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            _tqdm.write(f"[{i}/{len(processed_data)}] Writing {file_path} -> table: {table_name}")
        except Exception:
            print(f"[{i}/{len(processed_data)}] Writing {file_path} -> table: {table_name}")
        
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        except Exception as e:
            print(f"  Failed to write table {table_name} to sqlite: {e}", file=sys.stderr)
            continue

    conn.close()
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load CSV/SV/TSV files into duckdb or sqlite")
    parser.add_argument("--engine", choices=["duckdb", "sqlite"], default="duckdb",
                        help="Database engine to use (duckdb recommended)")
    parser.add_argument("--db-path", default=None, help="Path to output DB file")
    parser.add_argument("--input-dir", default="data/processed/deduped_hugging_csvs_v2",
                        help="Directory containing .csv/.sv/.tsv files")
    parser.add_argument("--sep", default=None, help="Force a separator (e.g. ',' or '\t'). By default the script will try to auto-detect")
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of parallel workers (default: min(cpu_count, file_count))")
    parser.add_argument("--no-parallel", action="store_true", 
                        help="Disable parallel processing (use sequential processing)")
    parser.add_argument("--use-zip", action="store_true",
                        help="Process zip files instead of individual CSV files (much faster for large datasets)")
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 3

    # choose sensible default db paths
    if args.db_path:
        db_path = args.db_path
    else:
        if args.engine == "duckdb":
            db_path = os.path.join(input_dir, "combined.duckdb")
        else:
            db_path = os.path.join(input_dir, "combined.sqlite")

    if args.use_zip:
        # Process zip files
        zip_files = find_zip_files(input_dir)
        if not zip_files:
            print(f"No zip files found in {input_dir}", file=sys.stderr)
            return 4
        
        print(f"Found {len(zip_files)} zip files. Writing to {args.engine} DB at: {db_path}")
        print("Note: This will process all CSV files inside the zip files directly without extraction.")
        
        if args.engine == "duckdb":
            return load_zip_to_duckdb(db_path, zip_files, args.sep)
        else:
            return load_zip_to_sqlite(db_path, zip_files, args.sep)
    else:
        # Process individual CSV files
    exts = ["*.csv", "*.sv", "*.tsv"]
    files = find_files(input_dir, exts)
    if not files:
        print(f"No files found in {input_dir} with extensions: {exts}", file=sys.stderr)
        return 4

    print(f"Found {len(files)} files. Writing to {args.engine} DB at: {db_path}")
        
        # Determine number of workers
        if args.no_parallel:
            max_workers = 1
        elif args.workers is not None:
            max_workers = args.workers
        else:
            max_workers = min(cpu_count(), len(files))

    if args.engine == "duckdb":
            return load_to_duckdb(db_path, files, args.sep, max_workers)
    else:
            return load_to_sqlite(db_path, files, args.sep, max_workers)


if __name__ == "__main__":
    raise SystemExit(main())
