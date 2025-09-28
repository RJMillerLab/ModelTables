#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-09-27
Description: Load all .csv/.sv/.tsv files from a directory into a single database.
Each file becomes a table named after the file's basename (sanitized).

Usage examples:
  python src/data_analysis/load_sv_to_db.py --engine duckdb --db-path deduped_hugging_csvs_v2.duckdb \
    --input-dir data/processed/deduped_hugging_csvs_v2

  python src/data_analysis/load_sv_to_db.py --engine sqlite --db-path deduped_hugging_csvs_v2.sqlite \
    --input-dir data/processed/deduped_hugging_csvs_v2
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import List
import io
import pandas as pd  # top-level import for type hints and potential fallback


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


def load_to_duckdb(db_path: str, files: List[str], sep: str | None) -> int:
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
        iterator = tqdm(enumerate(files, start=1), total=len(files), desc="Loading to duckdb")
    except Exception:
        iterator = enumerate(files, start=1)

    for i, f in iterator:
        base = os.path.splitext(os.path.basename(f))[0]
        table = sanitize_table_name(base)
        try:
            # when using tqdm, use tqdm.write to avoid clobbering
            from tqdm import tqdm as _tqdm  # type: ignore
            _tqdm.write(f"[{i}/{len(files)}] Reading {f} -> table: {table}")
        except Exception:
            print(f"[{i}/{len(files)}] Reading {f} -> table: {table}")
        try:
            df = read_csv_with_fallback(f, sep)
        except Exception as e:
            print(f"  Failed to read {f}: {e}", file=sys.stderr)
            continue

        tmp_name = f"__tmp_df_{i}"
        # Register pandas dataframe and create/replace table
        conn.register(tmp_name, df)
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM {tmp_name}')
        # unregistering is safe but duckdb will overwrite the name on next register
        try:
            conn.unregister(tmp_name)
        except Exception:
            # older duckdb versions may not have unregister
            pass

    conn.close()
    return 0


def load_to_sqlite(db_path: str, files: List[str], sep: str | None) -> int:
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
        iterator = tqdm(enumerate(files, start=1), total=len(files), desc="Loading to sqlite")
    except Exception:
        iterator = enumerate(files, start=1)

    for i, f in iterator:
        base = os.path.splitext(os.path.basename(f))[0]
        table = sanitize_table_name(base)
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            _tqdm.write(f"[{i}/{len(files)}] Reading {f} -> table: {table}")
        except Exception:
            print(f"[{i}/{len(files)}] Reading {f} -> table: {table}")
        try:
            df = read_csv_with_fallback(f, sep)
        except Exception as e:
            print(f"  Failed to read {f}: {e}", file=sys.stderr)
            continue

        try:
            df.to_sql(table, conn, if_exists="replace", index=False)
        except Exception as e:
            print(f"  Failed to write table {table} to sqlite: {e}", file=sys.stderr)
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

    exts = ["*.csv", "*.sv", "*.tsv"]
    files = find_files(input_dir, exts)
    if not files:
        print(f"No files found in {input_dir} with extensions: {exts}", file=sys.stderr)
        return 4

    print(f"Found {len(files)} files. Writing to {args.engine} DB at: {db_path}")

    if args.engine == "duckdb":
        return load_to_duckdb(db_path, files, args.sep)
    else:
        return load_to_sqlite(db_path, files, args.sep)


if __name__ == "__main__":
    raise SystemExit(main())
