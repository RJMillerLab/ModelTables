#!/usr/bin/env python3
"""
Quickly scan table shapes (rows / columns) for tables in all_valid_title_valid*.txt.

- Input: data/analysis/all_valid_title_valid{v2_suffix}{suffix}.txt
         (or a custom --list-file)
- For each CSV path, count:
    - n_rows (data rows, excluding header)
    - n_cols (number of columns from header)
- Uses fast file-level scanners from qc_stats, never pandas.read_csv.

Outputs:
- Parquet with columns: table_path, basename, resource, n_rows, n_cols
- Prints basic distribution stats (per resource and global).
"""

import argparse
import os
from typing import List, Tuple, Optional

import pandas as pd

from src.data_analysis.qc_stats import count_rows_fast, count_columns_from_header_fast
from src.utils import to_parquet


def infer_resource_from_path(path: str) -> str:
    """Infer resource label from table path."""
    p = path.replace("\\", "/")
    if "deduped_hugging_csvs" in p or "/hugging" in p:
        return "hugging"
    if "deduped_github_csvs" in p or "/github" in p:
        return "github"
    if "tables_output" in p or "/html" in p or "arxiv" in p:
        return "arxiv"
    if "llm_tables" in p or "/llm" in p:
        return "llm"
    return "unknown"


def load_table_list(list_file: str) -> List[str]:
    paths: List[str] = []
    with open(list_file, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    # dedup while preserving order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def resolve_path(path: str) -> str:
    """Return a usable absolute path; if already absolute and exists, keep it.
    If relative, resolve from repo root (current working directory)."""
    if os.path.isabs(path):
        return path
    # Many paths in txt are already "data/processed/...", which are relative to repo root
    return os.path.abspath(path)


def discover_tables_in_directories(directories: List[str]) -> List[str]:
    """Recursively discover CSV tables under the given directories."""
    all_paths: List[str] = []
    for d in directories:
        if not os.path.exists(d):
            print(f"Warning: directory does not exist, skipping: {d}")
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.lower().endswith(".csv"):
                    all_paths.append(os.path.join(root, fname))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in all_paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    print(f"Total unique table paths discovered from directories: {len(uniq):,}")
    return uniq


def scan_table_shapes(table_paths: List[str], output_parquet: str) -> None:
    """Core scanner given an in-memory list of table paths."""
    print(f"Total unique table paths to scan: {len(table_paths):,}")
    records: List[Tuple[str, str, str, int, int]] = []
    missing_paths: List[str] = []
    existing_count = 0

    for i, raw_path in enumerate(table_paths):
        abs_path = resolve_path(raw_path)
        if i % 1000 == 0:
            print(f"  Scanned {i}/{len(table_paths)} tables ...")

        if not os.path.exists(abs_path):
            # Skip missing files but keep a record with -1 for rows/cols
            basename = os.path.basename(raw_path)
            res = infer_resource_from_path(raw_path)
            records.append((raw_path, basename, res, -1, -1))
            missing_paths.append(raw_path)
            continue

        existing_count += 1
        try:
            n_cols = count_columns_from_header_fast(abs_path)
        except Exception:
            n_cols = -1
        try:
            # Exclude header row: head_flag=False matches qc_stats metric
            n_rows = count_rows_fast(abs_path, head_flag=False)
        except Exception:
            n_rows = -1

        basename = os.path.basename(raw_path)
        res = infer_resource_from_path(raw_path)
        records.append((raw_path, basename, res, int(n_rows), int(n_cols)))

    total = len(table_paths)
    print(f"\nShape scan summary: total={total:,}, existing={existing_count:,}, missing={total - existing_count:,}")
    if missing_paths:
        print("  Example missing paths (up to 5):")
        for p in missing_paths[:5]:
            print(f"    - {p}")

    df = pd.DataFrame(records, columns=["table_path", "basename", "resource", "n_rows", "n_cols"])
    df.to_parquet(output_parquet, index=False)
    print("saved to", output_parquet)
    print_distribution(df)


def print_distribution(df: pd.DataFrame) -> None:
    def _stats(series: pd.Series, name: str) -> None:
        valid = series[series >= 0]
        if valid.empty:
            print(f"  {name}: no valid values")
            return
        print(
            f"  {name}: count={len(valid):,}, "
            f"min={valid.min():,}, p50={int(valid.median()):,}, "
            f"p90={int(valid.quantile(0.9)):,}, max={valid.max():,}"
        )

    print("\n=== Global distribution (all resources) ===")
    _stats(df["n_rows"], "rows")
    _stats(df["n_cols"], "cols")

    for res, sub in df.groupby("resource"):
        print(f"\n=== Resource: {res} ===")
        _stats(sub["n_rows"], "rows")
        _stats(sub["n_cols"], "cols")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan table shapes (rows/cols) for tables listed in all_valid_title_valid*.txt")
    ap.add_argument('--tag', type=str, default=None, help="Tag suffix (e.g., 251117). Used for default list filename.")
    ap.add_argument('--v2_mode', action="store_true", help="Use v2 suffix in default list filename (i.e., all_valid_title_valid_v2{tag}.txt).")
    ap.add_argument('--from-hugging', action="store_true", help="Instead of using all_valid_title_valid*.txt, scan directly from deduped_hugging_csvs[_v2]_<tag>.")
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    output_parquet = os.path.join("data", "analysis", f"valid_table_shapes{v2_suffix}{suffix}.parquet")

    if args.from_hugging:
        base_dir = os.path.join("data", "processed", f"deduped_hugging_csvs{v2_suffix}{suffix}")
        table_paths: List[str] = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.lower().endswith(".csv"):
                    table_paths.append(os.path.join(root, fname))
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for p in table_paths:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        print(f"Scanning from HuggingFace dir: {base_dir}")
        scan_table_shapes(uniq, output_parquet)
    else:
        list_file = os.path.join("data", "analysis", f"all_valid_title_valid{v2_suffix}{suffix}.txt")
        print(f"Using list file: {list_file}")
        table_paths = load_table_list(list_file)
        scan_table_shapes(table_paths, output_parquet)

if __name__ == "__main__":
    main()

