#!/usr/bin/env python3
"""
Quick markdown diff viewer for a CSV basename across V1/V2 resource folders.

Author: Zhengyuan Dong
Date: 2025-10-15
Last Edited: 2025-10-15

Usage:
  python -m src.tools.show_table_diff_md 0ae65809ffffa20a2e5ead861e7408ac_table_0.csv
  python -m src.tools.show_table_diff_md --resource github 0ae65809ffffa20a2e5ead861e7408ac_table_0.csv
"""

import argparse
import os
import sys
import csv
from typing import Tuple

import pandas as pd

from src.data_analysis.qc_stats import (
    count_rows_fast,
    count_columns_from_header_fast,
)


def classify_resource(basename: str) -> str:
    # github: 32 hex + _tableN
    import re
    b = basename
    if re.fullmatch(r"[0-9a-f]{32}_table_\d+\.csv", b):
        return "github"
    # html/arxiv: 0705.2450v1_table39.csv or 1234.5678_table3.csv
    if re.fullmatch(r"\d+\.\d+(?:v\d+)?_table\d+\.csv", b):
        return "arxiv"
    # hugging: 10 hex + _tableN OR fallback
    if re.fullmatch(r"[0-9a-f]{10}_table\d+\.csv", b) or re.fullmatch(r".+_table\d+\.csv", b):
        return "hugging"
    return "hugging"


def resource_paths(resource: str, basename: str) -> Tuple[str, str]:
    if resource == "github":
        v1 = os.path.join("data/processed/deduped_github_csvs", basename)
        v2 = os.path.join("data/processed/deduped_github_csvs_v2", basename)
    elif resource == "arxiv":
        v1 = os.path.join("data/processed/tables_output", basename)
        v2 = os.path.join("data/processed/tables_output_v2", basename)
    else:  # hugging
        v1 = os.path.join("data/processed/deduped_hugging_csvs", basename)
        v2 = os.path.join("data/processed/deduped_hugging_csvs_v2", basename)
    return v1, v2


def file_stats(path: str) -> Tuple[bool, int, int, int]:
    if not os.path.exists(path):
        return False, 0, 0, 0
    try:
        size = os.path.getsize(path)
    except Exception:
        size = 0
    try:
        cols = count_columns_from_header_fast(path)
    except Exception:
        cols = 0
    try:
        rows = count_rows_fast(path, head_flag=True)
    except Exception:
        rows = 0
    return True, size, rows, cols


def safe_head_markdown(path: str, n: int = 5) -> str:
    if not os.path.exists(path):
        return "(missing)"
    try:
        df = pd.read_csv(path, nrows=n)
        # Drop fully-empty rows to avoid inflated blanks in preview
        try:
            df = df.replace(r'^\s*$', pd.NA, regex=True).dropna(axis=0, how='all')
        except Exception:
            pass
        return df.to_markdown(index=False)
    except Exception:
        # Very robust fallback: parse CSV manually and format as markdown table
        try:
            lines = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    lines.append(line.rstrip("\n"))
            if not lines:
                return "(empty file)"
            
            # Parse CSV lines manually
            import csv as csv_module
            from io import StringIO
            
            csv_content = "\n".join(lines)
            reader = csv_module.reader(StringIO(csv_content))
            rows = list(reader)
            
            if not rows:
                return "(empty file)"
            
            # Format as markdown table
            if len(rows) == 1:
                # Only header
                header = " | ".join(rows[0])
                separator = " | ".join(["---"] * len(rows[0]))
                return f"| {header} |\n| {separator} |"
            else:
                # Header + data rows
                header = " | ".join(rows[0])
                separator = " | ".join(["---"] * len(rows[0]))
                data_rows = []
                for row in rows[1:]:
                    data_rows.append(" | ".join(row))
                
                result = f"| {header} |\n| {separator} |\n"
                result += "\n".join([f"| {row} |" for row in data_rows])
                return result
                
        except Exception:
            return "(cannot preview)"


def save_markdown(resource: str, basename: str, v1: str, v2: str) -> str:
    v1_exist, v1_size, v1_rows, v1_cols = file_stats(v1)
    v2_exist, v2_size, v2_rows, v2_cols = file_stats(v2)

    content = f"""### Table diff: {basename} ({resource})

Paths:
- V1: `{v1}`
- V2: `{v2}`

Summary:
| version | exists | size (bytes) | rows | cols |
|---------|--------|--------------:|-----:|-----:|
| V1 | {v1_exist} | {v1_size} | {v1_rows} | {v1_cols} |
| V2 | {v2_exist} | {v2_size} | {v2_rows} | {v2_cols} |

V1 head(5):
{safe_head_markdown(v1)}

V2 head(5):
{safe_head_markdown(v2)}
"""

    # Create tmp/tab_compare directory if it doesn't exist
    os.makedirs("tmp/tab_compare", exist_ok=True)
    
    # Generate filename from basename
    safe_basename = basename.replace("/", "_").replace("\\", "_")
    md_file = f"tmp/tab_compare/{safe_basename}_{resource}_diff.md"
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    return md_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("basename", help="CSV basename like abc_table1.csv")
    ap.add_argument("--resource", choices=["hugging", "github", "arxiv"], default=None, help="Resource override if autodetect fails")
    args = ap.parse_args()

    res = args.resource or classify_resource(args.basename)
    v1, v2 = resource_paths(res, args.basename)
    md_file = save_markdown(res, args.basename, v1, v2)
    print(f"Saved markdown diff to: {md_file}")


if __name__ == "__main__":
    main()


