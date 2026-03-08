#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-03-26
Last Modified: 2026-03-08
Description:
  1. Load merged title list from Parquet file.
  2. Extract and extend all titles, deduplicate, and save as JSON.
  3. Simulate querying each unique title (or load existing query results).
  4. Map query results back to the DataFrame and save the final output.
Usage:
    python -m src.data_preprocess.step2_s2orc_save
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils import load_config, to_parquet, is_list_like, to_list_safe


def _normalize(s):
    """Remove: - (hyphen), space, . for cross-row dedup comparison."""
    if not isinstance(s, str):
        return ""
    return str(s).replace("-", "").replace(" ", "").replace(".", "").lower().strip()


def _count_symbols(s):
    return sum(1 for c in str(s) if c in "- .")


def _pick_kept(titles):
    """Among titles that normalize to same form, keep the one with fewest '-', ' ', '.'."""
    if not titles:
        return None
    return min(titles, key=lambda t: (_count_symbols(t), len(t), t))


def main():
    parser = argparse.ArgumentParser(description='Save deduplicated titles for querying Semantic Scholar')
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    args = parser.parse_args()

    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    data_type = 'modelcard'
    suffix = f"_{args.tag}" if args.tag else ""

    input_file = os.path.join(processed_base_path, f"{data_type}_all_title_list{suffix}.parquet")
    dedup_titles_path = os.path.join(processed_base_path, f"{data_type}_dedup_titles{suffix}.json")
    cross_row_groups_path = os.path.join(processed_base_path, f"s2orc_cross_row_dedup_groups{suffix}.json")

    print(f"📁 Input file: {input_file}")
    print(f"📁 Output files: Deduplicated titles: {dedup_titles_path}, Cross-row groups: {cross_row_groups_path}")
    print("Step 1: Loading data from parquet (modelcard_all_title_list)...")
    df_final = pd.read_parquet(input_file, columns=['modelId', 'all_title_list', 'all_bibtex_titles'])
    print(f"Loaded {len(df_final)} rows from {input_file}")

    # Step 3: Extract all titles from "all_title_list" column
    print("Step 3: Extracting and deduplicating all titles (cross-row dedup: remove -, space, .)...")
    all_titles = []
    for titles in df_final["all_title_list"]:
        if titles is None:
            continue
        if is_list_like(titles):
            all_titles.extend(to_list_safe(titles))
        elif isinstance(titles, str):
            all_titles.append(titles.strip())

    # Clean: trim, lowercase, remove empty
    all_titles_clean = [t.strip().lower() for t in all_titles if t.strip()]
    n_before_exact = len(all_titles_clean)
    n_unique_exact = len(set(all_titles_clean))

    # Cross-row dedup: group by normalized form (no -, space, .), keep one per group
    by_norm = {}
    for t in all_titles_clean:
        norm = _normalize(t)
        if norm:
            by_norm.setdefault(norm, []).append(t)

    cross_row_groups = []
    dedup_titles = []
    for norm, originals in by_norm.items():
        uniq = list(dict.fromkeys(originals))
        kept = _pick_kept(uniq)
        dedup_titles.append(kept)
        if len(uniq) > 1:
            cross_row_groups.append({"kept": kept, "duplicates": [x for x in uniq if x != kept]})

    n_after = len(dedup_titles)
    n_removed = n_unique_exact - n_after

    print(f"\n=== Cross-row dedup stats ===")
    print(f"  Before (total items):     {n_before_exact}")
    print(f"  After exact dedup:         {n_unique_exact} unique titles")
    print(f"  After cross-row dedup:     {n_after} query titles")
    print(f"  Removed (duplicate query items): {n_removed}")
    print(f"  Groups (kept->duplicates): {len(cross_row_groups)}")

    dup_to_kept = {}
    for g in cross_row_groups:
        for d in g["duplicates"]:
            dup_to_kept[d] = g["kept"]

    groups_output = {
        "groups": cross_row_groups,
        "duplicate_to_kept": dup_to_kept,
        "stats": {
            "before_total": n_before_exact,
            "after_exact": n_unique_exact,
            "after_cross_row": n_after,
            "removed": n_removed,
            "num_groups": len(cross_row_groups),
        },
    }
    with open(cross_row_groups_path, "w", encoding="utf-8") as f:
        json.dump(groups_output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved cross-row group mapping to {cross_row_groups_path}")

    dedup_titles = sorted(dedup_titles)
    with open(dedup_titles_path, "w", encoding="utf-8") as f:
        json.dump(dedup_titles, f, ensure_ascii=False, indent=2)
    print(f"✅ Deduplicated titles saved to {dedup_titles_path} (Total: {len(dedup_titles)})")

if __name__ == "__main__":
    main()