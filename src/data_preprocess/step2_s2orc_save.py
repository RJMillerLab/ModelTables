#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-03-26
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
from src.utils import load_config, to_parquet, is_list_like, to_list_safe  # Ensure this function is available
from src.data_preprocess.step2_arxiv_github_title import load_cache, save_cache  # if needed

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
    
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output files: Deduplicated titles: {dedup_titles_path}")
    print("Step 1: Loading data from parquet (modelcard_all_title_list)...")
    df_final = pd.read_parquet(input_file, columns=['modelId', 'all_title_list', 'all_bibtex_titles'])
    print(f"Loaded {len(df_final)} rows from {input_file}")

    # Step 3: Extract and deduplicate all titles from "all_title_list" column
    print("Step 3: Extracting and deduplicating all titles...")
    all_titles = []
    for titles in df_final["all_title_list"]:
        if titles is None:
            continue
        if is_list_like(titles):
            all_titles.extend(to_list_safe(titles))
        elif isinstance(titles, str):
            all_titles.append(titles.strip())

    # Clean titles: trim spaces, convert to lowercase, and remove empty strings
    all_titles_clean = [t.strip().lower() for t in all_titles if t.strip()]
    dedup_titles = sorted(list(set(all_titles_clean)))
    with open(dedup_titles_path, "w", encoding="utf-8") as f:
        json.dump(dedup_titles, f, ensure_ascii=False, indent=2)
    print(f"✅ Deduplicated titles saved to {dedup_titles_path} (Total: {len(dedup_titles)})")

if __name__ == "__main__":
    main()