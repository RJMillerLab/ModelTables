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
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--input-title-list', dest='input_title_list', default=None,
                        help='Path to all_title_list parquet file (default: auto-detect from tag)')
    args = parser.parse_args()
    
    # Step 0: Load configuration and basic paths
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    data_type = 'modelcard'
    tag = args.tag
    
    # Determine input file based on tag
    if args.input_title_list:
        input_file = args.input_title_list
    else:
        input_suffix = f"_{tag}" if tag else ""
        input_file = os.path.join(processed_base_path, f"{data_type}_all_title_list{input_suffix}.parquet")
    
    # Determine output files based on tag
    output_suffix = f"_{tag}" if tag else ""
    dedup_titles_path = os.path.join(processed_base_path, f"{data_type}_dedup_titles{output_suffix}.json")
    query_results_path = os.path.join(processed_base_path, f"{data_type}_title_query_results{output_suffix}.json")
    mapped_output_path = os.path.join(processed_base_path, f"{data_type}_all_title_list_mapped{output_suffix}.parquet")
    
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output files:")
    print(f"   - Deduplicated titles: {dedup_titles_path}")
    print(f"   - Query results: {query_results_path}")
    print(f"   - Mapped output: {mapped_output_path}")
    
    print("Step 1: Loading data from parquet (modelcard_all_title_list)...")
    df_final = pd.read_parquet(input_file, columns=['modelId', 'all_title_list', 'all_bibtex_titles'])
    print(f"Loaded {len(df_final)} rows from {input_file}")

    # Step 3: Extract and deduplicate all titles from "all_title_list" column
    print("Step 3: Extracting and deduplicating all titles...")
    all_titles = []
    for titles in df_final["all_title_list"]: # all_bibtex_titles: reliable, all_title_list: need to be cleaned ...
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

    # Step 4: Simulate querying titles (or load existing query results) - COMMENTED OUT
    # Uncomment if you need to simulate querying or map results back
    """
    print("Step 4: Querying titles (simulated query)...")
    if os.path.exists(query_results_path):
        with open(query_results_path, "r", encoding="utf-8") as f:
            title_query_results = json.load(f)
        print(f"Loaded existing title query results from {query_results_path}")
    else:
        title_query_results = {}
        for title in dedup_titles:
            # Here you can call your actual query API. This example simulates a result.
            title_query_results[title] = {"info": f"Simulated result for '{title}'"}
        with open(query_results_path, "w", encoding="utf-8") as f:
            json.dump(title_query_results, f, ensure_ascii=False, indent=2)
        print(f"Simulated query results saved to {query_results_path}")

    # Step 5: Map query results back to the DataFrame
    print("Step 5: Mapping query results back to DataFrame...")

    def map_title_info(titles):
        mapped = []
        if isinstance(titles, list):
            for t in titles:
                t_clean = t.strip().lower()
                mapped.append(title_query_results.get(t_clean, {}))
        elif isinstance(titles, str):
            t_clean = titles.strip().lower()
            mapped.append(title_query_results.get(t_clean, {}))
        return mapped

    df_final["mapped_title_info"] = df_final["all_title_list"].apply(map_title_info)
    print("✅ Mapped query results back to DataFrame in column 'mapped_title_info'.")

    # Save final DataFrame to a new Parquet file
    to_parquet(df_final, mapped_output_path)
    print(f"✅ Final DataFrame saved to {mapped_output_path}")
    """

if __name__ == "__main__":
    main()