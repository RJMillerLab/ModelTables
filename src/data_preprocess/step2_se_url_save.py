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
    python -m src.data_preprocess.step2_se_url_save
"""

import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils import load_config  # Ensure this function is available
from src.data_preprocess.step2_se_url_title import load_cache, save_cache  # if needed

# Step 0: Load configuration and basic paths
config = load_config('config.yaml')
processed_base_path = os.path.join(config.get('base_path'), 'processed')
data_type = 'modelcard'
# If needed, load GitHub cache:
# GITHUB_PATH_CACHE = load_github_cache(config)
# print(f"Loaded {len(GITHUB_PATH_CACHE)} GitHub cache entries.")

print("Step 1: Loading data from parquet (modelcard_step1)...")
df_final = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_all_title_list.parquet"), columns=['modelId', 'all_title_list', 'all_bibtex_titles'])
print(f"Loaded {len(df_final)} rows from {data_type}_all_title_list.parquet")

# Step 3: Extract and deduplicate all titles from "all_title_list" column
print("Step 3: Extracting and deduplicating all titles...")
all_titles = []
for titles in df_final["all_title_list"]: # all_bibtex_titles: reliable, all_title_list: need to be cleaned ...
    if isinstance(titles, (list, tuple, np.ndarray)):
        all_titles.extend(titles)
    elif isinstance(titles, str):
        all_titles.append(titles.strip())

# Clean titles: trim spaces, convert to lowercase, and remove empty strings
all_titles_clean = [t.strip().lower() for t in all_titles if t.strip()]
dedup_titles = sorted(list(set(all_titles_clean)))
dedup_titles_path = os.path.join(processed_base_path, f"{data_type}_dedup_titles.json")
with open(dedup_titles_path, "w", encoding="utf-8") as f:
    json.dump(dedup_titles, f, ensure_ascii=False, indent=2)
print(f"✅ Deduplicated titles saved to {dedup_titles_path} (Total: {len(dedup_titles)})")

"""# Step 4: Simulate querying titles (or load existing query results)
print("Step 4: Querying titles (simulated query)...")
query_results_path = os.path.join(processed_base_path, f"{data_type}_title_query_results.json")
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
output_parquet = os.path.join(processed_base_path, f"{data_type}_all_title_list_mapped.parquet")
pq.write_table(pa.Table.from_pandas(df_final), output_parquet)
print(f"✅ Final DataFrame saved to {output_parquet}")

"""