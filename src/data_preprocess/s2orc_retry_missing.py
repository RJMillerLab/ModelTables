#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Created: 2025-04-13
Last Modified: 2025-04-13
Description:
    This script checks for paper IDs that are missing citations or references (e.g. due to 429 errors)
    based on the titles mapping cache, then re-queries those missing items by calling the imported API functions.
    The re-queried results are saved into separate Parquet files (one for citations, one for references).
    
    Later, you can merge these new results with your existing caches in your main merge function.
    
Usage:
    python -m src.data_preprocess.s2orc_retry_missing
"""

import os
import pandas as pd
from tqdm import tqdm
from src.data_preprocess.s2orc_API_query import get_single_citations_row, get_single_references_row
from src.utils import to_parquet

prefix = "_429"  # Set prefix if needed (e.g. "_429")
DATA_FOLDER = "data/processed"
TITLES_CACHE_FILE = f"{DATA_FOLDER}/s2orc_titles2ids{prefix}.parquet"        # Titles mapping cache
CITATIONS_CACHE_FILE = f"{DATA_FOLDER}/s2orc_citations_cache{prefix}.parquet"  # Original citations cache
REFERENCES_CACHE_FILE = f"{DATA_FOLDER}/s2orc_references_cache{prefix}.parquet"  # Original references cache

CITATIONS_MISSING_FILE = f"{DATA_FOLDER}/s2orc_citations_missing{prefix}.parquet"
REFERENCES_MISSING_FILE = f"{DATA_FOLDER}/s2orc_references_missing{prefix}.parquet"

def load_parquet_or_empty(file_path, columns):
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        return pd.DataFrame(columns=columns)

if __name__ == "__main__":
    # --- Load the Titles Mapping Cache ---
    if not os.path.exists(TITLES_CACHE_FILE):
        print(f"❌ Titles cache file {TITLES_CACHE_FILE} not found. Exiting.")
        exit(1)
    df_titles = pd.read_parquet(TITLES_CACHE_FILE)
    print(f"Loaded {len(df_titles)} records from titles mapping cache.")

    # --- Load original caches ---
    df_citations = load_parquet_or_empty(CITATIONS_CACHE_FILE, ["paperId", "original_response", "parsed_response"])
    df_references = load_parquet_or_empty(REFERENCES_CACHE_FILE, ["paperId", "original_response", "parsed_response"])
    print(f"Original citations cache has {len(df_citations)} records.")
    print(f"Original references cache has {len(df_references)} records.")

    # --- Make sure paper IDs are strings ---
    df_titles["paperId"] = df_titles["paperId"].astype(str)
    df_citations["paperId"] = df_citations["paperId"].astype(str)
    df_references["paperId"] = df_references["paperId"].astype(str)
    
    # --- Identify missing paper IDs from the original caches ---
    missing_citations_ids = df_titles.loc[~df_titles["paperId"].isin(df_citations["paperId"]), "paperId"].unique().tolist()
    missing_references_ids = df_titles.loc[~df_titles["paperId"].isin(df_references["paperId"]), "paperId"].unique().tolist()
    
    print(f"Missing citations count (original): {len(missing_citations_ids)}")
    print(f"Missing references count (original): {len(missing_references_ids)}")

    # --- Load already re-queried missing results, if any ---
    df_citations_missing = load_parquet_or_empty(CITATIONS_MISSING_FILE, ["paperId", "original_response", "parsed_response"])
    df_references_missing = load_parquet_or_empty(REFERENCES_MISSING_FILE, ["paperId", "original_response", "parsed_response"])
    
    # Exclude IDs already re-queried in our missing files
    existing_missing_citations = set(df_citations_missing["paperId"].tolist())
    existing_missing_references = set(df_references_missing["paperId"].tolist())
    
    # Filter missing lists
    requery_citations_ids = [pid for pid in missing_citations_ids if pid not in existing_missing_citations]
    requery_references_ids = [pid for pid in missing_references_ids if pid not in existing_missing_references]
    
    print(f"Re-querying {len(requery_citations_ids)} missing citations.")
    print(f"Re-querying {len(requery_references_ids)} missing references.")
    
    # --- Re-query Missing Citations ---
    new_citations = []
    if requery_citations_ids:
        for pid in tqdm(requery_citations_ids, desc="Re-querying Citations"):
            # Call the imported function with the new missing file as cache file
            record = get_single_citations_row(pid, sleep_time=1, timeout=60, cache_file=CITATIONS_MISSING_FILE)
            # Only add if the returned record is not empty
            if record and record.get("paperId"):
                new_citations.append(record)
    else:
        print("No missing citations to re-query.")
    
    # Save/update the missing citations parquet (append new results to the existing ones).
    if new_citations:
        df_new_citations = pd.DataFrame(new_citations)
        df_citations_missing = pd.concat([df_citations_missing, df_new_citations], ignore_index=True)
        to_parquet(df_citations_missing, CITATIONS_MISSING_FILE)
        print(f"Saved updated missing citations to {CITATIONS_MISSING_FILE}.")
    else:
        print(f"No new citations re-queried; {CITATIONS_MISSING_FILE} remains unchanged.")
    
    # --- Re-query Missing References ---
    new_references = []
    if requery_references_ids:
        for pid in tqdm(requery_references_ids, desc="Re-querying References"):
            record = get_single_references_row(pid, sleep_time=1, timeout=60, cache_file=REFERENCES_MISSING_FILE)
            if record and record.get("paperId"):
                new_references.append(record)
    else:
        print("No missing references to re-query.")
    
    # Save/update the missing references parquet.
    if new_references:
        df_new_references = pd.DataFrame(new_references)
        df_references_missing = pd.concat([df_references_missing, df_new_references], ignore_index=True)
        to_parquet(df_references_missing, REFERENCES_MISSING_FILE)
        print(f"Saved updated missing references to {REFERENCES_MISSING_FILE}.")
    else:
        print(f"No new references re-queried; {REFERENCES_MISSING_FILE} remains unchanged.")
    
    print("\n✅ Re-query process for missing items complete.")
    print("You may now merge these missing results with your main caches in your merge process.")

