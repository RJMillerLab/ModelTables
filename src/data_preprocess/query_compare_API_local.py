#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Your Name
Date: 2025-04-13
Description:
    This script compares local database queries with SORC API queries, as well as comparing the bibtex titles query results.
    It loads and processes data from the local 'query_cache' and 's2orc_rerun' files, identifies missing queries, and then uses a ground truth
    (all bibtex titles from a separate file) to show which queries were missed in the local database while being retrieved by the SORC API.
Usage:
    python -m src.data_analysis.query_compare_API_local
"""

import os
import json
import numpy as np
import pandas as pd
from src.utils import load_config

def main():
    # --------------------------
    # Load the query cache file (local database query)
    query_cache_path = "data/processed/query_cache.parquet"
    df_query = pd.read_parquet(query_cache_path)
    df_query = df_query[df_query['rank'] == 1]  # Keep only records where rank == 1
    print(f"Loaded query_cache: shape = {df_query.shape}")
    
    # --------------------------
    # Load the SORC rerun file (API query)
    s2orc_path = "data/processed/s2orc_rerun.parquet"
    df_s2orc = pd.read_parquet(s2orc_path)
    print(f"Loaded s2orc_rerun: shape = {df_s2orc.shape}")
    
    # Rename 'corpusId' column to 'corpusid' if present
    if 'corpusId' in df_s2orc.columns:
        df_s2orc.rename(columns={'corpusId': 'corpusid'}, inplace=True)
    
    # --------------------------
    # Compare corpusid values between local query and API query datasets
    query_ids = set(df_query['corpusid'])
    s2orc_ids = set(df_s2orc['corpusid'])
    missing_ids = query_ids - s2orc_ids
    print("(corpusid) in Query Cache:", len(query_ids))
    print("(corpusid) in SORC Rerun:", len(s2orc_ids))
    print("(corpusid) in Query Cache but missing in SORC Rerun:", len(missing_ids))

    if missing_ids:
        df_missing = df_query[df_query['corpusid'].isin(missing_ids)]
        df_missing['query'] = df_missing['query'].str.lower().str.strip('" ')
        
        if 'query_title' in df_s2orc.columns:
            df_s2orc['query_title'] = df_s2orc['query_title'].str.lower().str.strip('" ')
        df_s2orc['retrieved_title'] = df_s2orc['retrieved_title'].str.lower().str.strip('" ')
        
        df_missing = df_missing.merge(df_s2orc[['query_title', 'retrieved_title', 'corpusid']], how='inner', 
                                      left_on='query', right_on='query_title')
        
        df_missing = df_missing[['corpusid_x', 'query', 'corpusid_y', 'retrieved_title_x', 'retrieved_title_y']]
        df_missing.rename(columns={
            'corpusid_x': 'corpusid', 
            'corpusid_y': 'corpusid_s2orc', 
            'retrieved_title_x': 'retrieved_title', 
            'retrieved_title_y': 'retrieved_title_s2orc'
        }, inplace=True)
        df_missing.drop_duplicates(inplace=True)
        df_missing.reset_index(drop=True, inplace=True)
        
        print(f"Missing IDs in SORC Rerun: {len(missing_ids)}")
        print(f"Missing IDs in SORC Rerun with matching query in SORC: {len(df_missing)}")
        print(f"Detail of matching missing IDs: {df_missing.shape}")

    # --------------------------
    # Process and compare query strings between local and API query datasets
    df_query['query_processed'] = df_query['query'].str.lower().str.strip('" ')
    if 'query_title' in df_s2orc.columns:
        df_s2orc['query_title_processed'] = df_s2orc['query_title'].str.lower().str.strip('" ')
    else:
        df_s2orc['query_title_processed'] = ""

    query_set_1 = set(df_query['query_processed'])
    query_set_2 = set(df_s2orc['query_title_processed'])
    missing_queries = query_set_1 - query_set_2
    print(f"Number of queries in df_query but not in df_s2orc: {len(missing_queries)}")

    # Save missing queries to a temporary text file for later review
    with open("tmp.txt", "w", encoding="utf-8") as f:
        for query in missing_queries:
            f.write(query + "\n")

    # Save rows corresponding to missing queries to a Parquet file for later inspection
    missing_rows = df_query[df_query['query_processed'].isin(missing_queries)]
    missing_rows.to_parquet("missing_queries_rows.parquet", index=False)
    print(f"Missing query rows saved to missing_queries_rows.parquet, shape: {missing_rows.shape}")

    # Filter final results: rows with score > 100 and where the retrieved title exactly matches the original query
    final = missing_rows[(missing_rows['score'] > 100) & (missing_rows['retrieved_title'] == missing_rows['query'])]
    print(final)
    # Note: The local search engine might have specific rules that retrieve more queries than the API query.

    # Additional code for ground truth comparison using all bibtex titles
    # Load ground truth file (all bibtex titles)
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    df_all = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_all_title_list.parquet"))
    print(f"Loaded ground truth all title list: shape = {df_all.shape}")

    # Extract and deduplicate all bibtex titles (assumed stored in the "all_bibtex_titles" column)
    all_titles = []
    for titles in df_all["all_bibtex_titles"]:
        if isinstance(titles, (list, tuple, np.ndarray)):
            all_titles.extend(titles)
        elif isinstance(titles, str):
            all_titles.append(titles.strip())
    gt_titles = set([t.strip().lower() for t in all_titles if t.strip()])
    print(f"Total ground truth deduplicated titles: {len(gt_titles)}")

    # Get the processed query set from the local database (df_query)
    query_file_queries = set(df_query['query_processed'])

    # Calculate ground truth titles missing in the local database query
    missing_in_first = gt_titles - query_file_queries
    print(f"Number of ground truth titles missing in first file: {len(missing_in_first)}")

    # Obtain the query set from the SORC API if available
    if 'query_title_processed' in df_s2orc.columns:
        query_second = set(df_s2orc['query_title_processed'])
    else:
        query_second = set()

    # Identify ground truth titles that are missing in the local database but found in the API query
    present_in_second_missing_in_first = missing_in_first & query_second
    print(f"Number of ground truth titles missing in first file but found in second file: {len(present_in_second_missing_in_first)}")

    # Print out sample titles for comparison
    print("Examples of titles missing in first file but found in second file:")
    for title in list(present_in_second_missing_in_first):
        print(title)

if __name__ == '__main__':
    main()
