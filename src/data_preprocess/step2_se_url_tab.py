#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Created: 2025-03-17
Last updated: 2025-04-13
Description:
  Demonstration of how to batch extract from Parquet-based ES query cache,
  then for each top-K retrieved title (or corpusid), query the SQLite index
  and NDJSON to extract table/figure data.

Optimized batch extraction from ES cache → SQLite → NDJSON lines → Annotations extraction.

python step2_se_url_tab.py \
    --directory /u4/z6dong/shared_data/se_s2orc_250218 \
    --db_path /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db \
    --parquet_cache data/processed/query_cache.parquet \
    --output_parquet data/processed/extracted_annotations.parquet \
    --n_jobs -1
"""

import os
import json
import sqlite3
import argparse
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import subprocess
from extract_table import extract_references

def quality_filter_cache(parquet_cache):
    """
    Quality filter the Elasticsearch query cache.

    Reads the Parquet-based query cache and filters records based on quality metrics:
      - Only records with rank == 1.
      - Records where the query matches the retrieved_title (after lowercasing and stripping quotes/spaces).
      - Only keeps records with a score greater than or equal to the minimum score among exact matches.

    Args:
        parquet_cache (str): Path to the Parquet query cache.

    Returns:
        DataFrame: Filtered DataFrame of eligible records.
    """
    df_es = pd.read_parquet(parquet_cache)
    exact_matches = df_es[
        (df_es['rank'] == 1) &
        (df_es['query'].str.lower().str.strip('" ') == df_es['retrieved_title'].str.lower().str.strip('" '))
    ]
    min_score = exact_matches['score'].min()
    eligible_matches = df_es[(df_es['score'] >= min_score) & (df_es['rank'] == 1)].copy()
    return eligible_matches

def query_db_by_corpusid(filtered_df, db_path):
    """
    Query the SQLite database using corpusid values from the filtered ES query cache.

    Extracts unique corpusid values from the filtered DataFrame, queries the database 
    to retrieve corresponding records, and then merges the DB results with the filtered data.
    (Updated to use corpusid instead of title) ########

    Args:
        filtered_df (DataFrame): The filtered Elasticsearch query cache DataFrame.
        db_path (str): Path to the SQLite database.

    Returns:
        DataFrame: Merged DataFrame containing both ES data and the corresponding DB records.
    """
    # Extract unique corpusid values, converting to lowercase and stripping spaces ########
    corpusids = filtered_df['corpusid'].str.lower().str.strip().unique()
    placeholders = ','.join(['?'] * len(corpusids))
    with sqlite3.connect(db_path) as conn:
        # Query the database using corpusid instead of title ########
        query = f"""
                SELECT corpusid, filename, line_index, LOWER(TRIM(title)) as title
                FROM papers
                WHERE corpusid IN ({placeholders})
                """
        db_df = pd.read_sql(query, conn, params=list(corpusids))
    # Merge on corpusid only
    merged_df = filtered_df.merge(db_df, on='corpusid', how='left')
    return merged_df

def extract_lines_to_parquet(merged_df, data_directory, temp_parquet, n_jobs):
    import os
    import subprocess
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm import tqdm

    def extract_lines(filename, indices):
        filepath = os.path.join(data_directory, filename)
        if not os.path.exists(filepath):
            return []
        sed_cmd = ";".join(f"{idx+1}p" for idx in indices)
        cmd = f"sed -n '{sed_cmd}' '{filepath}'"
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
            return output.strip().split('\n')
        except subprocess.CalledProcessError:
            return []

    dedup_df = merged_df.drop_duplicates(subset=['filename', 'line_index'], keep='first')
    grouped = list(dedup_df.groupby('filename'))
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda filename, group: (filename, group, extract_lines(filename, group['line_index'].tolist())))(
            filename, group
        ) for filename, group in tqdm(grouped, total=len(grouped), desc="Extracting lines")
    )
    lines_expanded = []
    for filename, group, lines in results:
        group_sorted = group.sort_values('line_index')
        for (_, row), line in zip(group_sorted.iterrows(), lines):
            row_dict = row.to_dict()
            row_dict['raw_json'] = line
            lines_expanded.append(row_dict)
    pd.DataFrame(lines_expanded).to_parquet(temp_parquet, index=False)

def parse_annotations(row):
    try:
        paper_json = json.loads(row['raw_json'])
        content_text = (paper_json.get("content") or {}).get("text", "")
        extracted = extract_references(paper_json, content_text)
        new_row = dict(row)
        new_row.update({
            "extracted_openaccessurl": (((paper_json.get("content") or {}).get("source", {}) or {}).get("oainfo", {}) or {}).get("openaccessurl", None),
            "extracted_tables": extracted.get("extracted_tables", []),
            "extracted_tablerefs": extracted.get("extracted_tablerefs", []),
            "extracted_figures": extracted.get("extracted_figures", []),
            "extracted_figure_captions": extracted.get("extracted_figure_captions", []),
            "extracted_figurerefs": extracted.get("extracted_figurerefs", [])
        })
        return new_row
    except Exception as e:
        print(f"Error: {e}")
        return None

def final_annotation_extraction(temp_parquet, output_parquet, n_jobs):
    df_temp = pd.read_parquet(temp_parquet)
    parsed_data = Parallel(n_jobs=n_jobs)(
        delayed(parse_annotations)(row)
        for _, row in tqdm(df_temp.iterrows(), total=len(df_temp), desc="Parsing annotations")
    )
    parsed_data_clean = [item for item in parsed_data if item]
    pd.DataFrame(parsed_data_clean).to_parquet(output_parquet, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, help="Directory of NDJSON files")
    parser.add_argument("--db_path", required=True, help="SQLite database path")
    parser.add_argument("--parquet_cache", required=True, help="Input Parquet cache")
    parser.add_argument("--output_parquet", default="final_annotations.parquet", help="Output Parquet path")
    parser.add_argument("--temp_parquet", default="tmp_extracted_lines.parquet", help="Temporary Parquet path")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Parallel jobs")

    args = parser.parse_args()

    # Step 1: Quality filter the Elasticsearch query cache.
    filtered_df = quality_filter_cache(args.parquet_cache)
    # Step 2: Query the SQLite database using corpusid from the filtered data. ########
    merged_df = query_db_by_corpusid(filtered_df, args.db_path)
    merged_df.to_parquet("data/processed/merged_df.parquet", index=False)
    # Step 3: Extract specific lines from NDJSON files and write to a temporary Parquet file.
    extract_lines_to_parquet(merged_df, args.directory, args.temp_parquet, args.n_jobs)
    # Step 4: Parse annotations from the extracted lines and write final annotated data to output Parquet.
    final_annotation_extraction(args.temp_parquet, args.output_parquet, args.n_jobs)

if __name__ == "__main__":
    main()
