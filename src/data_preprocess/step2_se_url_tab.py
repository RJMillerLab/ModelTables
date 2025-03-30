#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Created: 2025-03-17
Last updated: 2025-03-27
Description:
  Demonstration of how to batch extract from Parquet-based ES query cache,
  then for each top-K retrieved title (or corpusid), query the SQLite index
  and NDJSON to extract table/figure data.

Optimized batch extraction from ES cache → SQLite → NDJSON lines → Annotations extraction.

python step2_se_url_tab.py \
    --directory /u4/z6dong/shared_data/se_s2orc_250218 \
    --db_path /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db \
    --parquet_cache query_cache.parquet \
    --output_parquet extracted_annotations.parquet \
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

def fetch_db_entries(parquet_cache, db_path):
    df_es = pd.read_parquet(parquet_cache)
    exact_matches = df_es[
        (df_es['rank'] == 1) &
        (df_es['query'].str.lower().str.strip('" ') == df_es['retrieved_title'].str.lower().str.strip('" '))
    ]
    min_score = exact_matches['score'].min()
    eligible_matches = df_es[(df_es['score'] >= min_score) & (df_es['rank']==1)].copy()
    titles = eligible_matches['retrieved_title'].str.lower().str.strip().unique()
    placeholders = ','.join(['?'] * len(titles))
    with sqlite3.connect(db_path) as conn:
        query = f"SELECT corpusid, filename, line_index, LOWER(TRIM(title)) as title FROM papers WHERE title IN ({placeholders})"
        db_df = pd.read_sql(query, conn, params=titles.tolist())
    merged_df = eligible_matches.merge(
        db_df,
        left_on=['retrieved_title', 'corpusid'],
        right_on=['title', 'corpusid'],
        how='left'
    )
    #merged_df = merged_df.drop(columns=['rank', 'title_clean'])  ########
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

    merged_df = fetch_db_entries(args.parquet_cache, args.db_path)
    merged_df.to_parquet("merged_df.parquet", index=False)
    extract_lines_to_parquet(merged_df, args.directory, args.temp_parquet, args.n_jobs)
    final_annotation_extraction(args.temp_parquet, args.output_parquet, args.n_jobs)

if __name__ == "__main__":
    main()
