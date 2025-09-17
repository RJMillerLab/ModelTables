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
    --parquet_cache data/processed/s2orc_rerun.parquet \
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


def extract_annotations(data, key):
    raw_data = data.get("content", {}).get("annotations", {}).get(key, "[]")
    if not raw_data:  # Explicitly handle None values
        return []
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError:
        return []

def extract_references(data, content_text):
    """Extract tables, figures, captions, and references."""
    extracted_data = {}
    annotations = {
        "extracted_abstract": "abstract",
        "extracted_authors": "authors",
        "extracted_references": "references",
        "extracted_author_affiliations": "authoraffiliation",
        "extracted_author_firstnames": "authorfirstname",
        "extracted_author_lastnames": "authorlastname",
        "extracted_bib_authors": "bibauthor",
        "extracted_bib_author_firstnames": "bibauthorfirstname",
        "extracted_bib_author_lastnames": "bibauthorlastname",
        "extracted_bib_entries": "bibentry",
        "extracted_bib_refs": "bibref",
        "extracted_bib_titles": "bibtitle",
        "extracted_bib_venues": "bibvenue",
        "extracted_tablerefs": "tableref",
        "extracted_tables": "table",
        "extracted_figures": "figure",
        "extracted_figure_captions": "figurecaption",
        "extracted_figurerefs": "figureref",
        "extracted_formula": "formula",
        "extracted_publisher": "publisher",
        "extracted_sectionheader": "sectionheader",
        "extracted_title": "title",
        "extracted_venue": "venue",
    }
    for key, annotation_key in annotations.items():
        extracted = [
            {
                "start": item["start"],
                "end": item["end"],
                "extracted_text": content_text[item["start"]:item["end"]],
                **({"id": item.get("attributes", {}).get("id", "unknown")} if "id" in item.get("attributes", {}) else {}),
                **({"figure_ref_id": item.get("attributes", {}).get("ref_id", "unknown")} if "ref_id" in item.get("attributes", {}) else {})
            }
            for item in extract_annotations(data, annotation_key)
            if isinstance(item.get("start"), int) and isinstance(item.get("end"), int) and item["start"] < item["end"]
        ]
        if extracted:
            extracted_data[key] = extracted
    return extracted_data

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

def query_db_by_corpusid(filtered_df, db_path, batch_size=1000):
    """
    Query the SQLite database using corpusid values from the filtered ES query cache.

    Extracts unique corpusid values from the filtered DataFrame, queries the database 
    to retrieve corresponding records, and then merges the DB results with the filtered data.
    (Updated to use corpusid instead of title) 

    Args:
        filtered_df (DataFrame): The filtered Elasticsearch query cache DataFrame.
        db_path (str): Path to the SQLite database.

    Returns:
        DataFrame: Merged DataFrame containing both ES data and the corresponding DB records.
    """
    # Extract unique corpusid values, converting to lowercase and stripping spaces 
    #corpusids = filtered_df['corpusid'].str.lower().str.strip().unique()
    corpusids = (
        filtered_df['corpusid']
        .astype(str)
        .str.lower().str.strip()
        .dropna()
        .unique()
    )
    # batch querying
    frames = []
    with sqlite3.connect(db_path) as conn:
        for i in range(0, len(corpusids), batch_size):
            batch = corpusids[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            query = (
                "SELECT corpusid, filename, line_index, "              
                "LOWER(TRIM(title)) AS title "                         
                "FROM papers "                                         
                f"WHERE corpusid IN ({placeholders})"
            )
            frames.append(pd.read_sql(query, conn, params=list(batch)))
    db_df = pd.concat(frames, ignore_index=True)
    # Merge on corpusid only
    filtered_df['corpusid'] = filtered_df['corpusid'].astype(str).str.lower().str.strip()
    db_df['corpusid'] = db_df['corpusid'].astype(str).str.lower().str.strip()
    merged_df = filtered_df.merge(db_df, on='corpusid', how='left')
    return merged_df

def extract_lines_to_parquet(merged_df_parquet, data_directory, temp_parquet, n_jobs):
    def extract_lines(filename, indices):
        filepath = os.path.join(data_directory, filename)
        if not os.path.exists(filepath):
            tqdm.write(f"❌ Missing file: {filepath}")
            return []
        #sed_cmd = ";".join(f"{idx+1}p" for idx in indices)
        sed_cmd = ";".join(f"{int(idx)+1}p" for idx in indices)

        cmd = f"sed -n '{sed_cmd}' '{filepath}'"
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
            if not output.strip():                                   
                tqdm.write(f"⚠️  Empty sed result {filepath}")       
            return output.strip().split('\n')
        except Exception as e:
            print(f"[sed error] {filepath}: {e}")
            return []
    
    merged_df = pd.read_parquet(merged_df_parquet, columns=['filename', 'line_index'])
    dedup_df = merged_df.drop_duplicates(subset=['filename', 'line_index'], keep='first')
    grouped = list(dedup_df.groupby('filename'))

    os.makedirs(os.path.dirname(temp_parquet), exist_ok=True)
    if os.path.exists(temp_parquet):
        os.remove(temp_parquet)
    
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
    print('now parsing annotations')
    df_temp = pd.DataFrame(lines_expanded)
    parsed_data = Parallel(n_jobs=n_jobs)(
        delayed(parse_annotations)(row)
        for _, row in tqdm(df_temp.iterrows(), total=len(df_temp), desc="Parsing annotations")
    )
    df_parsed = pd.DataFrame([item for item in parsed_data if item])
    df_parsed.to_parquet(temp_parquet, compression="zstd", engine="pyarrow", index=False)

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
        placeholder = dict(row)
        placeholder.update({
            "extracted_openaccessurl": None,                 
            "extracted_tables": [],                          
            "extracted_tablerefs": [],                       
            "extracted_figures": [],                         
            "extracted_figure_captions": [],                 
            "extracted_figurerefs": []                       
        })
        return placeholder

def preprocess_custom_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={
        'query_title': 'query',
        'corpusId': 'corpusid'
    })
    df['rank'] = 1
    df['score'] = 1000
    return df

def merge_full_df(merged_df_parquet, temp_parquet, output_parquet):
    merged_df = pd.read_parquet(merged_df_parquet)
    df_extracted = pd.read_parquet(temp_parquet)
    merged_full = merged_df.merge(
        df_extracted[['filename', 'line_index',
                      'raw_json',
                      'extracted_openaccessurl',
                      'extracted_tables',
                      'extracted_tablerefs',
                      'extracted_figures',
                      'extracted_figure_captions',
                      'extracted_figurerefs']],
        on=['filename', 'line_index'],
        how='left'
    )
    merged_full.to_parquet(output_parquet, compression="zstd", engine="pyarrow", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, help="Directory of NDJSON files")
    parser.add_argument("--db_path", required=True, help="SQLite database path")
    parser.add_argument("--parquet_cache", required=True, help="Input Parquet cache")
    parser.add_argument("--output_parquet", default="data/processed/extracted_annotations.parquet", help="Output Parquet path")
    parser.add_argument("--temp_parquet", default="data/processed/tmp_extracted_lines.parquet", help="Temporary Parquet path")
    parser.add_argument("--merged_df", default="data/processed/merged_df.parquet", help="Merged DataFrame path")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Parallel jobs")

    args = parser.parse_args()

    # Step 1: Quality filter the Elasticsearch query cache.
    if 'query_cache' in args.parquet_cache: # this is queried from local, thus it might include multiple choice for one item, need to filter out then
        filtered_df = quality_filter_cache(args.parquet_cache)
    else:
        filtered_df = preprocess_custom_parquet(args.parquet_cache)
    # Step 2: Query the SQLite database using corpusid from the filtered data. 
    merged_df = query_db_by_corpusid(filtered_df, args.db_path, batch_size=1000)
    merged_df.to_parquet(args.merged_df, compression="zstd", engine="pyarrow", index=False)
    print('merged_df saved to', args.merged_df)
    # Step 3: Extract specific lines from NDJSON files and write to a temporary Parquet file.
    extract_lines_to_parquet(args.merged_df, args.directory, args.temp_parquet, args.n_jobs)
    # Step 4: Parse annotations from the extracted lines and write final annotated data to output Parquet.
    merge_full_df(args.merged_df, args.temp_parquet, args.output_parquet)
    print('output_parquet saved to', args.output_parquet)

if __name__ == "__main__":
    main()
