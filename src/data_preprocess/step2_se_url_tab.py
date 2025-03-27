#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Created: 2025-03-17
Last updated: 2025-03-27
Description:
  Demonstration of how to batch extract from Parquet-based ES query cache,
  then for each top-K retrieved title (or corpusid), query the SQLite index
  and NDJSON to extract table/figure data.

Usage:
  python step2_se_url_tab.py \
    --directory /u4/z6dong/shared_data/se_s2orc_250218 \
    --output_json data/processed/extracted_results.json \
    --db_path /u4/z6dong/shared_data/se_s2orc_250218/paper_index_mini.db \
    --parquet_cache query_cache.parquet
"""

import os
import json
import sqlite3
import argparse
import pandas as pd
from tqdm import tqdm

DATABASE_FILE = "paper_index_mini.db"

def query_paper_info(title, data_directory, db_path=DATABASE_FILE):
    """
    Query the database for a paper using its title (matching in lower-case).
    Returns a dictionary with corpusid, filename, line_index, and full filepath.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT corpusid, filename, line_index FROM papers WHERE title = ?", (title.lower(),))
    result = cur.fetchone()
    conn.close()
    if not result:
        return None
    corpusid, filename, line_index = result
    filepath = os.path.join(data_directory, filename)
    return {
        "corpusid": corpusid,
        "filename": filename,
        "line_index": line_index,
        "filepath": filepath
    }

def load_paper_json(filepath, line_index):
    """
    Load the JSON data from the NDJSON file at the given filepath and line number.
    """
    if not os.path.exists(filepath):
        print(f"❌ File {filepath} does not exist")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if line_index >= len(lines):
        print(f"❌ Line index {line_index} exceeds total lines {len(lines)} in file {filepath}")
        return None
    try:
        paper = json.loads(lines[line_index].strip())
        return paper
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON at line {line_index} in {filepath}: {e}")
        return None

def extract_annotations_from_paper(paper):
    """
    Extract table, figure, figure caption, and figure reference annotations from the paper's JSON data.
    The NDJSON data is assumed to have an 'annotations' field under paper["content"] with keys:
      - "tableref"           -> extracted_tables
      - "figure"             -> extracted_figures
      - "figurecaption"      -> extracted_figure_captions
      - "figureref"          -> extracted_figurerefs
    """
    content = paper.get("content", {})
    text = content.get("text", "")
    annotations = content.get("annotations", {})

    def load_annotations(key):
        raw = annotations.get(key, "[]")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    tablerefs = load_annotations("tableref")
    figures = load_annotations("figure")
    figurecaptions = load_annotations("figurecaption")
    figurerefs = load_annotations("figureref")

    extracted_tables = []
    for ref in tablerefs:
        start = ref.get("start")
        end = ref.get("end")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_tables.append({
                "tableref": ref,
                "extracted_text": text[start:end]
            })
    extracted_figures = []
    for fig in figures:
        start = fig.get("start")
        end = fig.get("end")
        fig_id = fig.get("attributes", {}).get("id", "unknown")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_figures.append({
                "figure_id": fig_id,
                "start": start,
                "end": end,
                "extracted_text": text[start:end]
            })
    extracted_figure_captions = []
    for cap in figurecaptions:
        start = cap.get("start")
        end = cap.get("end")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_figure_captions.append({
                "start": start,
                "end": end,
                "caption_text": text[start:end]
            })
    extracted_figurerefs = []
    for fref in figurerefs:
        start = fref.get("start")
        end = fref.get("end")
        ref_id = fref.get("attributes", {}).get("ref_id", "unknown")
        if isinstance(start, int) and isinstance(end, int) and start < end:
            extracted_figurerefs.append({
                "figure_ref_id": ref_id,
                "start": start,
                "end": end,
                "extracted_text": text[start:end]
            })
    extracted_data = {}
    if extracted_tables:
        extracted_data["extracted_tables"] = extracted_tables
    if extracted_figures:
        extracted_data["extracted_figures"] = extracted_figures
    if extracted_figure_captions:
        extracted_data["extracted_figure_captions"] = extracted_figure_captions
    if extracted_figurerefs:
        extracted_data["extracted_figurerefs"] = extracted_figurerefs
    return extracted_data

def extract_full_paper_data(title, data_directory, db_path=DATABASE_FILE):
    """
    For a given EXACT 'title', query the database to get the file location,
    load the corresponding NDJSON line as JSON, and extract all annotations.
    Returns a dict with corpusid, modelid, title, filename, line_index, and extracted data.
    """
    info = query_paper_info(title, data_directory, db_path=db_path)
    if info is None:
        print(f"❌ Paper with title='{title}' not found in the database!")
        return None

    paper = load_paper_json(info["filepath"], info["line_index"])
    if paper is None:
        return None

    # Extract annotations from the paper
    extracted_annotations = extract_annotations_from_paper(paper)
    # Get modelid if exists
    modelid = paper.get("modelid", "unknown")
    result = {
        "corpusid": info["corpusid"],
        "modelid": modelid,
        "title": title,
        "filename": info["filename"],
        "line_index": info["line_index"],
        "extracted": extracted_annotations
    }
    return result

def batch_extract_from_es_cache(parquet_cache, data_directory, db_path, output_json):
    if not os.path.exists(parquet_cache):
        print(f"❌ Parquet cache file '{parquet_cache}' does not exist!")
        return

    df_es = pd.read_parquet(parquet_cache)
    print(f"Loaded {len(df_es)} rows from '{parquet_cache}'")

    results = []
    df_es_unique = df_es.drop_duplicates(subset=["query", "retrieved_title"]).reset_index(drop=True)

    for idx, row in tqdm(df_es_unique.iterrows(), total=len(df_es_unique),
                         desc="Batch Extract from ES Cache"):
        retrieved_title = row["retrieved_title"]
        score = row["score"]
        rank = row["rank"]
        query_str = row["query"]

        if not retrieved_title or not isinstance(retrieved_title, str):
            continue

        paper_data = extract_full_paper_data(
            title=retrieved_title,
            data_directory=data_directory,
            db_path=db_path
        )
        if paper_data is not None:
            paper_data["rank"] = rank
            paper_data["score"] = score
            paper_data["query"] = query_str
            results.append(paper_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✅ Extraction from ES cache complete. Processed {len(results)} items. Saved to '{output_json}'")

def main():
    parser = argparse.ArgumentParser(description="Batch query papers and extract table/figure annotations")
    parser.add_argument("--directory", type=str, required=True,
                        help="Directory containing NDJSON (stepXX_file) files")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output JSON file to save extraction results")
    parser.add_argument("--db_path", type=str, default=DATABASE_FILE,
                        help="Path to the SQLite database file")

    parser.add_argument("--parquet_cache", type=str, default=None,
                        help="Path to Parquet file from ES batch query (e.g. 'query_cache.parquet')")

    parser.add_argument("--legacy_mode", action="store_true",
                        help="Use the old CSV->DB approach if set, otherwise use the ES-based approach")

    args = parser.parse_args()

    if args.legacy_mode:
        batch_query_extract(args.directory, args.output_json, db_path=args.db_path)
    else:
        if not args.parquet_cache:
            print("❌ Must provide --parquet_cache unless using legacy mode.")
            return
        batch_extract_from_es_cache(
            parquet_cache=args.parquet_cache,
            data_directory=args.directory,
            db_path=args.db_path,
            output_json=args.output_json
        )

def batch_query_extract(data_directory, output_json, db_path=DATABASE_FILE):
    with open("modelcard_dedup_titles.json", "r", encoding="utf-8") as f:
        titles_data = json.load(f)
    
    results = []
    for title in tqdm(titles_data, desc="Legacy Mode: DB Query"):
        if not title or not isinstance(title, str):
            continue
        paper_data = extract_full_paper_data(title, data_directory, db_path=db_path)
        if paper_data is not None:
            results.append(paper_data)
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✅ Legacy extraction complete. Found {len(results)} papers. Saved to {output_json}")

if __name__ == "__main__":
    main()
