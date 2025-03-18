"""
Author: Zhengyuan Dong
Created: 2025-03-17
Description:
  Batch query papers by title from a pre-built SQLite index, then extract table, figure,
  figure caption, and figure reference annotations from the corresponding NDJSON files.
  The script retains corpusid, modelid, title, and file position information.
  Finally, all results are saved to an output JSON file.
Usage:
  python step2_se_url_tab.py --directory /u4/z6dong/shared_data/se_s2orc_250218 --output_json extracted_results.json
"""

import os
import json
import sqlite3
import argparse
import pandas as pd
from tqdm import tqdm

DATABASE_FILE = "paper_index_mini.db"  # Database file, consistent with build_mini_s2orc.py

def query_paper_info(title, data_directory, db_path=DATABASE_FILE):
    """
    Query the database for a paper using its title.
    Returns a dictionary with corpusid, filename, line_index, and full filepath.
    The title is matched in lower-case.
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
    The text is extracted from paper["content"]["text"] using the provided start and end positions.
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
    For a given title, query the database to get the file location,
    load the corresponding NDJSON line as JSON, and extract all annotations.
    Retains corpusid, modelid, title, and file position information.
    """
    info = query_paper_info(title, data_directory, db_path=db_path)
    if info is None:
        print(f"❌ Paper with title '{title}' not found in the database!")
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

def batch_query_extract(data_directory, output_json, db_path=DATABASE_FILE):
    """
    Load a CSV file containing paper titles, batch query the database for each paper,
    and extract the corresponding paper data (annotations and metadata).
    The final results are saved to an output JSON file.
    The CSV must have at least a column "title".
    """
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    df = pd.read_parquet(
        os.path.join(processed_base_path, f"{data_type}_ext_title.parquet"),
        columns=['modelId', 'card_tags', 'github_link', 'pdf_link']
    )
    
    results = []
    # Iterate over each title in the DataFrame
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Batch Query & Extraction"):
        title = row.get("title")
        if not title or not isinstance(title, str):
            continue
        paper_data = extract_full_paper_data(title, data_directory, db_path=db_path)
        if paper_data is not None:
            results.append(paper_data)
    # Save all results to the output JSON file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✅ Extraction complete. Processed {len(results)} items. Results saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch query papers and extract table/figure annotations")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing NDJSON (stepXX_file) files")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file to save extraction results")
    parser.add_argument("--db_path", type=str, default=DATABASE_FILE, help="Path to the SQLite database file")
    args = parser.parse_args()
    
    batch_query_extract(args.directory, args.output_json, db_path=args.db_path)
