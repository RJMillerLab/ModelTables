"""
Author: Zhengyuan Dong
Created: 2025-04-08
Last Modified: 2025-04-08
Description: Generate a markdown report for table retrieval results, automatically from JSON files.
Usage: 
 python -m src.data_analysis.report_generation
"""

import json
import os
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict
import numpy as np
from tqdm import tqdm

BASE_PATH = "/Users/doradong/Repo/CitationLake"

DATA_DIR  = os.path.join(BASE_PATH, "data/processed")
FILES = {
    "step3": f"{DATA_DIR}/modelcard_step3_dedup.parquet",    
    "valid_title"  : f"{DATA_DIR}/all_title_list_valid.parquet"
}

TABLE_SOURCE_PARQUET = os.path.join(BASE_PATH, "data/processed/modelcard_step3_dedup.parquet")
VALID_TITLE_PARQUET = os.path.join(BASE_PATH, "data/processed/all_title_list_valid.parquet")

def get_file_path(filename):
    hugging_pattern = r'^[0-9a-f]{10}_table\d+\.csv$'
    tables_output_pattern = r'^\d{4}\.\d{5}.*\.csv$'
    llm_tables_pattern = r'^\d+_table\d+\.csv$'
    github_pattern = r'^[0-9a-f]{32}_table_\d+\.csv$'
    
    if re.match(hugging_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/deduped_hugging_csvs', filename)
    elif re.match(tables_output_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/tables_output', filename)
    elif re.match(llm_tables_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/llm_tables', filename)
    elif re.match(github_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/deduped_github_csvs', filename)
    else:
        raise ValueError(f"Unknown: {filename}")

def df_to_markdown(df, max_rows=5):
    if isinstance(df, pd.DataFrame):
        return df.head(max_rows).to_markdown(index=False) + "\n..."
    return df

def build_table_model_title_maps():
    """Return mapping dictionaries:
       - table_to_models: csv filename → set of modelIds
       - model_to_titles: modelId → list of valid titles
    """
    df_tables = pd.read_parquet(FILES["step3"])
    print('df_tables keys: ', df_tables.keys())
    df_titles = pd.read_parquet(FILES["valid_title"], columns=["modelId", "all_title_list", "all_title_list_valid"])
    df_tables = df_tables.merge(df_titles, on="modelId", how="left")
    table_cols = [c for c in df_tables.columns if c.endswith("_sym") or c.endswith("_dedup")]
    table_to_models = defaultdict(set)
    model_to_titles = {}
    for _, row in df_tables.iterrows():
        mid = row["modelId"]
        raw_vals = row.get("all_title_list", [])
        if isinstance(raw_vals, np.ndarray):
            raw_vals = raw_vals.tolist()
        if not isinstance(raw_vals, list):
            raw_vals = [raw_vals]
        valid_vals = row.get("all_title_list_valid", [])
        if isinstance(valid_vals, np.ndarray):
            valid_vals = valid_vals.tolist()
        if not isinstance(valid_vals, list):
            valid_vals = [valid_vals]
        model_to_titles[mid] = {"raw": raw_vals, "valid": valid_vals}
        for col in table_cols:
            col_val = row.get(col, [])
            if isinstance(col_val, np.ndarray):
                col_val = col_val.tolist()
            if not isinstance(col_val, list):
                col_val = [col_val]
            for tbl in col_val:
                if pd.notna(tbl):
                    table_to_models[os.path.basename(tbl)].add(mid)
    return table_to_models, model_to_titles

def generate_md_report(json_path, include_raw, include_valid, output_file=None):
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = f"table_report_{timestamp}.md"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # only print first 10
    print(f"Loaded {len(data)} records from {json_path}")
    # change index!
    data = {k: v for i, (k, v) in enumerate(data.items()) if 10 <= i < 20}
    print(f"Filtered to {len(data)} records for report generation.")
    table_to_models, model_to_titles = build_table_model_title_maps()
    print('Build table to model and model to titles mapping')
    
    report = []
    for query_file, retrieved_files in tqdm(data.items()):
        report.append(f"# Query Table: `{query_file}`\n")
        try:
            query_path = get_file_path(query_file)
            query_df = pd.read_csv(query_path)
            report.append("## Query Table Content\n")
            report.append(df_to_markdown(query_df))
        except Exception as e:
            report.append(f"**Loading failure: {str(e)}**")
        
        # Query Table Titles
        query_key = os.path.basename(query_file)
        models = table_to_models.get(query_key, [])
        if models:
            report.append("\n**Query Table Model → Titles**")
            for m in sorted(models):
                titles = model_to_titles.get(m, {"raw": [], "valid": []})
                report.append(f"- **{m}**:")
                if include_raw:
                    raw_titles = titles.get("raw", [])
                    report.append(f"    - Raw Titles: {raw_titles}")
                if include_valid:
                    valid_titles = titles.get("valid", [])
                    report.append(f"    - Valid Titles: {valid_titles}")
        
        report.append("\n## Retrieved Tables\n")
        for idx, file in enumerate(retrieved_files, 1):
            if file == query_file:
                continue
            report.append(f"### Top {idx}: `{file}`\n")
            try:
                file_path = get_file_path(file)
                df = pd.read_csv(file_path)
                report.append(df_to_markdown(df))
            except Exception as e:
                report.append(f"*Loading failure: {str(e)}*")

            # Retrieved Table Titles
            tbl_key = os.path.basename(file)
            models = table_to_models.get(tbl_key, [])
            if models:
                report.append("\n**Model → Titles**")
                for m in sorted(models):
                    titles = model_to_titles.get(m, {"raw": [], "valid": []})
                    report.append(f"- **{m}**:")
                    if include_raw:
                        raw_titles = titles.get("raw", [])
                        report.append(f"    - Raw Titles: {raw_titles}")
                    if include_valid:
                        valid_titles = titles.get("valid", [])
                        report.append(f"    - Valid Titles: {valid_titles}")
            report.append("\n")
        
        report.append("\n---\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"Report generated: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a markdown report from JSON files.")
    parser.add_argument("--json_path", type=str, default="results/scilake_final/test_hnsw_search_drop_cell_tfidf_entity_full.json", help="Path to the JSON file.")
    include_valid = True
    include_raw = False
    args = parser.parse_args()
    generate_md_report(args.json_path, include_raw, include_valid)
