"""
Author: Zhengyuan Dong
Created: 2025-04-11
Updated: 2025-09-01
Description: Given a CSV file name, return associated modelIds and their titles.
Uses step3_merged.parquet for better performance (includes titles directly).
Usage:
    python -m src.data_analysis.check_related --csv 201646309_table4.csv
"""

import os
import pandas as pd
import argparse
import numpy as np
from collections import defaultdict

BASE_PATH = "/Users/doradong/Repo/CitationLake"
DATA_DIR = os.path.join(BASE_PATH, "data/processed")

FILES = {
    "step3_merged": f"{DATA_DIR}/modelcard_step3_merged.parquet",
}

def build_table_model_title_maps():
    """Build mappings:
        - table_to_models: csv filename -> set of modelIds
        - model_to_titles: modelId -> dict of raw/valid titles
    """
    # Load step3_merged which already contains titles and table lists
    df = pd.read_parquet(FILES["step3_merged"])

    # Get table columns (mapped versions from step3)
    table_cols = [c for c in df.columns if c.endswith("_mapped") or c.endswith("_list")]
    table_cols = [c for c in table_cols if c not in ["all_title_list", "all_bibtex_titles"]]
    
    table_to_models = defaultdict(set)
    model_to_titles = {}

    for _, row in df.iterrows():
        mid = row["modelId"]
        title_vals = row.get("all_title_list", [])

        # Handle title values
        if isinstance(title_vals, np.ndarray): title_vals = title_vals.tolist()
        if not isinstance(title_vals, list): title_vals = [title_vals] if title_vals else []

        model_to_titles[mid] = {"raw": title_vals, "valid": title_vals}

        # Process table columns
        for col in table_cols:
            col_val = row.get(col, [])
            if isinstance(col_val, np.ndarray): col_val = col_val.tolist()
            if not isinstance(col_val, list): col_val = [col_val] if col_val else []

            for tbl in col_val:
                if pd.notna(tbl) and tbl:
                    table_to_models[os.path.basename(tbl)].add(mid)

    return table_to_models, model_to_titles

def print_csv_info(csv_name):
    print(f"🔍 Looking up CSV: {csv_name}")
    table_to_models, model_to_titles = build_table_model_title_maps()

    models = table_to_models.get(csv_name, [])
    if not models:
        print("❌ No associated models found.")
        return

    print(f"✅ Found {len(models)} associated model(s):")
    for mid in sorted(models):
        titles = model_to_titles.get(mid, {"raw": [], "valid": []})
        print(f"- Model ID: {mid}")
        print(f"    • Valid Titles: {titles.get('valid', [])}")
        print(f"    • Raw Titles  : {titles.get('raw', [])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query related modelIds and titles for a given CSV file.")
    parser.add_argument("--csv", type=str, required=True, help="CSV filename to query (e.g. 201646309_table4.csv)")
    args = parser.parse_args()
    print_csv_info(args.csv)
