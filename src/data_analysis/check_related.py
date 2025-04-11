"""
Author: Zhengyuan Dong
Created: 2025-04-11
Description: Given a CSV file name, return associated modelIds and their titles.
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
    "step4_symlink": f"{DATA_DIR}/modelcard_step4.parquet",
    "valid_title": f"{DATA_DIR}/all_title_list_valid.parquet",
}

def build_table_model_title_maps():
    """Build mappings:
        - table_to_models: csv filename -> set of modelIds
        - model_to_titles: modelId -> dict of raw/valid titles
    """
    df_tables = pd.read_parquet(FILES["step4_symlink"])
    df_titles = pd.read_parquet(FILES["valid_title"], columns=["modelId", "all_title_list", "all_title_list_valid"])
    df_tables = df_tables.merge(df_titles, on="modelId", how="left")

    table_cols = [c for c in df_tables.columns if c.endswith("_sym") or c.endswith("_dedup")]
    table_to_models = defaultdict(set)
    model_to_titles = {}

    for _, row in df_tables.iterrows():
        mid = row["modelId"]
        raw_vals = row.get("all_title_list", [])
        valid_vals = row.get("all_title_list_valid", [])

        if isinstance(raw_vals, np.ndarray): raw_vals = raw_vals.tolist()
        if isinstance(valid_vals, np.ndarray): valid_vals = valid_vals.tolist()
        if not isinstance(raw_vals, list): raw_vals = [raw_vals]
        if not isinstance(valid_vals, list): valid_vals = [valid_vals]

        model_to_titles[mid] = {"raw": raw_vals, "valid": valid_vals}

        for col in table_cols:
            col_val = row.get(col, [])
            if isinstance(col_val, np.ndarray): col_val = col_val.tolist()
            if not isinstance(col_val, list): col_val = [col_val]

            for tbl in col_val:
                if pd.notna(tbl):
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
