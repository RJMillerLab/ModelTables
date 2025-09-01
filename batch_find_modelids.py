#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-09-01
Description: Batch script to find modelIds for CSV files using step3_merged.parquet
Usage:
    python batch_find_modelids_step3.py
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def build_table_model_title_maps_step3():
    """Build mappings using step3_merged.parquet:
        - table_to_models: csv filename -> set of modelIds
        - model_to_titles: modelId -> dict of raw/valid titles
    """
    # Load step3_merged which already contains titles and table lists
    data_dir = "data/processed"
    df = pd.read_parquet(f"{data_dir}/modelcard_step3_merged.parquet")

    # Get table columns (mapped versions from step3)
    table_cols = [c for c in df.columns if c.endswith("_mapped") or c.endswith("_list")]
    table_cols = [c for c in table_cols if c not in ["all_title_list", "all_bibtex_titles"]]
    
    print(f"📊 Processing {len(df)} models with table columns: {table_cols}")
    
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
                if pd.notna(tbl) and tbl and isinstance(tbl, str):
                    table_to_models[os.path.basename(tbl)].add(mid)

    print(f"✅ Built mapping: {len(table_to_models)} unique CSV files -> {len(model_to_titles)} models")
    return table_to_models, model_to_titles

def process_top_tables(input_file="tmp/top_tables.txt", output_file="tmp/top_tables_with_modelids.txt"):
    """
    Process the top_tables.txt file and add modelId information to each line using step3 data.
    
    Args:
        input_file: Path to input file with CSV names and scores
        output_file: Path to output file with CSV names, scores, and modelIds
    """
    print(f"🔍 Processing {input_file}...")
    
    # Build the mapping from CSV filename to modelIds
    print("📊 Building table to modelId mapping using step3_merged...")
    table_to_models, model_to_titles = build_table_model_title_maps_step3()
    
    # Read the input file
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    results = []
    not_found = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                print(f"⚠️  Line {line_num}: Invalid format - {line}")
                continue
                
            csv_name = parts[0]
            score = parts[1]
            
            # Find modelIds for this CSV
            model_ids = table_to_models.get(csv_name, [])
            
            if model_ids:
                # Join multiple modelIds with semicolon if there are multiple
                model_id_str = '; '.join(sorted(model_ids))
                results.append(f"{csv_name}\t{score}\t{model_id_str}")
                print(f"✅ {csv_name} -> {model_id_str}")
            else:
                not_found.append(csv_name)
                results.append(f"{csv_name}\t{score}\tNOT_FOUND")
                print(f"❌ {csv_name} -> NOT_FOUND")
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')
    
    print(f"\n📝 Results saved to: {output_file}")
    print(f"✅ Found modelIds for: {len(results) - len(not_found)} files")
    print(f"❌ Not found: {len(not_found)} files")
    
    if not_found:
        print(f"\n⚠️  Files without modelIds:")
        for csv_name in not_found:
            print(f"   - {csv_name}")
    
    return output_file

if __name__ == "__main__":
    input_file = "tmp/top_tables.txt"
    output_file = "tmp/top_tables_with_modelids.txt"
    
    print("🚀 Starting batch modelId lookup using step3_merged...")
    result_file = process_top_tables(input_file, output_file)
    print(f"🎉 Done! Check {result_file} for results.")
