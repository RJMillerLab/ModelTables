#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-09-01
Description: Batch script to find modelIds for CSV files listed in top_tables.txt
Usage:
    python tmp/batch_find_modelids.py
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the check_related module
from src.data_analysis.check_related import build_table_model_title_maps

def process_top_tables(input_file="tmp/top_tables.txt", output_file="tmp/top_tables_with_modelids.txt"):
    """
    Process the top_tables.txt file and add modelId information to each line.
    
    Args:
        input_file: Path to input file with CSV names and scores
        output_file: Path to output file with CSV names, scores, and modelIds
    """
    print(f"🔍 Processing {input_file}...")
    
    # Build the mapping from CSV filename to modelIds
    print("📊 Building table to modelId mapping...")
    table_to_models, model_to_titles = build_table_model_title_maps()
    
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
    
    print("🚀 Starting batch modelId lookup...")
    result_file = process_top_tables(input_file, output_file)
    print(f"🎉 Done! Check {result_file} for results.")
