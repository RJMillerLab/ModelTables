#!/usr/bin/env python3
"""
Script to analyze HuggingFace table files and compare column counts between v1 and v2
"""

import pandas as pd
import os
import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def count_columns_fast(csv_file):
    """Count the number of columns in a CSV file by reading only the first line"""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return 0
            # Count commas + 1 (simple approach, works for most CSV files)
            return first_line.count(',') + 1
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def count_rows_ultra_fast(csv_file):
    """Count rows using memory-mapped file (fastest method)"""
    try:
        import mmap
        with open(csv_file, 'r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # Count newlines, subtract 1 for header
                return mm.count(b'\n') - 1
    except Exception as e:
        # Fallback to simple method
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                return sum(1 for line in f) - 1
        except Exception as e2:
            print(f"Error counting rows in {csv_file}: {e2}")
            return None


def build_file_index(v1_dir, v2_dir):
    """Build a fast lookup index of all CSV files"""
    v1_index = {}
    v2_index = {}
    
    print("Building file index...")
    
    # Index v1 files
    if os.path.exists(v1_dir):
        for root, dirs, files in os.walk(v1_dir):
            for file in files:
                if file.endswith('.csv'):
                    v1_index[file] = os.path.join(root, file)
    
    # Index v2 files  
    if os.path.exists(v2_dir):
        for root, dirs, files in os.walk(v2_dir):
            for file in files:
                if file.endswith('.csv'):
                    v2_index[file] = os.path.join(root, file)
    
    print(f"Indexed {len(v1_index)} v1 files, {len(v2_index)} v2 files")
    return v1_index, v2_index

def find_csv_files_fast(table_file, v1_index, v2_index):
    """Find CSV files using pre-built index"""
    v1_path = v1_index.get(table_file)
    v2_path = v2_index.get(table_file)
    
    v1_csvs = [v1_path] if v1_path else []
    v2_csvs = [v2_path] if v2_path else []
    
    return v1_csvs, v2_csvs

def process_single_table(args):
    """Process a single table file - designed for parallel execution"""
    table_file, model_id, v1_index, v2_index = args
    
    # Find corresponding CSV files in both versions using index
    v1_csvs, v2_csvs = find_csv_files_fast(table_file, v1_index, v2_index)
    
    # Check if files exist and count columns, rows
    v1_exists = len(v1_csvs) > 0
    v2_exists = len(v2_csvs) > 0
    
    v1_cols = None
    v1_rows = None
    v2_cols = None
    v2_rows = None
    
    if v1_exists:
        v1_path = v1_csvs[0]
        v1_cols = count_columns_fast(v1_path)
        v1_rows = count_rows_ultra_fast(v1_path)
    
    if v2_exists:
        v2_path = v2_csvs[0]
        v2_cols = count_columns_fast(v2_path)
        v2_rows = count_rows_ultra_fast(v2_path)
    
    return {
        'table_file': table_file,
        'modelId': model_id,
        'v1_cols': v1_cols,
        'v2_cols': v2_cols,
        'v1_rows': v1_rows,
        'v2_rows': v2_rows
    }

def main():
    # Load the keyword CSV to get model information
    keyword_csv = "tmp/top_tables_with_keywords.csv"
    if not os.path.exists(keyword_csv):
        print(f"Error: {keyword_csv} not found. Please run batch_process_tables.py first.")
        return
    
    df = pd.read_csv(keyword_csv)
    
    # Filter for HuggingFace source only
    hugging_df = df[df['source'] == 'huggingface'].copy()
    print(f"Found {len(hugging_df)} HuggingFace table entries")
    
    if len(hugging_df) == 0:
        print("No HuggingFace entries found in the keyword CSV")
        return
    
    # Define directories to search
    v1_dir = "data/processed/deduped_hugging_csvs"
    v2_dir = "data/processed/deduped_hugging_csvs_v2"
    
    print(f"Building file index for:")
    print(f"  V1 directory: {v1_dir}")
    print(f"  V2 directory: {v2_dir}")
    
    # Build file index once
    v1_index, v2_index = build_file_index(v1_dir, v2_dir)
    
    # Prepare arguments for parallel processing
    args_list = []
    for idx, row in hugging_df.iterrows():
        table_file = row['table_file']
        model_id = row['modelId']
        args_list.append((table_file, model_id, v1_index, v2_index))
    
    print(f"Processing {len(args_list)} tables in parallel...")
    
    # Process in parallel
    results = []
    max_workers = min(cpu_count(), len(args_list))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_table, args): args for args in args_list}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Print progress every 50 completions
                if completed % 50 == 0:
                    print(f"Processed {completed}/{len(args_list)} files...")
                    
            except Exception as e:
                args = future_to_args[future]
                print(f"Error processing {args[0]}: {e}")
                completed += 1
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = "tmp/huggingface_table_analysis.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Total HuggingFace tables analyzed: {len(results_df)}")
    
    # Summary statistics
    v1_available = results_df['v1_cols'].notna().sum()
    v2_available = results_df['v2_cols'].notna().sum()
    both_available = (results_df['v1_cols'].notna() & results_df['v2_cols'].notna()).sum()
    
    print(f"\nSummary:")
    print(f"V1 files available: {v1_available}")
    print(f"V2 files available: {v2_available}")
    print(f"Both V1 and V2 available: {both_available}")
    
    # Show some examples
    print(f"\nFirst 10 results:")
    print(results_df[['table_file', 'modelId', 'v1_cols', 'v2_cols', 'v1_rows', 'v2_rows']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
