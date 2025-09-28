#!/usr/bin/env python3
"""
Script to analyze HuggingFace table files and compare column counts between v1 and v2
"""

import pandas as pd
import os
import csv
import json
from pathlib import Path

def count_columns(csv_file):
    """Count the number of columns in a CSV file"""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            first_row = next(reader)
            return len(first_row)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def count_rows(csv_file):
    """Count the number of rows in a CSV file"""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return sum(1 for row in reader)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def get_file_size(csv_file):
    """Get file size in bytes"""
    try:
        return os.path.getsize(csv_file)
    except Exception as e:
        print(f"Error getting size of {csv_file}: {e}")
        return None

def load_huggingface_mappings():
    """Load both v1 and v2 HuggingFace mappings"""
    v1_mapping_path = "data/processed/hugging_deduped_mapping.json"
    v2_mapping_path = "data/processed/hugging_deduped_mapping_v2.json"
    
    v1_mapping = {}
    v2_mapping = {}
    
    if os.path.exists(v1_mapping_path):
        with open(v1_mapping_path, 'r', encoding='utf-8') as f:
            v1_mapping = json.load(f)
        print(f"Loaded v1 mapping with {len(v1_mapping)} entries")
    else:
        print(f"Warning: v1 mapping not found at {v1_mapping_path}")
    
    if os.path.exists(v2_mapping_path):
        with open(v2_mapping_path, 'r', encoding='utf-8') as f:
            v2_mapping = json.load(f)
        print(f"Loaded v2 mapping with {len(v2_mapping)} entries")
    else:
        print(f"Warning: v2 mapping not found at {v2_mapping_path}")
    
    return v1_mapping, v2_mapping

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
    
    # Load mappings
    v1_mapping, v2_mapping = load_huggingface_mappings()
    
    # Prepare results
    results = []
    
    for idx, row in hugging_df.iterrows():
        table_file = row['table_file']
        expected_cols = row['value']  # This is the expected column count from the CSV
        model_id = row['modelId']
        
        # Find corresponding CSV files in both versions
        v1_csvs = []
        v2_csvs = []
        
        # Look for the table file in v1 mapping
        for readme_hash, csv_list in v1_mapping.items():
            for csv_path in csv_list:
                if os.path.basename(csv_path) == table_file:
                    v1_csvs.append(csv_path)
        
        # Look for the table file in v2 mapping
        for readme_hash, csv_list in v2_mapping.items():
            for csv_path in csv_list:
                if os.path.basename(csv_path) == table_file:
                    v2_csvs.append(csv_path)
        
        
        # Check if files exist and count columns, rows, and file size
        v1_exists = len(v1_csvs) > 0
        v2_exists = len(v2_csvs) > 0
        
        v1_cols = None
        v1_rows = None
        v1_size = None
        v2_cols = None
        v2_rows = None
        v2_size = None
        
        if v1_exists:
            # Use the first matching CSV file
            v1_path = v1_csvs[0]
            v1_cols = count_columns(v1_path)
            v1_rows = count_rows(v1_path)
            v1_size = get_file_size(v1_path)
        
        if v2_exists:
            # Use the first matching CSV file
            v2_path = v2_csvs[0]
            v2_cols = count_columns(v2_path)
            v2_rows = count_rows(v2_path)
            v2_size = get_file_size(v2_path)
        
        # Calculate changes
        col_reduction_pct = None
        if v1_cols is not None and v2_cols is not None and v1_cols > 0:
            col_reduction_pct = ((v1_cols - v2_cols) / v1_cols) * 100
        
        row_change_pct = None
        if v1_rows is not None and v2_rows is not None and v1_rows > 0:
            row_change_pct = ((v2_rows - v1_rows) / v1_rows) * 100
        
        size_change_pct = None
        if v1_size is not None and v2_size is not None and v1_size > 0:
            size_change_pct = ((v2_size - v1_size) / v1_size) * 100
        
        result = {
            'table_file': table_file,
            'modelId': model_id,
            'expected_cols': expected_cols,
            'v1_exists': v1_exists,
            'v1_cols': v1_cols,
            'v1_rows': v1_rows,
            'v1_size': v1_size,
            'v2_exists': v2_exists,
            'v2_cols': v2_cols,
            'v2_rows': v2_rows,
            'v2_size': v2_size,
            'col_reduction_pct': col_reduction_pct,
            'row_change_pct': row_change_pct,
            'size_change_pct': size_change_pct,
            'downloads': row['downloads'],
            'card_length': row['card_length'],
            'readme_length': row['readme_length']
        }
        
        results.append(result)
        
        # Print progress for every 50 files
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(hugging_df)} files...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = "tmp/huggingface_table_analysis.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Total HuggingFace tables analyzed: {len(results_df)}")
    
    # Summary statistics
    v1_available = results_df['v1_exists'].sum()
    v2_available = results_df['v2_exists'].sum()
    both_available = (results_df['v1_exists'] & results_df['v2_exists']).sum()
    
    print(f"\nSummary:")
    print(f"V1 files available: {v1_available}")
    print(f"V2 files available: {v2_available}")
    print(f"Both V1 and V2 available: {both_available}")
    
    if both_available > 0:
        # Column changes
        valid_col_changes = results_df.dropna(subset=['col_reduction_pct'])
        if len(valid_col_changes) > 0:
            avg_col_reduction = valid_col_changes['col_reduction_pct'].mean()
            print(f"Average column reduction: {avg_col_reduction:.1f}%")
            print(f"Max column reduction: {valid_col_changes['col_reduction_pct'].max():.1f}%")
            print(f"Min column reduction: {valid_col_changes['col_reduction_pct'].min():.1f}%")
        
        # Row changes
        valid_row_changes = results_df.dropna(subset=['row_change_pct'])
        if len(valid_row_changes) > 0:
            avg_row_change = valid_row_changes['row_change_pct'].mean()
            print(f"Average row change: {avg_row_change:.1f}%")
            print(f"Max row increase: {valid_row_changes['row_change_pct'].max():.1f}%")
            print(f"Max row decrease: {valid_row_changes['row_change_pct'].min():.1f}%")
        
        # Size changes
        valid_size_changes = results_df.dropna(subset=['size_change_pct'])
        if len(valid_size_changes) > 0:
            avg_size_change = valid_size_changes['size_change_pct'].mean()
            print(f"Average size change: {avg_size_change:.1f}%")
            print(f"Max size increase: {valid_size_changes['size_change_pct'].max():.1f}%")
            print(f"Max size decrease: {valid_size_changes['size_change_pct'].min():.1f}%")
    
    # Show some examples
    print(f"\nFirst 10 results:")
    print(results_df[['table_file', 'expected_cols', 'v1_cols', 'v2_cols', 'v1_rows', 'v2_rows', 'col_reduction_pct', 'row_change_pct', 'size_change_pct']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
