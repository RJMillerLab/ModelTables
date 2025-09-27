#!/usr/bin/env python3
"""
Script to analyze HTML table files and compare column counts between v1 and v2
"""

import pandas as pd
import os
import csv
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

def main():
    # Load the keyword CSV
    keyword_csv = "tmp/top_tables_with_keywords.csv"
    df = pd.read_csv(keyword_csv)
    
    # Filter for HTML source only
    html_df = df[df['source'] == 'html'].copy()
    print(f"Found {len(html_df)} HTML table entries")
    
    # Prepare results
    results = []
    
    for idx, row in html_df.iterrows():
        table_file = row['table_file']
        expected_cols = row['value']  # This is the expected column count from the CSV
        
        # Paths for v1 and v2
        v1_path = f"data/processed/tables_output/{table_file}"
        v2_path = f"data/processed/tables_output_v2/{table_file}"
        
        # Check if files exist
        v1_exists = os.path.exists(v1_path)
        v2_exists = os.path.exists(v2_path)
        
        # Count columns
        v1_cols = count_columns(v1_path) if v1_exists else None
        v2_cols = count_columns(v2_path) if v2_exists else None
        
        # Calculate reduction
        reduction_pct = None
        if v1_cols is not None and v2_cols is not None and v1_cols > 0:
            reduction_pct = ((v1_cols - v2_cols) / v1_cols) * 100
        
        result = {
            'table_file': table_file,
            'modelId': row['modelId'],
            'expected_cols': expected_cols,
            'v1_exists': v1_exists,
            'v1_cols': v1_cols,
            'v2_exists': v2_exists,
            'v2_cols': v2_cols,
            'reduction_pct': reduction_pct,
            'downloads': row['downloads'],
            'card_length': row['card_length'],
            'readme_length': row['readme_length']
        }
        
        results.append(result)
        
        # Print progress for every 50 files
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(html_df)} files...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = "tmp/html_table_analysis.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Total HTML tables analyzed: {len(results_df)}")
    
    # Summary statistics
    v1_available = results_df['v1_exists'].sum()
    v2_available = results_df['v2_exists'].sum()
    both_available = (results_df['v1_exists'] & results_df['v2_exists']).sum()
    
    print(f"\nSummary:")
    print(f"V1 files available: {v1_available}")
    print(f"V2 files available: {v2_available}")
    print(f"Both V1 and V2 available: {both_available}")
    
    if both_available > 0:
        valid_reductions = results_df.dropna(subset=['reduction_pct'])
        if len(valid_reductions) > 0:
            avg_reduction = valid_reductions['reduction_pct'].mean()
            print(f"Average column reduction: {avg_reduction:.1f}%")
            print(f"Max reduction: {valid_reductions['reduction_pct'].max():.1f}%")
            print(f"Min reduction: {valid_reductions['reduction_pct'].min():.1f}%")
    
    # Show some examples
    print(f"\nFirst 10 results:")
    print(results_df[['table_file', 'expected_cols', 'v1_cols', 'v2_cols', 'reduction_pct']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
