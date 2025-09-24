"""
Author: Zhengyuan Dong
Created: 2025-08-30
Description: Merge WDC data into existing benchmark_results.parquet file
"""

import pandas as pd
import os
from src.utils import to_parquet

# WDC data from qc_stats.py
WDC_DATA = {
    "Benchmark": "WDC",
    "# Tables": 50000000,
    "# Cols": 250000000,
    "Avg # Rows": 14,
    "Size (GB)": 500.00
}

def merge_wdc_data():
    """Merge WDC data into existing benchmark_results.parquet file"""
    
    # Paths
    results_path = "data/analysis/benchmark_results.parquet"
    backup_path = "data/analysis/benchmark_results_backup.parquet"
    
    print("🔍 Reading existing benchmark results...")
    
    # Check if file exists
    if not os.path.exists(results_path):
        print(f"❌ File not found: {results_path}")
        return
    
    # Read existing data
    df = pd.read_parquet(results_path)
    print(f"✅ Loaded existing data: {df.shape}")
    
    # Check if WDC already exists
    if 'WDC' in df['Benchmark'].values:
        print("✅ WDC data already exists in the file")
        print(df[df['Benchmark'] == 'WDC'])
        return
    
    # Create backup
    print("💾 Creating backup...")
    to_parquet(df, backup_path)
    print(f"✅ Backup saved to: {backup_path}")
    
    # Add WDC data
    print("➕ Adding WDC data...")
    
    # Create WDC row
    wdc_row = pd.DataFrame([WDC_DATA])
    
    # Insert WDC after SANTOS Large (at position 4)
    df_with_wdc = pd.concat([
        df.iloc[:4],  # First 4 rows (SANTOS Small, TUS Small, TUS Large, SANTOS Large)
        wdc_row,      # WDC data
        df.iloc[4:]   # Remaining rows (scilake data)
    ], ignore_index=True)
    
    # Calculate Avg # Cols for WDC
    if "Avg # Cols" in df_with_wdc.columns:
        df_with_wdc.loc[4, "Avg # Cols"] = WDC_DATA["# Cols"] / WDC_DATA["# Tables"]
    
    print(f"✅ New data shape: {df_with_wdc.shape}")
    
    # Save updated data
    print("💾 Saving updated data...")
    to_parquet(df_with_wdc, results_path)
    print(f"✅ Updated data saved to: {results_path}")
    
    # Show the merged data
    print("\n📊 Updated baseline benchmarks:")
    baseline_mask = df_with_wdc['Benchmark'].isin(["SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large", "WDC"])
    print(df_with_wdc[baseline_mask][['Benchmark', '# Tables', '# Cols', 'Avg # Rows', 'Size (GB)']])
    
    print(f"\n🎉 Successfully merged WDC data!")
    print(f"Original rows: {len(df)}")
    print(f"New rows: {len(df_with_wdc)}")
    print(f"WDC data: {WDC_DATA}")

if __name__ == "__main__":
    merge_wdc_data()
