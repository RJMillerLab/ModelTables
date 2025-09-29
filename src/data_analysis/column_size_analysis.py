#!/usr/bin/env python3
"""
Analyze column sizes from parquet files and rank by memory usage.
Performs real-time computation by scanning and analyzing parquet files directly.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0

def analyze_single_parquet_file(file_path: str) -> Dict:
    """Analyze column sizes for a single parquet file."""
    try:
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Get basic file info
        file_size_mb = get_file_size_mb(file_path)
        num_rows = len(df)
        num_cols = len(df.columns)
        
        # Calculate memory usage per column
        column_info = {}
        total_memory = 0
        
        for col in df.columns:
            # Get memory usage in bytes
            memory_usage = df[col].memory_usage(deep=True)
            memory_mb = memory_usage / (1024 * 1024)
            
            # Get data type
            dtype = str(df[col].dtype)
            
            # Get null count
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / num_rows) * 100 if num_rows > 0 else 0
            
            # For list columns, get average list length
            avg_list_length = None
            if dtype.startswith('object') and num_rows > 0:
                try:
                    # Check if it's a list column by sampling first few non-null values
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0 and isinstance(sample_values.iloc[0], list):
                        list_lengths = df[col].dropna().apply(lambda x: len(x) if isinstance(x, list) else 0)
                        avg_list_length = list_lengths.mean()
                except:
                    pass
            
            column_info[col] = {
                'memory_mb': round(memory_mb, 2),
                'dtype': dtype,
                'null_count': int(null_count),
                'null_percentage': round(null_percentage, 2),
                'avg_list_length': round(avg_list_length, 2) if avg_list_length is not None else None
            }
            
            total_memory += memory_mb
        
        return {
            'file_path': file_path,
            'file_size_mb': round(file_size_mb, 2),
            'num_rows': num_rows,
            'num_cols': num_cols,
            'total_memory_mb': round(total_memory, 2),
            'columns': column_info
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'file_size_mb': get_file_size_mb(file_path)
        }

def find_parquet_files(directories: List[str]) -> List[str]:
    """Find all parquet files in the given directories."""
    parquet_files = []
    
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, file))
        else:
            print(f"Warning: Directory {directory} does not exist")
    
    return parquet_files

def analyze_column_sizes(directories: List[str] = None, exclude_modelid: bool = True):
    """Analyze column sizes and create rankings from parquet files."""
    
    if directories is None:
        directories = ['data/processed', 'data/gt', 'data/analysis']
    
    print(f"Scanning directories: {directories}")
    parquet_files = find_parquet_files(directories)
    
    if not parquet_files:
        print("No parquet files found in the specified directories!")
        return [], [], {}
    
    print(f"Found {len(parquet_files)} parquet files to analyze...")
    
    # Dictionary to store column information
    column_info = defaultdict(list)
    
    # Process each file
    for i, file_path in enumerate(parquet_files, 1):
        print(f"Analyzing file {i}/{len(parquet_files)}: {os.path.basename(file_path)}")
        
        file_data = analyze_single_parquet_file(file_path)
        
        if 'error' in file_data:
            print(f"  ❌ Error: {file_data['error']}")
            continue
            
        file_size_mb = file_data.get('file_size_mb', 0)
        num_rows = file_data.get('num_rows', 0)
        
        if 'columns' not in file_data:
            continue
            
        for col_name, col_data in file_data['columns'].items():
            # Skip modelId columns if requested
            if exclude_modelid and col_name.lower() == 'modelid':
                continue
                
            memory_mb = col_data.get('memory_mb', 0)
            dtype = col_data.get('dtype', 'unknown')
            null_count = col_data.get('null_count', 0)
            null_percentage = col_data.get('null_percentage', 0)
            avg_list_length = col_data.get('avg_list_length')
            
            column_info[col_name].append({
                'file_path': file_path,
                'memory_mb': memory_mb,
                'dtype': dtype,
                'null_count': null_count,
                'null_percentage': null_percentage,
                'avg_list_length': avg_list_length,
                'file_size_mb': file_size_mb,
                'num_rows': num_rows
            })
    
    # Calculate total memory usage per column across all files
    column_totals = {}
    for col_name, occurrences in column_info.items():
        total_memory = sum(occ['memory_mb'] for occ in occurrences)
        total_rows = sum(occ['num_rows'] for occ in occurrences)
        file_count = len(occurrences)
        avg_memory_per_file = total_memory / file_count if file_count > 0 else 0
        
        # Calculate average null percentage across files
        avg_null_percentage = sum(occ['null_percentage'] for occ in occurrences) / file_count if file_count > 0 else 0
        
        # Get most common dtype
        dtype_counts = {}
        for occ in occurrences:
            dtype = occ['dtype']
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        most_common_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0] if dtype_counts else 'unknown'
        
        column_totals[col_name] = {
            'total_memory_mb': total_memory,
            'total_rows': total_rows,
            'file_count': file_count,
            'avg_memory_per_file': avg_memory_per_file,
            'avg_null_percentage': avg_null_percentage,
            'most_common_dtype': most_common_dtype,
            'occurrences': occurrences
        }
    
    # Sort by total memory usage (descending)
    sorted_by_memory = sorted(column_totals.items(), 
                            key=lambda x: x[1]['total_memory_mb'], 
                            reverse=True)
    
    # Sort by average memory per file (descending)
    sorted_by_avg = sorted(column_totals.items(), 
                          key=lambda x: x[1]['avg_memory_per_file'], 
                          reverse=True)
    
    return sorted_by_memory, sorted_by_avg, column_totals

def print_rankings(sorted_by_memory, sorted_by_avg, top_n=20):
    """Print the rankings in a formatted way."""
    
    print("=" * 120)
    print("COLUMN SIZE RANKINGS (Real-time Analysis)")
    print("=" * 120)
    
    print(f"\n📊 TOP {top_n} COLUMNS BY TOTAL MEMORY USAGE:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Column Name':<35} {'Total MB':<12} {'Files':<6} {'Avg MB/File':<12} {'Type':<15} {'Null %':<8}")
    print("-" * 100)
    
    for i, (col_name, info) in enumerate(sorted_by_memory[:top_n], 1):
        print(f"{i:<4} {col_name:<35} {info['total_memory_mb']:<12.2f} {info['file_count']:<6} {info['avg_memory_per_file']:<12.2f} {info['most_common_dtype']:<15} {info['avg_null_percentage']:<8.1f}")
    
    print(f"\n📈 TOP {top_n} COLUMNS BY AVERAGE MEMORY PER FILE:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Column Name':<35} {'Avg MB/File':<12} {'Total MB':<12} {'Files':<6} {'Type':<15} {'Null %':<8}")
    print("-" * 100)
    
    for i, (col_name, info) in enumerate(sorted_by_avg[:top_n], 1):
        print(f"{i:<4} {col_name:<35} {info['avg_memory_per_file']:<12.2f} {info['total_memory_mb']:<12.2f} {info['file_count']:<6} {info['most_common_dtype']:<15} {info['avg_null_percentage']:<8.1f}")
    
    # Summary statistics
    print(f"\n📋 SUMMARY:")
    print(f"Total columns analyzed: {len(sorted_by_memory)}")
    print(f"Total memory across all columns: {sum(info['total_memory_mb'] for _, info in sorted_by_memory):.2f} MB")
    print(f"Total files processed: {sum(info['file_count'] for _, info in sorted_by_memory)}")
    print(f"Total rows across all files: {sum(info['total_rows'] for _, info in sorted_by_memory):,}")

def print_single_file_analysis(column_totals, top_n=15):
    """Print columns with highest memory usage in single files."""
    print(f"\n🔍 COLUMNS WITH HIGHEST SINGLE-FILE MEMORY USAGE:")
    print("-" * 100)
    print(f"{'Column Name':<35} {'File':<50} {'Memory MB':<12} {'Type':<15}")
    print("-" * 100)
    
    single_file_max = []
    for col_name, info in column_totals.items():
        for occ in info['occurrences']:
            single_file_max.append((col_name, occ['file_path'], occ['memory_mb'], occ['dtype']))
    
    single_file_max.sort(key=lambda x: x[2], reverse=True)
    for col_name, file_path, memory_mb, dtype in single_file_max[:top_n]:
        print(f"{col_name:<35} {os.path.basename(file_path):<50} {memory_mb:<12.2f} {dtype:<15}")

def print_file_summary(column_totals):
    """Print summary of files analyzed."""
    print(f"\n📁 FILES ANALYZED:")
    print("-" * 80)
    
    # Get unique files and their info
    file_info = {}
    for col_name, info in column_totals.items():
        for occ in info['occurrences']:
            file_path = occ['file_path']
            if file_path not in file_info:
                file_info[file_path] = {
                    'file_size_mb': occ['file_size_mb'],
                    'num_rows': occ['num_rows'],
                    'columns': set()
                }
            file_info[file_path]['columns'].add(col_name)
    
    # Sort by file size
    sorted_files = sorted(file_info.items(), key=lambda x: x[1]['file_size_mb'], reverse=True)
    
    print(f"{'File Name':<50} {'Size (MB)':<12} {'Rows':<10} {'Columns':<8}")
    print("-" * 80)
    
    for file_path, info in sorted_files:
        print(f"{os.path.basename(file_path):<50} {info['file_size_mb']:<12.2f} {info['num_rows']:<10,} {len(info['columns']):<8}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze column sizes in parquet files')
    parser.add_argument('--directories', nargs='+', 
                       default=['data/processed', 'data/gt', 'data/analysis'],
                       help='Directories to scan for parquet files')
    parser.add_argument('--include-modelid', action='store_true',
                       help='Include modelId columns in analysis (excluded by default)')
    parser.add_argument('--top-n', type=int, default=30,
                       help='Number of top columns to display')
    parser.add_argument('--single-file-top', type=int, default=15,
                       help='Number of top single-file columns to display')
    
    args = parser.parse_args()
    
    print("🔍 Analyzing column sizes from parquet files (real-time computation)...")
    print(f"Directories: {args.directories}")
    print(f"Include modelId: {args.include_modelid}")
    print()
    
    sorted_by_memory, sorted_by_avg, column_totals = analyze_column_sizes(
        directories=args.directories, 
        exclude_modelid=not args.include_modelid
    )
    
    if not sorted_by_memory:
        print("❌ No data to analyze. Check if parquet files exist in the specified directories.")
        return
    
    print_rankings(sorted_by_memory, sorted_by_avg, top_n=args.top_n)
    print_single_file_analysis(column_totals, top_n=args.single_file_top)
    print_file_summary(column_totals)
    
    print(f"\n✅ Analysis complete! Processed {len(column_totals)} unique columns.")

if __name__ == "__main__":
    main()
