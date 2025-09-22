#!/usr/bin/env python3
"""
Analyze column sizes from parquet files and rank by memory usage.
Excludes modelId columns as they are main keys.
"""

import json
from collections import defaultdict

def analyze_column_sizes(json_file_path):
    """Analyze column sizes and create rankings."""
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store column information
    column_info = defaultdict(list)
    
    # Process each file
    for file_data in data:
        if 'error' in file_data:
            print(f"Skipping file with error: {file_data['file_path']}")
            continue
            
        file_path = file_data['file_path']
        file_size_mb = file_data.get('file_size_mb', 0)
        num_rows = file_data.get('num_rows', 0)
        
        if 'columns' not in file_data:
            continue
            
        for col_name, col_data in file_data['columns'].items():
            # Skip modelId columns as requested
            if col_name.lower() == 'modelid':
                continue
                
            memory_mb = col_data.get('memory_mb', 0)
            dtype = col_data.get('dtype', 'unknown')
            null_count = col_data.get('null_count', 0)
            null_percentage = col_data.get('null_percentage', 0)
            
            column_info[col_name].append({
                'file_path': file_path,
                'memory_mb': memory_mb,
                'dtype': dtype,
                'null_count': null_count,
                'null_percentage': null_percentage,
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
        
        column_totals[col_name] = {
            'total_memory_mb': total_memory,
            'total_rows': total_rows,
            'file_count': file_count,
            'avg_memory_per_file': avg_memory_per_file,
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
    
    print("=" * 100)
    print("COLUMN SIZE RANKINGS (Excluding modelId columns)")
    print("=" * 100)
    
    print(f"\n📊 TOP {top_n} COLUMNS BY TOTAL MEMORY USAGE:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Column Name':<35} {'Total MB':<12} {'Files':<6} {'Avg MB/File':<12}")
    print("-" * 80)
    
    for i, (col_name, info) in enumerate(sorted_by_memory[:top_n], 1):
        print(f"{i:<4} {col_name:<35} {info['total_memory_mb']:<12.2f} {info['file_count']:<6} {info['avg_memory_per_file']:<12.2f}")
    
    print(f"\n📈 TOP {top_n} COLUMNS BY AVERAGE MEMORY PER FILE:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Column Name':<35} {'Avg MB/File':<12} {'Total MB':<12} {'Files':<6}")
    print("-" * 80)
    
    for i, (col_name, info) in enumerate(sorted_by_avg[:top_n], 1):
        print(f"{i:<4} {col_name:<35} {info['avg_memory_per_file']:<12.2f} {info['total_memory_mb']:<12.2f} {info['file_count']:<6}")
    
    # Summary statistics
    total_memory = sum(info['total_memory_mb'] for info in sorted_by_memory[0][1].values() if isinstance(info, dict) and 'total_memory_mb' in info)
    print(f"\n📋 SUMMARY:")
    print(f"Total columns analyzed: {len(sorted_by_memory)}")
    print(f"Total memory across all columns: {sum(info['total_memory_mb'] for _, info in sorted_by_memory):.2f} MB")

def main():
    json_file_path = '/Users/doradong/Repo/CitationLake/column_analysis_results.json'
    
    print("Analyzing column sizes from parquet files...")
    sorted_by_memory, sorted_by_avg, column_totals = analyze_column_sizes(json_file_path)
    
    print_rankings(sorted_by_memory, sorted_by_avg, top_n=30)
    
    # Additional analysis: columns with highest memory in single files
    print(f"\n🔍 COLUMNS WITH HIGHEST SINGLE-FILE MEMORY USAGE:")
    print("-" * 80)
    print(f"{'Column Name':<35} {'File':<50} {'Memory MB':<12}")
    print("-" * 80)
    
    single_file_max = []
    for col_name, info in column_totals.items():
        for occ in info['occurrences']:
            single_file_max.append((col_name, occ['file_path'], occ['memory_mb']))
    
    single_file_max.sort(key=lambda x: x[2], reverse=True)
    for col_name, file_path, memory_mb in single_file_max[:15]:
        print(f"{col_name:<35} {file_path:<50} {memory_mb:<12.2f}")

if __name__ == "__main__":
    main()
