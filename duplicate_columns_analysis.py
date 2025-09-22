#!/usr/bin/env python3
"""
Analyze columns that are large AND appear in multiple files to identify duplicates.
Focus on columns that should be deduplicated first.
"""

import json
from collections import defaultdict

def analyze_duplicate_columns(json_file_path):
    """Analyze columns that appear in multiple files and are large."""
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store column information
    column_info = defaultdict(list)
    
    # Process each file
    for file_data in data:
        if 'error' in file_data:
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
    
    # Calculate statistics for each column
    column_stats = {}
    for col_name, occurrences in column_info.items():
        total_memory = sum(occ['memory_mb'] for occ in occurrences)
        total_rows = sum(occ['num_rows'] for occ in occurrences)
        file_count = len(occurrences)
        avg_memory_per_file = total_memory / file_count if file_count > 0 else 0
        max_memory_single_file = max(occ['memory_mb'] for occ in occurrences)
        min_memory_single_file = min(occ['memory_mb'] for occ in occurrences)
        
        # Calculate memory efficiency (how much memory per row)
        memory_per_row = total_memory / total_rows if total_rows > 0 else 0
        
        column_stats[col_name] = {
            'total_memory_mb': total_memory,
            'total_rows': total_rows,
            'file_count': file_count,
            'avg_memory_per_file': avg_memory_per_file,
            'max_memory_single_file': max_memory_single_file,
            'min_memory_single_file': min_memory_single_file,
            'memory_per_row': memory_per_row,
            'occurrences': occurrences
        }
    
    return column_stats

def find_duplicate_candidates(column_stats, min_files=2, min_total_memory=100):
    """Find columns that appear in multiple files and are large."""
    
    candidates = []
    
    for col_name, stats in column_stats.items():
        if stats['file_count'] >= min_files and stats['total_memory_mb'] >= min_total_memory:
            # Calculate duplication impact (total memory that could be saved)
            potential_savings = stats['total_memory_mb'] - stats['max_memory_single_file']
            savings_percentage = (potential_savings / stats['total_memory_mb']) * 100 if stats['total_memory_mb'] > 0 else 0
            
            candidates.append({
                'column_name': col_name,
                'file_count': stats['file_count'],
                'total_memory_mb': stats['total_memory_mb'],
                'avg_memory_per_file': stats['avg_memory_per_file'],
                'max_memory_single_file': stats['max_memory_single_file'],
                'potential_savings_mb': potential_savings,
                'savings_percentage': savings_percentage,
                'memory_per_row': stats['memory_per_row'],
                'files': [occ['file_path'] for occ in stats['occurrences']]
            })
    
    # Sort by potential savings (descending)
    candidates.sort(key=lambda x: x['potential_savings_mb'], reverse=True)
    
    return candidates

def print_duplicate_analysis(candidates, top_n=20):
    """Print the duplicate column analysis."""
    
    print("=" * 120)
    print("DUPLICATE COLUMN ANALYSIS - PRIORITY FOR DEDUPLICATION")
    print("=" * 120)
    
    print(f"\n🎯 TOP {top_n} COLUMNS TO DEDUPLICATE (by potential memory savings):")
    print("-" * 120)
    print(f"{'Rank':<4} {'Column Name':<35} {'Files':<6} {'Total MB':<12} {'Max MB':<10} {'Savings MB':<12} {'Savings %':<10} {'Files List'}")
    print("-" * 120)
    
    for i, candidate in enumerate(candidates[:top_n], 1):
        files_short = [f.split('/')[-1] for f in candidate['files']]
        files_str = ', '.join(files_short[:3]) + ('...' if len(files_short) > 3 else '')
        
        print(f"{i:<4} {candidate['column_name']:<35} {candidate['file_count']:<6} "
              f"{candidate['total_memory_mb']:<12.2f} {candidate['max_memory_single_file']:<10.2f} "
              f"{candidate['potential_savings_mb']:<12.2f} {candidate['savings_percentage']:<10.1f}% "
              f"{files_str}")
    
    # Group by file count for better understanding
    print(f"\n📊 COLUMNS BY FILE COUNT:")
    print("-" * 80)
    
    by_file_count = defaultdict(list)
    for candidate in candidates:
        by_file_count[candidate['file_count']].append(candidate)
    
    for file_count in sorted(by_file_count.keys(), reverse=True):
        cols = by_file_count[file_count]
        total_savings = sum(c['potential_savings_mb'] for c in cols)
        print(f"\nFiles: {file_count} | Columns: {len(cols)} | Total Potential Savings: {total_savings:.2f} MB")
        for col in cols[:5]:  # Show top 5 for each file count
            print(f"  - {col['column_name']:<30} {col['potential_savings_mb']:<12.2f} MB savings")
        if len(cols) > 5:
            print(f"  ... and {len(cols) - 5} more")

def analyze_specific_columns(column_stats, target_columns):
    """Analyze specific columns in detail."""
    
    print(f"\n🔍 DETAILED ANALYSIS OF TARGET COLUMNS:")
    print("-" * 100)
    
    for col_name in target_columns:
        if col_name in column_stats:
            stats = column_stats[col_name]
            print(f"\nColumn: {col_name}")
            print(f"  Total Memory: {stats['total_memory_mb']:.2f} MB")
            print(f"  Files: {stats['file_count']}")
            print(f"  Avg per file: {stats['avg_memory_per_file']:.2f} MB")
            print(f"  Max single file: {stats['max_memory_single_file']:.2f} MB")
            print(f"  Memory per row: {stats['memory_per_row']:.6f} MB")
            print(f"  Files:")
            for occ in stats['occurrences']:
                print(f"    - {occ['file_path']}: {occ['memory_mb']:.2f} MB ({occ['num_rows']} rows)")

def main():
    json_file_path = '/Users/doradong/Repo/CitationLake/column_analysis_results.json'
    
    print("Analyzing duplicate columns for deduplication priority...")
    column_stats = analyze_duplicate_columns(json_file_path)
    
    # Find candidates (appear in 2+ files, total memory > 100MB)
    candidates = find_duplicate_candidates(column_stats, min_files=2, min_total_memory=100)
    
    print_duplicate_analysis(candidates, top_n=25)
    
    # Analyze the most promising candidates in detail
    top_candidates = [c['column_name'] for c in candidates[:10]]
    analyze_specific_columns(column_stats, top_candidates)
    
    # Summary
    total_potential_savings = sum(c['potential_savings_mb'] for c in candidates)
    print(f"\n💰 TOTAL POTENTIAL MEMORY SAVINGS: {total_potential_savings:.2f} MB")
    print(f"📈 COLUMNS WITH DUPLICATES: {len(candidates)}")

if __name__ == "__main__":
    main()
