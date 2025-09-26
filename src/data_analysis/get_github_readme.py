#!/usr/bin/env python3
"""
Get GitHub README paths from GitHub CSV paths

Date: 2025-09-25

This script takes GitHub CSV file paths as input and retrieves the corresponding GitHub README file paths.
It maps CSV files back to their source README files using the md_to_csv_mapping.json file.

Usage:
    python -m src.data_analysis.get_github_readme --csv-path "data/processed/deduped_github_csvs/abc123_table1.csv"
    python -m src.data_analysis.get_github_readme --input-file csv_paths.txt
    python -m src.data_analysis.get_github_readme --csv-path "abc123_table1.csv" --output results.json
"""

import argparse
import os
import sys
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Union

def load_csv_to_readme_mapping() -> Dict[str, str]:
    """
    Load CSV filename to README path mapping from md_to_csv_mapping.json.
    Returns a dictionary mapping CSV filenames to README file paths.
    """
    mapping_file = "data/processed/deduped_github_csvs/md_to_csv_mapping.json"
    
    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file not found: {mapping_file}")
        return {}
    
    print(f"Loading CSV to README mapping from {mapping_file}...")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        md_to_csv = json.load(f)
    
    # Reverse the mapping: CSV filename -> README path
    csv_to_readme = {}
    
    for readme_hash, csv_files in md_to_csv.items():
        if isinstance(csv_files, list):
            for csv_file in csv_files:
                csv_to_readme[csv_file] = f"data/downloaded_github_readmes/{readme_hash}.md"
        else:
            csv_to_readme[csv_files] = f"data/downloaded_github_readmes/{readme_hash}.md"
    
    print(f"Loaded {len(csv_to_readme)} CSV to README mappings")
    return csv_to_readme


def get_readme_path_from_csv(csv_path: str) -> Optional[Dict[str, str]]:
    """
    Get README path from a GitHub CSV file path.
    
    Args:
        csv_path: Path to the GitHub CSV file
        
    Returns:
        Dictionary with CSV path and corresponding README path, or None if not found
    """
    # Extract just the filename from the path
    csv_filename = os.path.basename(csv_path)
    
    # Load the mapping
    csv_to_readme = load_csv_to_readme_mapping()
    
    if not csv_to_readme:
        print("❌ No CSV to README mappings available")
        return None
    
    # Look up the README path
    readme_path = csv_to_readme.get(csv_filename)
    
    if not readme_path:
        print(f"❌ No README path found for CSV: {csv_filename}")
        return None
    
    # Check if the README file actually exists
    if not os.path.exists(readme_path):
        print(f"⚠️  README file not found: {readme_path}")
        return {
            'csv_path': csv_path,
            'csv_filename': csv_filename,
            'readme_path': readme_path,
            'readme_exists': False
        }
    
    print(f"✅ Found README path for {csv_filename}: {readme_path}")
    return {
        'csv_path': csv_path,
        'csv_filename': csv_filename,
        'readme_path': readme_path,
        'readme_exists': True
    }

def process_csv_input_file(input_file: str) -> List[Dict[str, str]]:
    """
    Process a file containing multiple CSV paths.
    
    Args:
        input_file: Path to input file containing CSV paths
        
    Returns:
        List of results
    """
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return []
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            print(f"\n📝 Processing line {line_num}: {line}")
            
            result = get_readme_path_from_csv(line)
            if result:
                results.append(result)
    
    return results


def save_results(results: List[Dict[str, str]], output_file: Optional[str] = None):
    """
    Save results to file or print to console.
    
    Args:
        results: List of result dictionaries
        output_file: Optional output file path
    """
    if not results:
        print("❌ No results to save")
        return
    
    if output_file:
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_file}")
    else:
        # Print to console
        for i, result in enumerate(results, 1):
            print(f"\n{'='*50}")
            print(f"Result {i}:")
            print(f"CSV Path: {result.get('csv_path', 'Unknown')}")
            print(f"CSV Filename: {result.get('csv_filename', 'Unknown')}")
            print(f"README Path: {result.get('readme_path', 'Unknown')}")
            print(f"README Exists: {result.get('readme_exists', False)}")
            print("-" * 30)

def main():
    parser = argparse.ArgumentParser(
        description="Get GitHub README paths from GitHub CSV paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get README path from single CSV file
  python -m src.data_analysis.get_github_readme --csv-path "data/processed/deduped_github_csvs/abc123_table1.csv"
  
  # Get README path from CSV filename only
  python -m src.data_analysis.get_github_readme --csv-path "abc123_table1.csv"
  
  # Process multiple CSV paths from file
  python -m src.data_analysis.get_github_readme --input-file csv_paths.txt --output results.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv-path', help='GitHub CSV file path or filename')
    input_group.add_argument('--input-file', help='File containing multiple CSV paths (one per line)')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    print("🚀 GitHub CSV to README Path Mapping Tool")
    print("=" * 50)
    
    results = []
    
    if args.csv_path:
        print(f"📁 Processing CSV path: {args.csv_path}")
        result = get_readme_path_from_csv(args.csv_path)
        if result:
            results = [result]
    
    elif args.input_file:
        print(f"📄 Processing input file: {args.input_file}")
        results = process_csv_input_file(args.input_file)
    
    # Save or display results
    if results:
        print(f"\n✅ Found {len(results)} README path(s)")
        save_results(results, args.output)
    else:
        print("\n❌ No README paths found")
        sys.exit(1)

if __name__ == "__main__":
    main()
