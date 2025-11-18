#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Created: 2025-05-28
Description: Create *_s / *_t / *_s_t variants for CSV filenames in a CSV list.

Usage:
    python -m src.data_gt.create_csvlist_variants --level direct
    or
    python -m src.data_gt.create_csvlist_variants --csvlist data/gt/csv_list_direct_label.pkl
"""

import os
import argparse
import pickle
from pathlib import Path

DATA_DIR = "data/gt"

# Mapping of level to CSV list file
LEVEL_CSVLIST = {
    "direct": "csv_list_direct_label.pkl",
    "direct_influential": "csv_list_direct_label_influential.pkl",
    "direct_methodology_or_result": "csv_list_direct_label_methodology_or_result.pkl",
    "direct_methodology_or_result_influential": "csv_list_direct_label_methodology_or_result_influential.pkl",
    "max_pr": "csv_list_max_pr.pkl",
    "max_pr_influential": "csv_list_max_pr_influential.pkl",
    "max_pr_methodology_or_result": "csv_list_max_pr_methodology_or_result.pkl",
    "max_pr_methodology_or_result_influential": "csv_list_max_pr_methodology_or_result_influential.pkl",
    "model": "scilake_gt_modellink_model_adj_csv_list_processed.pkl",
    "dataset": "scilake_gt_modellink_dataset_adj_csv_list_processed.pkl",
    "union": "csv_pair_union_direct_processed_csv_list.pkl",
}

SUFFIXES = {
    #"":      "",
    "_s":    "_s",
    "_t":    "_t",
    #"_s_t":  "_s_t",
}

def add_suffix_to_filename(filename, suffix):
    """
    Add suffix before the file extension.
    Example: 'table1.csv' + '_s' -> 'table1_s.csv'
    """
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix}"

def process_csvlist(csvlist_path):
    """Load CSV list and create variants with different suffixes."""
    print(f"Loading CSV list from: {csvlist_path}")
    with open(csvlist_path, 'rb') as f:
        base_list = pickle.load(f)
    print(f"Loaded {len(base_list)} CSV filenames")

    # Create variants for each suffix
    base_path = os.path.splitext(csvlist_path)[0]
    for tag, suf in SUFFIXES.items():
        if not tag:  # Skip empty suffix
            continue
            
        # Create new list with suffixed filenames
        suffixed_list = [add_suffix_to_filename(fname, suf) for fname in base_list]
        
        # Save the new list
        output_path = f"{base_path}{tag}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(suffixed_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Saved {len(suffixed_list)} suffixed filenames to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create variants of CSV filenames in a CSV list")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--level', choices=list(LEVEL_CSVLIST.keys()),
                      help='Which level to process (e.g., direct, max_pr)')
    group.add_argument('--csvlist', help='Path to the CSV list file to process')
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode for GT files.')
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    if args.level:
        # For processed files, add suffix before _processed if it exists, otherwise before .pkl
        base_name = LEVEL_CSVLIST[args.level]
        if '_processed' in base_name:
            csvlist_path = os.path.join(DATA_DIR, base_name.replace('_processed', f'{suffix}_processed'))
        else:
            csvlist_path = os.path.join(DATA_DIR, base_name.replace('.pkl', f'{suffix}.pkl'))
    else:
        csvlist_path = args.csvlist
        # If csvlist is provided and tag is set, add suffix
        if args.tag and csvlist_path.endswith('.pkl'):
            if '_processed' in csvlist_path:
                csvlist_path = csvlist_path.replace('_processed', f'{suffix}_processed')
            else:
                csvlist_path = csvlist_path.replace('.pkl', f'{suffix}.pkl')

    process_csvlist(csvlist_path)

if __name__ == "__main__":
    main() 