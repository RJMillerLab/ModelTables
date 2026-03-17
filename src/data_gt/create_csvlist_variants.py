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
from src.data_gt.debug_npz import get_npz_path
# Mapping of level to CSV list file

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
    group.add_argument('--level', choices=['direct', 'max_pr', 'model', 'dataset', 'direct_influential', 'direct_methodology_or_result', 'direct_methodology_or_result_influential', 'max_pr_influential', 'max_pr_methodology_or_result', 'max_pr_methodology_or_result_influential', 'union'], required=True, help='Which level to process (e.g., direct, max_pr)')
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode for GT files.')
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    NPZ_PATH, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix)
    process_csvlist(os.path.join("data", "gt", LEVEL_CSVLIST[args.level]))

if __name__ == "__main__":
    main() 