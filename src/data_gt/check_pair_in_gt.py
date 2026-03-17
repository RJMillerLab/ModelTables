"""
Author: Zhengyuan Dong
Date: 2025-05-31

This script is used to check if a csv pair is related in GT matrices.
Usage:
    python -m src.data_gt.check_pair_in_gt --gt-dir data/gt --csv1 1810.04805_table1.csv --csv2 1908.04577_table4.csv 2001.09694_table12.csv 2410.06581_table1.csv 2307.06942_table3.csv 2209.06638_table11.csv 2405.18406v3_table8.csv 237485280_table3.csv 2205.12644_table10.csv 2010.12148_table10.csv 
"""
import os
import argparse
import pickle
import numpy as np
from scipy.sparse import load_npz
import gc

LEVELS = [
    "direct",
    "max_pr",
    "union",
    "model",
    "dataset",
]

from src.data_gt.debug_npz import get_npz_path

def check_pair_fast(gt_dir, csv1, csv2_list):
    for level in LEVELS:
        npz_path = os.path.join(gt_dir, LEVEL_NPZ[level])
        csvlist_path = os.path.join(gt_dir, LEVEL_CSVLIST[level])
        if not (os.path.isfile(npz_path) and os.path.isfile(csvlist_path)):
            print(f"{level:8}: missing file")
            continue
        with open(csvlist_path, 'rb') as f:
            csv_list = pickle.load(f)
        try:
            idx1 = csv_list.index(csv1)
        except ValueError:
            print(f"{level:8}: {csv1} not found in csv_list")
            continue
        M = load_npz(npz_path)
        for csv2 in csv2_list:
            try:
                idx2 = csv_list.index(csv2)
            except ValueError:
                print(f"{level:8}: {csv2} not found in csv_list")
                continue
            related = M[idx1, idx2] != 0 or M[idx2, idx1] != 0
            print(f"{level:8}: {csv1} <-> {csv2}: {'related' if related else 'not related'}")
        del M
        del csv_list
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if a csv pair is related in GT matrices.")
    parser.add_argument('--gt-dir', required=True, help="Directory containing the ground-truth .npz and .pkl files")
    parser.add_argument('--csv1', required=True, help="First CSV filename (with extension)")
    parser.add_argument('--csv2', required=True, nargs='+', help="Second CSV filename(s) (with extension)")
    parser.add_argument('--v2_mode', action='store_true', help="Use v2 mode.")
    parser.add_argument('--tag', default=None, help="Tag suffix for versioning (e.g., 251117). Enables versioning mode for GT files.")
    args = parser.parse_args()

    v2_suffix = "_v2" if args.v2_mode else ""
    suffix = f"_{args.tag}" if args.tag else ""

    LEVEL_NPZ, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix)

    print(f"\nResults for {args.csv1} <-> {args.csv2}")
    check_pair_fast(args.gt_dir, args.csv1, args.csv2) 