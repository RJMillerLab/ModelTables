"""
Author: Zhengyuan Dong
Date: 2025-05-31

This script is used to check if a csv pair is related in GT matrices.
Usage:
    python -m src.data_gt.check_pair_in_gt --gt-dir data/gt --csv1 csv1.csv --csv2 csv2.csv
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

LEVEL_NPZ = {
    "direct": "csv_pair_matrix_direct_label.npz",
    "max_pr": "csv_pair_matrix_max_pr.npz",
    "union": "csv_pair_union_direct_processed.npz",
    "model": "scilake_gt_modellink_model_adj_processed.npz",
    "dataset": "scilake_gt_modellink_dataset_adj_processed.npz",
}

LEVEL_CSVLIST = {
    "direct": "csv_list_direct_label.pkl",
    "max_pr": "csv_list_max_pr.pkl",
    "union": "csv_pair_union_direct_processed_csv_list.pkl",
    "model": "scilake_gt_modellink_model_adj_csv_list_processed.pkl",
    "dataset": "scilake_gt_modellink_dataset_adj_csv_list_processed.pkl",
}

def check_pair_fast(gt_dir, csv1, csv2):
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
            idx2 = csv_list.index(csv2)
        except ValueError:
            print(f"{level:8}: not found in csv_list")
            continue
        M = load_npz(npz_path)
        related = M[idx1, idx2] != 0 or M[idx2, idx1] != 0
        print(f"{level:8}: {'related' if related else 'not related'}")
        # 释放内存
        del M
        del csv_list
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if a csv pair is related in GT matrices.")
    parser.add_argument('--gt-dir', required=True, help="Directory containing the ground-truth .npz and .pkl files")
    parser.add_argument('--csv1', required=True, help="First CSV filename (with extension)")
    parser.add_argument('--csv2', required=True, help="Second CSV filename (with extension)")
    args = parser.parse_args()

    print(f"\nResults for pair: {args.csv1} <-> {args.csv2}")
    check_pair_fast(args.gt_dir, args.csv1, args.csv2) 