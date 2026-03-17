"""
Author: Zhengyuan Dong
Date: 2025-05-28

This script is used to debug the .npz files and their corresponding CSV lists.
Usage:
    python -m src.data_gt.debug_npz --gt-dir data/gt
"""
import os
import argparse
try:
    import pickle5 as pickle
except ImportError:
    import pickle
import numpy as np
from scipy.sparse import load_npz

def get_npz_path(v2_suffix, suffix):
    LEVEL_NPZ = {
        "direct": f"csv_pair_matrix_direct_label{v2_suffix}{suffix}.npz",
        "direct_influential": f"csv_pair_matrix_direct_label_influential{v2_suffix}{suffix}.npz",
        "direct_methodology_or_result": f"csv_pair_matrix_direct_label_methodology_or_result{v2_suffix}{suffix}.npz",
        "direct_methodology_or_result_influential": f"csv_pair_matrix_direct_label_methodology_or_result_influential{v2_suffix}{suffix}.npz",
        "max_pr": f"csv_pair_matrix_max_pr{v2_suffix}{suffix}.npz",
        "max_pr_influential": f"csv_pair_matrix_max_pr_influential{v2_suffix}{suffix}.npz",
        "max_pr_methodology_or_result": f"csv_pair_matrix_max_pr_methodology_or_result{v2_suffix}{suffix}.npz",
        "max_pr_methodology_or_result_influential": f"csv_pair_matrix_max_pr_methodology_or_result_influential{v2_suffix}{suffix}.npz",
        "union": f"csv_pair_union_direct_processed{v2_suffix}{suffix}.npz",
        "model": f"scilake_gt_modellink_model_adj_processed{v2_suffix}{suffix}.npz",
        "dataset": f"scilake_gt_modellink_dataset_adj_processed{v2_suffix}{suffix}.npz",
    }

    # Mapping of level names to CSV list pickle filenames
    CANONICAL_CSVLIST = f"csv_list{v2_suffix}{suffix}.pkl"
    LEVEL_CSVLIST = {
        "direct": f"{CANONICAL_CSVLIST}",
        "direct_influential": f"{CANONICAL_CSVLIST}",
        "direct_methodology_or_result": f"{CANONICAL_CSVLIST}",
        "direct_methodology_or_result_influential": f"{CANONICAL_CSVLIST}",
        "max_pr": f"{CANONICAL_CSVLIST}",
        "max_pr_influential": f"{CANONICAL_CSVLIST}",
        "max_pr_methodology_or_result": f"{CANONICAL_CSVLIST}",
        "max_pr_methodology_or_result_influential": f"{CANONICAL_CSVLIST}",
        "union": f"{CANONICAL_CSVLIST}",
        "model": f"scilake_gt_modellink_model_adj_csv_list{v2_suffix}{suffix}_processed.pkl",
        "dataset": f"scilake_gt_modellink_dataset_adj_csv_list{v2_suffix}{suffix}_processed.pkl",
    }
    return LEVEL_NPZ, LEVEL_CSVLIST

def inspect_npz(matrix_path, csvlist_path, row_idx):
    print(f"\n=== Inspecting matrix: {os.path.basename(matrix_path)} ===")
    M = load_npz(matrix_path).tocsr()
    print(f"  shape: {M.shape}, nnz: {M.nnz}, dtype: {M.dtype}")

    # Check whether all diagonal entries are zero
    diag = M.diagonal()
    if np.any(diag):
        nz = np.where(diag)[0]
        print(f"  WARNING: Non-zero diagonal entries at positions: {nz.tolist()}")
    else:
        print("✅ OK: All diagonal entries are zero")

    # Check whether the matrix is symmetric
    diff = M - M.T
    if diff.nnz == 0:
        print("✅ OK: Matrix is symmetric")
    else:
        diff_coo = diff.tocoo()
        print(f"⚠️ WARNING: Matrix is not symmetric; number of asymmetric entries: {diff_coo.nnz}")
        sample = list(zip(diff_coo.row[:5], diff_coo.col[:5], diff_coo.data[:5]))
        print(f"    Sample asymmetries (row, col, M[i,j]-M[j,i]): {sample}")
    
    # Check for any row fully connected to all others (excluding self)
    n = M.shape[0]
    row_counts = np.diff(M.indptr)
    fully_connected = np.where(row_counts == n - 1)[0]
    if fully_connected.size > 0:
        print(f"⚠️ WARNING: Rows fully connected to all others: {fully_connected.tolist()}")
    else:
        print("✅ OK: No row is fully connected to all others")

    # Inspect a single row
    if 0 <= row_idx < M.shape[0]:
        start, end = M.indptr[row_idx], M.indptr[row_idx + 1]
        cols, vals = M.indices[start:end], M.data[start:end]
        print(f"  Row {row_idx} nnz: {end-start}, cols[:5]={cols[:5].tolist()}, vals[:5]={vals[:5].tolist()}")
    else:
        print(f"⚠️ WARNING: Row index {row_idx} is out of bounds")

    # Load and sample the CSV list
    with open(csvlist_path, "rb") as f:
        csv_list = pickle.load(f)
    print(f"  CSV list length: {len(csv_list)}, sample entries: {csv_list[:3]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inspect multiple CSR matrices (.npz) and their CSV lists")
    parser.add_argument("--gt-dir", required=True, help="Directory containing the ground-truth .npz and .pkl files")
    parser.add_argument("--row", type=int, default=0, help="Row index to inspect in each matrix (0-based)")
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 mode.")
    parser.add_argument("--tag", default=None, help="Tag suffix for versioning (e.g., 251117). Enables versioning mode for GT files.")
    args = parser.parse_args()

    v2_suffix = "_v2" if args.v2_mode else ""
    suffix = f"_{args.tag}" if args.tag else ""

    # Mapping of level names to NPZ filenames
    LEVEL_NPZ, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix)
    
    for level, npz_name in LEVEL_NPZ.items():
        npz_path = os.path.join(args.gt_dir, npz_name)
        csvlist_name = LEVEL_CSVLIST[level]
        csvlist_path = os.path.join(args.gt_dir, csvlist_name)

        if os.path.isfile(npz_path) and os.path.isfile(csvlist_path):
            inspect_npz(npz_path, csvlist_path, args.row)
        else:
            print(f"\nSkipping '{level}': missing file {npz_name} or {csvlist_name}")
