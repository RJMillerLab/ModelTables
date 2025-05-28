#!/usr/bin/env python3
"""
test_modelcard_update.py

Compare model or dataset link adjacency matrices after:
1. Trimming all-zero rows/cols from original .npz;
2. Aligning both matrices via their corresponding CSV-list;
3. Checking value equality after reordering.

Usage:
    python -m src.data_gt.test_modelcard_update --mode dataset
    python -m src.data_gt.test_modelcard_update --mode model
"""
import argparse
import os
import pickle
import numpy as np
from scipy.sparse import load_npz

def main():
    parser = argparse.ArgumentParser(description="Compare aligned model/dataset .npz matrices with CSV-lists")
    parser.add_argument("--mode", choices=["model", "dataset"], required=True, help="Choose mode: 'model' or 'dataset'")
    args = parser.parse_args()

    GT_DIR = "data/gt"  ########

    # Auto-resolve paths based on mode
    base = f"scilake_gt_modellink_{args.mode}"
    matrix1_path = os.path.join(GT_DIR, f"{base}.npz")                  ########
    matrix2_path = os.path.join(GT_DIR, f"{base}_adj.npz")              ########
    csv1_path    = os.path.join(GT_DIR, f"{base}_csv_list.pkl")         ########
    csv2_path    = os.path.join(GT_DIR, f"{base}_adj_csv_list.pkl")     ########

    # Load matrices
    A = load_npz(matrix1_path).tocsr()
    B = load_npz(matrix2_path).tocsr()

    print(f"[INFO] Matrix 1: {matrix1_path} | shape = {A.shape}, nnz = {A.nnz}")
    print(f"[INFO] Matrix 2: {matrix2_path} | shape = {B.shape}, nnz = {B.nnz}")

    # Trim zero rows/cols from A
    row_sums = np.array(A.sum(axis=1)).ravel()
    keep_idx = np.where(row_sums > 0)[0]
    print(f"[INFO] Dropping {A.shape[0] - keep_idx.size} all-zero rows from Matrix 1")
    A = A[keep_idx][:, keep_idx]

    # Load and trim CSV-lists
    with open(csv1_path, "rb") as f1, open(csv2_path, "rb") as f2:
        list1_full = pickle.load(f1)
        list1_full = [os.path.basename(p) for p in list1_full]
        list2_full = pickle.load(f2)
        list2_full = [os.path.basename(p) for p in list2_full]
    list1_trimmed = [list1_full[i] for i in keep_idx]

    # Intersect by common entries
    set1, set2 = set(list1_trimmed), set(list2_full)
    common = sorted(set1 & set2, key=list2_full.index)
    if len(common) == 0:
        print("❌ No common entries between trimmed CSV-list 1 and CSV-list 2.")
        print(f"  len(list1_trimmed) = {len(list1_trimmed)}")
        print(f"  len(list2_full) = {len(list2_full)}")
        print(list1_trimmed[:10])
        print(list2_full[:10])
        return
    print(f"[INFO] {len(common)} common items found.")

    if missing := sorted(set(list2_full) - set1):
        print(f"⚠️  {len(missing)} entries in Matrix 2 not found after trimming Matrix 1")
    if missing := sorted(set(list1_trimmed) - set2):
        print(f"⚠️  {len(missing)} entries in trimmed Matrix 1 not found in Matrix 2")

    # Index remapping
    idx_map_A = {v: i for i, v in enumerate(list1_trimmed)}
    idx_map_B = {v: i for i, v in enumerate(list2_full)}
    perm_A = [idx_map_A[v] for v in common]
    perm_B = [idx_map_B[v] for v in common]

    # Subset and align both matrices
    A_sub = A[perm_A][:, perm_A]
    B_sub = B[perm_B][:, perm_B]
    print(f"[INFO] Aligned shapes → A: {A_sub.shape}, B: {B_sub.shape}")

    # Final equality check
    diff = A_sub - B_sub
    if diff.nnz == 0:
        print("✅ Matrices are equal after alignment.")
    else:
        print(f"❌ Matrices differ at {diff.nnz} entries.")
        diff_coo = diff.tocoo()
        for r, c, v in zip(diff_coo.row[:10], diff_coo.col[:10], diff_coo.data[:10]):
            print(f"    ({r}, {c}) → {v}")

if __name__ == "__main__":
    main()
