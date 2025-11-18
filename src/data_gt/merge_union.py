#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-05-28
Last Edited: 2025-05-28
Description: Merge multiple boolean adjacency matrices into a single union matrix,
             with progress tracking and support for different citation pair levels.
             Also handles CSV list files for each input matrix.
Usage:
    python -m src.data_gt.merge_union --level direct
"""
import os
import argparse
import pickle
import numpy as np
from scipy.sparse import load_npz, csr_matrix, save_npz

DATA_DIR = "data/gt"

LEVEL_NPZ = {                                                ########
    "direct"  : "csv_pair_matrix_direct_label.npz",
    "direct_influential": "csv_pair_matrix_direct_label_influential.npz",
    "direct_methodology_or_result": "csv_pair_matrix_direct_label_methodology_or_result.npz",
    "direct_methodology_or_result_influential": "csv_pair_matrix_direct_label_methodology_or_result_influential.npz",
    "max_pr": "csv_pair_matrix_max_pr.npz",
    "max_pr_influential": "csv_pair_matrix_max_pr_influential.npz",
    "max_pr_methodology_or_result": "csv_pair_matrix_max_pr_methodology_or_result.npz",
    "max_pr_methodology_or_result_influential": "csv_pair_matrix_max_pr_methodology_or_result_influential.npz",
    #"union" : "csv_pair_union.npz",
    "model" : "scilake_gt_modellink_model_adj.npz",
    "dataset": "scilake_gt_modellink_dataset_adj.npz",
}

LEVEL_CSVLIST = {                                            ########
    "direct"  : "csv_list_direct_label.pkl",
    "direct_influential": "csv_list_direct_label_influential.pkl",
    "direct_methodology_or_result": "csv_list_direct_label_methodology_or_result.pkl",
    "direct_methodology_or_result_influential": "csv_list_direct_label_methodology_or_result_influential.pkl",
    "max_pr": "csv_list_max_pr.pkl",
    "max_pr_influential": "csv_list_max_pr_influential.pkl",
    "max_pr_methodology_or_result": "csv_list_max_pr_methodology_or_result.pkl",
    "max_pr_methodology_or_result_influential": "csv_list_max_pr_methodology_or_result_influential.pkl",
    #"union" : "csv_pair_union_csv_list.pkl",
    "model" : "scilake_gt_modellink_model_adj_csv_list.pkl",
    "dataset": "scilake_gt_modellink_dataset_adj_csv_list.pkl",
}

def _full(p):
    """Return DATA_DIR/p (only if p is not an absolute path)."""
    return p if os.path.isabs(p) else os.path.join(DATA_DIR, p)

def load_csv_list(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_bool_matrix(path):
    M = load_npz(path)
    M = M.astype(bool).tocsr()
    print(f"Loaded matrix '{path}' → shape={M.shape}, nnz={M.nnz}, dtype={M.dtype}")
    return M

def infer_list_path(npz_path):
    base = os.path.splitext(npz_path)[0]
    candidate = base + '_csv_list.pkl'
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Could not infer list file for {npz_path}")

def build_union_matrix(matrix_paths, list_paths=None):
    if list_paths is None:
        list_paths = [p.replace('.npz', '_csv_list.pkl') for p in matrix_paths]

    # Load ID lists
    print("Loading CSV lists…")
    lists = []
    for p in list_paths:
        lst = load_csv_list(p)
        print(f"Loaded list '{p}' → {len(lst)} ids")
        lists.append(lst)

    # Load matrices
    print("Loading boolean matrices…")
    mats = [load_bool_matrix(p) for p in matrix_paths]

    # Pre‑trim each input matrix & its ID list to drop all‑zero rows/cols
    trimmed_lists, trimmed_mats = [], []
    for lst, M in zip(lists, mats):
        row_sums = np.array(M.sum(axis=1)).ravel()
        keep_idx = np.where(row_sums > 0)[0]
        print(f"[INFO] Trimming {len(lst) - keep_idx.size} zero‑row IDs")
        M = M[keep_idx][:, keep_idx]
        lst = [lst[i] for i in keep_idx]
        trimmed_mats.append(M)
        trimmed_lists.append(lst)
    mats, lists = trimmed_mats, trimmed_lists

    # Build global union of IDs (order preserved)
    seen, union_ids = set(), []
    for lst in lists:
        for id_ in lst:
            if id_ not in seen:
                seen.add(id_)
                union_ids.append(id_)
    n = len(union_ids)
    print(f"Union of IDs → {n} total entries")
    id_to_idx = {id_: i for i, id_ in enumerate(union_ids)}

    # Merge matrices
    U = csr_matrix((n, n), dtype=bool)
    for idx, (lst, M, path) in enumerate(zip(lists, mats, matrix_paths), 1):
        print(f"[{idx}/{len(mats)}] Merging '{path}'…")
        old_to_union = np.array([id_to_idx[id_] for id_ in lst], dtype=int)
        coo = M.tocoo()
        rows, cols = old_to_union[coo.row], old_to_union[coo.col]
        chunk = csr_matrix((np.ones_like(coo.data, bool), (rows, cols)), shape=(n, n))
        U += chunk
        print(f"    chunk nnz={chunk.nnz}, cumulative nnz={U.nnz}")

    U.data = U.data.astype(bool)
    U.eliminate_zeros()
    print(f"Final union matrix nnz={U.nnz}")
    return U, union_ids

def process_matrix_by_paper_list(matrix_path, list_path, paper_list):
    """Process a matrix to only keep rows/cols that exist in paper_list and remove all-zero rows/cols."""
    # Load matrix and its list
    M = load_bool_matrix(matrix_path)
    with open(list_path, 'rb') as f:
        matrix_list = pickle.load(f)
    
    print(f"\nProcessing {os.path.basename(matrix_path)}:")
    print(f"  Original matrix shape: {M.shape}, nnz: {M.nnz}")
    print(f"  Original list length: {len(matrix_list)}")
    
    # Create mapping from matrix list to paper list
    paper_set = set(paper_list)
    keep_indices = [i for i, id_ in enumerate(matrix_list) if id_ in paper_set]
    
    if not keep_indices:
        raise ValueError(f"No matching IDs found between {list_path} and paper list")
    
    # Trim matrix to only keep matching rows/cols
    M = M[keep_indices][:, keep_indices]
    new_list = [matrix_list[i] for i in keep_indices]
    
    print(f"  After paper list filtering:")
    print(f"    Matrix shape: {M.shape}, nnz: {M.nnz}")
    print(f"    List length: {len(new_list)}")
    print(f"    Removed {len(matrix_list) - len(new_list)} entries not in paper list")
    
    # Remove all-zero rows/cols
    row_sums = np.array(M.sum(axis=1)).ravel()
    col_sums = np.array(M.sum(axis=0)).ravel()
    keep_idx = np.where((row_sums > 0) & (col_sums > 0))[0]
    
    M = M[keep_idx][:, keep_idx]
    new_list = [new_list[i] for i in keep_idx]
    
    print(f"  After removing zero rows/cols:")
    print(f"    Matrix shape: {M.shape}, nnz: {M.nnz}")
    print(f"    List length: {len(new_list)}")
    print(f"    Removed {len(new_list) - len(keep_idx)} zero rows/cols")
    
    return M, new_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute union of multiple boolean adjacency NPZs with progress")
    parser.add_argument('--level', required=True, choices=list(LEVEL_NPZ.keys()), help='Which citation‑pair level to use as PRIMARY (e.g. direct, max_pr_influential, union …)')
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode for GT files.')
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    primary_key = args.level
    primary_npz = _full(LEVEL_NPZ[primary_key].replace('.npz', f'{suffix}.npz'))
    primary_lst = _full(LEVEL_CSVLIST[primary_key].replace('.pkl', f'{suffix}.pkl'))

    # Load paper list first
    with open(primary_lst, 'rb') as f:
        paper_list = pickle.load(f)
    print(f"Loaded paper list with {len(paper_list)} entries")

    # Process model and dataset matrices
    model_npz = _full(LEVEL_NPZ['model'].replace('.npz', f'{suffix}.npz'))
    model_lst = _full(LEVEL_CSVLIST['model'].replace('.pkl', f'{suffix}.pkl'))
    ds_npz = _full(LEVEL_NPZ['dataset'].replace('.npz', f'{suffix}.npz'))
    ds_lst = _full(LEVEL_CSVLIST['dataset'].replace('.pkl', f'{suffix}.pkl'))

    # Process model matrix
    print("Processing model matrix...")
    model_matrix, model_list = process_matrix_by_paper_list(model_npz, model_lst, paper_list)
    model_processed_npz = _full(LEVEL_NPZ['model'].replace('.npz', f'{suffix}_processed.npz'))
    model_processed_lst = _full(LEVEL_CSVLIST['model'].replace('.pkl', f'{suffix}_processed.pkl'))
    save_npz(model_processed_npz, model_matrix, compressed=True)
    with open(model_processed_lst, 'wb') as f:
        pickle.dump(model_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved processed model matrix → {model_processed_npz}")
    print(f"Saved processed model list → {model_processed_lst}")

    # Process dataset matrix
    print("Processing dataset matrix...")
    ds_matrix, ds_list = process_matrix_by_paper_list(ds_npz, ds_lst, paper_list)
    ds_processed_npz = _full(LEVEL_NPZ['dataset'].replace('.npz', f'{suffix}_processed.npz'))
    ds_processed_lst = _full(LEVEL_CSVLIST['dataset'].replace('.pkl', f'{suffix}_processed.pkl'))
    save_npz(ds_processed_npz, ds_matrix, compressed=True)
    with open(ds_processed_lst, 'wb') as f:
        pickle.dump(ds_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved processed dataset matrix → {ds_processed_npz}")
    print(f"Saved processed dataset list → {ds_processed_lst}")

    # Build union matrix with processed files
    matrices = [primary_npz, model_processed_npz, ds_processed_npz]
    csvlists = [primary_lst, model_processed_lst, ds_processed_lst]

    # Output prefix reflects primary level
    output_prefix = _full(f"csv_pair_union_{primary_key}_processed")

    U, union_ids = build_union_matrix(matrices, csvlists)

    save_npz(output_prefix + '.npz', U, compressed=True)
    with open(output_prefix + '_csv_list.pkl', 'wb') as f:
        pickle.dump(union_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✔️  Saved union matrix → {output_prefix}.npz  (shape={U.shape}, nnz={U.nnz})")
    print(f"✔️  Saved union list   → {output_prefix}_csv_list.pkl  ({len(union_ids)} entries)")