#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
from scipy.sparse import load_npz, csr_matrix, save_npz

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
    # If list_paths not provided, infer them
    if list_paths is None:
        list_paths = [infer_list_path(p) for p in matrix_paths]

    # Load ID lists
    print("Loading CSV lists...")
    lists = []
    for p in list_paths:
        lst = load_csv_list(p)
        print(f"Loaded list '{p}' → {len(lst)} ids")
        lists.append(lst)

    # Load matrices
    print("Loading boolean matrices...")
    mats = [load_bool_matrix(p) for p in matrix_paths]

    # Build global union of IDs
    union_ids = sorted({id_ for lst in lists for id_ in lst})
    n = len(union_ids)
    print(f"Union of IDs → {n} total entries")
    id_to_idx = {id_: i for i, id_ in enumerate(union_ids)}

    # Initialize empty boolean CSR
    U = csr_matrix((n, n), dtype=bool)

    # Merge each source matrix
    total = len(mats)
    for idx, (lst, M, path) in enumerate(zip(lists, mats, matrix_paths), 1):
        print(f"[{idx}/{total}] Merging matrix '{path}' → remapping indices...")
        old_to_union = np.array([id_to_idx[id_] for id_ in lst], dtype=int)
        coo = M.tocoo()
        rows = old_to_union[coo.row]
        cols = old_to_union[coo.col]
        data = np.ones_like(coo.data, dtype=bool)
        chunk = csr_matrix((data, (rows, cols)), shape=(n, n))
        U = U + chunk
        print(f"  → chunk nnz={chunk.nnz}, cumulative nnz={U.nnz}")

    # Finalize
    U.data = U.data.astype(bool)
    U.eliminate_zeros()
    print(f"Final union matrix nnz={U.nnz}")
    return U, union_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute union of multiple boolean adjacency NPZs with progress"
    )
    parser.add_argument(
        '--matrices', nargs='+', required=True,
        help='Paths to one or more .npz boolean CSR matrices'
    )
    parser.add_argument(
        '--lists', nargs='+', default=None,
        help='(Optional) Paths to matching csv_list .pkl files; inferred if omitted'
    )
    parser.add_argument(
        '--output-prefix', required=True,
        help='Prefix for output files'
    )
    args = parser.parse_args()

    U, union_ids = build_union_matrix(args.matrices, args.lists)

    out_npz = args.output_prefix + '.npz'
    out_list = args.output_prefix + '_csv_list.pkl'
    save_npz(out_npz, U, compressed=True)
    with open(out_list, 'wb') as f:
        pickle.dump(union_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved union matrix to {out_npz} (shape={U.shape}, nnz={U.nnz})")
    print(f"Saved union CSV list to {out_list} ({len(union_ids)} entries)")
