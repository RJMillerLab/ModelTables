#!/usr/bin/env python3

import os
import argparse
from tkinter.font import names
from typing import Dict, List, Tuple

from dask.rewrite import args

from data_gt.merge_union import LEVEL_CSVLIST
from scipy.sparse import load_npz, csr_matrix, csc_matrix
from src.data_gt.step3_gt import get_npz_path

def load_index_map(csv_index_path: str) -> Tuple[List[str], Dict[str, int]]:
    names: List[str]
    import pickle
    with open(csv_index_path, 'rb') as f:
        lst = pickle.load(f)
    names = [os.path.basename(str(x)) for x in lst if str(x).strip()]
    name2idx = {n: i for i, n in enumerate(names)}
    return names, name2idx

def read_query_list(csv_list_path: str) -> List[str]:
    with open(csv_list_path, 'r', encoding='utf-8') as f:
        return [os.path.basename(line.strip()) for line in f if line.strip()]

def query_neighbors(A: csr_matrix, idx: int):
    indptr = A.indptr
    indices = A.indices
    start = indptr[idx]
    end = indptr[idx + 1]
    return indices[start:end]

def compute_coverage(A: csr_matrix, names_to_check: List[str], name2idx: Dict[str, int]) -> Tuple[int, int, int]:
    present = [n for n in names_to_check if n in name2idx]
    if not present:
        return 0, 0, 0
    # CSR gives fast row nnz via indptr
    csrA: csr_matrix = A.tocsr(copy=False)
    row_indptr = csrA.indptr
    row_nnz = lambda i: int(row_indptr[i+1] - row_indptr[i])

    cscA: csc_matrix = A.tocsc(copy=True)  # build once per GT
    col_indptr = cscA.indptr
    col_nnz = lambda j: int(col_indptr[j+1] - col_indptr[j])

    nonempty_row = 0
    nonempty_both = 0
    for n in present:
        i = name2idx[n]
        has_row = row_nnz(i) > 0
        has_col = (col_nnz(i) > 0)  # type: ignore
        nonempty_row += int(has_row)  # also report row coverage
        nonempty_both += int(has_row or has_col)

    return len(present), nonempty_row, nonempty_both

def query_table_related_from_gt(query_table: str, csv_list_path: str, matrix_path: str, output_path: str = "tmp/related_tables.txt") -> List[str]:
    # load index
    names, name2idx = load_index_map(csv_list_path)
    if query_table not in name2idx:
        print(f"Query not found in index")
        return
    i = name2idx[query_table]
    A: csr_matrix = load_npz(matrix_path).tocsr(copy=False)
    # get neighbors
    nbr_idx = query_neighbors(A, i)
    nbr_names = [names[j] for j in nbr_idx]
    # stats
    deg = len(nbr_idx)
    N = A.shape[0]
    rate = 100.0 * deg / max(1, N - 1)
    print(f"query={query_table} (idx={i})")
    print(f"degree={deg} / {N} ({rate:.4f}%)")
    # save
    with open(output_path, "w") as f:
        for n in nbr_names:
            f.write(n + "\n")
    print(f"saved → {output_path} ({deg} neighbors)")

def query_table_related(M: csr_matrix, names: List[str], name2idx: Dict[str, int], query_table: str,):
    """
    Return neighbor table names for a query table.
    """
    idx = name2idx.get(query_table)
    if idx is None:
        return [], 0, 0
    indptr = M.indptr
    indices = M.indices
    nbr_idx = indices[indptr[idx]:indptr[idx + 1]]
    nbr_names = [names[j] for j in nbr_idx]
    deg = len(nbr_idx)
    N = M.shape[0]
    return nbr_names, deg, N

def query_tablepair_related(M: csr_matrix, name2idx: Dict[str, int], csv1: str, csv2_list: List[str]):
    """
    Fast pair check using CSR row slice + dict lookup.
    """
    idx1 = name2idx.get(csv1)
    if idx1 is None:
        return []
    indptr = M.indptr
    indices = M.indices
    # O(degree)
    nbrs = set(indices[indptr[idx1]:indptr[idx1+1]])
    results = []
    for csv2 in csv2_list:
        idx2 = name2idx.get(csv2)
        if idx2 is None:
            results.append(False)
        else:
            # O(1)
            results.append(idx2 in nbrs)
    return results

def query_gt(
    M: csr_matrix,
    names: List[str],
    name2idx: Dict[str, int],
    query_table: str,
    target_tables: List[str] = None,
):
    """
    Unified GT query:
    - if target_tables is None → return neighbors
    - else → return pair relation
    """

    idx = name2idx.get(query_table)
    if idx is None:
        return None

    indptr = M.indptr
    indices = M.indices

    nbr_idx = indices[indptr[idx]:indptr[idx + 1]]

    # mode 1: neighbor retrieval
    if target_tables is None:
        nbr_names = [names[j] for j in nbr_idx]
        return {
            "neighbors": nbr_names,
            "degree": len(nbr_idx),
            "N": M.shape[0],
        }

    # mode 2: pair check
    nbrs = set(nbr_idx)
    results = [(name2idx.get(t) in nbrs) if t in name2idx else False for t in target_tables]

    return results
def main():
    parser = argparse.ArgumentParser(description='GT query tool')
    parser.add_argument('--query', required=True, help='Query table (basename)')
    parser.add_argument('--targets', nargs='*', default=None, help='Optional target tables')
    parser.add_argument('--level', default='direct', help='GT level')
    parser.add_argument('--v2_mode', action='store_true')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--out', default=None, help='Optional output file (for neighbors)')
    args = parser.parse_args()

    # ==== resolve paths ====
    v2_suffix = "_v2" if args.v2_mode else ""
    suffix = f"_{args.tag}" if args.tag else ""
    LEVEL_NPZ, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix)

    assert args.level in LEVEL_NPZ, f"Invalid level {args.level}"

    npz_path = LEVEL_NPZ[args.level]
    csv_path = LEVEL_CSVLIST[args.level]

    # ==== preload once ====
    M = load_npz(npz_path).tocsr(copy=False)
    names, name2idx = load_index_map(csv_path)

    # ==== unified query ====
    res = query_gt(
        M=M,
        names=names,
        name2idx=name2idx,
        query_table=args.query,
        target_tables=args.targets
    )

    # ==== handle output ====
    if res is None:
        print("Query not found in GT index")
        return

    # 🔹 case 1: neighbor mode
    if args.targets is None or len(args.targets) == 0:
        neighbors = res["neighbors"]
        deg = res["degree"]
        N = res["N"]

        rate = 100.0 * deg / max(1, N - 1)

        print(f"query={args.query}")
        print(f"degree={deg}/{N} ({rate:.4f}%)")

        if args.out:
            with open(args.out, "w") as f:
                for n in neighbors:
                    f.write(n + "\n")
            print(f"saved → {args.out} ({deg} neighbors)")

    # 🔹 case 2: pair mode
    else:
        for t, r in zip(args.targets, res):
            print(f"{args.query} <-> {t}: {'related' if r else 'not related'}")
    

if __name__ == '__main__':
    main()


