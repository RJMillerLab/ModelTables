#!/usr/bin/env python3

import os
import sys
import argparse
import glob
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import load_npz, csr_matrix, csc_matrix


DEFAULT_GT_DIR = os.path.join('data', 'gt')

# Preferred filename patterns for each logical GT level.
# The script will try these patterns (in order) to locate a GT .npz file.
GT_LEVEL_TO_PATTERNS: Dict[str, List[str]] = {
    # paper-level direct label
    'direct': [
        '*paper*direct*label*.npz',
        '*paper*direct*.npz',
    ],
    # model-level graph (naming may vary across repos)
    'model': [
        '*model*adj*processed*.npz',
        '*model*adj*.npz',
        '*model*.npz',
    ],
    # dataset-level graph
    'dataset': [
        '*modellink*dataset*adj*processed*.npz',
        '*dataset*adj*.npz',
        '*dataset*.npz',
    ],
    # union-level
    'union': [
        '*union*.npz',
    ],
}

# Optional alias mapping from user level names to step3_gt rel_keys
# step3_gt.py saves files as csv_pair_matrix_{rel_key}.npz and csv_list_{rel_key}.pkl
LEVEL_TO_RELKEY: Dict[str, str] = {
    'direct': 'direct_label',
    'direct_influential': 'direct_label_influential',
    'direct_methodology_or_result': 'direct_label_methodology_or_result',
    'direct_methodology_or_result_influential': 'direct_label_methodology_or_result_influential',
    'max_pr': 'max_pr',
    'max_pr_influential': 'max_pr_influential',
    'max_pr_methodology_or_result': 'max_pr_methodology_or_result',
    'max_pr_methodology_or_result_influential': 'max_pr_methodology_or_result_influential',
}


def find_file_by_patterns(root: str, patterns: List[str]) -> str:
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(root, pat)))
        if matches:
            return matches[0]
    return ''


def find_csv_index_file(gt_dir: str) -> str:
    """Find a CSV index file under gt_dir.
    Preference order:
      1) csv_list_*.pkl (step3_gt convention)
      2) csv_list.pkl
      3) csv_list.txt
      4) *csv*list*.pkl or *csv*list*.txt (largest by size)
    """
    # 1) Any rel_key-specific pkl
    pkl_specific = sorted(glob.glob(os.path.join(gt_dir, 'csv_list_*.pkl')))
    if pkl_specific:
        return pkl_specific[0]
    # 2) Generic pkl
    generic_pkl = os.path.join(gt_dir, 'csv_list.pkl')
    if os.path.exists(generic_pkl):
        return generic_pkl
    # 3) Generic txt
    generic_txt = os.path.join(gt_dir, 'csv_list.txt')
    if os.path.exists(generic_txt):
        return generic_txt
    # 4) Largest fallback among pkl/txt patterns
    candidates = glob.glob(os.path.join(gt_dir, '*csv*list*.pkl')) + \
                 glob.glob(os.path.join(gt_dir, '*csv*list*.txt'))
    if not candidates:
        return ''
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


def load_index_map(csv_index_path: str) -> Tuple[List[str], Dict[str, int]]:
    names: List[str]
    if csv_index_path.endswith('.pkl'):
        import pickle
        with open(csv_index_path, 'rb') as f:
            lst = pickle.load(f)
        # step3_gt stores basenames already, but normalize just in case
        names = [os.path.basename(str(x)) for x in lst if str(x).strip()]
    else:
        with open(csv_index_path, 'r', encoding='utf-8') as f:
            names = [os.path.basename(line.strip()) for line in f if line.strip()]
    name2idx = {n: i for i, n in enumerate(names)}
    return names, name2idx


def read_query_list(csv_list_path: str) -> List[str]:
    with open(csv_list_path, 'r', encoding='utf-8') as f:
        return [os.path.basename(line.strip()) for line in f if line.strip()]


def compute_coverage(A: csr_matrix, names_to_check: List[str], name2idx: Dict[str, int], mode: str) -> Tuple[int, int, int]:
    present = [n for n in names_to_check if n in name2idx]
    if not present:
        return 0, 0, 0
    # CSR gives fast row nnz via indptr
    csrA: csr_matrix = A.tocsr(copy=False)
    row_indptr = csrA.indptr
    row_nnz = lambda i: int(row_indptr[i+1] - row_indptr[i])

    if mode in ('col', 'both'):
        cscA: csc_matrix = A.tocsc(copy=True)  # build once per GT
        col_indptr = cscA.indptr
        col_nnz = lambda j: int(col_indptr[j+1] - col_indptr[j])
    else:
        col_nnz = None  # type: ignore

    nonempty_row = 0
    nonempty_both = 0
    for n in present:
        i = name2idx[n]
        has_row = row_nnz(i) > 0
        if mode == 'row':
            nonempty_row += int(has_row)
        elif mode == 'col':
            has_col = (col_nnz(i) > 0)  # type: ignore
            nonempty_row += int(has_col)  # reuse variable to report "nonempty"
        else:  # both
            has_col = (col_nnz(i) > 0)  # type: ignore
            nonempty_row += int(has_row)  # also report row coverage
            nonempty_both += int(has_row or has_col)

    return len(present), nonempty_row, nonempty_both


def main():
    parser = argparse.ArgumentParser(description='Check GT coverage for a CSV list across levels.')
    parser.add_argument('--csv-list', help='Path to CSV list to check (one filename per line).')
    parser.add_argument('--csv-name', help='Check a single CSV filename (basename). Overrides --csv-list if provided.')
    parser.add_argument('--levels', nargs='+', default=['direct'], help='Levels to check, e.g., direct model dataset union or exact rel_keys')
    parser.add_argument('--gt-dir', default=DEFAULT_GT_DIR, help='Directory containing GT .npz and csv index.')
    parser.add_argument('--mode', choices=['row', 'col', 'both'], default='both', help='Coverage mode: row/col/both')
    args = parser.parse_args()

    if not os.path.isdir(args.gt_dir):
        print(f"GT dir not found: {args.gt_dir}")
        sys.exit(1)

    if args.csv_name:
        q_names = [os.path.basename(args.csv_name.strip())]
        print(f"GT dir     : {args.gt_dir}")
        print(f"Single CSV : {q_names[0]}\n")
    else:
        if not args.csv_list:
            print("Either --csv-name or --csv-list is required.")
            sys.exit(1)
        q_names = read_query_list(args.csv_list)
        print(f"GT dir     : {args.gt_dir}")
        print(f"Check list : {args.csv_list} (N={len(q_names)})\n")

    for level in args.levels:
        # 1) Try step3_gt conventions first
        rel_key = LEVEL_TO_RELKEY.get(level, level)  # allow direct rel_key by passing exact name
        gt_file = os.path.join(args.gt_dir, f"csv_pair_matrix_{rel_key}.npz")
        csv_index = os.path.join(args.gt_dir, f"csv_list_{rel_key}.pkl")
        if not os.path.exists(gt_file) or not os.path.exists(csv_index):
            # 2) Fallback to legacy patterns
            patterns = GT_LEVEL_TO_PATTERNS.get(level, [])
            gt_file = find_file_by_patterns(args.gt_dir, patterns) if patterns else ''
            if not gt_file:
                print(f"[{level}] GT file not found (rel_key={rel_key}) and no match by patterns {patterns}")
                continue
            # try to locate a csv index as well
            csv_index = find_csv_index_file(args.gt_dir)
            if not csv_index:
                print(f"[{level}] CSV index not found under {args.gt_dir}")
                continue

        try:
            A = load_npz(gt_file)
        except Exception as e:
            print(f"[{level}] Failed to load {gt_file}: {e}. Hint: ensure you ran step3_gt for rel_key={rel_key}")
            continue

        gt_names, name2idx = load_index_map(csv_index)
        missing = [n for n in q_names if n not in name2idx]
        if missing:
            print(f"[{level}] Note: {len(missing)} names not present in GT index (showing up to 10): {missing[:10]}")

        present, nonempty_row, nonempty_both = compute_coverage(A, q_names, name2idx, args.mode)
        if present == 0:
            print(f"[{level}] present=0 (no overlap with GT index)")
            continue

        if args.csv_name:
            # Detailed single-CSV stats: row/col degree and rates
            n_total = A.shape[0]
            name = q_names[0]
            i = name2idx.get(name)
            if i is None:
                print(f"[{level}] {name} not found in index {os.path.basename(csv_index)}")
                continue
            csrA = A.tocsr(copy=False)
            row_deg = int(csrA.indptr[i+1] - csrA.indptr[i])
            if args.mode in ('col', 'both'):
                cscA = A.tocsc(copy=True)
                col_deg = int(cscA.indptr[i+1] - cscA.indptr[i])
            else:
                col_deg = 0
            denom = max(1, n_total - 1)  # exclude self; diag is already 0, but safer
            row_rate = 100.0 * row_deg / denom
            col_rate = 100.0 * col_deg / denom
            both_deg = int((row_deg > 0) or (col_deg > 0))
            print(f"[{level}] file={os.path.basename(gt_file)} | index={os.path.basename(csv_index)}")
            print(f"         name={name} | N={n_total} | row_deg={row_deg} ({row_rate:.2f}%), col_deg={col_deg} ({col_rate:.2f}%), any_edge={both_deg}")
        else:
            if args.mode == 'row':
                pct = 100.0 * nonempty_row / present
                print(f"[{level}] file={os.path.basename(gt_file)} | index={os.path.basename(csv_index)} | present={present} nonempty(row)={nonempty_row} coverage={pct:.2f}%")
            elif args.mode == 'col':
                pct = 100.0 * nonempty_row / present
                print(f"[{level}] file={os.path.basename(gt_file)} | index={os.path.basename(csv_index)} | present={present} nonempty(col)={nonempty_row} coverage={pct:.2f}%")
            else:
                pct_row = 100.0 * nonempty_row / present
                pct_both = 100.0 * nonempty_both / present
                print(f"[{level}] file={os.path.basename(gt_file)} | index={os.path.basename(csv_index)} | present={present} nonempty(row)={nonempty_row} ({pct_row:.2f}%), nonempty(both)={nonempty_both} ({pct_both:.2f}%)")


if __name__ == '__main__':
    main()


