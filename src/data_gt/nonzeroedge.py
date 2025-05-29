"""
Print number of non-zero elements and density for each matrix file.
"""

import os
from scipy.sparse import load_npz
import pickle
import numpy as np

def load_gt_npz(matrix_path, csv_list_path):
    """Load a boolean CSR matrix (npz) and its csv_list (pickle)."""
    M = load_npz(matrix_path).tocsr()
    with open(csv_list_path, 'rb') as f:
        csv_list = pickle.load(f)
    return M, csv_list

def get_nonzero_dims(M):
    """Get number of non-zero rows and columns."""
    nonzero_rows = np.unique(M.nonzero()[0])
    nonzero_cols = np.unique(M.nonzero()[1])
    return len(nonzero_rows), len(nonzero_cols)

def main(gt_dir):
    LEVELS = [
        "direct_label",
        "direct_label_influential",
        "direct_label_methodology_or_result",
        "direct_label_methodology_or_result_influential",
        "max_pr",
        "max_pr_influential",
        "max_pr_methodology_or_result",
        "max_pr_methodology_or_result_influential",
    ]

    print(f"{'Level':<40}{'NNZ':>12}{'Density':>12}")
    print('-' * 65)

    for lvl in LEVELS:
        npz_path = os.path.join(gt_dir, f"csv_pair_matrix_{lvl}.npz")
        csvlist_path = os.path.join(gt_dir, f"csv_list_{lvl}.pkl")
        
        if os.path.exists(npz_path) and os.path.exists(csvlist_path):
            print(f"Loading {lvl}...")
            M, _ = load_gt_npz(npz_path, csvlist_path)
            nnz = M.nnz
            
            # Get non-zero dimensions and calculate density
            nz_rows, nz_cols = get_nonzero_dims(M)
            assert nz_rows == nz_cols, f"Non-zero rows ({nz_rows}) != non-zero cols ({nz_cols}) for {lvl}"
            density = nnz / (nz_rows * nz_cols) if nz_rows > 0 else 0
            
            print(f"{lvl:<40}{nnz:>12,}{density:>12.6f}")
            del M  # Explicitly delete to free memory
        else:
            print(f"{lvl:<40}{'MISSING':>12}{'MISSING':>12}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print number of non-zero elements and density for each matrix')
    parser.add_argument('--gt_dir', type=str, default='data/gt', help='Directory of GT files')
    args = parser.parse_args()
    main(args.gt_dir)
