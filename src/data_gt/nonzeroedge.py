
import os
import pickle
import numpy as np
from scipy.sparse import load_npz


# Keep output format identical to the original script, only excluding generic CSVs.
GENERIC_TABLE_PATTERNS = [
    "1910.09700_table",
    "204823751_table",
]


def load_index_names(path):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            lst = pickle.load(f)
        return [os.path.basename(str(x)) for x in lst]
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [os.path.basename(line.strip()) for line in f if line.strip()]
    return []


def compute_nnz_density(npz_path, keep_idx=None):
    M = load_npz(npz_path).tocsr()
    if keep_idx is not None:
        M = M[keep_idx][:, keep_idx].tocsr()
    nnz = M.nnz
    row_nnz = M.getnnz(axis=1)
    nz_rows = int(np.sum(row_nnz > 0))
    density = nnz / (nz_rows * nz_rows) if nz_rows > 0 else 0.0
    return nnz, density, nz_rows, M.shape[0]


def format_sci(n: int) -> str:
    if n == 0:
        return "0"
    import math
    exp = int(math.floor(math.log10(abs(n))))
    mant = n / (10 ** exp)
    return f"{mant:.2f} x 10^{exp}"


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

    # Columns: Level, NNZ, NNZ_sci, Density, Density_pct
    print(f"{'Level':<40}{'NNZ':>14}{'NNZ_sci':>16}{'Density':>12}{'Percent':>10}")
    print('-' * 92)

    for lvl in LEVELS:
        npz_path = os.path.join(gt_dir, f"csv_pair_matrix_{lvl}.npz")
        # optional index for filtering
        idx_pkl = os.path.join(gt_dir, f"csv_list_{lvl}.pkl")
        idx_txt = os.path.join(gt_dir, f"csv_list_{lvl}.txt")
        keep_idx = None
        idx_names = load_index_names(idx_pkl) or load_index_names(idx_txt)
        if idx_names:
            name2idx = {n: i for i, n in enumerate(idx_names)}
            filtered_names = [n for n in idx_names if not any(p in n for p in GENERIC_TABLE_PATTERNS)]
            keep_idx = np.array([name2idx[n] for n in filtered_names], dtype=np.int64)
            keep_idx.sort()
        nnz, density, nz_rows, n_total = compute_nnz_density(npz_path, keep_idx)
        nnz_sci = format_sci(nnz)
        density_pct = density * 100.0
        print(f"{lvl:<40}{nnz:>14,}{nnz_sci:>16}{density:>12.6f}{density_pct:>9.2f}%")

    # extra union (no index available; print as-is)
    npz_path = os.path.join(gt_dir, f"csv_pair_union_direct_processed.npz")
    nnz, density, nz_rows, n_total = compute_nnz_density(npz_path)
    nnz_sci = format_sci(nnz)
    density_pct = density * 100.0
    print(f"{'Union':<40}{nnz:>14,}{nnz_sci:>16}{density:>12.6f}{density_pct:>9.2f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print number of non-zero elements and density for each matrix (generic excluded).')
    parser.add_argument('--gt_dir', type=str, default='data/gt', help='Directory of GT files')
    args = parser.parse_args()
    main(args.gt_dir)
