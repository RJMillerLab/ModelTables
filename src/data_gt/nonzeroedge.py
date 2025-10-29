
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


def compute_nnz_density(npz_path, keep_mask=None):
    """Compute nnz and density quickly without constructing a sliced submatrix.
    If keep_mask is provided (boolean array of length n), only counts edges where
    both endpoints are kept, and only counts rows among kept rows.
    """
    M = load_npz(npz_path).tocsr()
    n = M.shape[0]
    if keep_mask is None:
        # Fast path: original stats
        row_nnz = M.getnnz(axis=1)
        nz_rows = int(np.sum(row_nnz > 0))
        nnz = int(M.nnz)
        density = nnz / (nz_rows * nz_rows) if nz_rows > 0 else 0.0
        return nnz, density, nz_rows, n
    # Masked fast path: linear scan over rows and use column mask on indices
    indptr = M.indptr
    indices = M.indices
    nnz = 0
    nz_rows = 0
    for i in range(n):
        if not keep_mask[i]:
            continue
        start = indptr[i]
        end = indptr[i + 1]
        if start == end:
            continue
        cols = indices[start:end]
        cnt = int(np.count_nonzero(keep_mask[cols]))
        if cnt > 0:
            nz_rows += 1
            nnz += cnt
    density = nnz / (nz_rows * nz_rows) if nz_rows > 0 else 0.0
    return nnz, density, nz_rows, n


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

    # First: Model and Dataset (so you can stop early after they print)
    model_path = os.path.join(gt_dir, "scilake_gt_modellink_model_adj_processed.npz")
    if os.path.exists(model_path):
        nnz, density, nz_rows, n_total = compute_nnz_density(model_path)
        nnz_sci = format_sci(nnz)
        density_pct = density * 100.0
        print(f"{'Model':<40}{nnz:>14,}{nnz_sci:>16}{density:>12.6f}{density_pct:>9.2f}%")

    dataset_path = os.path.join(gt_dir, "scilake_gt_modellink_dataset_adj_processed.npz")
    if os.path.exists(dataset_path):
        nnz, density, nz_rows, n_total = compute_nnz_density(dataset_path)
        nnz_sci = format_sci(nnz)
        density_pct = density * 100.0
        print(f"{'Dataset':<40}{nnz:>14,}{nnz_sci:>16}{density:>12.6f}{density_pct:>9.2f}%")

    # Then: CSV-level with generic filtering
    for lvl in LEVELS:
        npz_path = os.path.join(gt_dir, f"csv_pair_matrix_{lvl}.npz")
        # optional index for filtering
        idx_pkl = os.path.join(gt_dir, f"csv_list_{lvl}.pkl")
        idx_txt = os.path.join(gt_dir, f"csv_list_{lvl}.txt")
        keep_mask = None
        idx_names = load_index_names(idx_pkl) or load_index_names(idx_txt)
        if idx_names:
            name2idx = {n: i for i, n in enumerate(idx_names)}
            filtered_names = [n for n in idx_names if not any(p in n for p in GENERIC_TABLE_PATTERNS)]
            keep_mask = np.zeros(len(idx_names), dtype=bool)
            if filtered_names:
                keep_mask[[name2idx[n] for n in filtered_names]] = True
        nnz, density, nz_rows, n_total = compute_nnz_density(npz_path, keep_mask)
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

"""
Level                                              NNZ         NNZ_sci     Density   Percent
--------------------------------------------------------------------------------------------
Model                                       17,138,364     1.71 x 10^7    0.001987     0.20%
Dataset                                     35,194,914     3.52 x 10^7    0.004082     0.41%
direct_label                               706,375,631     7.06 x 10^8    0.081736     8.17%
direct_label_influential                   301,569,941     3.02 x 10^8    0.034895     3.49%
direct_label_methodology_or_result         498,067,647     4.98 x 10^8    0.057633     5.76%
direct_label_methodology_or_result_influential   267,788,315     2.68 x 10^8    0.030986     3.10%
max_pr                                   3,719,462,957     3.72 x 10^9    0.430388    43.04%
max_pr_influential                         859,363,909     8.59 x 10^8    0.099439     9.94%
max_pr_methodology_or_result             2,232,772,267     2.23 x 10^9    0.258359    25.84%
max_pr_methodology_or_result_influential   695,309,213     6.95 x 10^8    0.080456     8.05%
Union                                      718,163,240     7.18 x 10^8    0.083099     8.31%
"""