
import os
import pickle
import numpy as np
from scipy.sparse import load_npz
from src.data_gt.step3_gt import get_npz_path

# Keep output format identical to the original script, only excluding generic CSVs.
GENERIC_TABLE_PATTERNS = ["1910.09700_table", "204823751_table"]

######################## DEBUG FUNCTIONS ########################
def inspect_npz(M, csv_list):
    print(f"  CSV list length: {len(csv_list)}, sample entries: {csv_list[:3]}")
    print(f"  shape: {M.shape}, nnz: {M.nnz}, dtype: {M.dtype}")

    diag = M.diagonal()
    nz_diag = np.where(diag)[0]
    assert nz_diag.size == 0, f"Non-zero diagonal entries at positions: {nz_diag[:20].tolist()} (showing up to 20)"
    print("✅ OK: diagonal is all zeros")

    diff = (M != M.T)
    if diff.nnz != 0:
        diff_coo = diff.tocoo()
        sample = list(zip(diff_coo.row[:5], diff_coo.col[:5]))
        raise AssertionError(f"Matrix is not symmetric; asymmetric nnz={diff_coo.nnz}; sample (row,col)={sample}")
    print("✅ OK: matrix is symmetric")

    n = M.shape[0]
    row_counts = np.diff(M.indptr)
    fully_connected = np.where(row_counts == n - 1)[0]
    assert fully_connected.size == 0, f"Found rows fully connected to all others: {fully_connected[:20].tolist()} (showing up to 20)"
    print("✅ OK: no row is fully connected")

def inspect_row(M, row_idx):
    if 0 <= row_idx < M.shape[0]:
        start, end = M.indptr[row_idx], M.indptr[row_idx + 1]
        cols, vals = M.indices[start:end], M.data[start:end]
        print(f"  Row {row_idx} nnz: {end-start}, cols[:5]={cols[:5].tolist()}, vals[:5]={vals[:5].tolist()}")
    else:
        print(f"⚠️ WARNING: Row index {row_idx} is out of bounds")

def main_debug(v2_suffix, suffix):
    LEVEL_NPZ, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix)
    for level, npz_name in LEVEL_NPZ.items():
        csvlist_name = LEVEL_CSVLIST[level]
        print(f"\n=== Level: {level} ===")
        M = load_npz(npz_name).tocsr()
        with open(csvlist_name, "rb") as f:
            csv_list = pickle.load(f)
        inspect_npz(M, csv_list)

def main_check_row(v2_suffix, suffix):
    LEVEL_NPZ, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix)
    for level, npz_name in LEVEL_NPZ.items():
        csvlist_name = LEVEL_CSVLIST[level]
        print(f"\n=== Level: {level} ===")
        M = load_npz(npz_name).tocsr()
        with open(csvlist_name, "rb") as f:
            csv_list = pickle.load(f)
        inspect_row(M, args.row)

######################## MAIN FUNCTIONS ########################
def load_index_names(path):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            lst = pickle.load(f)
        return [os.path.basename(str(x)) for x in lst]
    with open(path, 'r', encoding='utf-8') as f:
        return [os.path.basename(line.strip()) for line in f if line.strip()]

def compute_nnz_density(npz_path, key_name, keep_mask=None):
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
        if cols.size == 0:
            continue
        # Some matrices may have column indices outside the range of the
        # provided `keep_mask` (index list mismatch). Only consider column
        # indices that fall within the mask; out-of-range columns are treated
        # as not-kept.
        valid_cols = cols[cols < len(keep_mask)]
        if valid_cols.size == 0:
            continue
        cnt = int(np.count_nonzero(keep_mask[valid_cols]))
        if cnt > 0:
            nz_rows += 1
            nnz += cnt
    density = nnz / (nz_rows * nz_rows) if nz_rows > 0 else 0.0
    print(f"{key_name:<50}{nnz:>14,}{format_sci(nnz):>16}{density:>12.6f}{density * 100.0:>9.2f}%{n_total:>9,}")
    return nnz, density, nz_rows, n


def format_sci(n: int) -> str:
    if n == 0:
        return "0"
    import math
    exp = int(math.floor(math.log10(abs(n))))
    mant = n / (10 ** exp)
    return f"{mant:.2f} x 10^{exp}"


def main(v2_suffix, suffix):
    # Columns: Level, NNZ, NNZ_sci, Density, Density_pct
    print(f"{'Level':<50}{'NNZ':>14}{'NNZ_sci':>16}{'Density':>12}{'Percent':>10}{'Total':>9}")
    print('-' * 92)
    LEVEL_NPZ, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix)

    # First: Model and Dataset (so you can stop early after they print)
    nnz, density, nz_rows, n_total = compute_nnz_density(LEVEL_NPZ["model"], "Model")
    nnz, density, nz_rows, n_total = compute_nnz_density(LEVEL_NPZ["dataset"], "Dataset")

    # Then: CSV-level with generic filtering
    for lvl in LEVEL_NPZ.keys():
        idx_pkl = LEVEL_CSVLIST[lvl]
        idx_names = load_index_names(idx_pkl)
        name2idx = {n: i for i, n in enumerate(idx_names)}
        filtered_names = [n for n in idx_names if not any(p in n for p in GENERIC_TABLE_PATTERNS)]
        keep_mask = np.zeros(len(idx_names), dtype=bool)
        keep_mask[[name2idx[n] for n in filtered_names]] = True
        nnz, density, nz_rows, n_total = compute_nnz_density(LEVEL_NPZ[lvl], lvl, keep_mask)
    
    # extra union (no index available; print as-is)
    nnz, density, nz_rows, n_total = compute_nnz_density(LEVEL_NPZ["union"], "Union")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print number of non-zero elements and density for each matrix (generic excluded).')
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode for GT files.')
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    main(v2_suffix=v2_suffix, suffix=suffix)


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