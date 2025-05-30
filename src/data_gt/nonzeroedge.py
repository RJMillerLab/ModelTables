"""
Print number of non-zero elements and density for each matrix file.
"""

import os
from scipy.sparse import load_npz
import numpy as np

def compute_nnz_density(npz_path):
    M = load_npz(npz_path).tocsr()
    nnz = M.nnz
    row_nnz = M.getnnz(axis=1)
    nz_rows = np.sum(row_nnz > 0)
    density = nnz / (nz_rows * nz_rows) if nz_rows > 0 else 0
    return nnz,density

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
        nnz, density = compute_nnz_density(npz_path)
        print(f"{lvl:<40}{nnz:>12,}{density:>12.6f}")
    # extra model
    npz_path = os.path.join(gt_dir, f"scilake_gt_modellink_model_adj.npz")
    nnz, density = compute_nnz_density(npz_path)
    print(f"{'Model':<40}{nnz:>12,}{density:>12.6f}")
    # extra dataset
    npz_path = os.path.join(gt_dir, f"scilake_gt_modellink_dataset_adj.npz")
    nnz, density = compute_nnz_density(npz_path)
    print(f"{'Dataset':<40}{nnz:>12,}{density:>12.6f}")
    # extra union
    npz_path = os.path.join(gt_dir, f"csv_pair_union_direct.npz")
    nnz, density = compute_nnz_density(npz_path)
    print(f"{'Union':<40}{nnz:>12,}{density:>12.6f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print number of non-zero elements and density for each matrix')
    parser.add_argument('--gt_dir', type=str, default='data/gt', help='Directory of GT files')
    args = parser.parse_args()
    main(args.gt_dir)

"""
Level                                            NNZ     Density
-----------------------------------------------------------------
direct_label                             708,281,447    0.081936
direct_label_influential                 302,904,269    0.035041
direct_label_methodology_or_result       499,726,167    0.057810
direct_label_methodology_or_result_influential 269,056,043    0.031125
max_pr                                  3,721,596,269    0.430524
max_pr_influential                       860,908,477    0.099592
max_pr_methodology_or_result            2,234,737,339    0.258520
max_pr_methodology_or_result_influential 696,729,269    0.080599
Model                                     18,308,662    0.001759
Dataset                                   36,894,918    0.003615
Union                                    720,963,231    0.068290
"""