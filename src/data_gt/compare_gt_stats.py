"""
python -m src.data_gt.compare_gt_stats --gt_dir data/gt
"""

import os, pickle, hashlib
from collections import Counter
from scipy.sparse import load_npz

def load_gt(path):
    """Load a pickled adjacency dict."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_gt_npz(matrix_path, csv_list_path):
    """
    Load a boolean CSR matrix (npz) and its csv_list (pickle).
    Returns: (csr_matrix, csv_list)
    """
    M = load_npz(matrix_path).tocsr()
    with open(csv_list_path, 'rb') as f:
        csv_list = pickle.load(f)
    return M, csv_list

def hash_file(path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(adj):
    """Return number of nodes, edges, avg degree."""
    '''num_nodes = len(adj)
    total_links = sum(len(v) for v in adj.values())
    avg_degree = total_links / num_nodes if num_nodes else 0
    return num_nodes, total_links // 2, avg_degree'''

    if isinstance(adj, dict):
        num_nodes = len(adj)
        total_links = sum(len(v) for v in adj.values())
        return num_nodes, total_links // 2, total_links / num_nodes if num_nodes else 0
    # else assume csr matrix + list
    M, csv_list = adj
    num_nodes = len(csv_list)
    # since undirected and boolean, sum of row nnz gives 2*edges
    row_counts = M.sum(axis=1).A1  # each row's degree
    total_links = int(row_counts.sum() / 2)
    avg_degree = row_counts.mean() if num_nodes else 0
    return num_nodes, total_links, avg_degree

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

    adjs = {}
    hashes = {}
    for lvl in LEVELS:
        npz_path     = os.path.join(gt_dir, f"csv_pair_matrix_{lvl}.npz")
        csvlist_path = os.path.join(gt_dir, f"csv_list_{lvl}.pkl")
        if os.path.exists(npz_path) and os.path.exists(csvlist_path):
            print(f"[NPZ] loading {lvl}")
            adjs[lvl]   = load_gt_npz(npz_path, csvlist_path)
            hashes[lvl] = hash_file(npz_path)
        else:
            pkl_path = os.path.join(gt_dir, f"csv_pair_adj_{lvl}_processed.pkl")
            if os.path.exists(pkl_path):
                print(f"[PKL] loading {lvl}")
                adjs[lvl]   = load_gt_pickle(pkl_path)
                hashes[lvl] = hash_file(pkl_path)
            else:
                print(f"{lvl}: MISSING")
                continue

    # 1) File hashes
    print("\nFile MD5 hashes:")
    for lvl, h in hashes.items():
        print(f"  {lvl}: {h}")

    # 2) Basic stats
    print("\nBasic GT stats:")
    print(f"{'Level':<40}{'Nodes':>8}{'Edges':>10}{'AvgDeg':>10}")
    print('-' * 70)
    base = LEVELS[0]
    for lvl, adj in adjs.items():
        nodes, edges, avg_deg = summarize(adj)
        print(f"{lvl:<40}{nodes:>8}{edges:>10}{avg_deg:>10.2f}")

    # 3) Equality checks against base level
    print(f"\nComparisons to '{base}':")
    '''for lvl in LEVELS[1:]:
        if lvl in adjs:
            equal = (adjs[lvl] == adjs[base])'''
    for lvl in LEVELS[1:]:
        if lvl in adjs and base in adjs:
            a1, a2 = adjs[base], adjs[lvl]
            # dict vs tuple: only compare shapes if both npz
            if isinstance(a1, tuple) and isinstance(a2, tuple):
                M1, _ = a1; M2, _ = a2
                equal = ( (M1 != M2).nnz == 0 )
            else:
                equal = (a1 == a2)
            print(f"  {lvl} == {base}: {equal}")
            if not equal:
                #extra = set(adjs[lvl].keys()) - set(adjs[base].keys())
                #missing = set(adjs[base].keys()) - set(adjs[lvl].keys())
                if isinstance(adjs[lvl], dict):
                    extra   = set(adjs[lvl].keys()) - set(adjs[base].keys())
                    missing = set(adjs[base].keys()) - set(adjs[lvl].keys())
                    print(f"    extra keys: {sorted(list(extra))[:5]}...")
                    print(f"    missing keys: {sorted(list(missing))[:5]}...")
                else:
                    # CSR, only report different
                    print("    (Matrices differ in non-zero pattern)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare GT adjacency pickle files with deeper debug')
    parser.add_argument('--gt_dir', type=str, default='data/gt', help='Directory of GT pickles')
    args = parser.parse_args()
    main(args.gt_dir)


"""
Basic GT stats:
Level                                      Nodes     Edges    AvgDeg
----------------------------------------------------------------------
direct_label                               92975 354094236   7616.98
direct_label_influential                   92975 151405647   3256.91
direct_label_methodology_or_result         92975 249816596   5373.84
direct_label_methodology_or_result_influential   92975 134481534   2892.85
max_pr_influential                         92975 430407751   9258.57
max_pr_methodology_or_result               929751117322182  24034.90
max_pr_methodology_or_result_influential   92975 348318147   7492.73
"""