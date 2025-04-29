"""
Print statistics for the four adjacency matrices saved by modelcard_matrix.py.

Usage
-----
python -m src.data_gt.print_relations_stats data/tmp/relations_all.pkl
"""

import sys
import pickle
import numpy as np

def report(pkl_path: str) -> None:
    # -------- load ----------
    with open(pkl_path, "rb") as f:
        g = pickle.load(f)

    index          = g["model_index"]
    num_models     = len(index)
    matrices_order = ["base_adj", "hfmodel_adj", "dataset_adj", "tags_adj"]

    print(f"✔ Total modelId nodes: {num_models}")

    # -------- stats per matrix ----------
    for key in matrices_order:
        mat = g.get(key)
        if mat is None:
            continue

        nnz          = mat.nnz
        non_self     = nnz - num_models           # self-loops sit on the diagonal
        deg          = mat.sum(axis=1).A1         # out-degree (row sum) because matrices are symmetric
        linked_cnt   = int((deg > 1).sum())       # degree > 1  → has at least one real link
        unlinked_cnt = int((deg == 1).sum())      # only self-loop -> isolated

        print(f"\n[{key}]")
        print(f"  non-self edges : {non_self}")
        print(f"  linked models  : {linked_cnt}")
        print(f"  unlinked models: {unlinked_cnt}")

    # -------- missing models ----------
    present = set(index)
    # any model IDs you expected but didn’t appear in the index
    # can be checked here by passing an optional list.
    # Example stub (kept for extension):
    # missing = expected_ids - present
    # print(f"\nMissing modelIds not in graph: {len(missing)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_relations_stats.py <path/to/relations_all.pkl>")
        sys.exit(1)
    report(sys.argv[1])

