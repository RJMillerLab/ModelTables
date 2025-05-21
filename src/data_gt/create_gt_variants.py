#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Created: 2025-05-20
Last modified: 2025-05-21
Description: Create *_s / *_t / *_s_t ground-truth variants from a base GT pickle.

Usage:
    python -m src.data_gt.create_gt_variants data/gt/csv_pair_adj_overlap_rate_processed.pkl
"""

import os, argparse, pickle

SUFFIXES = {
    "":      "",
    "_s":    "_s",
    "_t":    "_t",
    "_s_t":  "_s_t",
}

def add_suffix_to_adj(adj_dict, suf):
    """
    Append suffix *before* the file extension, e.g.
    'table1.csv' + '_s' -> 'table1_s.csv'
    """
    out = {}
    for k, neigh in adj_dict.items():
        root, ext = os.path.splitext(k)                                ########
        k2 = f"{root}{suf}{ext}"                                       ########
        neigh2 = []
        for n in neigh:
            r, e = os.path.splitext(n)                                 ########
            neigh2.append(f"{r}{suf}{e}")                             ########
        out[k2] = neigh2
    return out  

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Path to the *_processed.pkl adjacency file")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        base_adj = pickle.load(f)

    base, _ = os.path.splitext(args.src)
    for tag, suf in SUFFIXES.items():
        tgt = f"{base}{tag}.pkl" if tag else args.src
        adj_tag = add_suffix_to_adj(base_adj, suf)
        with open(tgt, "wb") as fout:
            pickle.dump(adj_tag, fout)
        print(f"✅ Saved {tgt}")

if __name__ == "__main__":
    main()

