#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Date: 2025-05-18

Combine the intermediate pickle files in data/gt_tmp into a complete GT.
"""
import os, gzip, pickle, argparse

GT_DIR = "data/gt"
GT_TMP_DIR = "data/gt_tmp"
GT_COMBINED_PATH = os.path.join(GT_DIR, "scilake_gt_all_matrices.pkl.gz")

def _load_fragment(name: str, suffix: str):
    path = os.path.join(GT_TMP_DIR, f"{name}_{suffix}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find the intermediate file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def main(rel_mode: str):
    suffix = f"__{rel_mode}"
    paper_index        = _load_fragment("paper_index", suffix)
    paper_adj          = _load_fragment("paper_paper_adj", suffix)
    model_adj          = _load_fragment("model_model_adj", suffix)
    model_index        = _load_fragment("model_index", suffix)
    csv_adj            = _load_fragment("csv_csv_adj", suffix)
    csv_index          = _load_fragment("csv_index", suffix)
    csv_symlink_adj    = _load_fragment("csv_symlink_adj", suffix)
    csv_symlink_index  = _load_fragment("csv_symlink_index", suffix)
    csv_real_adj       = _load_fragment("csv_real_adj", suffix)
    csv_real_index     = _load_fragment("csv_real_index", suffix)
    csv_real_gt        = _load_fragment("csv_real_gt", suffix)
    csv_real_count     = _load_fragment("csv_real_count", suffix)

    combined = {
        "paper_adj":        paper_adj,
        "paper_index":      paper_index,
        "model_adj":        model_adj,
        "model_index":      model_index,
        "csv_adj":          csv_adj,
        "csv_index":        csv_index,
        "csv_symlink_adj":  csv_symlink_adj,
        "csv_symlink_index":csv_symlink_index,
        "csv_real_adj":     csv_real_adj,
        "csv_real_index":   csv_real_index,
        "csv_real_gt":      csv_real_gt,
        "csv_real_count":   csv_real_count,
    }

    out_path = GT_COMBINED_PATH.replace(".pkl.gz", f"{suffix}.pkl.gz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"✔️  Combined GT saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine the intermediate files in data/gt_tmp into a complete GT")
    parser.add_argument(
        "--rel_mode",
        choices=["overlap_rate", "direct_label"],
        required=True,
        help="Specify the relation mode (suffix)"
    )
    args = parser.parse_args()
    main(args.rel_mode)
