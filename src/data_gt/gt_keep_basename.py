#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Created: 2025-05-20
Last modified: 2025-05-21
Description: keep basename of gt
"""
import os
import pickle
import json

files = [
    ("data/gt/csv_pair_adj_direct_label.pkl", "pickle"),
    ("data/gt/csv_pair_adj_overlap_rate.pkl", "pickle"),
    ("data/gt/scilake_gt_modellink_model_adj.json", "json"),
    ("data/gt/scilake_gt_modellink_dataset_adj.json", "json"),
]

for path, ftype in files:
    if ftype == "pickle":
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    new_data = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_k = os.path.basename(k)
            if isinstance(v, list):
                new_v = [os.path.basename(x) for x in v]
            else:
                new_v = v
            new_data[new_k] = new_v
    else:
        new_data = data

    base, _ = os.path.splitext(path)
    output_path = f"{base}_processed.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(new_data, f)

    print(f"✅ Saved processed pickle to {output_path}")
