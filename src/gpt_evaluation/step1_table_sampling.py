#!/usr/bin/env python3
"""
Table Relatedness Sampling - 8-Way Auto-Balanced (Rare-first)
Author: Zhengyuan Dong
Date: 2025-10-28
"""

import os
import json
import pickle
import argparse
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_evaluation.sparse_matrix_loader import SparseMatrixLoader


class TableSamplerV2:
    def __init__(self, gt_dir="data/gt", output_dir="output/gpt_evaluation", seed=42):
        self.gt_dir = gt_dir
        self.output_dir = output_dir
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.loaders = {}
        self.csv_lists = {}
        self.csv_to_idx = {}
        print(f"✓ Initialized TableSamplerV2 with seed={seed}")

    def load_gt_level(self, name, npz_path, list_path):
        loader = SparseMatrixLoader(npz_path, cache_positive_pairs=False)
        with open(list_path, "rb") as f:
            csv_list = pickle.load(f)
        self.loaders[name] = loader
        self.csv_lists[name] = csv_list
        self.csv_to_idx[name] = {c: i for i, c in enumerate(csv_list)}
        stats = loader.get_statistics()
        print(f"  ✓ {name}: shape={stats['shape']}, nnz={stats['nnz']:,}")
        return True

    def load_all_levels(self):
        print("=" * 60)
        print("LOADING GROUND TRUTH MATRICES")
        print("=" * 60)
        self.load_gt_level("paper",
            os.path.join(self.gt_dir, "csv_pair_matrix_direct_label.npz"),
            os.path.join(self.gt_dir, "csv_list_direct_label.pkl"))
        self.load_gt_level("modelcard",
            os.path.join(self.gt_dir, "scilake_gt_modellink_model_adj_processed.npz"),
            os.path.join(self.gt_dir, "scilake_gt_modellink_model_adj_csv_list_processed.pkl"))
        self.load_gt_level("dataset",
            os.path.join(self.gt_dir, "scilake_gt_modellink_dataset_adj_processed.npz"),
            os.path.join(self.gt_dir, "scilake_gt_modellink_dataset_adj_csv_list_processed.pkl"))
        print("=" * 60)

    def sample_all_levels_unified(self, n_samples_pool=1000000, total_target=200):
        print("=" * 60)
        print("UNIFIED MULTI-LEVEL SAMPLING (8-way balanced, rare-first)")
        print("=" * 60)

        csv_list = self.csv_lists["paper"]
        n_nodes = len(csv_list)

        print(f"Sampling {n_samples_pool} random pairs ...")
        seen = set()
        pairs = []
        while len(pairs) < n_samples_pool:
            i, j = self.rng.choice(n_nodes, 2, replace=False)
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                pairs.append((i, j))
        print(f"✓ Sampled {len(pairs)} pairs")

        # ---- Query all GTs ----
        print("\nQuerying GT matrices ...")
        all_pairs = []
        for i, j in pairs:
            a, b = csv_list[i], csv_list[j]
            def get_label(level):
                idx_a = self.csv_to_idx[level].get(a, -1)
                idx_b = self.csv_to_idx[level].get(b, -1)
                if idx_a >= 0 and idx_b >= 0:
                    return 1 if self.loaders[level].get_value(idx_a, idx_b) > 0 else 0
                return 0
            labels = {
                "paper": get_label("paper"),
                "modelcard": get_label("modelcard"),
                "dataset": get_label("dataset")
            }
            all_pairs.append({
                "csv_a": a, "csv_b": b, "labels": labels,
                "combination": (labels["paper"], labels["modelcard"], labels["dataset"])
            })

        # ---- Group by combination ----
        by_combo = defaultdict(list)
        for p in all_pairs:
            by_combo[p["combination"]].append(p)
        combo_names = {
            (0, 0, 0): "None",
            (1, 0, 0): "Paper only",
            (0, 1, 0): "ModelCard only",
            (0, 0, 1): "Dataset only",
            (1, 1, 0): "Paper + ModelCard",
            (1, 0, 1): "Paper + Dataset",
            (0, 1, 1): "ModelCard + Dataset",
            (1, 1, 1): "All three"
        }

        print("\n[3/4] 8-Way distribution in pool:")
        for c in sorted(by_combo.keys()):
            print(f"  {c}: {combo_names[c]:20s} - {len(by_combo[c])}")

        # ---- Rare-first balanced selection ----
        print("\n[4/4] Selecting rare-first balanced 8-way samples ...")

        target_per_combo = max(1, total_target // 8)
        selected = []
        remaining_target = total_target

        # 1️⃣ 全选稀缺组合
        small_combos, large_combos = [], []
        for c, items in by_combo.items():
            n = len(items)
            if n <= target_per_combo:
                selected += items
                small_combos.append(c)
                remaining_target -= n
            else:
                large_combos.append(c)

        # 2️⃣ 从剩余组合按比例抽样补齐
        if remaining_target > 0 and large_combos:
            total_large = sum(len(by_combo[c]) for c in large_combos)
            for c in large_combos:
                n = len(by_combo[c])
                share = int(remaining_target * n / total_large)
                np.random.shuffle(by_combo[c])
                selected += by_combo[c][:share]

        # 截断超额
        selected = selected[:total_target]

        # 3️⃣ --- per-level rebalancing ---
        def level_ratio(sel, lvl):
            pos = sum(1 for p in sel if p["labels"][lvl] == 1)
            return pos / len(sel)

        def adjust_balance(sel, lvl, target_ratio=0.5):
            """Try to balance level positives near 50%"""
            current = level_ratio(sel, lvl)
            if current >= target_ratio:
                return sel
            need = int(target_ratio * len(sel)) - sum(p["labels"][lvl] for p in sel)
            if need <= 0:
                return sel
            # find positive examples from unselected samples
            pool = [p for p in all_pairs if p["labels"][lvl] == 1 and p not in sel]
            np.random.shuffle(pool)
            add = pool[:need]
            # delete same number of pure negative examples (0,0,0)
            none_idx = [i for i, p in enumerate(sel) if p["combination"] == (0,0,0)]
            remove_n = min(len(add), len(none_idx))
            for i in range(remove_n):
                sel.pop(none_idx[i])
            sel += add
            return sel

        for lvl in ["paper", "modelcard", "dataset"]:
            selected = adjust_balance(selected, lvl, 0.5)

        # Final truncate
        np.random.shuffle(selected)
        selected = selected[:total_target]

        # ---- Print post-selection distributions ----
        print("\n[5/5] Post-selection statistics:")
        # 8-way on selected
        sel_combo_counts = Counter(tuple(p["combination"]) for p in selected)
        total_sel = len(selected)
        print("  8-Way (selected):")
        for c in sorted(sel_combo_counts.keys()):
            name = {
                (0,0,0): "None",
                (1,0,0): "Paper only",
                (0,1,0): "ModelCard only",
                (0,0,1): "Dataset only",
                (1,1,0): "Paper + ModelCard",
                (1,0,1): "Paper + Dataset",
                (0,1,1): "ModelCard + Dataset",
                (1,1,1): "All three",
            }[c]
            cnt = sel_combo_counts[c]
            pct = 100.0 * cnt / max(1, total_sel)
            print(f"    {c}: {name:20s} - {cnt:4d} ({pct:5.2f}%)")

        # Per-level
        print("\n  Per-level (selected):")
        for lvl in ["paper", "modelcard", "dataset"]:
            pos = sum(1 for p in selected if p["labels"][lvl] == 1)
            neg = total_sel - pos
            ppos = 100.0 * pos / max(1, total_sel)
            pneg = 100.0 * neg / max(1, total_sel)
            print(f"    {lvl.title():10s}: {pos:4d} pos ({ppos:5.2f}%) / {neg:4d} neg ({pneg:5.2f}%)")
        # Build outputs
        samples = {"all": selected}
        cross_stats = {
            "pool_size": len(all_pairs),
            "final_unique_pairs": len(selected),
            "combination_counts": {str(k): len(v) for k, v in by_combo.items()}
        }
        return samples, cross_stats
    def save_samples(self, samples, prefix):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{prefix}_unique_pairs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in samples["all"]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\n✓ Saved {len(samples['all'])} pairs → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", default="data/gt")
    parser.add_argument("--output-dir", default="output/gpt_evaluation")
    parser.add_argument("--n-samples-pool", type=int, default=1000000)
    parser.add_argument("--total-target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="table_1M_8way_rarefirst")
    args = parser.parse_args()

    sampler = TableSamplerV2(args.gt_dir, args.output_dir, args.seed)
    sampler.load_all_levels()
    samples, _ = sampler.sample_all_levels_unified(args.n_samples_pool, args.total_target)
    sampler.save_samples(samples, args.prefix)


if __name__ == "__main__":
    main()
