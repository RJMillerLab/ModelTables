#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-27
Description: Table Relatedness Sampling - Multi-Level Balanced Sampling

Strategy:
1. Sample N random CSV pairs ONCE (e.g., 100k pairs)
2. Batch query ALL THREE GT matrices to get labels
3. Each pair has 3 labels: [paper_label, modelcard_label, dataset_label]
4. This creates 8 possible combinations (2^3):
   - (0,0,0): None
   - (1,0,0): Paper only
   - (0,1,0): ModelCard only
   - (0,0,1): Dataset only
   - (1,1,0): Paper + ModelCard
   - (1,0,1): Paper + Dataset
   - (0,1,1): ModelCard + Dataset
   - (1,1,1): All three
5. Filter to maintain balance PER GT LEVEL:
   - Paper GT: 50% positive (label=1) / 50% negative (label=0)
   - ModelCard GT: 50% positive / 50% negative
   - Dataset GT: 50% positive / 50% negative

Note: A pair can be positive in one GT but negative in another!

Usage:
    python src/gpt_evaluation/step1_table_sampling_v2.py \
        --n-samples 100000 \
        --target-positive 150 \
        --target-negative 150 \
        --seed 42
"""

import os
import json
import pickle
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_evaluation.sparse_matrix_loader import SparseMatrixLoader


class TableSamplerV2:
    """
    Fast table pair sampler using batch query + filter strategy
    """

    def __init__(self,
                 gt_dir: str = "data/gt",
                 output_dir: str = "output/gpt_evaluation",
                 seed: int = 42):
        self.gt_dir = gt_dir
        self.output_dir = output_dir
        self.seed = seed

        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Loaders and lists
        self.loaders = {}
        self.csv_lists = {}
        self.csv_to_idx = {}  # O(1) lookup: csv_path -> index in each GT

        # Sampling log
        self.sampling_log = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "levels": {}
        }

        print(f"✓ Initialized TableSamplerV2 with seed={seed}")

    def load_gt_level(self, level_name: str, npz_path: str, list_path: str):
        """Load one GT level"""
        if not os.path.exists(npz_path) or not os.path.exists(list_path):
            print(f"  ✗ Missing files for {level_name}")
            return False

        # Fast load (no cache)
        loader = SparseMatrixLoader(npz_path, cache_positive_pairs=False)

        with open(list_path, 'rb') as f:
            csv_list = pickle.load(f)

        self.loaders[level_name] = loader
        self.csv_lists[level_name] = csv_list

        # Build O(1) lookup dict
        self.csv_to_idx[level_name] = {csv: idx for idx, csv in enumerate(csv_list)}

        stats = loader.get_statistics()
        print(f"  ✓ {level_name}: shape={stats['shape']}, nnz={stats['nnz']:,}, index_map={len(self.csv_to_idx[level_name])}")

        return True

    def load_all_levels(self):
        """Load all three GT levels"""
        print("\n" + "="*60)
        print("LOADING GROUND TRUTH MATRICES")
        print("="*60)

        # Paper level
        print("\n[Level 1] Paper")
        self.load_gt_level(
            "paper",
            os.path.join(self.gt_dir, "csv_pair_matrix_direct_label.npz"),
            os.path.join(self.gt_dir, "csv_list_direct_label.pkl")
        )

        # ModelCard level
        print("\n[Level 2] ModelCard")
        self.load_gt_level(
            "modelcard",
            os.path.join(self.gt_dir, "scilake_gt_modellink_model_adj_processed.npz"),
            os.path.join(self.gt_dir, "scilake_gt_modellink_model_adj_csv_list_processed.pkl")
        )

        # Dataset level
        print("\n[Level 3] Dataset")
        self.load_gt_level(
            "dataset",
            os.path.join(self.gt_dir, "scilake_gt_modellink_dataset_adj_processed.npz"),
            os.path.join(self.gt_dir, "scilake_gt_modellink_dataset_adj_csv_list_processed.pkl")
        )

        print("="*60)

    def sample_and_query_level(self,
                               level_name: str,
                               n_samples: int,
                               target_positive: int,
                               target_negative: int) -> List[Dict[str, Any]]:
        """
        Sample random pairs, batch query GT, then filter for balance

        Args:
            level_name: Level name
            n_samples: Number of random pairs to sample
            target_positive: Target number of positive pairs
            target_negative: Target number of negative pairs

        Returns:
            List of balanced pairs
        """
        print(f"\n{'='*60}")
        print(f"SAMPLING LEVEL: {level_name.upper()}")
        print(f"{'='*60}")

        if level_name not in self.loaders:
            print(f"  ✗ Level not loaded")
            return []

        loader = self.loaders[level_name]
        csv_list = self.csv_lists[level_name]
        n_nodes = loader.shape[0]

        print(f"  Strategy: Sample {n_samples} random pairs → batch query → filter to {target_positive}+{target_negative}")

        # Step 1: Sample random pair indices
        print(f"\n  [1/3] Sampling {n_samples} random pair indices...")
        pair_indices = []
        seen_pairs = set()

        while len(pair_indices) < n_samples:
            i, j = self.rng.choice(n_nodes, size=2, replace=False)
            pair_key = (min(i, j), max(i, j))

            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                pair_indices.append((i, j))

        print(f"    ✓ Sampled {len(pair_indices)} unique pairs")

        # Step 2: Batch query GT matrix (FAST - O(n_samples * log deg))
        print(f"\n  [2/3] Batch querying GT matrix...")
        positive_pairs = []
        negative_pairs = []

        for i, j in pair_indices:
            value = loader.get_value(i, j)

            pair_dict = {
                "csv_a": csv_list[i],
                "csv_b": csv_list[j],
                "csv_a_idx": int(i),
                "csv_b_idx": int(j),
                "relation_value": float(value)
            }

            if value > 0:
                positive_pairs.append(pair_dict)
            else:
                negative_pairs.append(pair_dict)

        print(f"    ✓ Found {len(positive_pairs)} positive, {len(negative_pairs)} negative")

        # Step 3: Filter to maintain balance
        print(f"\n  [3/3] Filtering for balance...")

        # Shuffle before selecting
        self.rng.shuffle(positive_pairs)
        self.rng.shuffle(negative_pairs)

        # Select target amounts
        selected_positive = positive_pairs[:target_positive]
        selected_negative = negative_pairs[:target_negative]

        print(f"    ✓ Selected {len(selected_positive)} positive, {len(selected_negative)} negative")

        # Add metadata
        final_pairs = []

        for idx, pair in enumerate(selected_positive):
            pair["id"] = f"{level_name}_pos_{idx+1}"
            pair["level"] = level_name
            pair["is_positive"] = True
            final_pairs.append(pair)

        for idx, pair in enumerate(selected_negative):
            pair["id"] = f"{level_name}_neg_{idx+1}"
            pair["level"] = level_name
            pair["is_positive"] = False
            final_pairs.append(pair)

        # Shuffle final list
        self.rng.shuffle(final_pairs)

        # Log
        self.sampling_log["levels"][level_name] = {
            "n_sampled": n_samples,
            "n_positive_found": len(positive_pairs),
            "n_negative_found": len(negative_pairs),
            "n_positive_selected": len(selected_positive),
            "n_negative_selected": len(selected_negative),
            "total_selected": len(final_pairs)
        }

        print(f"\n  ✓ Total: {len(final_pairs)} balanced pairs")
        print("="*60)

        return final_pairs

    def sample_all_levels_unified(self,
                                   n_samples_pool: int = 100000,
                                   target_positive: int = 250,
                                   target_negative: int = 250,
                                   total_target: int = 500) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Unified sampling: sample from large pool, query all 3 GTs, filter per-level, then analyze

        Strategy:
        1. Sample large pool (e.g., 100k pairs) and query all GTs
        2. Filter to maintain per-level balance (150 pos + 150 neg per level)
        3. Analyze 8 combinations based on FINAL selected pairs (not pool)

        Returns:
            - Per-level balanced samples
            - Cross-level statistics (8 combinations) based on final selection
        """
        print(f"\n{'='*60}")
        print(f"UNIFIED MULTI-LEVEL SAMPLING")
        print(f"{'='*60}")
        print(f"  Strategy: Sample {n_samples_pool} pairs pool → query 3 GTs → filter per-level → analyze")

        # Get common CSV list (intersection of all three levels)
        if not all(level in self.csv_lists for level in ["paper", "modelcard", "dataset"]):
            raise ValueError("All three levels must be loaded")

        # Use paper CSV list as base (largest)
        csv_list = self.csv_lists["paper"]
        n_nodes = len(csv_list)

        print(f"  Using {n_nodes} CSVs from Paper level as base")

        # Step 1: Sample random pairs from large pool
        print(f"\n  [1/4] Sampling {n_samples_pool} random pair indices (pool)...")
        pair_indices = []
        seen_pairs = set()

        while len(pair_indices) < n_samples_pool:
            i, j = self.rng.choice(n_nodes, size=2, replace=False)
            pair_key = (min(i, j), max(i, j))

            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                pair_indices.append((i, j))

        print(f"    ✓ Sampled {len(pair_indices)} unique pairs (pool)")

        # Step 2: Query all 3 GTs for each pair
        print(f"\n  [2/4] Batch querying ALL 3 GT matrices...")

        all_pairs = []
        for i, j in pair_indices:
            csv_a = csv_list[i]
            csv_b = csv_list[j]

            # Query paper GT (O(1) lookup with dict)
            paper_idx_a = self.csv_to_idx["paper"].get(csv_a, -1)
            paper_idx_b = self.csv_to_idx["paper"].get(csv_b, -1)
            paper_label = 0
            if paper_idx_a >= 0 and paper_idx_b >= 0:
                paper_label = 1 if self.loaders["paper"].get_value(paper_idx_a, paper_idx_b) > 0 else 0

            # Query modelcard GT (O(1) lookup with dict)
            model_idx_a = self.csv_to_idx["modelcard"].get(csv_a, -1)
            model_idx_b = self.csv_to_idx["modelcard"].get(csv_b, -1)
            model_label = 0
            if model_idx_a >= 0 and model_idx_b >= 0:
                model_label = 1 if self.loaders["modelcard"].get_value(model_idx_a, model_idx_b) > 0 else 0

            # Query dataset GT (O(1) lookup with dict)
            dataset_idx_a = self.csv_to_idx["dataset"].get(csv_a, -1)
            dataset_idx_b = self.csv_to_idx["dataset"].get(csv_b, -1)
            dataset_label = 0
            if dataset_idx_a >= 0 and dataset_idx_b >= 0:
                dataset_label = 1 if self.loaders["dataset"].get_value(dataset_idx_a, dataset_idx_b) > 0 else 0

            pair_dict = {
                "csv_a": csv_a,
                "csv_b": csv_b,
                "csv_a_idx": int(i),
                "csv_b_idx": int(j),
                "labels": {
                    "paper": paper_label,
                    "modelcard": model_label,
                    "dataset": dataset_label
                },
                "combination": (paper_label, model_label, dataset_label)
            }

            all_pairs.append(pair_dict)

        print(f"    ✓ Queried {len(all_pairs)} pairs across 3 GTs (pool)")

        # Step 3: Filter per-level for balance FIRST
        print(f"\n  [3/4] Filtering for per-level balance...")

        # Group by each level's label
        paper_positive = [p for p in all_pairs if p["labels"]["paper"] == 1]
        paper_negative = [p for p in all_pairs if p["labels"]["paper"] == 0]

        model_positive = [p for p in all_pairs if p["labels"]["modelcard"] == 1]
        model_negative = [p for p in all_pairs if p["labels"]["modelcard"] == 0]

        dataset_positive = [p for p in all_pairs if p["labels"]["dataset"] == 1]
        dataset_negative = [p for p in all_pairs if p["labels"]["dataset"] == 0]

        print(f"\n    Per-Level Distribution:")
        print(f"      Paper:     {len(paper_positive):5d} pos / {len(paper_negative):5d} neg")
        print(f"      ModelCard: {len(model_positive):5d} pos / {len(model_negative):5d} neg")
        print(f"      Dataset:   {len(dataset_positive):5d} pos / {len(dataset_negative):5d} neg")

        # Shuffle and select
        self.rng.shuffle(paper_positive)
        self.rng.shuffle(paper_negative)
        self.rng.shuffle(model_positive)
        self.rng.shuffle(model_negative)
        self.rng.shuffle(dataset_positive)
        self.rng.shuffle(dataset_negative)

        # Create per-level balanced sets
        paper_selected = paper_positive[:target_positive] + paper_negative[:target_negative]
        model_selected = model_positive[:target_positive] + model_negative[:target_negative]
        dataset_selected = dataset_positive[:target_positive] + dataset_negative[:target_negative]

        print(f"\n    ✓ Selected per level:")
        print(f"      Paper:     {len([p for p in paper_selected if p['labels']['paper']==1])} pos + {len([p for p in paper_selected if p['labels']['paper']==0])} neg")
        print(f"      ModelCard: {len([p for p in model_selected if p['labels']['modelcard']==1])} pos + {len([p for p in model_selected if p['labels']['modelcard']==0])} neg")
        print(f"      Dataset:   {len([p for p in dataset_selected if p['labels']['dataset']==1])} pos + {len([p for p in dataset_selected if p['labels']['dataset']==0])} neg")

        # Step 4: Select unique pairs with balanced combinations
        print(f"\n  [4/4] Selecting unique pairs with balanced 8-way combinations...")

        from collections import Counter, defaultdict

        combination_names = {
            (0, 0, 0): "None",
            (1, 0, 0): "Paper only",
            (0, 1, 0): "ModelCard only",
            (0, 0, 1): "Dataset only",
            (1, 1, 0): "Paper + ModelCard",
            (1, 0, 1): "Paper + Dataset",
            (0, 1, 1): "ModelCard + Dataset",
            (1, 1, 1): "All three"
        }

        # Group all selected pairs by combination
        pairs_by_combo = defaultdict(list)
        seen_pairs = set()

        for pair in paper_selected + model_selected + dataset_selected:
            key = (pair["csv_a"], pair["csv_b"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                pairs_by_combo[pair["combination"]].append(pair)

        print(f"\n    Available pairs by combination:")
        for combo in sorted(pairs_by_combo.keys()):
            name = combination_names[combo]
            print(f"      {combo}: {name:25s} - {len(pairs_by_combo[combo]):4d} pairs")

        # Keep some None (0,0,0) pairs but limit them
        # total_target comes from parameter (e.g., 500)
        none_target = min(100, len(pairs_by_combo.get((0, 0, 0), [])))  # Keep max 100 None pairs

        # First pass: Calculate initial target for other combinations
        other_combinations = [c for c in pairs_by_combo.keys() if c != (0, 0, 0)]
        n_other = len(other_combinations)
        remaining_target = total_target - none_target
        initial_target_per_combo = remaining_target // n_other if n_other > 0 else 0

        print(f"\n    Target distribution (initial):")
        print(f"      None (0,0,0): {none_target} pairs")
        print(f"      Other combinations: ~{initial_target_per_combo} pairs each")
        print(f"      Total target: {total_target} pairs")

        # Select from each combination, tracking shortfall
        all_selected_pairs = {}
        total_selected = 0
        shortfall = 0  # Track how many pairs we're short of target

        for combo, pairs in pairs_by_combo.items():
            self.rng.shuffle(pairs)

            if combo == (0, 0, 0):
                target = none_target
            else:
                target = initial_target_per_combo

            selected = pairs[:min(target, len(pairs))]

            # Track shortfall if this combination doesn't have enough pairs
            if len(selected) < target:
                shortfall += target - len(selected)

            for pair in selected:
                key = (pair["csv_a"], pair["csv_b"])
                all_selected_pairs[key] = pair

            total_selected += len(selected)

        # Second pass: Redistribute shortfall to combinations with extra pairs
        if shortfall > 0 and total_selected < total_target:
            print(f"\n    Redistributing {shortfall} pairs from shortfall...")

            for combo, pairs in pairs_by_combo.items():
                if combo == (0, 0, 0):
                    continue  # Don't add more None pairs

                # How many pairs did we already select?
                already_selected = len([p for p in all_selected_pairs.values() if p["combination"] == combo])
                available = len(pairs)

                # Can we add more from this combination?
                can_add = available - already_selected
                if can_add > 0:
                    # Add more pairs up to shortfall
                    to_add = min(can_add, shortfall)
                    additional_pairs = pairs[already_selected:already_selected + to_add]

                    for pair in additional_pairs:
                        key = (pair["csv_a"], pair["csv_b"])
                        if key not in all_selected_pairs:
                            all_selected_pairs[key] = pair
                            shortfall -= 1
                            total_selected += 1

                            if shortfall <= 0 or total_selected >= total_target:
                                break

                if shortfall <= 0 or total_selected >= total_target:
                    break

        # Count final combinations
        combination_counts = Counter(pair["combination"] for pair in all_selected_pairs.values())

        print(f"\n    Final unique pairs selected: {len(all_selected_pairs)}")
        print(f"\n    Final 8-Way Combinations:")

        for combo in sorted(combination_counts.keys()):
            count = combination_counts[combo]
            name = combination_names.get(combo, str(combo))
            pct = 100 * count / len(all_selected_pairs)
            print(f"      {combo}: {name:25s} - {count:4d} pairs ({pct:5.2f}%)")

        # Now build per-level results from unique pairs
        # Each unique pair appears in its relevant levels
        paper_final = []
        model_final = []
        dataset_final = []

        for idx, pair in enumerate(all_selected_pairs.values()):
            # Paper level
            if pair["labels"]["paper"] == 1 or pair["labels"]["paper"] == 0:
                pair_copy = pair.copy()
                pair_copy["id"] = f"paper_{len(paper_final)+1}"
                pair_copy["level"] = "paper"
                pair_copy["is_positive"] = pair["labels"]["paper"] == 1
                paper_final.append(pair_copy)

            # ModelCard level
            if pair["labels"]["modelcard"] == 1 or pair["labels"]["modelcard"] == 0:
                pair_copy = pair.copy()
                pair_copy["id"] = f"modelcard_{len(model_final)+1}"
                pair_copy["level"] = "modelcard"
                pair_copy["is_positive"] = pair["labels"]["modelcard"] == 1
                model_final.append(pair_copy)

            # Dataset level
            if pair["labels"]["dataset"] == 1 or pair["labels"]["dataset"] == 0:
                pair_copy = pair.copy()
                pair_copy["id"] = f"dataset_{len(dataset_final)+1}"
                pair_copy["level"] = "dataset"
                pair_copy["is_positive"] = pair["labels"]["dataset"] == 1
                dataset_final.append(pair_copy)

        # Shuffle each level
        self.rng.shuffle(paper_final)
        self.rng.shuffle(model_final)
        self.rng.shuffle(dataset_final)

        results = {
            "paper": paper_final,
            "modelcard": model_final,
            "dataset": dataset_final
        }

        print(f"\n    Per-level pair counts (same unique pairs, different views):")
        print(f"      Paper:     {len(paper_final)} pairs")
        print(f"      ModelCard: {len(model_final)} pairs")
        print(f"      Dataset:   {len(dataset_final)} pairs")

        # Cross-level statistics (based on final unique pairs)
        cross_stats = {
            "pool_size": len(all_pairs),
            "final_unique_pairs": len(all_selected_pairs),
            "combination_counts": {combination_names[k]: v for k, v in combination_counts.items()},
            "per_level_pair_counts": {
                "paper": {
                    "positive": len([p for p in paper_final if p["is_positive"]]),
                    "negative": len([p for p in paper_final if not p["is_positive"]]),
                    "total": len(paper_final)
                },
                "modelcard": {
                    "positive": len([p for p in model_final if p["is_positive"]]),
                    "negative": len([p for p in model_final if not p["is_positive"]]),
                    "total": len(model_final)
                },
                "dataset": {
                    "positive": len([p for p in dataset_final if p["is_positive"]]),
                    "negative": len([p for p in dataset_final if not p["is_positive"]]),
                    "total": len(dataset_final)
                }
            }
        }

        # Log
        self.sampling_log["cross_level_statistics"] = cross_stats

        print("="*60)

        return results, cross_stats

    def save_samples(self, samples: Dict[str, List[Dict[str, Any]]], prefix: str = "table_v2"):
        """Save samples"""
        print(f"\n{'='*60}")
        print(f"SAVING SAMPLES")
        print(f"{'='*60}")

        os.makedirs(self.output_dir, exist_ok=True)

        # Save each level
        for level_name, pairs in samples.items():
            output_path = os.path.join(self.output_dir, f"{prefix}_{level_name}_pairs.jsonl")
            with open(output_path, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            print(f"  ✓ {level_name}: {len(pairs)} pairs → {output_path}")

        # Save combined
        combined_path = os.path.join(self.output_dir, f"{prefix}_all_levels_pairs.jsonl")
        all_pairs = [p for pairs in samples.values() for p in pairs]
        with open(combined_path, 'w', encoding='utf-8') as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        print(f"  ✓ Combined: {len(all_pairs)} pairs → {combined_path}")

        # Save log
        log_path = os.path.join(self.output_dir, f"{prefix}_sampling_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.sampling_log, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Log → {log_path}")

    def generate_report(self, samples: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate report with cross-level analysis"""
        lines = []
        lines.append("="*60)
        lines.append("TABLE RELATEDNESS SAMPLING REPORT")
        lines.append("Multi-Level Balanced Sampling")
        lines.append("="*60)
        lines.append(f"Timestamp: {self.sampling_log['timestamp']}")
        lines.append(f"Seed: {self.seed}")
        lines.append("")

        # Cross-level statistics
        if "cross_level_statistics" in self.sampling_log:
            stats = self.sampling_log["cross_level_statistics"]

            lines.append("CROSS-LEVEL COMBINATION ANALYSIS")
            lines.append("-" * 60)
            lines.append(f"Pool size sampled: {stats['pool_size']}")
            lines.append(f"Final unique pairs selected: {stats['final_unique_pairs']}")
            lines.append("")
            lines.append("8-Way Combinations (Paper, ModelCard, Dataset):")
            lines.append("(Based on final selected pairs, not pool)")

            for combo_name, count in sorted(stats["combination_counts"].items(),
                                           key=lambda x: x[1], reverse=True):
                pct = 100 * count / stats["final_unique_pairs"]
                lines.append(f"  {combo_name:30s}: {count:4d} ({pct:5.2f}%)")

            lines.append("")

        # Per-level balance
        lines.append("PER-LEVEL BALANCED SAMPLES")
        lines.append("-" * 60)

        for level_name, pairs in samples.items():
            n_pos = sum(1 for p in pairs if p["is_positive"])
            n_neg = len(pairs) - n_pos

            lines.append(f"{level_name.upper()}")
            lines.append(f"  Total: {len(pairs)}")
            lines.append(f"  Positive: {n_pos} ({100*n_pos/len(pairs):.1f}%)")
            lines.append(f"  Negative: {n_neg} ({100*n_neg/len(pairs):.1f}%)")
            lines.append("")

        lines.append("="*60)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Table Sampling V2 (Batch Query)")

    parser.add_argument("--gt-dir", default="data/gt")
    parser.add_argument("--output-dir", default="output/gpt_evaluation")
    parser.add_argument("--n-samples-pool", type=int, default=100000,
                       help="Size of random pairs pool to sample from")
    parser.add_argument("--target-positive", type=int, default=250,
                       help="Target positive pairs per level (for pool filtering)")
    parser.add_argument("--target-negative", type=int, default=250,
                       help="Target negative pairs per level (for pool filtering)")
    parser.add_argument("--total-target", type=int, default=500,
                       help="Total unique pairs to select (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="table")

    args = parser.parse_args()

    sampler = TableSamplerV2(
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )

    sampler.load_all_levels()

    samples, cross_stats = sampler.sample_all_levels_unified(
        n_samples_pool=args.n_samples_pool,
        target_positive=args.target_positive,
        target_negative=args.target_negative,
        total_target=args.total_target
    )

    sampler.save_samples(samples, prefix=args.prefix)

    report = sampler.generate_report(samples)
    print("\n" + report)

    report_path = os.path.join(args.output_dir, f"{args.prefix}_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")


if __name__ == "__main__":
    main()
