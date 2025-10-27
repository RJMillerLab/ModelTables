#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-27
Description: Model Relatedness Sampling for GPT Evaluation

This script samples model pairs based on model card metadata to evaluate
whether two models are related. Since we don't have ground truth for model
relatedness yet, this creates a diverse sample for human annotation.

Sampling strategy:
- Sample model pairs from deduplicated model cards
- Balance across different sources (hugging, github, html, llm)
- Ensure models have sufficient table information
- Apply ground truth construction filters (nonempty paper/csv lists)
- Random seed for reproducibility

Output:
- JSONL files with sampled model pairs
- Each pair contains: modelId, metadata, associated tables
- Sampling log with statistics

Usage:
    python src/gpt_evaluation/step1_model_sampling.py \
        --n-samples 200 \
        --seed 42 \
        --output-dir output/gpt_evaluation
"""

import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path


class ModelSampler:
    """
    Model pair sampler for model relatedness evaluation.

    Samples pairs of models based on their metadata and associated tables.
    Applies filtering logic similar to ground truth construction.
    """

    def __init__(self,
                 processed_dir: str = "data/processed",
                 output_dir: str = "output/gpt_evaluation",
                 seed: int = 42):
        """
        Initialize model sampler

        Args:
            processed_dir: Processed data directory
            output_dir: Output directory
            seed: Random seed
        """
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        self.seed = seed

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        # Data storage
        self.model_data = None
        self.model_metadata = None
        self.filtered_models = None

        # Table sources
        self.table_sources = ["hugging", "github", "html", "llm"]

        # Sampling log
        self.sampling_log = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "statistics": {}
        }

        print(f"✓ Initialized ModelSampler with seed={seed}")

    def load_model_data(self):
        """Load model data and metadata"""
        print("\n" + "="*60)
        print("LOADING MODEL DATA")
        print("="*60)

        # Load deduplicated model data (step3_dedup)
        model_file = os.path.join(self.processed_dir, "modelcard_step3_dedup.parquet")
        if os.path.exists(model_file):
            self.model_data = pd.read_parquet(model_file)
            print(f"  ✓ Loaded {len(self.model_data)} deduplicated models")

            self.sampling_log["initial_model_count"] = len(self.model_data)
        else:
            print(f"  ✗ Missing: {model_file}")
            raise FileNotFoundError(f"Model data not found: {model_file}")

        # Load model metadata (step1) if available
        metadata_file = os.path.join(self.processed_dir, "modelcard_step1.parquet")
        if os.path.exists(metadata_file):
            self.model_metadata = pd.read_parquet(metadata_file)
            print(f"  ✓ Loaded {len(self.model_metadata)} model metadata records")
        else:
            print(f"  ℹ No metadata file found (optional)")

        print("="*60)

    def apply_filters(self):
        """
        Apply ground truth construction filters to ensure quality samples

        Filters applied (similar to modelcard_matrix.py):
        1. Nonempty paper lists
        2. Nonempty CSV lists (at least one source has tables)
        3. Valid modelId
        """
        print("\n" + "="*60)
        print("APPLYING FILTERS (Ground Truth Construction Logic)")
        print("="*60)

        if self.model_data is None:
            raise ValueError("Model data not loaded")

        original_count = len(self.model_data)
        df = self.model_data.copy()

        # Filter 1: Valid modelId
        df = df[df['modelId'].notna() & (df['modelId'] != '')]
        print(f"  After valid modelId filter: {len(df)} models ({original_count - len(df)} removed)")

        # Filter 2: Nonempty paper lists
        if 'paper_list' in df.columns:
            df = df[df['paper_list'].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )]
            print(f"  After nonempty paper filter: {len(df)} models")

        # Filter 3: At least one table source has tables
        has_tables = df.apply(self._has_any_tables, axis=1)
        df = df[has_tables]
        print(f"  After nonempty table filter: {len(df)} models")

        self.filtered_models = df
        self.sampling_log["filtered_model_count"] = len(df)
        self.sampling_log["filter_removal_count"] = original_count - len(df)

        print(f"\n  ✓ Final filtered models: {len(df)} ({len(df)/original_count*100:.1f}% retained)")
        print("="*60)

    def _has_any_tables(self, row) -> bool:
        """Check if model has tables in any source"""
        for source in self.table_sources:
            col_name = f"{source}_table_list_dedup"
            if col_name in row.index:
                table_list = row[col_name]
                if isinstance(table_list, list) and len(table_list) > 0:
                    return True
        return False

    def _get_table_count(self, row) -> int:
        """Get total table count across all sources"""
        total = 0
        for source in self.table_sources:
            col_name = f"{source}_table_list_dedup"
            if col_name in row.index:
                table_list = row[col_name]
                if isinstance(table_list, list):
                    total += len(table_list)
        return total

    def sample_model_pairs(self, n_samples: int = 200) -> List[Dict[str, Any]]:
        """
        Sample model pairs for relatedness evaluation

        Args:
            n_samples: Number of model pairs to sample

        Returns:
            List of sampled model pairs
        """
        print("\n" + "="*60)
        print(f"SAMPLING {n_samples} MODEL PAIRS")
        print("="*60)

        if self.filtered_models is None:
            raise ValueError("Must apply filters before sampling")

        models = self.filtered_models

        if len(models) < 2:
            raise ValueError("Not enough models to sample pairs")

        # Calculate available pairs
        max_pairs = len(models) * (len(models) - 1) // 2
        if n_samples > max_pairs:
            print(f"  ⚠ Requested {n_samples} pairs but only {max_pairs} available")
            n_samples = max_pairs

        pairs = []
        sampled_pairs_set = set()

        print(f"  Sampling from {len(models)} models...")

        attempts = 0
        max_attempts = n_samples * 10

        while len(pairs) < n_samples and attempts < max_attempts:
            # Sample two different models
            idx_a, idx_b = self.rng.choice(len(models), size=2, replace=False)

            # Create normalized pair key
            pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))

            if pair_key in sampled_pairs_set:
                attempts += 1
                continue

            sampled_pairs_set.add(pair_key)

            # Get model records
            model_a = models.iloc[idx_a]
            model_b = models.iloc[idx_b]

            # Extract table information
            tables_a = self._extract_tables(model_a)
            tables_b = self._extract_tables(model_b)

            # Create pair record
            pair = {
                "id": f"model_pair_{len(pairs)+1}",
                "model_a_id": str(model_a['modelId']),
                "model_b_id": str(model_b['modelId']),
                "model_a_tables": tables_a,
                "model_b_tables": tables_b,
                "model_a_table_count": len(tables_a),
                "model_b_table_count": len(tables_b),
                "model_a_paper_count": len(model_a.get('paper_list', [])),
                "model_b_paper_count": len(model_b.get('paper_list', [])),
            }

            # Add optional metadata if available
            if 'tags' in model_a.index:
                pair["model_a_tags"] = model_a.get('tags', [])
            if 'tags' in model_b.index:
                pair["model_b_tags"] = model_b.get('tags', [])

            pairs.append(pair)
            attempts += 1

        print(f"  ✓ Sampled {len(pairs)} pairs in {attempts} attempts")

        # Statistics
        self.sampling_log["sampled_pairs"] = len(pairs)
        self.sampling_log["sampling_attempts"] = attempts

        # Analyze distribution
        table_counts_a = [p["model_a_table_count"] for p in pairs]
        table_counts_b = [p["model_b_table_count"] for p in pairs]

        print(f"\n  Statistics:")
        print(f"    Avg tables per model A: {np.mean(table_counts_a):.1f}")
        print(f"    Avg tables per model B: {np.mean(table_counts_b):.1f}")
        print(f"    Total unique models: {len(set([p['model_a_id'] for p in pairs] + [p['model_b_id'] for p in pairs]))}")

        print("="*60)

        return pairs

    def _extract_tables(self, model_row) -> List[str]:
        """Extract all table paths from a model record"""
        tables = []
        for source in self.table_sources:
            col_name = f"{source}_table_list_dedup"
            if col_name in model_row.index:
                table_list = model_row[col_name]
                if isinstance(table_list, list):
                    tables.extend(table_list)
        return tables

    def save_samples(self, samples: List[Dict[str, Any]], prefix: str = "model"):
        """
        Save sampled pairs to files

        Args:
            samples: List of sampled pairs
            prefix: Output file prefix
        """
        print(f"\n{'='*60}")
        print(f"SAVING SAMPLES")
        print(f"{'='*60}")

        os.makedirs(self.output_dir, exist_ok=True)

        # Save pairs
        output_path = os.path.join(self.output_dir, f"{prefix}_pairs.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in samples:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        print(f"  ✓ Saved {len(samples)} pairs to: {output_path}")

        # Save sampling log
        log_path = os.path.join(self.output_dir, f"{prefix}_sampling_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.sampling_log, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved sampling log to: {log_path}")

    def generate_report(self, samples: List[Dict[str, Any]]) -> str:
        """Generate sampling report"""
        lines = []
        lines.append("="*60)
        lines.append("MODEL RELATEDNESS SAMPLING REPORT")
        lines.append("="*60)
        lines.append(f"Timestamp: {self.sampling_log['timestamp']}")
        lines.append(f"Random Seed: {self.seed}")
        lines.append("")

        lines.append("STATISTICS")
        lines.append("-" * 40)
        lines.append(f"  Initial models: {self.sampling_log['initial_model_count']}")
        lines.append(f"  Filtered models: {self.sampling_log['filtered_model_count']}")
        lines.append(f"  Removed by filters: {self.sampling_log['filter_removal_count']}")
        lines.append(f"  Sampled pairs: {len(samples)}")
        lines.append("")

        # Analyze table distribution
        table_counts_a = [p["model_a_table_count"] for p in samples]
        table_counts_b = [p["model_b_table_count"] for p in samples]
        all_table_counts = table_counts_a + table_counts_b

        lines.append("TABLE DISTRIBUTION")
        lines.append("-" * 40)
        lines.append(f"  Avg tables per model: {np.mean(all_table_counts):.1f}")
        lines.append(f"  Median tables per model: {np.median(all_table_counts):.1f}")
        lines.append(f"  Max tables per model: {np.max(all_table_counts)}")
        lines.append(f"  Min tables per model: {np.min(all_table_counts)}")
        lines.append("")

        # Unique models
        unique_models = set()
        for p in samples:
            unique_models.add(p['model_a_id'])
            unique_models.add(p['model_b_id'])

        lines.append(f"  Total unique models in sample: {len(unique_models)}")
        lines.append("")

        lines.append("="*60)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Model Relatedness Sampling for GPT Evaluation"
    )

    parser.add_argument("--processed-dir", default="data/processed",
                       help="Processed data directory")
    parser.add_argument("--output-dir", default="output/gpt_evaluation",
                       help="Output directory")
    parser.add_argument("--n-samples", type=int, default=200,
                       help="Number of model pairs to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--prefix", default="model",
                       help="Output file prefix")

    args = parser.parse_args()

    # Initialize sampler
    sampler = ModelSampler(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # Load model data
    sampler.load_model_data()

    # Apply filters
    sampler.apply_filters()

    # Sample pairs
    samples = sampler.sample_model_pairs(n_samples=args.n_samples)

    # Save samples
    sampler.save_samples(samples, prefix=args.prefix)

    # Generate and print report
    report = sampler.generate_report(samples)
    print("\n" + report)

    # Save report
    report_path = os.path.join(args.output_dir, f"{args.prefix}_sampling_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ Saved report to: {report_path}")


if __name__ == "__main__":
    main()
