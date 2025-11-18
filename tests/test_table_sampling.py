#!/usr/bin/env python3
"""Quick test of table sampling with dataset level only"""

import sys
import os
sys.path.insert(0, 'src')

from gpt_evaluation.step1_table_sampling import TableSampler

# Test with small dataset matrix only
sampler = TableSampler(
    gt_dir="data/gt",
    output_dir="output/gpt_evaluation_test",
    seed=42
)

# Manually load only dataset level
print("Loading dataset level only for quick test...")
from gpt_evaluation.sparse_matrix_loader import SparseMatrixLoader
import pickle

dataset_matrix_path = "data/gt/scilake_gt_modellink_dataset_adj_processed.npz"
dataset_list_path = "data/gt/scilake_gt_modellink_dataset_adj_csv_list_processed.pkl"

sampler.dataset_loader = SparseMatrixLoader(
    dataset_matrix_path,
    cache_positive_pairs=False  # Build on demand
)

with open(dataset_list_path, 'rb') as f:
    sampler.dataset_list = pickle.load(f)

print(f"✓ Loaded dataset matrix: {sampler.dataset_loader.shape}")

# Sample only dataset level (small sample for testing)
print("\nSampling 10 pos + 10 neg pairs...")
samples = {}

if sampler.dataset_loader is not None:
    samples["dataset"] = sampler.sample_level(
        sampler.dataset_loader,
        sampler.dataset_list,
        "dataset",
        n_positive=10,
        n_negative=10
    )

# Save
if samples:
    sampler.save_samples(samples, prefix="test_table")
    report = sampler.generate_report(samples)
    print("\n" + report)

    # Show first few samples
    print("\n" + "="*60)
    print("SAMPLE PAIRS (first 3)")
    print("="*60)
    for pair in samples["dataset"][:3]:
        print(f"\nPair ID: {pair['id']}")
        print(f"  Positive: {pair['is_positive']}")
        print(f"  CSV A: {os.path.basename(pair['csv_a'])}")
        print(f"  CSV B: {os.path.basename(pair['csv_b'])}")
        print(f"  Value: {pair['relation_value']}")
