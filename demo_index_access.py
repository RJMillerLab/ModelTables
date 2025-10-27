#!/usr/bin/env python3
"""
Demo: How to use SparseMatrixLoader for on-demand index access
Based on checkPrecisionRecall.py's indptr-based querying approach
"""

import numpy as np
from src.gpt_evaluation.sparse_matrix_loader import SparseMatrixLoader

print("="*80)
print("DEMO: On-Demand Index Access to Large Sparse Matrices")
print("="*80)

# Load matrix (fast, no cache building)
loader = SparseMatrixLoader(
    "data/gt/scilake_gt_modellink_dataset_adj_processed.npz",
    cache_positive_pairs=False  # Don't build cache for fast loading
)

print(f"\n✓ Loaded {loader.shape[0]:,} × {loader.shape[1]:,} matrix with {loader.nnz:,} edges")

# ============================================================================
# Use Case 1: Check if specific pair exists
# ============================================================================
print("\n" + "="*80)
print("USE CASE 1: Check if specific pair (i, j) exists")
print("="*80)

test_pairs = [(0, 1), (10, 20), (100, 200)]

for i, j in test_pairs:
    exists = loader.has_edge(i, j)  # O(log deg) without cache
    value = loader.get_value(i, j)  # O(log deg)
    print(f"  Edge ({i:3d}, {j:3d}): exists={exists}, value={value:.1f}")

# ============================================================================
# Use Case 2: Get all neighbors of a node (like checkPrecisionRecall.py:231)
# ============================================================================
print("\n" + "="*80)
print("USE CASE 2: Get all neighbors of a node")
print("="*80)

node_idx = 10
neighbors = loader.get_neighbors(node_idx)  # O(deg) using indptr slicing
print(f"  Node {node_idx} has {len(neighbors)} neighbors:")
print(f"  First 10 neighbors:")
for j, val in neighbors[:10]:
    print(f"    -> {j} (value={val:.2f})")

# ============================================================================
# Use Case 3: Batch query multiple edges efficiently
# ============================================================================
print("\n" + "="*80)
print("USE CASE 3: Batch query multiple edges")
print("="*80)

# Query 1000 random pairs
rng = np.random.default_rng(42)
n_queries = 1000
query_pairs = [(rng.integers(0, loader.shape[0]), rng.integers(0, loader.shape[1]))
               for _ in range(n_queries)]

import time
start = time.time()
results = [(i, j, loader.has_edge(i, j), loader.get_value(i, j))
           for i, j in query_pairs]
elapsed = time.time() - start

positive_count = sum(1 for _, _, exists, _ in results if exists)
print(f"  Queried {n_queries} random pairs in {elapsed:.3f}s")
print(f"  Speed: {elapsed/n_queries*1000:.2f}ms per query")
print(f"  Found {positive_count} positive edges ({positive_count/n_queries*100:.1f}%)")

# ============================================================================
# Use Case 4: Iterate over neighbors like checkPrecisionRecall.py
# ============================================================================
print("\n" + "="*80)
print("USE CASE 4: Iterate pattern from checkPrecisionRecall.py:224-236")
print("="*80)

# Simulate the calcMetrics_idx pattern:
# for qi, cand in valid_items.items():
#     for k in range(max_k):
#         cj = cand[k]
#         if csrM[qi, cj]:  # <- This is O(log deg)

# Example: Check if top-10 candidates for query 5 are in ground truth
query_idx = 5
candidate_indices = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]

print(f"  Query: {query_idx}")
print(f"  Checking {len(candidate_indices)} candidates:")

hit_count = 0
for k, cand_idx in enumerate(candidate_indices, 1):
    is_hit = loader.has_edge(query_idx, cand_idx)
    if is_hit:
        value = loader.get_value(query_idx, cand_idx)
        print(f"    k={k}: candidate {cand_idx} is a HIT (value={value:.2f})")
        hit_count += 1

print(f"  Total hits: {hit_count}/{len(candidate_indices)}")

# ============================================================================
# Use Case 5: Get degree distribution (using indptr directly)
# ============================================================================
print("\n" + "="*80)
print("USE CASE 5: Get degree distribution (checkPrecisionRecall.py:207 pattern)")
print("="*80)

# From checkPrecisionRecall.py:207: deg = np.diff(csrM.indptr)
csrM = loader.matrix
deg = np.diff(csrM.indptr)

print(f"  Total nodes: {len(deg):,}")
print(f"  Non-isolated nodes: {np.sum(deg > 0):,}")
print(f"  Avg degree: {np.mean(deg):.2f}")
print(f"  Max degree: {np.max(deg):,}")
print(f"  Median degree: {np.median(deg):.2f}")

# Find high-degree nodes
top_k = 5
top_degree_nodes = np.argsort(-deg)[:top_k]
print(f"\n  Top {top_k} highest-degree nodes:")
for rank, node_idx in enumerate(top_degree_nodes, 1):
    print(f"    #{rank}: Node {node_idx} with degree {deg[node_idx]:,}")

# ============================================================================
# Use Case 6: Extract values for specific indices
# ============================================================================
print("\n" + "="*80)
print("USE CASE 6: Extract edge values for a list of pairs")
print("="*80)

# Example: You have sampled pairs from your balanced sampler
sampled_pairs = [
    (10, 69107),
    (20, 30),
    (100, 200),
    (0, 69107)
]

print(f"  Extracting values for {len(sampled_pairs)} sampled pairs:")
for i, j in sampled_pairs:
    value = loader.get_value(i, j)
    exists = value != 0.0
    print(f"    ({i:5d}, {j:6d}): value={value:.3f}, exists={exists}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: Why this approach is efficient")
print("="*80)
print("""
1. **No full matrix loading**: Only CSR structure (indptr, indices, data) loaded
   - Memory: O(nnz) instead of O(n²)
   - For this matrix: ~403 MB instead of ~64 GB (if dense)

2. **O(log deg) edge queries**: Using CSR's binary search on row indices
   - Fast for sparse matrices (typical deg << n)
   - From checkPrecisionRecall.py:231: csrM[qi, cj]

3. **O(1) degree queries**: Using indptr differences
   - From checkPrecisionRecall.py:207: deg = np.diff(csrM.indptr)

4. **O(deg) neighbor iteration**: Using indptr slicing
   - From checkPrecisionRecall.py:224-236 pattern

5. **Optional O(1) edge existence**: Build positive_pairs_set cache only when needed
   - Trade-off: Build time vs. query speed
   - Good for negative sampling (many queries)

Recommendation for your sampling:
- For balanced sampling (150 pos + 150 neg), use cache_positive_pairs=True
- Only for negative sampling, to get O(1) lookup
- Cache builds once, then reused for all negative queries
""")

print("="*80)
print("✓ Demo complete!")
print("="*80)
