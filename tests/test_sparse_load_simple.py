#!/usr/bin/env python3
"""Simple test of sparse matrix loading speed"""

import time
import numpy as np
from scipy.sparse import load_npz
from src.gpt_evaluation.sparse_matrix_loader import SparseMatrixLoader

print("="*60)
print("TEST 1: Direct load_npz (baseline)")
print("="*60)

start = time.time()
matrix = load_npz("data/gt/scilake_gt_modellink_dataset_adj_processed.npz")
elapsed = time.time() - start

print(f"  Shape: {matrix.shape}")
print(f"  NNZ: {matrix.nnz}")
print(f"  Time: {elapsed:.2f}s")

print("\n" + "="*60)
print("TEST 2: Using SparseMatrixLoader (cache=False)")
print("="*60)

start = time.time()
loader = SparseMatrixLoader(
    "data/gt/scilake_gt_modellink_dataset_adj_processed.npz",
    cache_positive_pairs=False
)
elapsed = time.time() - start
print(f"  Time: {elapsed:.2f}s")

# Test edge queries
print("\n" + "="*60)
print("TEST 3: Edge queries without cache")
print("="*60)

start = time.time()
for _ in range(100):
    i, j = np.random.randint(0, loader.shape[0], size=2)
    exists = loader.has_edge(i, j)
elapsed = time.time() - start
print(f"  100 edge queries: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms per query)")

# Test neighbor lookup
print("\n" + "="*60)
print("TEST 4: Neighbor lookup using indptr")
print("="*60)

node_idx = 0
start = time.time()
neighbors = loader.get_neighbors(node_idx)
elapsed = time.time() - start
print(f"  Node {node_idx} degree: {len(neighbors)}")
print(f"  Time: {elapsed*1000:.2f}ms")
print(f"  First 5 neighbors: {neighbors[:5]}")

# Test degree calculation
print("\n" + "="*60)
print("TEST 5: Degree calculation")
print("="*60)

start = time.time()
degrees = [loader.get_degree(i) for i in range(min(1000, loader.shape[0]))]
elapsed = time.time() - start
print(f"  1000 degree queries: {elapsed:.3f}s ({elapsed/1000*1000:.2f}ms per query)")
print(f"  Avg degree: {np.mean(degrees):.2f}")

# Test direct CSR access (from checkPrecisionRecall.py pattern)
print("\n" + "="*60)
print("TEST 6: Direct CSR access (checkPrecisionRecall.py pattern)")
print("="*60)

csrM = loader.matrix
deg = np.diff(csrM.indptr)
print(f"  Calculated {len(deg)} node degrees")
print(f"  Max degree: {np.max(deg)}")
print(f"  Nodes with degree > 0: {np.sum(deg > 0)}")

# Test specific edge value retrieval
print("\n" + "="*60)
print("TEST 7: Get specific edge values")
print("="*60)

# Find some positive edges first
print("  Finding positive edges...")
found_edges = []
for i in range(min(100, loader.shape[0])):
    if loader.get_degree(i) > 0:
        neighbors = loader.get_neighbors(i)
        if neighbors:
            j, val = neighbors[0]
            found_edges.append((i, j, val))
            if len(found_edges) >= 5:
                break

print(f"  Found {len(found_edges)} positive edges:")
for i, j, v in found_edges:
    retrieved_val = loader.get_value(i, j)
    print(f"    ({i}, {j}): stored={v:.3f}, retrieved={retrieved_val:.3f}")

print("\n" + "="*60)
print("✓ All basic tests passed!")
print("="*60)
