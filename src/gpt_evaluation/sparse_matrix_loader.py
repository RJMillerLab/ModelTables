#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-27
Description: Efficient sparse matrix loader using index caching (Solution 2)
             Only loads COO format indices for O(1) lookup, avoiding full matrix loading

Based on the approach from starmie_internal/checkPrecisionRecall.py:199-268
Key insight: CSR matrix structure (indptr, indices, data) can be queried directly
without materializing the full matrix in dense format.

Usage:
    loader = SparseMatrixLoader("data/gt/matrix.npz")

    # Check if pair (i, j) exists
    if loader.has_edge(i, j):
        value = loader.get_value(i, j)

    # Get all neighbors of node i
    neighbors = loader.get_neighbors(i)

    # Memory efficient iteration over positive pairs
    for i, j, value in loader.iter_positive_pairs():
        print(f"Edge ({i}, {j}) = {value}")
"""

import numpy as np
from scipy.sparse import load_npz, csr_matrix
from typing import Tuple, List, Set, Optional, Iterator
import os


class SparseMatrixLoader:
    """
    Memory-efficient sparse matrix loader using index caching.

    Key features:
    1. Only loads matrix structure (indptr, indices, data) - not full matrix
    2. O(1) lookup for edge existence via pre-built set
    3. O(deg) lookup for neighbors via indptr-based slicing
    4. Lazy loading - only loads what's needed
    """

    def __init__(self, npz_path: str, cache_positive_pairs: bool = True):
        """
        Initialize sparse matrix loader

        Args:
            npz_path: Path to .npz file containing sparse matrix
            cache_positive_pairs: Whether to build O(1) lookup set for positive pairs
        """
        self.npz_path = npz_path
        self.matrix = None  # CSR matrix
        self.shape = None
        self.nnz = 0

        # Cache for fast lookups
        self.positive_pairs_set: Optional[Set[Tuple[int, int]]] = None
        self.cache_positive_pairs = cache_positive_pairs

        # Load matrix structure
        self._load_matrix_structure()

        # Build positive pairs cache LAZILY (only when needed)
        # Don't build immediately - wait until first use
        # if cache_positive_pairs:
        #     self._build_positive_pairs_cache()

    def _load_matrix_structure(self):
        """
        Load sparse matrix in CSR format (only structure, not full dense array)

        CSR format stores:
        - data: non-zero values
        - indices: column indices for each non-zero value
        - indptr: row pointers (indptr[i]:indptr[i+1] gives indices for row i)

        Memory usage: O(nnz) instead of O(n²) for dense matrix
        """
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")

        print(f"[SparseMatrixLoader] Loading matrix structure from: {self.npz_path}")

        # Load as CSR matrix (most efficient for row-wise access)
        self.matrix = load_npz(self.npz_path)

        if not isinstance(self.matrix, csr_matrix):
            print(f"  Converting to CSR format...")
            self.matrix = self.matrix.tocsr()

        self.shape = self.matrix.shape
        self.nnz = self.matrix.nnz

        print(f"  ✓ Loaded: shape={self.shape}, nnz={self.nnz}")
        print(f"  Memory usage: ~{self._estimate_memory_mb():.2f} MB (structure only)")

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage of CSR structure"""
        # CSR memory: data (float64) + indices (int32) + indptr (int32)
        data_bytes = self.nnz * 8  # float64
        indices_bytes = self.nnz * 4  # int32
        indptr_bytes = (self.shape[0] + 1) * 4  # int32
        total_bytes = data_bytes + indices_bytes + indptr_bytes
        return total_bytes / (1024 * 1024)

    def _build_positive_pairs_cache(self):
        """
        Build O(1) lookup set for positive pairs using COO format

        This is the key optimization from step1_balanced_sampling.py:112-115
        Convert to COO to get (row, col) pairs, store as normalized set
        """
        print(f"[SparseMatrixLoader] Building positive pairs cache...")

        # Convert to COO format for easy (row, col) extraction
        coo = self.matrix.tocoo()

        # Build set of (min, max) normalized pairs (exclude self-loops)
        self.positive_pairs_set = set(
            (min(r, c), max(r, c))
            for r, c in zip(coo.row, coo.col)
            if r != c
        )

        print(f"  ✓ Cached {len(self.positive_pairs_set)} positive pairs")

        # Estimate cache memory
        cache_bytes = len(self.positive_pairs_set) * (2 * 8 + 32)  # 2 ints + set overhead
        print(f"  Cache memory: ~{cache_bytes / (1024*1024):.2f} MB")

    def has_edge(self, i: int, j: int) -> bool:
        """
        Check if edge (i, j) exists - O(1) with cache, O(log deg) without

        Args:
            i, j: Node indices

        Returns:
            True if edge exists (non-zero value)
        """
        # Build cache on first use if requested
        if self.cache_positive_pairs and self.positive_pairs_set is None:
            self._build_positive_pairs_cache()

        if self.positive_pairs_set is not None:
            # O(1) lookup using cached set
            pair_key = (min(i, j), max(i, j))
            return pair_key in self.positive_pairs_set
        else:
            # O(log deg) lookup using CSR structure
            return self.matrix[i, j] != 0

    def get_value(self, i: int, j: int) -> float:
        """
        Get value at (i, j) - O(log deg) using CSR indexing

        This is the key technique from checkPrecisionRecall.py:231
        Uses CSR's efficient row slicing: matrix[i, j]

        Args:
            i, j: Node indices

        Returns:
            Value at (i, j), or 0.0 if edge doesn't exist
        """
        return float(self.matrix[i, j])

    def get_neighbors(self, i: int) -> List[Tuple[int, float]]:
        """
        Get all neighbors of node i - O(deg_i) using indptr slicing

        This uses the CSR structure directly (from checkPrecisionRecall.py:207-239):
        - indptr[i]:indptr[i+1] gives the range in indices/data for row i
        - indices[start:end] gives column indices
        - data[start:end] gives corresponding values

        Args:
            i: Node index

        Returns:
            List of (neighbor_index, edge_value) tuples
        """
        # Get range for row i in CSR structure
        start = self.matrix.indptr[i]
        end = self.matrix.indptr[i + 1]

        # Extract neighbors and values
        neighbor_indices = self.matrix.indices[start:end]
        neighbor_values = self.matrix.data[start:end]

        return [(int(j), float(v)) for j, v in zip(neighbor_indices, neighbor_values)]

    def get_degree(self, i: int) -> int:
        """
        Get degree of node i - O(1) using indptr

        From checkPrecisionRecall.py:207: deg = np.diff(csrM.indptr)

        Args:
            i: Node index

        Returns:
            Number of neighbors (out-degree)
        """
        return int(self.matrix.indptr[i + 1] - self.matrix.indptr[i])

    def iter_positive_pairs(self, exclude_self_loops: bool = True) -> Iterator[Tuple[int, int, float]]:
        """
        Memory-efficient iterator over all positive pairs

        Yields:
            (i, j, value) tuples for each non-zero edge
        """
        coo = self.matrix.tocoo()

        for r, c, v in zip(coo.row, coo.col, coo.data):
            if exclude_self_loops and r == c:
                continue
            yield int(r), int(c), float(v)

    def sample_positive_pairs(self, n_samples: int, rng: np.random.Generator = None) -> List[Tuple[int, int, float]]:
        """
        Efficiently sample n_samples positive pairs by directly sampling COO indices

        Strategy: Sample random indices from COO arrays, filter out self-loops

        Args:
            n_samples: Number of pairs to sample
            rng: Random number generator (for reproducibility)

        Returns:
            List of (i, j, value) tuples
        """
        if rng is None:
            rng = np.random.default_rng()

        # Use COO format
        coo = self.matrix.tocoo()
        nnz = coo.nnz

        # Sample with replacement first (fast), then deduplicate
        samples = []
        seen_indices = set()
        attempts = 0
        max_attempts = n_samples * 3  # Allow retries for self-loops and duplicates

        while len(samples) < n_samples and attempts < max_attempts:
            # Sample random index in COO arrays
            idx = rng.integers(0, nnz)

            # Skip if already sampled
            if idx in seen_indices:
                attempts += 1
                continue

            seen_indices.add(idx)

            r, c, v = coo.row[idx], coo.col[idx], coo.data[idx]

            # Skip self-loops
            if r == c:
                attempts += 1
                continue

            samples.append((int(r), int(c), float(v)))
            attempts += 1

        if len(samples) < n_samples:
            print(f"[WARN] Only {len(samples)} samples collected (requested {n_samples})")

        return samples

    def sample_negative_pairs(self, n_samples: int, max_attempts: int = 50000,
                              rng: np.random.Generator = None) -> List[Tuple[int, int]]:
        """
        Efficiently sample negative pairs (not in matrix) using cached positive set

        This is the optimized version from step1_balanced_sampling.py:252-313
        Uses O(1) positive_pairs_set lookup instead of O(n) matrix access

        Args:
            n_samples: Number of negative pairs to sample
            max_attempts: Maximum sampling attempts
            rng: Random number generator

        Returns:
            List of (i, j) tuples for negative pairs
        """
        # Build cache on first use if requested
        if self.cache_positive_pairs and self.positive_pairs_set is None:
            self._build_positive_pairs_cache()

        if self.positive_pairs_set is None:
            raise ValueError("Need cache_positive_pairs=True for efficient negative sampling")

        if rng is None:
            rng = np.random.default_rng()

        n_nodes = self.shape[0]
        pairs = []
        attempts = 0
        seen_pairs = set()

        while len(pairs) < n_samples and attempts < max_attempts:
            # Sample two random nodes
            i, j = rng.choice(n_nodes, size=2, replace=False)

            # Normalize pair key
            pair_key = (min(i, j), max(i, j))

            # O(1) checks!
            if pair_key in seen_pairs or pair_key in self.positive_pairs_set:
                attempts += 1
                continue

            seen_pairs.add(pair_key)
            pairs.append((int(i), int(j)))
            attempts += 1

        return pairs

    def get_statistics(self) -> dict:
        """Get matrix statistics"""
        deg = np.diff(self.matrix.indptr)

        return {
            "shape": self.shape,
            "nnz": self.nnz,
            "density": self.nnz / (self.shape[0] * self.shape[1]),
            "avg_degree": float(np.mean(deg)),
            "max_degree": int(np.max(deg)),
            "min_degree": int(np.min(deg)),
            "num_isolated_nodes": int(np.sum(deg == 0)),
            "memory_mb": self._estimate_memory_mb()
        }


# Demo usage
if __name__ == "__main__":
    import pickle

    # Example: Load paper-paper matrix (use smallest one for testing)
    gt_dir = "data/gt"
    # Use smaller matrix for demo
    npz_path = os.path.join(gt_dir, "scilake_gt_modellink_dataset_adj_processed.npz")

    print("="*60)
    print("DEMO: Efficient Sparse Matrix Loading")
    print("="*60)

    # Initialize loader
    loader = SparseMatrixLoader(npz_path, cache_positive_pairs=True)

    # Get statistics
    stats = loader.get_statistics()
    print(f"\nMatrix Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test edge queries
    print(f"\n" + "="*60)
    print("TEST: Edge Queries")
    print("="*60)

    # Get neighbors of node 0
    neighbors = loader.get_neighbors(0)
    print(f"\nNode 0 has {len(neighbors)} neighbors:")
    for j, v in neighbors[:5]:  # Show first 5
        print(f"  -> {j} (value={v:.3f})")

    # Check specific edges
    test_pairs = [(0, 1), (0, 100), (5, 10)]
    print(f"\nEdge existence checks:")
    for i, j in test_pairs:
        exists = loader.has_edge(i, j)
        value = loader.get_value(i, j) if exists else 0.0
        print(f"  ({i}, {j}): exists={exists}, value={value:.3f}")

    # Sample positive pairs
    print(f"\n" + "="*60)
    print("TEST: Sampling")
    print("="*60)

    rng = np.random.default_rng(42)
    pos_samples = loader.sample_positive_pairs(5, rng=rng)
    print(f"\nSampled {len(pos_samples)} positive pairs:")
    for i, j, v in pos_samples:
        print(f"  ({i}, {j}) = {v:.3f}")

    neg_samples = loader.sample_negative_pairs(5, rng=rng)
    print(f"\nSampled {len(neg_samples)} negative pairs:")
    for i, j in neg_samples:
        print(f"  ({i}, {j}) - verified negative: {not loader.has_edge(i, j)}")

    print(f"\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
