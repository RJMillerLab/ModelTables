# GPT Evaluation Scripts

This directory contains scripts for sampling and evaluating table/model relatedness using GPT.

## Overview

We evaluate relatedness at three levels:
1. **Paper-Paper**: Based on citation relationships
2. **ModelCard-ModelCard**: Based on model relationships
3. **Dataset-Dataset**: Based on dataset relationships

## Scripts

### 1. Table Relatedness Sampling (`step1_table_sampling.py`)

Samples balanced table pairs for GPT evaluation with multi-level balance and 8-way combination analysis.

**Key Innovation**:
- Samples pairs once, queries all 3 GTs simultaneously
- Maintains per-level balance (50/50 pos/neg)
- Analyzes 8 label combinations of final selected pairs

**Quick Start**:
```bash
python src/gpt_evaluation/step1_table_sampling.py \
    --n-samples-pool 100000 \
    --target-positive 150 \
    --target-negative 150 \
    --seed 42
```

**Output Example**:
```
Final unique pairs: 864

8-Way Combinations:
  None (0,0,0):           417 (48.26%)
  Paper only (1,0,0):     175 (20.25%)
  Paper + Dataset:         94 (10.88%)
  All three (1,1,1):       90 (10.42%)
  ...

Per-Level Balance:
  Paper:     150 pos + 150 neg
  ModelCard: 150 pos + 150 neg
  Dataset:   150 pos + 150 neg
```

### 2. Model Relatedness Sampling (`step1_model_sampling.py`)

Samples model pairs for model relatedness evaluation.

```bash
python src/gpt_evaluation/step1_model_sampling.py --n-samples 200
```

### 3. Sparse Matrix Loader (`sparse_matrix_loader.py`)

Utility for efficient loading and querying of large sparse matrices (>700MB NPZ files).

**Features**:
- Memory efficient: O(nnz) instead of O(n²)
- Fast queries: O(log deg) edge queries
- Lazy cache building

**Example**:
```python
from sparse_matrix_loader import SparseMatrixLoader

loader = SparseMatrixLoader("matrix.npz", cache_positive_pairs=False)

# Fast queries
exists = loader.has_edge(i, j)      # O(log deg)
value = loader.get_value(i, j)      # O(log deg)
degree = loader.get_degree(i)       # O(1)
neighbors = loader.get_neighbors(i) # O(deg)
```

## Workflow

1. **Sample**: Use `step1_table_sampling.py` to sample balanced pairs
2. **Evaluate**: (TODO) Use GPT to evaluate sampled pairs
3. **Analyze**: (TODO) Compute inter-annotator agreement metrics

## File Structure

```
src/gpt_evaluation/
├── step1_table_sampling.py      # Table pair sampling (main)
├── step1_model_sampling.py      # Model pair sampling
├── sparse_matrix_loader.py      # Efficient sparse matrix utilities
├── load_table_content.py        # Table content loading utilities
└── README.md                    # This file
```

## Requirements

- Python 3.8+
- numpy
- scipy
- pandas

## Performance

- Load 700MB NPZ: ~1s
- Sample 900 balanced pairs: ~40s
- Memory: ~8GB (vs ~64GB if dense)

## References

See `docs/scripts.md` for detailed documentation.
