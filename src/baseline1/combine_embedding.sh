#!/bin/bash
# Baseline1: Combine embeddings for mixed experiments
# Supports TAG environment variable for versioning (e.g., TAG=251117)

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

# ori+tr
python src/baseline1/merge_embeddings_and_jsonl.py \
  data/baseline/valid_tables${SUFFIX}_embeddings.npz data/baseline/valid_tables_tr${SUFFIX}_embeddings.npz \
  --out_npz data/baseline/valid_tables_ori_tr${SUFFIX}_embeddings.npz \
  data/baseline/valid_tables${SUFFIX}.jsonl data/baseline/valid_tables_tr${SUFFIX}.jsonl \
  --out_jsonl data/baseline/valid_tables_ori_tr${SUFFIX}.jsonl
# ori+str
python src/baseline1/merge_embeddings_and_jsonl.py \
  data/baseline/valid_tables${SUFFIX}_embeddings.npz data/baseline/valid_tables_str${SUFFIX}_embeddings.npz \
  --out_npz data/baseline/valid_tables_ori_str${SUFFIX}_embeddings.npz \
  data/baseline/valid_tables${SUFFIX}.jsonl data/baseline/valid_tables_str${SUFFIX}.jsonl \
  --out_jsonl data/baseline/valid_tables_ori_str${SUFFIX}.jsonl
# ori+tr+str
python src/baseline1/merge_embeddings_and_jsonl.py \
  data/baseline/valid_tables${SUFFIX}_embeddings.npz data/baseline/valid_tables_tr${SUFFIX}_embeddings.npz data/baseline/valid_tables_str${SUFFIX}_embeddings.npz \
  --out_npz data/baseline/valid_tables_mixed${SUFFIX}_embeddings.npz \
  data/baseline/valid_tables${SUFFIX}.jsonl data/baseline/valid_tables_tr${SUFFIX}.jsonl data/baseline/valid_tables_str${SUFFIX}.jsonl \
  --out_jsonl data/baseline/valid_tables_mixed${SUFFIX}.jsonl

