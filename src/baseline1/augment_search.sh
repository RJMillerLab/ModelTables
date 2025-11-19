#!/bin/bash
# Baseline1: Search with mixed embeddings
# Supports TAG environment variable for versioning (e.g., TAG=251117)

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

# ori+tr
python src/baseline1/table_retrieval_pipeline.py \
  search --emb_npz data/baseline/valid_tables_ori_tr${SUFFIX}_embeddings.npz \
  --faiss_index data/baseline/valid_tables_ori_tr${SUFFIX}.faiss \
  --top_k 11 \
  --output_json data/baseline/table_neighbors_ori_tr${SUFFIX}.json

# ori+str
python src/baseline1/table_retrieval_pipeline.py \
  search --emb_npz data/baseline/valid_tables_ori_str${SUFFIX}_embeddings.npz \
  --faiss_index data/baseline/valid_tables_ori_str${SUFFIX}.faiss \
  --top_k 11 \
  --output_json data/baseline/table_neighbors_ori_str${SUFFIX}.json

# ori+tr+str
python src/baseline1/table_retrieval_pipeline.py \
  search --emb_npz data/baseline/valid_tables_mixed${SUFFIX}_embeddings.npz \
  --faiss_index data/baseline/valid_tables_mixed${SUFFIX}.faiss \
  --top_k 11 \
  --output_json data/baseline/table_neighbors_mixed${SUFFIX}.json

