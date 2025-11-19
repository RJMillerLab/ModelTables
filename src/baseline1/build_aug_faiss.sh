#!/bin/bash
# Baseline1: Build FAISS index for mixed experiments
# Supports TAG environment variable for versioning (e.g., TAG=251117)

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

# or + tr
python src/baseline1/table_retrieval_pipeline.py \
  build_faiss --emb_npz data/baseline/valid_tables_ori_tr${SUFFIX}_embeddings.npz \
  --output_index data/baseline/valid_tables_ori_tr${SUFFIX}.faiss
# or + str
python src/baseline1/table_retrieval_pipeline.py \
  build_faiss --emb_npz data/baseline/valid_tables_ori_str${SUFFIX}_embeddings.npz \
  --output_index data/baseline/valid_tables_ori_str${SUFFIX}.faiss
# or + tr + str
python src/baseline1/table_retrieval_pipeline.py \
  build_faiss --emb_npz data/baseline/valid_tables_mixed${SUFFIX}_embeddings.npz \
  --output_index data/baseline/valid_tables_mixed${SUFFIX}.faiss

