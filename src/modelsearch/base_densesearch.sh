#!/usr/bin/env bash

python src/baseline1/build_modelcard_jsonl.py \
  --field card \
  --output_jsonl output/modelsearch/modelsearch_corpus.jsonl

python src/baseline1/table_retrieval_pipeline.py \
  encode \
  --jsonl output/modelsearch/modelsearch_corpus.jsonl \
  --model_name all-MiniLM-L6-v2 \
  --batch_size 256 \
  --output_npz output/modelsearch/modelsearch_embeddings.npz \
  --device cuda

python src/baseline1/table_retrieval_pipeline.py \
  build_faiss \
  --emb_npz output/modelsearch/modelsearch_embeddings.npz \
  --output_index output/modelsearch/modelsearch.faiss

python src/baseline1/table_retrieval_pipeline.py \
  search \
  --emb_npz output/modelsearch/modelsearch_embeddings.npz \
  --faiss_index output/modelsearch/modelsearch.faiss \
  --top_k 20 \
  --output_json output/modelsearch/modelsearch_neighbors.json
