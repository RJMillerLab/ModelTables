#!/bin/bash
# Baseline1: Dense Search Pipeline - TR MODE (transpose augmentation)
# Supports TAG environment variable for versioning (e.g., TAG=251117)

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

# build corpus - TR MODE (transpose augmentation)
python src/baseline1/table_retrieval_pipeline.py \
filter --base_path data/processed \
        --mask_file data/analysis/all_valid_title_valid${SUFFIX}.txt \
        --output_jsonl data/baseline/valid_tables_tr${SUFFIX}.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --device cuda \
        --mode tr

python src/baseline1/table_retrieval_pipeline.py \
encode --jsonl data/baseline/valid_tables_tr${SUFFIX}.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --batch_size 256 \
        --output_npz data/baseline/valid_tables_tr${SUFFIX}_embeddings.npz \
        --device cuda

# python src/baseline1/table_retrieval_pipeline.py \
# build_faiss --emb_npz data/baseline/valid_tables_tr_embeddings.npz \
#             --output_index data/baseline/valid_tables_tr.faiss

# python src/baseline1/table_retrieval_pipeline.py \
# search --emb_npz data/baseline/valid_tables_tr_embeddings.npz \
#         --faiss_index data/baseline/valid_tables_tr.faiss \
#         --top_k 11 \
#         --output_json data/baseline/table_neighbors_tr.json

# python src/baseline1/postprocess.py data/baseline/table_neighbors_tr.json data/baseline/baseline1_dense_tr.json 