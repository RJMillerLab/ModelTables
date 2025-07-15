# build corpus - STR MODE (string augmentation)
python src/baseline1/table_retrieval_pipeline.py \
filter --base_path data/processed \
        --mask_file data/analysis/all_valid_title_valid.txt \
        --output_jsonl data/baseline/valid_tables_str.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --device cuda \
        --mode str

python src/baseline1/table_retrieval_pipeline.py \
encode --jsonl data/baseline/valid_tables_str.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --batch_size 256 \
        --output_npz data/baseline/valid_tables_str_embeddings.npz \
        --device cuda

# python src/baseline1/table_retrieval_pipeline.py \
# build_faiss --emb_npz data/baseline/valid_tables_str_embeddings.npz \
#             --output_index data/baseline/valid_tables_str.faiss

# python src/baseline1/table_retrieval_pipeline.py \
# search --emb_npz data/baseline/valid_tables_str_embeddings.npz \
#         --faiss_index data/baseline/valid_tables_str.faiss \
#         --top_k 11 \
#         --output_json data/baseline/table_neighbors_str.json

# python src/baseline1/postprocess.py data/baseline/table_neighbors_str.json data/baseline/baseline1_dense_str.json 