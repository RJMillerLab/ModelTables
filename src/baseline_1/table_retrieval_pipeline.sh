# build corpus
python src/baseline_1/table_retrieval_pipeline.py \
filter --base_path data/processed \
        --mask_file data/analysis/all_valid_title_valid.txt \
        --output_jsonl data/baseline/valid_tables.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --device cuda

python src/baseline_1/table_retrieval_pipeline.py \
encode --jsonl data/baseline/valid_tables.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --batch_size 256 \
        --output_npz data/baseline/valid_tables_embeddings.npz \
        --device cuda

python src/baseline_1/table_retrieval_pipeline.py \
build_faiss --emb_npz data/baseline/valid_tables_embeddings.npz \
            --output_index data/baseline/valid_tables.faiss

python src/baseline_1/table_retrieval_pipeline.py \
search --emb_npz data/baseline/valid_tables_embeddings.npz \
        --faiss_index data/baseline/valid_tables.faiss \
        --top_k 11 \
        --output_json data/baseline/table_neighbors.json

python src/baseline_1/postprocess.py data/baseline/table_neighbors.json data/baseline/baseline1_dense.json
