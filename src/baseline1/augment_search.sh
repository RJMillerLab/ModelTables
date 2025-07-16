# ori+tr
python src/baseline1/table_retrieval_pipeline.py \
  search --emb_npz data/baseline/valid_tables_ori_tr_embeddings.npz \
  --faiss_index data/baseline/valid_tables_ori_tr.faiss \
  --top_k 11 \
  --output_json data/baseline/table_neighbors_ori_tr.json

# ori+str
python src/baseline1/table_retrieval_pipeline.py \
  search --emb_npz data/baseline/valid_tables_ori_str_embeddings.npz \
  --faiss_index data/baseline/valid_tables_ori_str.faiss \
  --top_k 11 \
  --output_json data/baseline/table_neighbors_ori_str.json

# ori+tr+str
python src/baseline1/table_retrieval_pipeline.py \
  search --emb_npz data/baseline/valid_tables_mixed_embeddings.npz \
  --faiss_index data/baseline/valid_tables_mixed.faiss \
  --top_k 11 \
  --output_json data/baseline/table_neighbors_mixed.json

