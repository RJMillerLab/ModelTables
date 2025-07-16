# or + tr
python src/baseline1/table_retrieval_pipeline.py \
  build_faiss --emb_npz data/baseline/valid_tables_ori_tr_embeddings.npz \
  --output_index data/baseline/valid_tables_ori_tr.faiss
# or + str
python src/baseline1/table_retrieval_pipeline.py \
  build_faiss --emb_npz data/baseline/valid_tables_ori_str_embeddings.npz \
  --output_index data/baseline/valid_tables_ori_str.faiss
# or + tr + str
python src/baseline1/table_retrieval_pipeline.py \
  build_faiss --emb_npz data/baseline/valid_tables_mixed_embeddings.npz \
  --output_index data/baseline/valid_tables_mixed.faiss

