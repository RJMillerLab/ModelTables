# ori+tr
python src/baseline1/merge_embeddings_and_jsonl.py \
  data/baseline/valid_tables_embeddings.npz data/baseline/valid_tables_tr_embeddings.npz \
  --out_npz data/baseline/valid_tables_ori_tr_embeddings.npz \
  data/baseline/valid_tables.jsonl data/baseline/valid_tables_tr.jsonl \
  --out_jsonl data/baseline/valid_tables_ori_tr.jsonl
# ori+str
python src/baseline1/merge_embeddings_and_jsonl.py \
  data/baseline/valid_tables_embeddings.npz data/baseline/valid_tables_str_embeddings.npz \
  --out_npz data/baseline/valid_tables_ori_str_embeddings.npz \
  data/baseline/valid_tables.jsonl data/baseline/valid_tables_str.jsonl \
  --out_jsonl data/baseline/valid_tables_ori_str.jsonl
# ori+tr+str
python src/baseline1/merge_embeddings_and_jsonl.py \
  data/baseline/valid_tables_embeddings.npz data/baseline/valid_tables_tr_embeddings.npz data/baseline/valid_tables_str_embeddings.npz \
  --out_npz data/baseline/valid_tables_mixed_embeddings.npz \
  data/baseline/valid_tables.jsonl data/baseline/valid_tables_tr.jsonl data/baseline/valid_tables_str.jsonl \
  --out_jsonl data/baseline/valid_tables_mixed.jsonl

