# hybrid search
# python src/baseline1/table_retrieval_pipeline.py \
# encode --jsonl data/tmp/corpus/collection.jsonl \
#         --model_name all-MiniLM-L6-v2 \
#         --batch_size 256 \
#         --output_npz data/tmp/index_dense/valid_tables_embeddings.npz \
#         --device cpu

# python -m pyserini.encode \
#   input   --corpus   data/tmp/corpus/collection_text.jsonl \
#           --fields   text \
#   output  --embeddings data/tmp/index_dense \
#           --to-faiss \
#   encoder --encoder sentence-transformers/all-MiniLM-L6-v2 \
#           --batch 64 --device cpu
# python src/baseline1/table_retrieval_pipeline.py \
# build_faiss \
#   --emb_npz     data/tmp/index_dense/valid_tables_embeddings.npz \
#   --output_index data/tmp/index_dense/index.faiss
# #python -m pyserini.search.lucene --index data/tmp/index --topics data/tmp/queries_table.tsv --output data/tmp/search_result_hybrid.txt --bm25 --hits 101 --threads 8 --batch-size 64
# python src/baseline2/search_with_pyserini_hybrid.py \
#   --sparse-index data/tmp/index \
#   --dense-index  data/tmp/index_dense \
#   --queries      data/tmp/queries_table.tsv \
#   --mapping      data/tmp/queries_table_mapping.json \
#   --k 11 --alpha 0.45 --device cpu

# first sparse, then dense
python src/baseline2/search_with_pyserini.py --hits 101 --output data/tmp/search_sparse_101.json
python src/baseline2/filter_sparse_to_dense.py data/tmp/search_sparse_101.json data/tmp/sparse_top101.tsv
# get subset jsonl
# python - <<'PY'
# import json, pathlib, sys
# subset = set(line.strip().split('\t')[1] for line in open(
#     'data/tmp/sparse_top101.tsv'))
# out   = pathlib.Path('data/tmp/corpus/subset.jsonl').open('w')
# for l in open('data/tmp/corpus/collection_text.jsonl'):
#     j = json.loads(l)
#     if j['id'] in subset:
#         out.write(l)
# PY
# add content keyword
#python scripts/make_subset_jsonl.py data/tmp/sparse_top101.tsv data/tmp/corpus/collection_text.jsonl data/tmp/corpus/subset.jsonl
# 2-a  Encode
python src/baseline1/table_retrieval_pipeline.py \
encode \
    --jsonl data/tmp/corpus/collection.jsonl \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --batch_size 256 \
    --output_npz data/tmp/index_dense/embeddings.npz \
    --device cuda 

# 2-b  Build Faiss index
#python src/baseline1/table_retrieval_pipeline.py build_faiss --emb_npz data/tmp/index_dense_subset/embeddings.npz --output_index data/tmp/index_dense_subset/index.faiss
# 2-c  Dense search (top-11 per query)
#python dense_rerank.py
python src/baseline2/hybrid_rerank.py \
     --sparse-json data/tmp/search_sparse_101.json \
     --dense-npz   data/tmp/index_dense/embeddings.npz \
     --corpus-jsonl data/tmp/corpus/collection.jsonl \
     --topk 11 --device cuda \
     --output data/tmp/search_result_hybrid.json

python src/baseline2/postprocess.py \
  --input  data/tmp/search_result_hybrid.json \
  --output data/tmp/baseline3_hybrid.json \
  --top1-list data/tmp/hybrid_selfhit_qids.txt

# check whether hybrid result is included in sparse 101 results
python src/baseline2/check_hybrid_vs_sparse.py \
  --hybrid   data/tmp/baseline3_hybrid.json \
  --sparse   data/tmp/search_sparse_101.json
#python src/baseline1/table_retrieval_pipeline.py search_faiss --faiss_index data/tmp/index_dense_subset/index.faiss --query_tsv   data/tmp/sparse_top101.tsv --topk 11 --output data/tmp/search_dense_11.json
