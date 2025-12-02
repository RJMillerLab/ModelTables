#!/bin/bash
# Baseline2: Sparse search with BM25
# Supports TAG environment variable for versioning (e.g., TAG=251117)
# Usage: TAG=251117 bash src/baseline2/sparse_search.sh

TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

# 3. build index: sparse retrieval by pyserini
python -m pyserini.index.lucene --collection JsonCollection --input data/tmp/corpus --index data/tmp/index${SUFFIX} --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw
# 4. build up tsv
python src/baseline2/create_queries_from_table.py
# or python src/baseline2/create_queries_from_corpus.py
# 4. search pyserini
#python -m pyserini.search.lucene --index data/tmp/index${SUFFIX} --topics data/tmp/queries_table.tsv --output data/tmp/search_result.txt --bm25 --hits 11 --threads 8 --batch-size 64 # as this can not solve truncating clause automatically
# or python batch_search.py
python src/baseline2/search_with_pyserini.py --hits 11
python src/baseline2/postprocess.py \
  --input  data/tmp/search_result${SUFFIX}.json \
  --output data/tmp/baseline2_sparse_results${SUFFIX}.json \
  --top1-list data/tmp/queries_with_top1_matches${SUFFIX}.txt
