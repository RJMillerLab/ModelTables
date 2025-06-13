# 1. generate mapping from csv_path:readme_path
python src/baseline2/create_raw_csv_to_text_mapping.py
python src/baseline2/view_mapping.py
# 2. get incontext embedding for each csv_path, save to data/tmp/corpus/collection.jsonl
python src/baseline2/create_dedup_table_to_text_mapping.py
# Build up corpus
# mkdir -p pyserini_demo/my_collection
# cat << 'EOF' > pyserini_demo/my_collection/collection.jsonl
# {"id":"1","contents":"The capital of France is Paris."}
# {"id":"2","contents":"Berlin is the capital of Germany."}
# {"id":"3","contents":"Tokyo is a large metropolis in Japan."}
# EOF
# 3. sparse retrieval by pyserini
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/tmp/corpus \
  --index data/tmp/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
# # prepare query corpus
# # overwrite with line 1
# echo -e "1\tWhat is the capital of France?" > pyserini_demo/queries.tsv
# # append lines 2–4
# echo -e "2\tWhich document talks about cooking?" >> pyserini_demo/queries.tsv
# echo -e "3\tWhere is Berlin located?"                 >> pyserini_demo/queries.tsv
# echo -e "4\tWhich text mentions a large metropolis in Japan?" >> pyserini_demo/queries.tsv
# echo -e "5\tHiHi Test?" >> pyserini_demo/queries.tsv
# echo -e "6\tTokyo is a large metropolis which is beautiful" >> pyserini_demo/queries.tsv

# search
python -m pyserini.search.lucene \
  --index data/tmp/index \
  --topics pyserini_demo/queries.tsv \
  --output data/tmp/search_result.txt \
  --bm25
# or python batch_search.py
python pyserini_run.py
