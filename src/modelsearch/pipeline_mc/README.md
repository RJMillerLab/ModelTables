# Pipeline MC → MC

Baseline retrieval where full model card text is used as query to retrieve other model cards.

Steps:

```bash
# 1. Build corpus jsonl (for BM25 / Lucene)
python build_corpus.py --field card

# 2. Encode dense embeddings
python encode_dense.py --jsonl ../../output/baseline_mc/corpus.jsonl --output ../../output/baseline_mc/embeddings.npy

# 3. Build FAISS index
python build_faiss.py --emb ../../output/baseline_mc/embeddings.npy

# 4. TODO: sparse_search.sh / dense_search.sh / hybrid_search.sh
```

All scripts rely on utilities from `modelsearch.common` to avoid code duplication.

