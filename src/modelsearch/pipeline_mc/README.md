# Model Card Retrieval Pipeline

This pipeline provides model card retrieval functionality using three different methods:
1. **Dense Retrieval**: SBERT + FAISS for semantic similarity
2. **Sparse Retrieval**: BM25 for keyword matching  
3. **Hybrid Retrieval**: BM25 + SBERT reranking for best of both worlds

## Overview

The pipeline reuses the existing baseline code structure but works with model cards instead of tables:
- `baseline1` → `modelcard_retrieval_pipeline.py` (dense)
- `baseline2` → `modelcard_sparse_search.py` (sparse) 
- `baseline3` → `modelcard_hybrid_search.py` (hybrid)

## Quick Start

### 1. Build All Indices

```bash
# Build all indices (dense + sparse)
python src/modelsearch/pipeline_mc/build_modelcard_indices.py \
    --parquet data/processed/modelcard_step1.parquet \
    --field card \
    --output_dir output/baseline_mc \
    --device cuda
```

### 2. Search Model Cards

```bash
# Dense search (semantic similarity)
python src/modelsearch/pipeline_mc/modelcard_search.py dense \
    --query "transformer model for text classification" \
    --topk 5

# Sparse search (keyword matching)
python src/modelsearch/pipeline_mc/modelcard_search.py sparse \
    --query "transformer model for text classification" \
    --topk 5

# Hybrid search (BM25 + SBERT reranking)
python src/modelsearch/pipeline_mc/modelcard_search.py hybrid \
    --query "transformer model for text classification" \
    --topk 5
```

## Detailed Usage

### Dense Retrieval Pipeline

```bash
# Step 1: Create corpus from model cards
python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py filter \
    --parquet data/processed/modelcard_step1.parquet \
    --field card \
    --output_jsonl output/baseline_mc/modelcard_corpus.jsonl \
    --model_name all-MiniLM-L6-v2 \
    --device cuda

# Step 2: Encode with SBERT
python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py encode \
    --jsonl output/baseline_mc/modelcard_corpus.jsonl \
    --model_name all-MiniLM-L6-v2 \
    --batch_size 256 \
    --output_npz output/baseline_mc/modelcard_embeddings.npz \
    --device cuda

# Step 3: Build FAISS index
python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py build_faiss \
    --emb_npz output/baseline_mc/modelcard_embeddings.npz \
    --output_index output/baseline_mc/modelcard.faiss

# Step 4: Search
python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py search \
    --emb_npz output/baseline_mc/modelcard_embeddings.npz \
    --faiss_index output/baseline_mc/modelcard.faiss \
    --top_k 5 \
    --output_json output/baseline_mc/modelcard_neighbors.json

# Single query search
python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py query \
    --query "transformer model for text classification" \
    --faiss_index output/baseline_mc/modelcard.faiss \
    --emb_npz output/baseline_mc/modelcard_embeddings.npz \
    --top_k 5
```

### Sparse Retrieval Pipeline

```bash
# Step 1: Create corpus JSONL
python src/modelsearch/pipeline_mc/modelcard_sparse_search.py create_corpus \
    --parquet data/processed/modelcard_step1.parquet \
    --field card \
    --output_jsonl output/baseline_mc/modelcard_corpus.jsonl

# Step 2: Build Pyserini index
python src/modelsearch/pipeline_mc/modelcard_sparse_search.py build_index \
    --corpus_jsonl output/baseline_mc/modelcard_corpus.jsonl \
    --index_path output/baseline_mc/sparse_index

# Step 3: Search
python src/modelsearch/pipeline_mc/modelcard_sparse_search.py query \
    --index_path output/baseline_mc/sparse_index \
    --query "transformer model for text classification" \
    --hits 5
```

### Hybrid Retrieval Pipeline

```bash
# First build both dense and sparse indices (see above)

# Then perform hybrid search
python src/modelsearch/pipeline_mc/modelcard_hybrid_search.py query \
    --query "transformer model for text classification" \
    --sparse_candidates candidate1 candidate2 candidate3 \
    --corpus_jsonl output/baseline_mc/modelcard_corpus.jsonl \
    --dense_npz output/baseline_mc/modelcard_embeddings.npz \
    --topk 5
```

## File Structure

```
src/modelsearch/pipeline_mc/
├── README.md                           # This file
├── build_modelcard_indices.py          # Build all indices script
├── modelcard_search.py                 # Unified search interface
├── modelcard_retrieval_pipeline.py     # Dense retrieval (SBERT + FAISS)
├── modelcard_sparse_search.py          # Sparse retrieval (BM25)
└── modelcard_hybrid_search.py          # Hybrid retrieval (BM25 + SBERT)
```

## Output Files

After building indices, you'll have:

```
output/baseline_mc/
├── modelcard_corpus.jsonl              # Corpus JSONL file
├── modelcard_embeddings.npz            # Dense embeddings
├── modelcard.faiss                     # FAISS dense index
├── sparse_index/                       # Pyserini sparse index directory
└── modelcard_neighbors.json            # Dense search results (optional)
```

## Configuration

### Model Selection

You can use different SBERT models:

```bash
# Use different model
python src/modelsearch/pipeline_mc/build_modelcard_indices.py \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --device cuda
```

### Field Selection

Extract different fields from the model card parquet:

```bash
# Use card_readme instead of card
python src/modelsearch/pipeline_mc/build_modelcard_indices.py \
    --field card_readme \
    --parquet data/processed/modelcard_step1.parquet
```

### Device Configuration

```bash
# Use CPU instead of GPU
python src/modelsearch/pipeline_mc/build_modelcard_indices.py \
    --device cpu
```

## Integration with Existing Baselines

This pipeline reuses the existing baseline code structure:

- **baseline1** (dense): `table_retrieval_pipeline.py` → `modelcard_retrieval_pipeline.py`
- **baseline2** (sparse): `search_with_pyserini.py` → `modelcard_sparse_search.py`  
- **baseline3** (hybrid): `hybrid_rerank.py` → `modelcard_hybrid_search.py`

The main differences:
1. Input: Model cards instead of tables
2. Data source: Parquet files instead of CSV files
3. Field extraction: `card` field instead of table content
4. Output: Model IDs instead of table IDs

## Examples

### Search for Specific Model Types

```bash
# Find BERT models
python src/modelsearch/pipeline_mc/modelcard_search.py dense \
    --query "BERT model for natural language processing" \
    --topk 10

# Find vision models
python src/modelsearch/pipeline_mc/modelcard_search.py hybrid \
    --query "computer vision model for image classification" \
    --topk 10

# Find models by task
python src/modelsearch/pipeline_mc/modelcard_search.py sparse \
    --query "sentiment analysis text classification" \
    --topk 10
```

### Batch Search

```bash
# Create queries file
echo -e "q1\ttransformer model for text classification\nq2\tBERT model for NLP\nq3\tvision model for image classification" > queries.tsv

# Run batch search
python src/modelsearch/pipeline_mc/modelcard_sparse_search.py search \
    --index_path output/baseline_mc/sparse_index \
    --queries_tsv queries.tsv \
    --output batch_results.json \
    --hits 10
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `--device cpu` or reduce `--batch_size`
2. **File not found**: Check parquet file path and ensure data exists
3. **Index not found**: Run the build script first to create indices
4. **Empty results**: Check if model cards contain the specified field

### Performance Tips

1. Use GPU for encoding if available (`--device cuda`)
2. Increase batch size for faster encoding (if memory allows)
3. Use hybrid search for best quality results
4. Pre-filter model cards to reduce corpus size if needed