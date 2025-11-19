#!/bin/bash
# Baseline1: Dense Search Pipeline (Unified)
# Supports multiple modes: base (default), str, tr
# Supports TAG environment variable for versioning (e.g., TAG=251117)
#
# Usage:
#   bash src/baseline1/table_retrieval_pipeline_unified.sh [mode] [--skip-search]
#   mode: base (default), str, tr
#   --skip-search: Skip build_faiss, search, and postprocess steps (useful for str/tr modes)
#
# Examples:
#   TAG=251117 bash src/baseline1/table_retrieval_pipeline_unified.sh base
#   TAG=251117 bash src/baseline1/table_retrieval_pipeline_unified.sh str --skip-search
#   TAG=251117 bash src/baseline1/table_retrieval_pipeline_unified.sh tr --skip-search

set -e

# Parse arguments
MODE="${1:-base}"  # Default to base mode
SKIP_SEARCH=false

if [[ "$2" == "--skip-search" ]]; then
    SKIP_SEARCH=true
fi

# Validate mode
if [[ ! "$MODE" =~ ^(base|str|tr)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be one of: base, str, tr"
    exit 1
fi

# Get TAG from environment
TAG="${TAG:-}"
SUFFIX="${TAG:+_${TAG}}"

# Determine file suffixes based on mode
if [[ "$MODE" == "base" ]]; then
    MODE_SUFFIX=""
    MODE_ARG=""
else
    MODE_SUFFIX="_${MODE}"
    MODE_ARG="--mode ${MODE}"
fi

echo "=========================================="
echo "Baseline1: Dense Search Pipeline"
echo "Mode: $MODE"
echo "Tag: ${TAG:-none}"
echo "Skip search: $SKIP_SEARCH"
echo "=========================================="

# Step 1: Filter and build corpus JSONL
echo ""
echo "Step 1: Filtering and building corpus JSONL..."
python src/baseline1/table_retrieval_pipeline.py \
filter --base_path data/processed \
        --mask_file data/analysis/all_valid_title_valid${SUFFIX}.txt \
        --output_jsonl data/baseline/valid_tables${MODE_SUFFIX}${SUFFIX}.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --device cuda \
        ${MODE_ARG}

# Step 2: Encode embeddings
echo ""
echo "Step 2: Encoding embeddings..."
python src/baseline1/table_retrieval_pipeline.py \
encode --jsonl data/baseline/valid_tables${MODE_SUFFIX}${SUFFIX}.jsonl \
        --model_name all-MiniLM-L6-v2 \
        --batch_size 256 \
        --output_npz data/baseline/valid_tables${MODE_SUFFIX}${SUFFIX}_embeddings.npz \
        --device cuda

# Steps 3-5: Build FAISS, search, and postprocess (skip for str/tr modes by default)
if [[ "$SKIP_SEARCH" == "false" ]]; then
    # Step 3: Build FAISS index
    echo ""
    echo "Step 3: Building FAISS index..."
    python src/baseline1/table_retrieval_pipeline.py \
    build_faiss --emb_npz data/baseline/valid_tables${MODE_SUFFIX}${SUFFIX}_embeddings.npz \
                --output_index data/baseline/valid_tables${MODE_SUFFIX}${SUFFIX}.faiss

    # Step 4: Search
    echo ""
    echo "Step 4: Searching..."
    python src/baseline1/table_retrieval_pipeline.py \
    search --emb_npz data/baseline/valid_tables${MODE_SUFFIX}${SUFFIX}_embeddings.npz \
            --faiss_index data/baseline/valid_tables${MODE_SUFFIX}${SUFFIX}.faiss \
            --top_k 11 \
            --output_json data/baseline/table_neighbors${MODE_SUFFIX}${SUFFIX}.json

    # Step 5: Postprocess
    echo ""
    echo "Step 5: Postprocessing..."
    python src/baseline1/postprocess.py \
        data/baseline/table_neighbors${MODE_SUFFIX}${SUFFIX}.json \
        data/baseline/baseline1_dense${MODE_SUFFIX}${SUFFIX}.json
else
    echo ""
    echo "Skipping search steps (build_faiss, search, postprocess) as requested."
    echo "To run full pipeline, omit --skip-search flag."
fi

echo ""
echo "=========================================="
echo "✅ Pipeline completed for mode: $MODE"
echo "=========================================="

