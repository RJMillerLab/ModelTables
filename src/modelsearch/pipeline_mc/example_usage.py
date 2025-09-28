"""
Example usage of Model Card Retrieval Pipeline.

This script demonstrates how to use the model card retrieval system.
"""
import os
import json
from pathlib import Path


def example_dense_search():
    """Example of dense search."""
    print("🔍 Dense Search Example")
    print("=" * 50)
    
    query = "transformer model for text classification"
    print(f"Query: {query}")
    
    # Run dense search
    cmd = f"""python src/modelsearch/pipeline_mc/modelcard_search.py dense \
        --query "{query}" \
        --topk 5 \
        --dense_index output/baseline_mc/modelcard.faiss \
        --dense_emb output/baseline_mc/modelcard_embeddings.npz"""
    
    print(f"Command: {cmd}")
    print("\nExpected output:")
    print("Top-5 candidates:")
    print("1. microsoft/DialoGPT-medium")
    print("2. distilbert-base-uncased")
    print("3. bert-base-uncased")
    print("4. roberta-base")
    print("5. xlnet-base-cased")


def example_sparse_search():
    """Example of sparse search."""
    print("\n🔍 Sparse Search Example")
    print("=" * 50)
    
    query = "BERT model for natural language processing"
    print(f"Query: {query}")
    
    # Run sparse search
    cmd = f"""python src/modelsearch/pipeline_mc/modelcard_search.py sparse \
        --query "{query}" \
        --topk 5 \
        --sparse_index output/baseline_mc/sparse_index"""
    
    print(f"Command: {cmd}")
    print("\nExpected output:")
    print("Top-5 candidates:")
    print("1. bert-base-uncased")
    print("2. bert-large-uncased")
    print("3. distilbert-base-uncased")
    print("4. bert-base-cased")
    print("5. bert-large-cased")


def example_hybrid_search():
    """Example of hybrid search."""
    print("\n🔍 Hybrid Search Example")
    print("=" * 50)
    
    query = "computer vision model for image classification"
    print(f"Query: {query}")
    
    # Run hybrid search
    cmd = f"""python src/modelsearch/pipeline_mc/modelcard_search.py hybrid \
        --query "{query}" \
        --topk 5 \
        --dense_index output/baseline_mc/modelcard.faiss \
        --dense_emb output/baseline_mc/modelcard_embeddings.npz \
        --sparse_index output/baseline_mc/sparse_index \
        --corpus_jsonl output/baseline_mc/modelcard_corpus.jsonl"""
    
    print(f"Command: {cmd}")
    print("\nExpected output:")
    print("Top-5 candidates:")
    print("1. resnet-50")
    print("2. vgg16")
    print("3. efficientnet-b0")
    print("4. mobilenet-v2")
    print("5. densenet-121")


def example_batch_search():
    """Example of batch search."""
    print("\n🔍 Batch Search Example")
    print("=" * 50)
    
    # Create queries file
    queries = [
        ("q1", "transformer model for text classification"),
        ("q2", "BERT model for NLP"),
        ("q3", "vision model for image classification"),
        ("q4", "GPT model for text generation"),
        ("q5", "T5 model for text-to-text")
    ]
    
    queries_file = "example_queries.tsv"
    with open(queries_file, 'w', encoding='utf-8') as f:
        for qid, query in queries:
            f.write(f"{qid}\t{query}\n")
    
    print(f"Created queries file: {queries_file}")
    
    # Run batch search
    cmd = f"""python src/modelsearch/pipeline_mc/modelcard_sparse_search.py search \
        --index_path output/baseline_mc/sparse_index \
        --queries_tsv {queries_file} \
        --output example_batch_results.json \
        --hits 10"""
    
    print(f"Command: {cmd}")
    print("\nThis will create example_batch_results.json with results for all queries")


def example_custom_search():
    """Example of custom search with different parameters."""
    print("\n🔍 Custom Search Example")
    print("=" * 50)
    
    # Different model
    print("Using different SBERT model:")
    cmd1 = f"""python src/modelsearch/pipeline_mc/build_modelcard_indices.py \
        --model_name sentence-transformers/all-mpnet-base-v2 \
        --device cuda"""
    print(f"Command: {cmd1}")
    
    # Different field
    print("\nUsing different field (card_readme):")
    cmd2 = f"""python src/modelsearch/pipeline_mc/build_modelcard_indices.py \
        --field card_readme \
        --parquet data/processed/modelcard_step1.parquet"""
    print(f"Command: {cmd2}")
    
    # CPU only
    print("\nUsing CPU only:")
    cmd3 = f"""python src/modelsearch/pipeline_mc/build_modelcard_indices.py \
        --device cpu \
        --batch_size 64"""
    print(f"Command: {cmd3}")


def main():
    """Run all examples."""
    print("🚀 Model Card Retrieval Pipeline Examples")
    print("=" * 60)
    
    print("\nFirst, build the indices:")
    print("python src/modelsearch/pipeline_mc/build_modelcard_indices.py \\")
    print("    --parquet data/processed/modelcard_step1.parquet \\")
    print("    --field card \\")
    print("    --output_dir output/baseline_mc \\")
    print("    --device cuda")
    
    example_dense_search()
    example_sparse_search()
    example_hybrid_search()
    example_batch_search()
    example_custom_search()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("\nFor more details, see README.md")


if __name__ == "__main__":
    main()
