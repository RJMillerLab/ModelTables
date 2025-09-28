"""
Build all indices for model card retrieval.

This script builds dense, sparse, and hybrid indices for model card retrieval.
It reuses the existing baseline code structure but works with model cards instead of tables.
"""
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print("❌ Error!")
        if result.stderr:
            print("Error:", result.stderr)
        return False
    
    return True


def build_dense_index(parquet_path: str, field: str, output_dir: str, 
                     model_name: str, device: str, batch_size: int = 256):
    """Build dense index using SBERT + FAISS."""
    print("\n🔍 Building Dense Index (SBERT + FAISS)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Filter to JSONL
    jsonl_path = os.path.join(output_dir, "modelcard_corpus.jsonl")
    cmd1 = f"""python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py filter \
        --parquet {parquet_path} \
        --field {field} \
        --output_jsonl {jsonl_path} \
        --model_name {model_name} \
        --device {device}"""
    
    if not run_command(cmd1, "Creating corpus JSONL"):
        return False
    
    # Step 2: Encode with SBERT
    npz_path = os.path.join(output_dir, "modelcard_embeddings.npz")
    cmd2 = f"""python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py encode \
        --jsonl {jsonl_path} \
        --model_name {model_name} \
        --batch_size {batch_size} \
        --output_npz {npz_path} \
        --device {device}"""
    
    if not run_command(cmd2, "Encoding with SBERT"):
        return False
    
    # Step 3: Build FAISS index
    faiss_path = os.path.join(output_dir, "modelcard.faiss")
    cmd3 = f"""python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py build_faiss \
        --emb_npz {npz_path} \
        --output_index {faiss_path}"""
    
    if not run_command(cmd3, "Building FAISS index"):
        return False
    
    print("✅ Dense index built successfully!")
    return True


def build_sparse_index(parquet_path: str, field: str, output_dir: str):
    """Build sparse index using Pyserini BM25."""
    print("\n🔍 Building Sparse Index (BM25)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create corpus JSONL
    jsonl_path = os.path.join(output_dir, "modelcard_corpus.jsonl")
    cmd1 = f"""python src/modelsearch/pipeline_mc/modelcard_sparse_search.py create_corpus \
        --parquet {parquet_path} \
        --field {field} \
        --output_jsonl {jsonl_path}"""
    
    if not run_command(cmd1, "Creating corpus JSONL"):
        return False
    
    # Step 2: Build Pyserini index
    index_path = os.path.join(output_dir, "sparse_index")
    cmd2 = f"""python src/modelsearch/pipeline_mc/modelcard_sparse_search.py build_index \
        --corpus_jsonl {jsonl_path} \
        --index_path {index_path}"""
    
    if not run_command(cmd2, "Building Pyserini index"):
        return False
    
    print("✅ Sparse index built successfully!")
    return True


def test_search(output_dir: str, query: str = "transformer model for text classification", topk: int = 5):
    """Test all search methods."""
    print(f"\n🧪 Testing Search with query: '{query}'")
    
    # Test dense search
    print("\n--- Dense Search ---")
    cmd_dense = f"""python src/modelsearch/pipeline_mc/modelcard_search.py dense \
        --query "{query}" \
        --topk {topk} \
        --dense_index {output_dir}/modelcard.faiss \
        --dense_emb {output_dir}/modelcard_embeddings.npz"""
    
    run_command(cmd_dense, "Dense search test")
    
    # Test sparse search
    print("\n--- Sparse Search ---")
    cmd_sparse = f"""python src/modelsearch/pipeline_mc/modelcard_search.py sparse \
        --query "{query}" \
        --topk {topk} \
        --sparse_index {output_dir}/sparse_index"""
    
    run_command(cmd_sparse, "Sparse search test")
    
    # Test hybrid search
    print("\n--- Hybrid Search ---")
    cmd_hybrid = f"""python src/modelsearch/pipeline_mc/modelcard_search.py hybrid \
        --query "{query}" \
        --topk {topk} \
        --dense_index {output_dir}/modelcard.faiss \
        --dense_emb {output_dir}/modelcard_embeddings.npz \
        --sparse_index {output_dir}/sparse_index \
        --corpus_jsonl {output_dir}/modelcard_corpus.jsonl"""
    
    run_command(cmd_hybrid, "Hybrid search test")


def main():
    parser = argparse.ArgumentParser(description="Build Model Card Retrieval Indices")
    parser.add_argument('--parquet', default='data/processed/modelcard_step1.parquet',
                       help='Path to modelcard parquet file')
    parser.add_argument('--field', default='card', 
                       help='Field name to extract from parquet (default: card)')
    parser.add_argument('--output_dir', default='output/baseline_mc',
                       help='Output directory for indices')
    parser.add_argument('--model_name', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='SBERT model name')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for encoding')
    parser.add_argument('--test_query', default='transformer model for text classification',
                       help='Test query for search validation')
    parser.add_argument('--test_topk', type=int, default=5, help='Top-k for test search')
    parser.add_argument('--skip_dense', action='store_true', help='Skip dense index building')
    parser.add_argument('--skip_sparse', action='store_true', help='Skip sparse index building')
    parser.add_argument('--test_only', action='store_true', help='Only run tests, skip building')
    
    args = parser.parse_args()
    
    print("🚀 Model Card Retrieval Index Builder")
    print(f"Parquet: {args.parquet}")
    print(f"Field: {args.field}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    
    if not args.test_only:
        # Build dense index
        if not args.skip_dense:
            if not build_dense_index(args.parquet, args.field, args.output_dir, 
                                   args.model_name, args.device, args.batch_size):
                print("❌ Failed to build dense index")
                return
        
        # Build sparse index
        if not args.skip_sparse:
            if not build_sparse_index(args.parquet, args.field, args.output_dir):
                print("❌ Failed to build sparse index")
                return
    
    # Test search functionality
    test_search(args.output_dir, args.test_query, args.test_topk)
    
    print("\n🎉 All done! Model card retrieval indices are ready.")
    print(f"\nUsage examples:")
    print(f"# Dense search")
    print(f"python src/modelsearch/pipeline_mc/modelcard_search.py dense --query 'your query' --topk 5")
    print(f"\n# Sparse search")
    print(f"python src/modelsearch/pipeline_mc/modelcard_search.py sparse --query 'your query' --topk 5")
    print(f"\n# Hybrid search")
    print(f"python src/modelsearch/pipeline_mc/modelcard_search.py hybrid --query 'your query' --topk 5")


if __name__ == "__main__":
    main()
