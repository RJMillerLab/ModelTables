"""
Test script for Model Card Retrieval Pipeline.

This script tests the model card retrieval system to ensure it works correctly.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path


def run_test(test_name: str, cmd: str, expected_files: list = None):
    """Run a test and check results."""
    print(f"\n🧪 Testing: {test_name}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Test passed!")
            if result.stdout:
                print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            
            # Check expected files
            if expected_files:
                for file_path in expected_files:
                    if os.path.exists(file_path):
                        print(f"✅ File created: {file_path}")
                    else:
                        print(f"❌ File missing: {file_path}")
                        return False
        else:
            print("❌ Test failed!")
            if result.stderr:
                print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out!")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    return True


def test_data_loading():
    """Test if we can load model card data."""
    print("\n🔍 Testing Data Loading")
    print("=" * 50)
    
    # Test if parquet files exist
    parquet_patterns = [
        "data/processed/modelcard_step1.parquet",
        "data/raw/**/modelcard_step1.parquet"
    ]
    
    found_files = []
    for pattern in parquet_patterns:
        import glob
        files = glob.glob(pattern, recursive=True)
        found_files.extend(files)
    
    if found_files:
        print(f"✅ Found {len(found_files)} parquet files:")
        for f in found_files[:3]:  # Show first 3
            print(f"  - {f}")
        return True
    else:
        print("❌ No parquet files found!")
        print("Please ensure modelcard_step1.parquet files exist in data/processed/ or data/raw/")
        return False


def test_dense_pipeline():
    """Test dense retrieval pipeline."""
    print("\n🔍 Testing Dense Pipeline")
    print("=" * 50)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test filter step
        cmd1 = f"""python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py filter \
            --parquet data/processed/modelcard_step1.parquet \
            --field card \
            --output_jsonl {output_dir}/test_corpus.jsonl \
            --model_name all-MiniLM-L6-v2 \
            --device cpu"""
        
        if not run_test("Filter step", cmd1, [f"{output_dir}/test_corpus.jsonl"]):
            return False
        
        # Test encode step
        cmd2 = f"""python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py encode \
            --jsonl {output_dir}/test_corpus.jsonl \
            --model_name all-MiniLM-L6-v2 \
            --batch_size 32 \
            --output_npz {output_dir}/test_embeddings.npz \
            --device cpu"""
        
        if not run_test("Encode step", cmd2, [f"{output_dir}/test_embeddings.npz"]):
            return False
        
        # Test build_faiss step
        cmd3 = f"""python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py build_faiss \
            --emb_npz {output_dir}/test_embeddings.npz \
            --output_index {output_dir}/test_index.faiss"""
        
        if not run_test("Build FAISS step", cmd3, [f"{output_dir}/test_index.faiss"]):
            return False
        
        # Test query step
        cmd4 = f"""python src/modelsearch/pipeline_mc/modelcard_retrieval_pipeline.py query \
            --query "test query" \
            --faiss_index {output_dir}/test_index.faiss \
            --emb_npz {output_dir}/test_embeddings.npz \
            --top_k 3"""
        
        if not run_test("Query step", cmd4):
            return False
    
    return True


def test_sparse_pipeline():
    """Test sparse retrieval pipeline."""
    print("\n🔍 Testing Sparse Pipeline")
    print("=" * 50)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test create_corpus step
        cmd1 = f"""python src/modelsearch/pipeline_mc/modelcard_sparse_search.py create_corpus \
            --parquet data/processed/modelcard_step1.parquet \
            --field card \
            --output_jsonl {output_dir}/test_corpus.jsonl"""
        
        if not run_test("Create corpus step", cmd1, [f"{output_dir}/test_corpus.jsonl"]):
            return False
        
        # Test build_index step
        cmd2 = f"""python src/modelsearch/pipeline_mc/modelcard_sparse_search.py build_index \
            --corpus_jsonl {output_dir}/test_corpus.jsonl \
            --index_path {output_dir}/test_sparse_index"""
        
        if not run_test("Build index step", cmd2, [f"{output_dir}/test_sparse_index"]):
            return False
        
        # Test query step
        cmd3 = f"""python src/modelsearch/pipeline_mc/modelcard_sparse_search.py query \
            --index_path {output_dir}/test_sparse_index \
            --query "test query" \
            --hits 3"""
        
        if not run_test("Query step", cmd3):
            return False
    
    return True


def test_unified_search():
    """Test unified search interface."""
    print("\n🔍 Testing Unified Search Interface")
    print("=" * 50)
    
    # This test requires pre-built indices, so we'll just test the help
    cmd = "python src/modelsearch/pipeline_mc/modelcard_search.py --help"
    
    if run_test("Unified search help", cmd):
        print("✅ Unified search interface is working!")
        return True
    else:
        return False


def main():
    """Run all tests."""
    print("🚀 Model Card Retrieval Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Dense Pipeline", test_dense_pipeline),
        ("Sparse Pipeline", test_sparse_pipeline),
        ("Unified Search", test_unified_search),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed!")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The model card retrieval system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Ensure modelcard_step1.parquet files exist in data/processed/ or data/raw/")
        print("2. Install required dependencies: pip install -r requirements.txt")
        print("3. Check if Pyserini is properly installed")
        print("4. Verify CUDA/CPU device availability")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
