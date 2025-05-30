import os
import pickle
import json
from src.baseline.table_embedding import TableEncoder, TableRetriever
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

def load_embeddings(embeddings_path: str):
    """Load table embeddings from pickle file"""
    with open(embeddings_path, "rb") as f:
        return pickle.load(f)

def run_retrieval(embeddings_path: str, output_path: str, top_k: int = 5, batch_size: int = 32):
    """Run retrieval for all tables and save results with GPU acceleration"""
    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings(embeddings_path)
    
    # Initialize retriever
    retriever = TableRetriever(embeddings)
    
    # Convert embeddings to GPU tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Pre-compute all embeddings on GPU
    all_embeddings = torch.tensor(
        np.stack([emb["table_embedding"] for emb in embeddings.values()]),
        device=device
    )
    all_embeddings = all_embeddings / torch.norm(all_embeddings, dim=1, keepdim=True)
    
    # Get all table IDs
    table_ids = list(embeddings.keys())
    
    # Run retrieval in batches
    results = {}
    for i in tqdm(range(0, len(table_ids), batch_size), desc="Running retrieval"):
        batch_ids = table_ids[i:i + batch_size]
        
        # Get batch embeddings
        batch_embeddings = all_embeddings[i:i + batch_size]
        
        # Calculate similarities for the batch
        with torch.no_grad():
            similarities = torch.matmul(all_embeddings, batch_embeddings.t())
            
            # Get top-k for each query in batch
            top_k_values, top_k_indices = torch.topk(similarities, k=top_k + 1, dim=0)
            
            # Process results for each query in batch
            for j, query_id in enumerate(batch_ids):
                # Remove self-similarity (first result)
                retrieved_indices = top_k_indices[1:, j].cpu().numpy()
                scores = top_k_values[1:, j].cpu().numpy()
                
                # Store results
                results[query_id] = {
                    "retrieved_tables": [table_ids[idx] for idx in retrieved_indices],
                    "similarity_scores": [float(score) for score in scores]
                }
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved retrieval results to {output_path}")

def analyze_results(results_path: str, structure_path: str):
    """Analyze retrieval results and table structure"""
    # Load results and structure info
    with open(results_path, "r") as f:
        results = json.load(f)
    with open(structure_path, "r") as f:
        structure_info = json.load(f)
    
    # Calculate average similarity scores
    avg_scores = []
    for query_id, result in results.items():
        avg_scores.append(np.mean(result["similarity_scores"]))
    
    print("\nRetrieval Analysis:")
    print(f"Average similarity score: {np.mean(avg_scores):.4f} ± {np.std(avg_scores):.4f}")
    
    # Analyze structure patterns
    structure_stats = {
        "num_rows": [],
        "num_cols": [],
        "column_types": {}
    }
    
    for table_id, info in structure_info.items():
        structure_stats["num_rows"].append(info["num_rows"])
        structure_stats["num_cols"].append(info["num_cols"])
        
        # Count column types
        for dtype in info["dtypes"].values():
            if dtype not in structure_stats["column_types"]:
                structure_stats["column_types"][dtype] = 0
            structure_stats["column_types"][dtype] += 1
    
    print("\nTable Structure Statistics:")
    print(f"Average rows: {np.mean(structure_stats['num_rows']):.2f} ± {np.std(structure_stats['num_rows']):.2f}")
    print(f"Average columns: {np.mean(structure_stats['num_cols']):.2f} ± {np.std(structure_stats['num_cols']):.2f}")
    print("\nColumn Type Distribution:")
    for dtype, count in structure_stats["column_types"].items():
        print(f"{dtype}: {count}")

def main():
    # Paths
    base_path = "data/processed"
    embeddings_path = os.path.join(base_path, "table_embeddings.pkl")
    output_path = os.path.join(base_path, "retrieval_results.json")
    structure_path = os.path.join(base_path, "table_structure.json")
    
    # Run retrieval with GPU acceleration
    run_retrieval(embeddings_path, output_path, batch_size=32)
    
    # Analyze results
    analyze_results(output_path, structure_path)

if __name__ == "__main__":
    main() 