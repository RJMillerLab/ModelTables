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
    """Run retrieval for all tables and save results"""
    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings(embeddings_path)
    
    # Initialize retriever
    print("Initializing retriever...")
    retriever = TableRetriever(embeddings)
    
    # Get all table IDs
    table_ids = list(embeddings.keys())
    print(f"Total number of tables: {len(table_ids)}")
    
    # Run retrieval in batches
    results = {}
    for i in tqdm(range(0, len(table_ids), batch_size), desc="Running retrieval"):
        batch_ids = table_ids[i:i + batch_size]
        
        # Process each query in the batch
        for query_id in batch_ids:
            query_embedding = embeddings[query_id]
            retrieved_results = retriever.retrieve(query_embedding, top_k=top_k)
            
            # Store results
            results[query_id] = {
                "retrieved_tables": [table_id for table_id, _ in retrieved_results],
                "similarity_scores": [float(score) for _, score in retrieved_results]
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
    for dtype, count in sorted(structure_stats["column_types"].items(), key=lambda x: x[1], reverse=True):
        print(f"{dtype}: {count}")

def main():
    # Paths
    base_path = "data/processed"
    embeddings_path = os.path.join(base_path, "embeddings_output", "table_embeddings_st.pkl")
    output_path = os.path.join(base_path, "embeddings_output", "retrieval_results_st.json")
    structure_path = os.path.join(base_path, "embeddings_output", "table_structure_st.json")
    
    # Run retrieval
    run_retrieval(embeddings_path, output_path, batch_size=32)
    
    # Analyze results
    analyze_results(output_path, structure_path)

if __name__ == "__main__":
    main() 