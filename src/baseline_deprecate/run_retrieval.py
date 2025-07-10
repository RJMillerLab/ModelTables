import os
import pickle
import json
from src.baseline.table_embedding import TableEncoder, TableRetriever
import pandas as pd
from tqdm import tqdm
import numpy as np

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
            # Get top_k + 1 results to account for self-retrieval
            retrieved_results = retriever.retrieve(query_embedding, top_k=top_k + 1)
            
            # Filter out self-retrieval and keep only top_k results
            filtered_results = [(tid, score) for tid, score in retrieved_results if tid != query_id][:top_k]
            
            # Store results
            results[query_id] = {
                "retrieved_tables": [table_id for table_id, _ in filtered_results],
                "similarity_scores": [float(score) for _, score in filtered_results]
            }
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved retrieval results to {output_path}")

def main():
    # Paths
    base_path = "data/processed"
    embeddings_path = os.path.join(base_path, "embeddings_output", "table_embeddings_st.pkl")
    output_path = os.path.join(base_path, "embeddings_output", "retrieval_results_st.json")
    
    # Run retrieval
    run_retrieval(embeddings_path, output_path, batch_size=32)

if __name__ == "__main__":
    main() 