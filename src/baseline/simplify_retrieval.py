import os
import json
from typing import Dict, List, Any

def load_results(results_path: str) -> Dict[str, Dict[str, Any]]:
    """Load retrieval results from JSON file"""
    with open(results_path, "r") as f:
        return json.load(f)

def simplify_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Convert detailed retrieval results to simplified format"""
    simplified = {}
    for query_id, result in results.items():
        # Only include queries that have retrieved tables
        if result["retrieved_tables"]:
            simplified[query_id] = result["retrieved_tables"]
    return simplified

def main():
    # Paths
    base_path = "data/processed"
    input_path = os.path.join(base_path, "embeddings_output", "retrieval_results_st.json")
    output_path = os.path.join(base_path, "embeddings_output", "retrieval_results_simplified.json")
    
    # Load results
    print("Loading retrieval results...")
    results = load_results(input_path)
    print(f"Loaded {len(results)} query results")
    
    # Simplify results
    print("Simplifying results...")
    simplified_results = simplify_results(results)
    print(f"Simplified {len(simplified_results)} results")
    
    # Save simplified results
    print("Saving simplified results...")
    with open(output_path, "w") as f:
        json.dump(simplified_results, f, indent=2)
    print(f"Saved simplified results to {output_path}")

if __name__ == "__main__":
    main() 