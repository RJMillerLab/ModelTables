import os
import json
from typing import Dict, List, Any

def load_results(results_path: str) -> Dict[str, Dict[str, Any]]:
    """Load retrieval results from JSON file"""
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Results file not found: {results_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {results_path}: {e}")
 
def simplify_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Convert detailed retrieval results to simplified format"""
    simplified = {}
    for query_id, result in results.items():
        # Check if result has retrieved_tables and it's not empty
        retrieved_tables = result.get("retrieved_tables", [])
        if retrieved_tables and len(retrieved_tables) > 0:
            simplified[query_id] = retrieved_tables
    return simplified

def main():
    # Paths
    base_path = "data/processed"
    input_path = os.path.join(base_path, "embeddings_output", "retrieval_results_st.json")
    output_path = os.path.join(base_path, "embeddings_output", "retrieval_results_simplified.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load results
    print("Loading retrieval results...")
    try:
        results = load_results(input_path)
        print(f"Loaded {len(results)} query results")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Simplify results
    print("Simplifying results...")
    simplified_results = simplify_results(results)
    print(f"Simplified {len(simplified_results)} results (filtered out {len(results) - len(simplified_results)} empty results)")
    
    # Save simplified results
    print("Saving simplified results...")
    try:
        with open(output_path, "w") as f:
            json.dump(simplified_results, f, indent=2)
        print(f"Saved simplified results to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 