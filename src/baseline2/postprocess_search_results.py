import json
import os
from collections import defaultdict

def load_search_results(result_file):
    """Load search results from JSON file."""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def add_csv_suffix(results):
    """Add .csv suffix to all results if not already present."""
    processed = {}
    for query_id, hits in results.items():
        # Add .csv to query_id if not present
        if not query_id.endswith('.csv'):
            query_id = f"{query_id}.csv"
        
        # Add .csv to all hits if not present
        processed_hits = [hit if hit.endswith('.csv') else f"{hit}.csv" for hit in hits]
        processed[query_id] = processed_hits
    
    return processed

def analyze_results(results):
    """
    Analyze search results and handle cases where top1 matches the query.
    Returns:
    - processed_results: Results with top1 matches removed
    - stats: Statistics about the matches
    """
    stats = {
        'total_queries': len(results),
        'top1_matches': 0,
        'matches_by_source': defaultdict(int),
        'queries_with_top1_matches': set()
    }
    
    processed_results = {}
    
    for query_id, hits in results.items():
        # Skip if no hits
        if not hits:
            processed_results[query_id] = []
            continue
            
        # Check if top1 matches the query
        if hits[0] == query_id:
            stats['top1_matches'] += 1
            stats['queries_with_top1_matches'].add(query_id)
            
            # Get source from query_id (assuming format: source/path)
            source = query_id.split('/')[0] if '/' in query_id else 'unknown'
            stats['matches_by_source'][source] += 1
            
            # Remove the top1 match and keep the rest
            processed_results[query_id] = hits[1:]
        else:
            processed_results[query_id] = hits
    
    return processed_results, stats

def main():
    # Load search results
    result_file = 'data/tmp/search_result.json'
    if not os.path.exists(result_file):
        print(f"Error: Search results file not found at {result_file}")
        return
        
    print("Loading search results...")
    results = load_search_results(result_file)
    print(f"Loaded {len(results)} query results")
    
    # Process results
    print("\nProcessing results...")
    processed_results, stats = analyze_results(results)
    
    # Add .csv suffix to all results
    print("\nAdding .csv suffix to all results...")
    processed_results = add_csv_suffix(processed_results)
    
    # Print statistics
    print("\nResults Analysis:")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Queries with top1 matches: {stats['top1_matches']} ({stats['top1_matches']/stats['total_queries']*100:.2f}%)")
    
    print("\nTop1 matches by source:")
    for source, count in sorted(stats['matches_by_source'].items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {count}")
    
    # Save processed results
    output_file = 'data/tmp/processed_search_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    print(f"\nProcessed results saved to {output_file}")
    
    # Save list of queries with top1 matches
    matches_file = 'data/tmp/queries_with_top1_matches.txt'
    with open(matches_file, 'w', encoding='utf-8') as f:
        for query_id in sorted(stats['queries_with_top1_matches']):
            if not query_id.endswith('.csv'):
                query_id = f"{query_id}.csv"
            f.write(f"{query_id}\n")
    print(f"List of queries with top1 matches saved to {matches_file}")

if __name__ == "__main__":
    main() 