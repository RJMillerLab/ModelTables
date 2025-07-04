from pyserini.search.lucene import LuceneSearcher
import json
import argparse

def load_id_mapping(mapping_file):
    """Load ID mapping from JSON file."""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_queries(tsv_file):
    """Load queries from TSV file."""
    queries = {}
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Line {i} has {len(parts)} parts, skipping: {line[:100]}...")
                    continue
                qid, text = parts
                if not qid or not text:
                    print(f"Warning: Line {i} has empty ID or text, skipping")
                    continue
                queries[qid] = text
            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue
    return queries

# --------------------
# Argument Parsing
# --------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Search with Pyserini")
    parser.add_argument(
        "--hits",
        type=int,
        default=11,
        help="Number of hits (documents) to retrieve per query."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tmp/search_result.json",
        help="Output JSON file to save search results."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize searcher
    searcher = LuceneSearcher('data/tmp/index')
    searcher.set_bm25()  # Use BM25 scoring
    
    # Load queries and mapping
    print("Loading queries...")
    queries = load_queries('data/tmp/queries_table.tsv')
    print(f"Loaded {len(queries)} queries")
    
    print("Loading ID mapping...")
    id_mapping = load_id_mapping('data/tmp/queries_table_mapping.json')
    print(f"Loaded {len(id_mapping)} ID mappings")
    
    # Perform search
    results = {}
    total = len(queries)
    for i, (qid, text) in enumerate(queries.items(), 1):
        print(f"Searching for query {qid} ({i}/{total})...")
        try:
            hits = searcher.search(text, k=args.hits)  # Use user-specified hits
            
            # Store results with original IDs
            original_id = id_mapping[qid]
            results[original_id] = [hit.docid for hit in hits]
        except Exception as e:
            print(f"Error searching for query {qid}: {e}")
            continue
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Search results saved to {args.output}")
    print(f"Total queries processed: {len(results)}")

if __name__ == "__main__":
    main() 