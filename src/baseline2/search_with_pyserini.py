from pyserini.search.lucene import LuceneSearcher
import json
import argparse
import os

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
        default=None,
        help="Output JSON file to save search results. If not specified, uses data/tmp/search_result.json or data/tmp/search_result_<TAG>.json if TAG env var is set."
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to Pyserini index. If not specified, uses data/tmp/index or data/tmp/index_<TAG> if TAG env var is set."
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to queries TSV file. If not specified, uses data/tmp/queries_table.tsv."
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Path to ID mapping JSON file. If not specified, uses data/tmp/queries_table_mapping.json."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Support TAG environment variable for versioning
    tag = os.environ.get('TAG', '')
    suffix = f"_{tag}" if tag else ""
    
    # Set default paths with tag support
    index_path = args.index or f'data/tmp/index{suffix}'
    output_path = args.output or f'data/tmp/search_result{suffix}.json'
    queries_path = args.queries or f'data/tmp/queries_table{suffix}.tsv'
    mapping_path = args.mapping or f'data/tmp/queries_table{suffix}_mapping.json'

    # Initialize searcher
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25()  # Use BM25 scoring
    
    # Load queries and mapping
    print("Loading queries...")
    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")
    
    print("Loading ID mapping...")
    id_mapping = load_id_mapping(mapping_path)
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Search results saved to {output_path}")
    print(f"Total queries processed: {len(results)}")

if __name__ == "__main__":
    main() 