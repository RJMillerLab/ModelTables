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

def main():
    parser = argparse.ArgumentParser(description="Search with Pyserini")
    parser.add_argument("--top_k", type=int, default=11, help="Number of hits (documents) to retrieve per query.")   
    parser.add_argument("--tag", type=str, default=None, help="Tag suffix for versioning (e.g., 251117).")
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 mode.")
    args = parser.parse_args()
    
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""
    
    index_path =  f'data/tmp/index_sparse{v2_suffix}{suffix}'
    output_path = f'data/tmp/search_result{v2_suffix}{suffix}.json'
    queries_path = f'data/tmp/queries_table{v2_suffix}{suffix}.tsv'
    mapping_path = f'data/tmp/queries_table{v2_suffix}{suffix}_mapping.json'

    # Initialize searcher
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25()  # Use BM25 scoring
    
    queries = load_queries(queries_path)
    id_mapping = load_id_mapping(mapping_path)
    
    results = {}
    total = len(queries)
    for i, (qid, text) in enumerate(queries.items(), 1):
        print(f"Searching for query {qid} ({i}/{total})...")
        try:
            hits = searcher.search(text, k=args.top_k)
            original_id = id_mapping[qid]
            results[original_id] = [hit.docid for hit in hits]
        except Exception as e:    
            print(f"Error searching for query {qid}: {e}")
            continue
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Search results saved to {output_path}")
    print(f"Total queries processed: {len(results)}")

if __name__ == "__main__":
    main() 