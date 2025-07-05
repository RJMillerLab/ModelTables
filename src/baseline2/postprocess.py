#!/usr/bin/env python3
import json
import os
import re
from collections import defaultdict
import argparse

def load_search_results(result_file):
    """Load search results from JSON file."""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def add_csv_suffix(results):
    """Add .csv suffix to all results if not already present."""
    processed = {}
    for query_id, hits in results.items():
        if not query_id.endswith('.csv'):
            query_id = f"{query_id}.csv"
        processed_hits = [hit if hit.endswith('.csv') else f"{hit}.csv" for hit in hits]
        processed[query_id] = processed_hits
    return processed

def classify_source(query_id):
    """Classify the source of a query based on its ID prefix."""
    base = query_id[:-4] if query_id.endswith('.csv') else query_id
    prefix = base.split('_', 1)[0]
    if re.fullmatch(r'\d{4}\.\d{4,5}(v\d+)?', prefix):
        return 'arxiv'
    elif re.fullmatch(r'[0-9a-f]{32}', prefix.lower()):
        return 'github'
    elif re.fullmatch(r'[0-9a-f]{6,31}', prefix.lower()):
        return 'hugging'
    elif prefix.isdigit():
        return 'numeric'
    else:
        return 'unknown'

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
        'total_by_source': defaultdict(int),
        'top1_by_source': defaultdict(int),
        'queries_with_top1_matches': set()
    }
    processed_results = {}
    for query_id, hits in results.items():
        source = classify_source(query_id)
        stats['total_by_source'][source] += 1
        if not hits:
            processed_results[query_id] = []
            continue
        if hits[0] == query_id:
            stats['top1_matches'] += 1
            stats['top1_by_source'][source] += 1
            stats['queries_with_top1_matches'].add(query_id)
            processed_results[query_id] = hits[1:]
        else:
            processed_results[query_id] = hits
    return processed_results, stats

def main():
    parser = argparse.ArgumentParser(description="Post-process search results: strip self-hit, add .csv suffix, output stats")
    parser.add_argument("--input", required=True, help="Input JSON search result file")
    parser.add_argument("--output", required=True, help="Output JSON file after processing")
    parser.add_argument("--top1-list", default=None, help="Optional path to save list of queries whose top1 == query itself")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    print("Loading search results …")
    results = load_search_results(args.input)
    print(f"Loaded {len(results)} query results")

    print("Processing …")
    processed_results, stats = analyze_results(results)

    print("Adding .csv suffix …")
    processed_results = add_csv_suffix(processed_results)

    # stats summary
    print("\nResults Analysis:")
    print(f"Total queries: {stats['total_queries']}")
    for source, count in sorted(stats['total_by_source'].items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {count}")
    pct = stats['top1_matches'] * 100 / stats['total_queries']
    print(f"\nQueries with top1 matches: {stats['top1_matches']} ({pct:.2f}%)")
    for source, count in sorted(stats['top1_by_source'].items(), key=lambda x: x[1], reverse=True):
        print(f"{source}: {count}")

    # Save processed JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    print(f"Processed results saved to {args.output}  (entries = {len(processed_results)})")

    # optional top1 list
    if args.top1_list:
        with open(args.top1_list, 'w', encoding='utf-8') as f:
            for qid in sorted(stats['queries_with_top1_matches']):
                if not qid.endswith('.csv'):
                    qid = f"{qid}.csv"
                f.write(qid + "\n")
        print(f"Queries with self-hit saved to {args.top1_list}")

if __name__ == "__main__":
    main()
