"""
Model Card Sparse Search using Pyserini BM25.

This script creates a sparse index from model card corpus and performs BM25 search.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexCreator
from pyserini.index import IndexReader


def load_combined_data(data_type: str, file_path: str, columns: List[str]) -> pd.DataFrame:
    """Load combined data from parquet files."""
    import glob
    from pathlib import Path
    
    file_path = Path(file_path).expanduser()
    pattern = f"{file_path}/**/{data_type}_step1.parquet"
    files = glob.glob(str(pattern), recursive=True)
    
    if not files:
        raise FileNotFoundError(f"No {data_type}_step1.parquet files found in {file_path}")
    
    print(f"Found {len(files)} parquet files: {files}")
    
    dfs = []
    for file in files:
        try:
            df = pd.read_parquet(file, columns=columns)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid parquet files could be loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    return combined_df


def create_corpus_jsonl(parquet_path: str, field: str, output_jsonl: str):
    """Create corpus JSONL from model card parquet data."""
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    # Load model card data
    print(f"Loading model card data...")
    df = load_combined_data("modelcard", file_path="~/Repo/CitationLake/data/raw", 
                           columns=['modelId', field])
    
    written = 0
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            model_id = row['modelId']
            contents = str(row[field])
            
            if not contents or contents.lower() == "nan" or contents.strip() == "":
                continue
                
            doc = {
                'id': model_id,
                'contents': contents
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
            written += 1
    
    print(f'Generated corpus JSONL: {output_jsonl}, entries written: {written}/{len(df)}')


def build_sparse_index(corpus_jsonl: str, index_path: str):
    """Build Pyserini sparse index from corpus JSONL."""
    os.makedirs(index_path, exist_ok=True)
    
    print(f"Building sparse index from {corpus_jsonl}...")
    index_creator = IndexCreator()
    
    # Create index
    index_creator.create(
        input=corpus_jsonl,
        index=index_path,
        storePositions=True,
        storeDocvectors=True,
        storeRaw=True
    )
    
    print(f"Sparse index created at: {index_path}")


def load_queries(tsv_file: str) -> Dict[str, str]:
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


def search_sparse(index_path: str, queries: Dict[str, str], hits: int = 11) -> Dict[str, List[str]]:
    """Perform sparse search using Pyserini."""
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25()  # Use BM25 scoring
    
    results = {}
    total = len(queries)
    for i, (qid, text) in enumerate(queries.items(), 1):
        print(f"Searching for query {qid} ({i}/{total})...")
        try:
            hits_list = searcher.search(text, k=hits)
            results[qid] = [hit.docid for hit in hits_list]
        except Exception as e:
            print(f"Error searching for query {qid}: {e}")
            continue
    
    return results


def search_single_query(index_path: str, query_text: str, hits: int = 11) -> List[str]:
    """Search for a single query and return top candidates."""
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25()
    
    try:
        hits_list = searcher.search(query_text, k=hits)
        return [hit.docid for hit in hits_list]
    except Exception as e:
        print(f"Error searching for query: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Model Card Sparse Search with Pyserini")
    sub = parser.add_subparsers(dest='cmd')
    
    # Create corpus
    p = sub.add_parser('create_corpus')
    p.add_argument('--parquet', required=True, help='Path to modelcard parquet file')
    p.add_argument('--field', default='card', help='Field name to extract (default: card)')
    p.add_argument('--output_jsonl', required=True, help='Output corpus JSONL file')
    
    # Build index
    b = sub.add_parser('build_index')
    b.add_argument('--corpus_jsonl', required=True, help='Input corpus JSONL file')
    b.add_argument('--index_path', required=True, help='Output index directory')
    
    # Search
    s = sub.add_parser('search')
    s.add_argument('--index_path', required=True, help='Path to sparse index')
    s.add_argument('--queries_tsv', help='TSV file with queries (qid<TAB>query_text)')
    s.add_argument('--output', default='output/baseline_mc/sparse_results.json', 
                   help='Output JSON file for search results')
    s.add_argument('--hits', type=int, default=11, help='Number of hits per query')
    
    # Single query search
    q = sub.add_parser('query')
    q.add_argument('--index_path', required=True, help='Path to sparse index')
    q.add_argument('--query', required=True, help='Query text to search')
    q.add_argument('--hits', type=int, default=11, help='Number of hits to return')
    
    args = parser.parse_args()
    
    if args.cmd == 'create_corpus':
        create_corpus_jsonl(args.parquet, args.field, args.output_jsonl)
    elif args.cmd == 'build_index':
        build_sparse_index(args.corpus_jsonl, args.index_path)
    elif args.cmd == 'search':
        if not args.queries_tsv:
            print("Error: --queries_tsv is required for search command")
            return
        queries = load_queries(args.queries_tsv)
        results = search_sparse(args.index_path, queries, args.hits)
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Search results saved to {args.output}")
    elif args.cmd == 'query':
        candidates = search_single_query(args.index_path, args.query, args.hits)
        print(f"Top-{args.hits} candidates for query '{args.query}':")
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. {candidate}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
