"""
Model Card Hybrid Search: Sparse (BM25) + Dense (SBERT) reranking.

This script combines sparse search results with dense reranking for model card retrieval.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_sparse_results(json_path: str) -> Dict[str, List[str]]:
    """Load sparse search results from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus_metadata(jsonl_path: str) -> Dict[str, str]:
    """Load corpus metadata from JSONL file."""
    corpus_text = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus_text[obj["id"]] = obj["contents"]
    return corpus_text


def load_dense_embeddings(npz_path: str):
    """Load dense embeddings from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    embs = data["embeddings"].astype("float32")
    ids = data["ids"].tolist()
    return ids, embs


def hybrid_rerank(sparse_results: Dict[str, List[str]], 
                  corpus_metadata: Dict[str, str],
                  doc_ids: List[str], 
                  doc_embeddings: np.ndarray,
                  model: SentenceTransformer,
                  topk: int = 11) -> Dict[str, List[str]]:
    """
    Perform hybrid reranking: sparse top-k -> dense SBERT rerank -> top-n.
    """
    # Build mapping for O(1) lookup
    id2row = {d: i for i, d in enumerate(doc_ids)}
    
    # L2 normalize embeddings
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    results = {}
    for qid, candidates in tqdm(sparse_results.items(), desc="Hybrid reranking"):
        # Get query text from corpus metadata
        query_text = corpus_metadata.get(qid)
        if not query_text:
            print(f"Warning: No metadata found for query {qid}, using sparse results")
            results[qid] = candidates[:topk]
            continue
        
        # Filter candidates that exist in dense index
        filtered_candidates = [d for d in candidates if d in id2row]
        if not filtered_candidates:
            print(f"Warning: No dense embeddings for query {qid}, using sparse results")
            results[qid] = candidates[:topk]
            continue
        
        # Encode query
        try:
            query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            print(f"Error encoding query {qid}: {e}")
            results[qid] = candidates[:topk]
            continue
        
        # Get candidate embeddings
        candidate_rows = [id2row[d] for d in filtered_candidates]
        candidate_embeddings = doc_embeddings[candidate_rows]
        
        # Compute similarity scores
        scores = candidate_embeddings @ query_emb[0]  # (n_candidates,)
        
        # Get top-k indices
        top_indices = np.argsort(-scores)[:topk]
        top_candidates = [filtered_candidates[i] for i in top_indices]
        
        # Remove self-hit if present
        top_candidates = [c for c in top_candidates if c != qid]
        
        # Fill remaining slots if needed
        if len(top_candidates) < topk:
            for c in candidates:
                if c not in top_candidates and c != qid:
                    top_candidates.append(c)
                if len(top_candidates) == topk:
                    break
        
        results[qid] = top_candidates
    
    return results


def search_single_query_hybrid(query_text: str, 
                              sparse_candidates: List[str],
                              corpus_metadata: Dict[str, str],
                              doc_ids: List[str], 
                              doc_embeddings: np.ndarray,
                              model: SentenceTransformer,
                              topk: int = 11) -> List[str]:
    """
    Perform hybrid search for a single query.
    """
    # Build mapping for O(1) lookup
    id2row = {d: i for i, d in enumerate(doc_ids)}
    
    # L2 normalize embeddings
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Filter candidates that exist in dense index
    filtered_candidates = [d for d in sparse_candidates if d in id2row]
    if not filtered_candidates:
        return sparse_candidates[:topk]
    
    # Encode query
    try:
        query_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    except Exception as e:
        print(f"Error encoding query: {e}")
        return sparse_candidates[:topk]
    
    # Get candidate embeddings
    candidate_rows = [id2row[d] for d in filtered_candidates]
    candidate_embeddings = doc_embeddings[candidate_rows]
    
    # Compute similarity scores
    scores = candidate_embeddings @ query_emb[0]  # (n_candidates,)
    
    # Get top-k indices
    top_indices = np.argsort(-scores)[:topk]
    top_candidates = [filtered_candidates[i] for i in top_indices]
    
    return top_candidates


def main():
    parser = argparse.ArgumentParser(description="Model Card Hybrid Search")
    sub = parser.add_subparsers(dest='cmd')
    
    # Batch hybrid search
    h = sub.add_parser('hybrid_search')
    h.add_argument('--sparse_json', required=True, help='Sparse search results JSON file')
    h.add_argument('--corpus_jsonl', required=True, help='Corpus metadata JSONL file')
    h.add_argument('--dense_npz', required=True, help='Dense embeddings NPZ file')
    h.add_argument('--output', default='output/baseline_mc/hybrid_results.json', 
                   help='Output JSON file for hybrid results')
    h.add_argument('--topk', type=int, default=11, help='Final top-k results')
    h.add_argument('--model_name', default='sentence-transformers/all-MiniLM-L6-v2',
                   help='SBERT model name')
    h.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    # Single query hybrid search
    q = sub.add_parser('query')
    q.add_argument('--query', required=True, help='Query text to search')
    q.add_argument('--sparse_candidates', nargs='+', help='List of sparse candidates')
    q.add_argument('--corpus_jsonl', required=True, help='Corpus metadata JSONL file')
    q.add_argument('--dense_npz', required=True, help='Dense embeddings NPZ file')
    q.add_argument('--topk', type=int, default=11, help='Number of results to return')
    q.add_argument('--model_name', default='sentence-transformers/all-MiniLM-L6-v2',
                   help='SBERT model name')
    q.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.cmd == 'hybrid_search':
        # Load data
        print("Loading sparse results...")
        sparse_results = load_sparse_results(args.sparse_json)
        print(f"Loaded {len(sparse_results)} sparse results")
        
        print("Loading corpus metadata...")
        corpus_metadata = load_corpus_metadata(args.corpus_jsonl)
        print(f"Loaded {len(corpus_metadata)} corpus entries")
        
        print("Loading dense embeddings...")
        doc_ids, doc_embeddings = load_dense_embeddings(args.dense_npz)
        print(f"Loaded {len(doc_ids)} dense embeddings, shape={doc_embeddings.shape}")
        
        # Initialize model
        print("Initializing SBERT model...")
        model = SentenceTransformer(args.model_name, device=args.device)
        
        # Perform hybrid search
        print("Performing hybrid search...")
        results = hybrid_rerank(sparse_results, corpus_metadata, doc_ids, 
                               doc_embeddings, model, args.topk)
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Hybrid search results saved to {args.output}")
        
    elif args.cmd == 'query':
        # Load data
        print("Loading corpus metadata...")
        corpus_metadata = load_corpus_metadata(args.corpus_jsonl)
        
        print("Loading dense embeddings...")
        doc_ids, doc_embeddings = load_dense_embeddings(args.dense_npz)
        
        # Initialize model
        print("Initializing SBERT model...")
        model = SentenceTransformer(args.model_name, device=args.device)
        
        # Perform hybrid search
        candidates = search_single_query_hybrid(
            args.query, args.sparse_candidates, corpus_metadata, 
            doc_ids, doc_embeddings, model, args.topk
        )
        
        print(f"Top-{args.topk} candidates for query '{args.query}':")
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. {candidate}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
