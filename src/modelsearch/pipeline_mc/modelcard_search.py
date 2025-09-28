"""
Unified Model Card Search Interface.

This script provides a unified interface for model card retrieval using:
1. Dense retrieval (SBERT + FAISS)
2. Sparse retrieval (BM25)
3. Hybrid retrieval (BM25 + SBERT reranking)

Usage:
# Dense search
python modelcard_search.py dense --query "transformer model for text classification" --topk 5

# Sparse search  
python modelcard_search.py sparse --query "transformer model for text classification" --topk 5

# Hybrid search
python modelcard_search.py hybrid --query "transformer model for text classification" --topk 5
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class ModelCardSearcher:
    """Unified model card searcher supporting dense, sparse, and hybrid retrieval."""
    
    def __init__(self, 
                 dense_index_path: str = "output/baseline_mc/modelcard.faiss",
                 dense_emb_path: str = "output/baseline_mc/modelcard_embeddings.npz",
                 sparse_index_path: str = "output/baseline_mc/sparse_index",
                 corpus_jsonl_path: str = "output/baseline_mc/modelcard_corpus.jsonl",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda"):
        """
        Initialize the searcher with pre-built indices.
        
        Args:
            dense_index_path: Path to FAISS dense index
            dense_emb_path: Path to dense embeddings NPZ file
            sparse_index_path: Path to Pyserini sparse index
            corpus_jsonl_path: Path to corpus JSONL file
            model_name: SBERT model name
            device: Device for model inference
        """
        self.dense_index_path = dense_index_path
        self.dense_emb_path = dense_emb_path
        self.sparse_index_path = sparse_index_path
        self.corpus_jsonl_path = corpus_jsonl_path
        self.model_name = model_name
        self.device = device
        
        # Lazy loading
        self._dense_index = None
        self._dense_embeddings = None
        self._doc_ids = None
        self._sparse_searcher = None
        self._corpus_metadata = None
        self._sbert_model = None
    
    def _load_dense_index(self):
        """Lazy load dense index and embeddings."""
        if self._dense_index is None:
            print("Loading dense index...")
            self._dense_index = faiss.read_index(self.dense_index_path)
            
            data = np.load(self.dense_emb_path)
            self._dense_embeddings = data['embeddings'].astype('float32')
            self._doc_ids = data['ids'].tolist()
            print(f"Loaded dense index with {self._dense_index.ntotal} vectors")
    
    def _load_sparse_index(self):
        """Lazy load sparse index."""
        if self._sparse_searcher is None:
            print("Loading sparse index...")
            from pyserini.search.lucene import LuceneSearcher
            self._sparse_searcher = LuceneSearcher(self.sparse_index_path)
            self._sparse_searcher.set_bm25()
            print(f"Loaded sparse index")
    
    def _load_corpus_metadata(self):
        """Lazy load corpus metadata."""
        if self._corpus_metadata is None:
            print("Loading corpus metadata...")
            self._corpus_metadata = {}
            with open(self.corpus_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    self._corpus_metadata[obj['id']] = obj['contents']
            print(f"Loaded {len(self._corpus_metadata)} corpus entries")
    
    def _load_sbert_model(self):
        """Lazy load SBERT model."""
        if self._sbert_model is None:
            print("Loading SBERT model...")
            self._sbert_model = SentenceTransformer(self.model_name, device=self.device)
            print("SBERT model loaded")
    
    def dense_search(self, query: str, topk: int = 5) -> List[str]:
        """Perform dense search using SBERT + FAISS."""
        self._load_dense_index()
        self._load_sbert_model()
        
        # Encode query
        query_emb = self._sbert_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search
        D, I = self._dense_index.search(query_emb, topk)
        
        # Return top-k candidates
        candidates = [self._doc_ids[i] for i in I[0]]
        return candidates
    
    def sparse_search(self, query: str, topk: int = 5) -> List[str]:
        """Perform sparse search using BM25."""
        self._load_sparse_index()
        
        # Search
        hits = self._sparse_searcher.search(query, k=topk)
        candidates = [hit.docid for hit in hits]
        return candidates
    
    def hybrid_search(self, query: str, topk: int = 5, sparse_topk: int = 50) -> List[str]:
        """Perform hybrid search: sparse retrieval + dense reranking."""
        # First get more candidates from sparse search
        sparse_candidates = self.sparse_search(query, sparse_topk)
        
        if not sparse_candidates:
            return []
        
        # Load required components
        self._load_dense_index()
        self._load_sbert_model()
        self._load_corpus_metadata()
        
        # Filter candidates that exist in dense index
        id2row = {d: i for i, d in enumerate(self._doc_ids)}
        filtered_candidates = [d for d in sparse_candidates if d in id2row]
        
        if not filtered_candidates:
            return sparse_candidates[:topk]
        
        # Encode query
        query_emb = self._sbert_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Get candidate embeddings
        candidate_rows = [id2row[d] for d in filtered_candidates]
        candidate_embeddings = self._dense_embeddings[candidate_rows]
        
        # L2 normalize embeddings
        candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Compute similarity scores
        scores = candidate_embeddings @ query_emb[0]
        
        # Get top-k indices
        top_indices = np.argsort(-scores)[:topk]
        top_candidates = [filtered_candidates[i] for i in top_indices]
        
        return top_candidates
    
    def search(self, query: str, method: str = "dense", topk: int = 5) -> List[str]:
        """
        Unified search interface.
        
        Args:
            query: Search query text
            method: Search method ("dense", "sparse", "hybrid")
            topk: Number of results to return
            
        Returns:
            List of model IDs (candidates)
        """
        if method == "dense":
            return self.dense_search(query, topk)
        elif method == "sparse":
            return self.sparse_search(query, topk)
        elif method == "hybrid":
            return self.hybrid_search(query, topk)
        else:
            raise ValueError(f"Unknown search method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Unified Model Card Search")
    parser.add_argument('method', choices=['dense', 'sparse', 'hybrid'], 
                       help='Search method to use')
    parser.add_argument('--query', required=True, help='Search query text')
    parser.add_argument('--topk', type=int, default=5, help='Number of results to return')
    parser.add_argument('--dense_index', default='output/baseline_mc/modelcard.faiss',
                       help='Path to dense FAISS index')
    parser.add_argument('--dense_emb', default='output/baseline_mc/modelcard_embeddings.npz',
                       help='Path to dense embeddings NPZ file')
    parser.add_argument('--sparse_index', default='output/baseline_mc/sparse_index',
                       help='Path to sparse Pyserini index')
    parser.add_argument('--corpus_jsonl', default='output/baseline_mc/modelcard_corpus.jsonl',
                       help='Path to corpus JSONL file')
    parser.add_argument('--model_name', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='SBERT model name')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', help='Output JSON file to save results')
    
    args = parser.parse_args()
    
    # Initialize searcher
    searcher = ModelCardSearcher(
        dense_index_path=args.dense_index,
        dense_emb_path=args.dense_emb,
        sparse_index_path=args.sparse_index,
        corpus_jsonl_path=args.corpus_jsonl,
        model_name=args.model_name,
        device=args.device
    )
    
    # Perform search
    print(f"Performing {args.method} search for: '{args.query}'")
    candidates = searcher.search(args.query, args.method, args.topk)
    
    # Display results
    print(f"\nTop-{args.topk} candidates:")
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate}")
    
    # Save results if output file specified
    if args.output:
        results = {
            'query': args.query,
            'method': args.method,
            'topk': args.topk,
            'candidates': candidates
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
