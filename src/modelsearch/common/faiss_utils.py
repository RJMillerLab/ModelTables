"""Common FAISS index build and search helpers."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray, index_path: str | Path, nlist: int = 100):
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    print(f"FAISS index written to {index_path}")


def search_index(index_path: str | Path, query_emb: np.ndarray, topk: int = 10):
    index = faiss.read_index(str(index_path))
    distances, indices = index.search(query_emb, topk)
    return distances, indices

