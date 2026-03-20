#!/usr/bin/env python3
"""
Hybrid retrieval with **BM25 (sparse)** + **Sentence‑BERT (dense)**
=================================================================

Usage example
-------------
```bash
python search_with_pyserini_hybrid.py \
  --sparse-index data/tmp/index \
  --dense-index  data/tmp/index_dense \
  --queries      data/tmp/queries_table.tsv \
  --mapping      data/tmp/queries_table_mapping.json \
  --k 11 --alpha 0.45 --device cpu
```

Revision highlights
-------------------
* **Robust encoder‑dim probe** – avoid the `len(float32)` crash by inspecting the ndarray shape.
* **Dimension guard** – abort early if encoder dim ≠ Faiss dim.
* **Graceful error logging** – never trip on empty exception strings.
* **Token truncation** – log original vs. truncated length.
"""

import argparse
import json
import os
import re
import traceback
from pathlib import Path
import torch
import faiss  # to read Faiss header only; cheap
from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher
from pyserini.search.hybrid import HybridSearcher
from pyserini.search.lucene import LuceneSearcher

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_queries(tsv_file: str):
    qs = {}
    with open(tsv_file, "r", encoding="utf-8") as f:
        for ln in f:
            qid, *txt = ln.rstrip("\n").split("\t", 1)
            if txt:
                qs[qid] = txt[0]
    return qs


def load_mapping(mapping_file: str):
    with open(mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=11)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--tag", type=str, default=None, help="Tag suffix for versioning (e.g., 251117).")
    ap.add_argument("--v2_mode", action="store_true", help="Use v2 mode.")
    args = ap.parse_args()
    
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""
    
    queries_path = f'data/tmp/queries_table{v2_suffix}{suffix}.tsv'
    mapping_path = f'data/tmp/queries_table{v2_suffix}{suffix}_mapping.json'
    sparse_index_path = f'data/tmp/index_sparse{v2_suffix}{suffix}'
    dense_index_path = f'data/tmp/index_dense{v2_suffix}{suffix}/index.faiss'
    out_file = f'data/tmp/search_result_hybrid{v2_suffix}{suffix}.json'
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384‑d
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # 1. dense side
    # ---------------------------------------------------------------------
    print("[init] dense‑Faiss searcher …")
    encoder = AutoQueryEncoder(encoder_name, device=device, pooling="mean", l2_norm=True)

    probe_vec = encoder.encode(["dim_probe"])
    if probe_vec.ndim == 1:
        enc_dim = probe_vec.shape[0]
    else:
        enc_dim = probe_vec.shape[1]

    if not os.path.exists(dense_index_path):
        raise FileNotFoundError(f"Faiss index not found under {dense_index_path}")
    idx_dim = faiss.read_index(str(dense_index_path)).d

    print(f"  ├─ Faiss dim:   {idx_dim}\n  └─ Encoder dim: {enc_dim}")
    if idx_dim != enc_dim:
        raise ValueError(
            "❌ Dimension mismatch – Faiss index is "
            f"{idx_dim}‑d but encoder outputs {enc_dim}‑d.\n"
            "   Re‑encode corpus with the same encoder, or load the encoder used for indexing."
        )

    dense = FaissSearcher(dense_index_path, encoder)

    # ---------------------------------------------------------------------
    # 2. sparse side
    # ---------------------------------------------------------------------
    print("[init] sparse‑Lucene (BM25) searcher …")
    sparse = LuceneSearcher(sparse_index_path)
    sparse.set_bm25()

    # ---------------------------------------------------------------------
    # 3. hybrid searcher
    # ---------------------------------------------------------------------
    hybrid = HybridSearcher(dense, sparse)

    # ---------------------------------------------------------------------
    # 4. data
    # ---------------------------------------------------------------------
    queries = load_queries(queries_path)
    id_map = load_mapping(mapping_path)
    
    debug_log = Path(queries_path).parent / "hybrid_debug.log"
    if debug_log.exists():
        debug_log.unlink()

    results = {}
    total = len(queries)
    max_terms = 1024
    token_pat = re.compile(r"\w+")

    for i, (qid, text) in enumerate(queries.items(), 1):
        if i % 500 == 0 or i == total:
            print(f"[{i}/{total}] {qid}")

        toks = token_pat.findall(text)
        query_txt = " ".join(toks[:max_terms]) if len(toks) > max_terms else text

        try:
            hits = hybrid.search(query_txt, k=args.top_k, alpha=args.alpha)
        except Exception as e:
            title = str(e).splitlines()[0] or type(e).__name__
            print(f"  !! Error for {qid}: {title}, logged to {debug_log}")
            with open(debug_log, "a", encoding="utf-8") as df:
                df.write(f"=== Error for QID={qid} ===\n")
                df.write(f"Original tokens: {len(toks)}, used: {len(token_pat.findall(query_txt))}\n")
                df.write("Query snippet: " + query_txt[:200] + "…\n")
                traceback.print_exc(file=df)
            continue

        orig = id_map.get(qid, qid)
        results[orig] = [h.docid for h in hits if h.docid != orig]

    # ---------------------------------------------------------------------
    # 5. output
    # ---------------------------------------------------------------------
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅  Saved {len(results)} hybrid results → {out_file}")


if __name__ == "__main__":
    main()
