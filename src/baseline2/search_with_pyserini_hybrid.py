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
    ap.add_argument("--sparse-index", required=True)
    ap.add_argument("--dense-index", required=True)
    ap.add_argument("--queries", default=None,
                    help="Path to queries TSV file. If not specified, uses queries_table.tsv or queries_table_<TAG>.tsv if TAG env var is set.")
    ap.add_argument("--mapping", default=None,
                    help="Path to ID mapping JSON file. If not specified, uses queries_table_mapping.json or queries_table_<TAG>_mapping.json if TAG env var is set.")
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSON file path. If not specified, uses search_result_hybrid.json or search_result_hybrid_<TAG>.json if TAG env var is set.")
    args = ap.parse_args()
    
    # Support TAG environment variable for versioning
    tag = os.environ.get('TAG', '')
    suffix = f"_{tag}" if tag else ""
    
    # Set default paths with tag support if not provided
    queries_path = args.queries or f'data/tmp/queries_table{suffix}.tsv'
    mapping_path = args.mapping or f'data/tmp/queries_table{suffix}_mapping.json'

    # ---------------------------------------------------------------------
    # 1. dense side
    # ---------------------------------------------------------------------
    print("[init] dense‑Faiss searcher …")
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384‑d
    encoder = AutoQueryEncoder(
        encoder_name,
        device=args.device,
        pooling="mean",
        l2_norm=True,
    )

    # ---- probe encoder dim (works for any return shape) ------------------
    probe_vec = encoder.encode(["dim_probe"])
    if probe_vec.ndim == 1:
        enc_dim = probe_vec.shape[0]
    else:
        enc_dim = probe_vec.shape[1]

    # ---- read Faiss index dim -------------------------------------------
    idx_path = Path(args.dense_index) / "index.faiss"
    if not idx_path.exists():
        idx_path = Path(args.dense_index) / "index"
    if not idx_path.exists():
        raise FileNotFoundError(f"Faiss index not found under {args.dense_index}")
    idx_dim = faiss.read_index(str(idx_path)).d

    print(f"  ├─ Faiss dim:   {idx_dim}\n  └─ Encoder dim: {enc_dim}")
    if idx_dim != enc_dim:
        raise ValueError(
            "❌ Dimension mismatch – Faiss index is "
            f"{idx_dim}‑d but encoder outputs {enc_dim}‑d.\n"
            "   Re‑encode corpus with the same encoder, or load the encoder used for indexing."
        )

    dense = FaissSearcher(str(Path(args.dense_index)), encoder)

    # ---------------------------------------------------------------------
    # 2. sparse side
    # ---------------------------------------------------------------------
    print("[init] sparse‑Lucene (BM25) searcher …")
    sparse = LuceneSearcher(args.sparse_index)
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
            hits = hybrid.search(query_txt, k=args.k, alpha=args.alpha)
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
    if args.output:
        out_file = Path(args.output)
    else:
        out_file = Path(queries_path).parent / f"search_result_hybrid{suffix}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅  Saved {len(results)} hybrid results → {out_file}")


if __name__ == "__main__":
    main()
