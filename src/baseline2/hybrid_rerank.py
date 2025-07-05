import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_sparse(json_path: str) -> Dict[str, List[str]]:
    """qid -> list[docid] (length <= 101)"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_queries(tsv_path: str) -> Dict[str, str]:
    """qid -> query text"""
    qs = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for ln in f:
            qid, *txt = ln.rstrip("\n").split("\t", 1)
            if txt:
                qs[qid] = txt[0]
    return qs


def load_dense(npz_path: str):
    """Return (docid list, ndarray[float32])"""
    data = np.load(npz_path, allow_pickle=True)
    embs = data["embeddings"].astype("float32")
    ids = data["ids"].tolist()
    return ids, embs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Hybrid rerank: sparse top-k → dense SBERT rerank → top-n"
    )
    ap.add_argument("--sparse-json", required=True,
                    help="JSON produced by search_with_pyserini.py (top-101)")
    ap.add_argument("--queries-tsv", required=True,
                    help="TSV with query_id<TAB>query_text")
    ap.add_argument("--dense-npz", required=True,
                    help="NPZ from table_retrieval_pipeline encode step")
    ap.add_argument("--topk", type=int, default=11,
                    help="final K after rerank (default: 11)")
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SBERT model (default: all-MiniLM-L6-v2)")
    ap.add_argument("--device", default="cpu",
                    help="cuda or cpu")
    ap.add_argument("--output", default="data/tmp/search_result_hybrid.json",
                    help="output JSON file")
    ap.add_argument("--mapping-json", required=True,
                    help="queries_table_mapping.json produced earlier (qid -> orig_id)")
    args = ap.parse_args()

    # 1. load data ---------------------------------------------------------
    print("[load] sparse results …")
    sparse = load_sparse(args.sparse_json)
    print(f"         queries: {len(sparse):,}")

    print("[load] query texts …")
    qtexts = load_queries(args.queries_tsv)
    with open(args.mapping_json, "r", encoding="utf-8") as mf:
        qid_to_orig = json.load(mf)
    # build reverse mapping: orig_id -> q_text
    orig_to_text = {orig: qtexts[qid] for qid, orig in qid_to_orig.items() if qid in qtexts}
    missing_txt = len(set(sparse) - set(orig_to_text))
    if missing_txt:
        print(f"⚠️  {missing_txt} queries have no text, they will be skipped.")

    print("[load] dense embeddings …")
    doc_ids, doc_vecs = load_dense(args.dense_npz)
    # build mapping for O(1) lookup
    id2row = {d: i for i, d in enumerate(doc_ids)}
    print(f"         vectors: {len(doc_ids):,}, dim={doc_vecs.shape[1]}")

    # L2 normalize once upfront
    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    # 2. init model --------------------------------------------------------
    print("[init] SBERT model …")
    model = SentenceTransformer(args.model_name, device=args.device)

    # 3. rerank ------------------------------------------------------------
    results = {}
    for qid, cands in tqdm(sparse.items(), desc="rerank"):
        txt = orig_to_text.get(qid)
        if not txt:
            continue

        # filter candidates that are missing in dense index
        filtered_cands = [d for d in cands if d in id2row]
        if not filtered_cands:
            # fall back to sparse order if no dense embeddings available
            results[qid] = cands[: args.topk]
            continue

        sub_rows = [id2row[d] for d in filtered_cands]
        q_vec = model.encode(txt, normalize_embeddings=True)
        sub_matrix = doc_vecs[sub_rows]  # (k, dim)
        scores = sub_matrix @ q_vec      # (k,)
        top_idx = np.argsort(-scores)[: args.topk]
        top_docs = [filtered_cands[i] for i in top_idx]
        # make sure not to include self docid if present at first position
        results[qid] = top_docs

    # 4. save --------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅  saved hybrid results for {len(results):,} queries → {out_path}")


if __name__ == "__main__":
    main() 