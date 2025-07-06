import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
#from sentence_transformers import SentenceTransformer
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
    ap.add_argument("--queries-tsv", required=False,
                    help="(optional) TSV with query_id<TAB>query_text; falls back to corpus JSONL if omitted")
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
    ap.add_argument("--mapping-json", required=False,
                    help="(optional) mapping JSON; only needed when using --queries-tsv")
    ap.add_argument("--corpus-jsonl", default="data/tmp/corpus/collection_text.jsonl",
                    help="Corpus JSONL that stores metadata text for every table id (default: collection_text.jsonl)")
    args = ap.parse_args()

    # 1. load data ---------------------------------------------------------
    print("[load] sparse results …")
    sparse = load_sparse(args.sparse_json)
    print(f"         queries: {len(sparse):,}")

    # ------------------------------------------------------------------
    # Load query -> text mapping
    # Priority: (1) corpus JSONL   (2) queries TSV + mapping JSON (back-compat)
    # ------------------------------------------------------------------
    print("[load] corpus metadata …")
    corpus_text = {}
    try:
        with open(args.corpus_jsonl, "r", encoding="utf-8") as cf:
            for ln in cf:
                obj = json.loads(ln)
                corpus_text[obj["id"]] = obj["contents"]
    except FileNotFoundError:
        print(f"⚠️  corpus JSONL not found: {args.corpus_jsonl} – will rely on TSV")

    # optional TSV fallback
    orig_to_text = {}
    if args.queries_tsv and args.mapping_json:
        print("[load] TSV query texts (fallback) …")
        qtexts = load_queries(args.queries_tsv)
        with open(args.mapping_json, "r", encoding="utf-8") as mf:
            qid_to_orig = json.load(mf)
        for qid, txt in qtexts.items():
            if qid in qid_to_orig:
                orig_to_text.setdefault(qid_to_orig[qid], txt)
            orig_to_text.setdefault(qid, txt)

    # merge corpus_text with orig_to_text (prefer corpus)
    for k, v in corpus_text.items():
        orig_to_text[k] = v

    missing_txt = len(set(sparse) - set(orig_to_text))
    if missing_txt:
        print(f"⚠️  {missing_txt} queries have no metadata text and will be skipped.")

    print("[load] dense embeddings …")
    doc_ids, doc_vecs = load_dense(args.dense_npz)
    # build mapping for O(1) lookup
    id2row = {d: i for i, d in enumerate(doc_ids)}
    print(f"         vectors: {len(doc_ids):,}, dim={doc_vecs.shape[1]}")

    # L2 normalize once upfront
    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    # 2. init model --------------------------------------------------------
    #print("[init] SBERT model …")
    #model = SentenceTransformer(args.model_name, device=args.device)

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
        # SentenceTransformer.encode returns a Python list by default; force numpy for efficient math
        #q_vec = model.encode(txt, convert_to_numpy=True, normalize_embeddings=True)
        # Retrieve the precomputed query embedding from the same NPZ
        if qid not in id2row:
            print(f"⚠️  query ID {qid} not in dense index, fall back to sparse order")
            # if query ID isn't in dense index, fall back to sparse order
            results[qid] = cands[: args.topk]
            continue
        q_vec = doc_vecs[id2row[qid]]

        sub_matrix = doc_vecs[sub_rows]  # (k, dim)
        scores = sub_matrix @ q_vec      # (k,)
        top_idx = np.argsort(-scores)[: args.topk]
        top_docs = [filtered_cands[i] for i in top_idx]

        # Ensure self-hit is removed (it can be anywhere, not necessarily position-0)
        top_docs = [d for d in top_docs if d != qid]

        # If we removed self-hit or there were < topk dense candidates, fill the remainder
        if len(top_docs) < args.topk:
            for d in cands:
                if d not in top_docs and d != qid:
                    top_docs.append(d)
                if len(top_docs) == args.topk:
                    break

        results[qid] = top_docs

    # 4. save --------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅  saved hybrid results for {len(results):,} queries → {out_path}")


if __name__ == "__main__":
    main() 