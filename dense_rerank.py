#!/usr/bin/env python3
# dense_rerank_subset.py
"""
  1. 载入  <docid, vector>  二进制文件 (np.load)
  2. 用 SBERT 编码 query 文本
  3. 对每个 query 的 101 个候选 doc 做点积, 取前 10
输出：
  {"qid": "...", "docids": ["d1", "d2", ...]}   (JSON Lines)
"""
import json, sys, pathlib, numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# ---------- paths ----------
sparse_tsv   = 'data/tmp/sparse_top101.tsv'
query_tsv    = 'data/tmp/queries_table.tsv'
doc_vec_npy  = 'data/tmp/index_dense_subset/embeddings.npy.npz'
doc_id_path  = 'data/tmp/index_dense_subset/docids'   # 一行一个 docid
out_path     = 'data/tmp/search_dense_11.json'

# ---------- load doc vectors ----------
doc_vecs = np.load(doc_vec_npy)              # shape (N, 384)
doc_ids  = [l.strip() for l in open(doc_id_path)]
id2vec   = {d:v for d, v in zip(doc_ids, doc_vecs)}

# ---------- build mapping: qid -> query text ----------
qid2text = {}
for line in open(query_tsv):
    qid, text = line.rstrip('\n').split('\t', 1)
    qid2text[qid] = text

# ---------- build mapping: qid -> list(docid) ----------
qid2cands = {}
for line in open(sparse_tsv):
    qid, did = line.rstrip('\n').split('\t')
    qid2cands.setdefault(qid, []).append(did)

# ---------- rerank ----------
with open(out_path, 'w') as fout:
    for qid, cand_list in qid2cands.items():
        q_vec = model.encode(qid2text[qid], normalize_embeddings=True)
        docs  = [d for d in cand_list if d in id2vec]      # 安全过滤
        if not docs:
            continue
        mat   = np.stack([id2vec[d] for d in docs])        # (k, 384)
        scores = mat @ q_vec                               # dot product
        top = np.argsort(-scores)[:10]
        fout.write(json.dumps({'qid': qid,
                               'docids': [docs[i] for i in top]}) + '\n')

print(f'written: {out_path}')
