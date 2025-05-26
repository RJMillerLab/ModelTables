"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-16
Description: Build SciLake union benchmark tables.

Usage:
python -m src.data_gt.step3_gt --overlap_rate_threshold 0.0 --rel_key [
    direct_label, 
    direct_label_influential, 
    direct_label_methodology_or_result, 
    direct_label_methodology_or_result_influential, 
    max_pr, 
    max_pr_influential, 
    max_pr_methodology_or_result, 
    max_pr_methodology_or_result_influential
]
"""

import os, json, gzip, pickle, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from enum import Enum
from collections import defaultdict, Counter
from itertools import combinations
from scipy.sparse import csr_matrix, coo_matrix, save_npz


# === Configuration ===
DATA_DIR = "data/processed"
GT_DIR = "data/gt"

# ---- default files (can be overridden by CLI/kwargs) ----
FILES = {
    "combined": f"{DATA_DIR}/modelcard_citation_all_matrices.pkl.gz",
    "step3_dedup": f"{DATA_DIR}/modelcard_step3_dedup.parquet",
    "integration": f"{DATA_DIR}/final_integration_with_paths.parquet",
    "title_list": f"{DATA_DIR}/modelcard_all_title_list.parquet",
    "symlink_mapping": f"{DATA_DIR}/symlink_mapping.pickle",
    "valid_title": f"{DATA_DIR}/all_title_list_valid.parquet"
}

# ===== FACTORIES =========================================================== #
def load_relationships(rel_key: str):
    """Factory loader for paperId‑level relationship graphs."""
    path = FILES["combined"]
    print(f"Loading relationships (combined) from: {path}")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    paper_index = data["paper_index"]
    if rel_key not in data:                           ########
        raise KeyError(
            f"Key '{rel_key}' not found. Available keys: {list(data.keys())[:10]} ...")  ########
    # Select the proper score matrix from the new keys
    score_matrix = data[rel_key]
    return paper_index, score_matrix

def load_table_source(file_key: str="step3_dedup"):
    """Factory loader for table‑list metadata."""
    print(f"Loading table source from: {file_key}")
    # load valid_title, and merge by modelId
    df_valid_title = pd.read_parquet(FILES["valid_title"], columns=["modelId", "all_title_list_valid"])
    df = pd.read_parquet(FILES[file_key], columns=["modelId", "hugging_table_list_dedup", "github_table_list_dedup", "html_table_list_mapped_dedup", "llm_table_list_mapped_dedup"])
    df['all_table_list_dedup'] = df[[
        'hugging_table_list_dedup',
        'github_table_list_dedup',
        'html_table_list_mapped_dedup',
        'llm_table_list_mapped_dedup'
    ]].apply(
        lambda row: [
            x
            for arr in row.tolist()
            if isinstance(arr, (list, tuple, np.ndarray))
            for x in list(arr)
        ],
        axis=1
    )
    df_tables = pd.merge(df[['modelId', 'all_table_list_dedup']], df_valid_title, how="left", on="modelId")
    # drop rows with no valid title or no table list
    print("before filtering invalid rows: ", len(df_tables))
    mask = (
        df_tables['all_title_list_valid'].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        ) &
        df_tables['all_table_list_dedup'].apply(
            lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0
        )
    )
    df_tables = df_tables.loc[mask, ['modelId', 'all_table_list_dedup', 'all_title_list_valid']]
    print("after filtering invalid rows: ", len(df_tables))
    return df_tables.set_index("modelId", drop=False)

def load_symlink_mapping():
    path = FILES["symlink_mapping"]
    print(f"Loading symlink mapping from: {path}")
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    return mapping

def build_paper_matrix(score_matrix: csr_matrix, rel_key: str, overlap_rate_threshold: float):
    if rel_key.startswith("direct_label"):
        paper_adj = (score_matrix >= 1.0).astype(np.bool_)
    else:
        paper_adj = (score_matrix > overlap_rate_threshold).astype(np.bool_)
    paper_adj.setdiag(True)
    return paper_adj.tocsr()

"""def naive_subset_adj(B: csr_matrix, A: csr_matrix):
    B_T = B.transpose().tocsr()
    C = (B_T @ A) @ B
    C.setdiag(True)
    return C"""

def build_title_model_matrix(df_titles_exploded):
    titles = df_titles_exploded['all_title_list'].dropna().unique()
    modelIds = df_titles_exploded['modelId'].dropna().unique()
    title_pos = {title: i for i, title in enumerate(titles)}
    model_pos = {mid: j for j, mid in enumerate(modelIds)}
    rows, cols = [], []
    for _, row in df_titles_exploded.iterrows():
        title, mid = row['all_title_list'], row['modelId']
        if pd.notna(title) and pd.notna(mid):
            rows.append(title_pos[title])
            cols.append(model_pos[mid])
    data = np.ones(len(rows), dtype=bool)
    return csr_matrix((data, (rows, cols)), shape=(len(titles), len(modelIds))), titles.tolist(), modelIds.tolist()

def build_paper_title_matrix(df_metadata, titles):
    paperIds = df_metadata['corpusid'].dropna().unique()
    title_pos = {title: i for i, title in enumerate(titles)}
    paper_pos = {pid: i for i, pid in enumerate(paperIds)}
    rows, cols = [], []
    for _, row in df_metadata.iterrows():
        pid, title = row['corpusid'], row['query']
        if pd.notna(pid) and pd.notna(title) and title in title_pos:
            rows.append(paper_pos[pid])
            cols.append(title_pos[title])
    data = np.ones(len(rows), dtype=bool)
    return csr_matrix((data, (rows, cols)), shape=(len(paperIds), len(titles))), paperIds.tolist()

def compute_subset_pt_tm(FILES):
    """
    Subset titles to only used ones, compute comb_sub full p x m.
    """
    df_metadata = pd.read_parquet(FILES["integration"], columns=["corpusid", "query"])
    df_titles = pd.read_parquet(FILES["title_list"], columns=["modelId", "all_title_list"])
    df_titles_exploded = df_titles.explode("all_title_list")
    tm_mat, titles_list, model_list = build_title_model_matrix(df_titles_exploded)
    pt_mat, paper_list = build_paper_title_matrix(df_metadata, titles_list)
    print(f"pt_mat: {pt_mat.shape}")
    print(f"tm_mat: {tm_mat.shape}")

    #comb_mat = (pt_mat @ tm_mat).tocsr()
    used_pt = set(pt_mat.nonzero()[1])
    used_tm = set(tm_mat.nonzero()[0])
    used = sorted(used_pt.union(used_tm))
    # Subset matrices
    pt_sub = pt_mat[:, used]      # p x k
    tm_sub = tm_mat[used, :]      # k x m
    comb_sub = (pt_sub @ tm_sub).astype(bool).tocsr()  # p x m
    return comb_sub, titles_list, model_list, paper_list

def build_ground_truth(rel_key, overlap_rate_threshold):
    # modelId-paperList-csvList, Our aim is to use paper-paper matrix to build {csv:[csv1, csv2]} related json
    """High‑level orchestration for building GT tables."""
    # ========== Step 1: Load paper-level info. Build paper-level adjacency matrix ==========
    paper_index, score_matrix = load_relationships(rel_key)
    paper_paper_adj = build_paper_matrix(score_matrix, rel_key, overlap_rate_threshold)
    print(f"Step1: [Paper-level] Adjacency shape: {paper_paper_adj.shape}: ", paper_paper_adj.nnz)

    title_df = pd.read_parquet(FILES["integration"], columns=["corpusid", "query"])
    cid2titles = defaultdict(list)
    for cid, title in zip(title_df["corpusid"].astype(str), title_df["query"]):
        cid2titles[cid].append(title)

    # ---------- Step 2: modelId → csv mapping ----------
    df_tables = load_table_source()
    # Filter rows with nonempty paper & csv lists
    rows = []
    for _, row in df_tables.iterrows():
        papers = row['all_title_list_valid']
        csvs   = row['all_table_list_dedup']  # updated col ########
        rows.append((papers, csvs))

    # Build reverse index: paper -> row IDs
    paper2rows = defaultdict(list)
    for rid, (papers, _) in enumerate(rows):
        for p in papers:
            paper2rows[p].append(rid)
    sample_pid = next(iter(paper2rows))
    print(f"[DEBUG] paper2rows first key → {sample_pid!r} , type={type(sample_pid)}")
    print(f"[DEBUG] idx2pid[0] → {paper_index[0]!r} , type={type(paper_index[0])}")
    print(f"[DEBUG] idx2pid[0] == sample?  {paper_index[0]==sample_pid}")

    # Dictionary for flat tuple-key counts
    csv_pair_cnt = defaultdict(int)

    # -- Vectorized CSV pair counting via sparse matrices --
    # build global CSV list & index
    all_csvs = sorted({c for _, cs in rows for c in cs})
    csv2idx  = {c:i for i,c in enumerate(all_csvs)}

    # 1) intra-row: construct B for same-model CSV pairs
    row_b, col_b = [], []
    for _, cs in tqdm(rows, desc="Building intra-row B"):
        for a, b in combinations(sorted(set(cs)), 2):
            ia, ib = csv2idx[a], csv2idx[b]
            row_b += [ia, ib]; col_b += [ib, ia]
    B = coo_matrix((np.ones(len(row_b), int), (row_b, col_b)),
                   shape=(len(all_csvs), len(all_csvs))).tocsr()

    # 2) inter-row: build A (paper→CSV) and compute C = Aᵀ·P·A
    corpus2pidx = {cid:i for i,cid in enumerate(paper_index)}
    title2cid   = {t:cid for cid,titles in cid2titles.items() for t in titles}
    row_i, col_i = [], []
    for titles, cs in tqdm(rows, desc="Building inter-row A indices"):
        for t in titles:
            cid = title2cid.get(t); p = corpus2pidx.get(cid)
            if p is None: continue
            for c in cs:
                row_i.append(p); col_i.append(csv2idx[c])
    A = coo_matrix((np.ones(len(row_i), bool),(row_i,col_i)), shape=(len(paper_index), len(all_csvs))).tocsr()
    C = A.T.dot(paper_paper_adj).dot(A).tocsr()

    # 3) sum and extract
    '''M = (B + C).tocsr()
    processed_csv_adj = {}
    for i in tqdm(range(M.shape[0]), desc="Building processed CSV adjacency mapping"):
        start, end = M.indptr[i], M.indptr[i+1]
        # collect all non-zero columns except self
        nbrs = [j for j in M.indices[start:end] if j != i]
        if nbrs:
            processed_csv_adj[all_csvs[i]] = [ all_csvs[j] for j in nbrs ]'''
    # try to accelerate
    M = (B + C).tocsr()
    matrix_path = f"{GT_DIR}/csv_pair_matrix_{rel_key}.npz"
    save_npz(matrix_path, M)
    print(f"✅ Sparse matrix saved to {matrix_path}")

    indptr   = M.indptr
    indices  = M.indices
    nnz_per_row = np.diff(indptr)                     # 每行非零数

    processed_csv_adj = {}
    rows_with_nbrs = np.where(nnz_per_row > 1)[0]     # 跳过全 0 或仅 self
    print(f"get {len(rows_with_nbrs)} non-empty rows, total {len(indptr)} rows")

    for i in tqdm(rows_with_nbrs, desc=f"Building CSV adjacency ({len(rows_with_nbrs)} non-empty rows)"):
        start, end = indptr[i], indptr[i+1]
        nbrs = indices[start:end]
        if nbrs.size:                                 # 这里一定 >1，但再保险
            # 排除对角线 i
            nbrs = nbrs[nbrs != i]
            if nbrs.size:
                processed_csv_adj[all_csvs[i]] = [all_csvs[j] for j in nbrs.tolist()]
    # keep basename
    processed_csv_adj = {os.path.basename(k): [os.path.basename(x) for x in v] for k, v in processed_csv_adj.items()}
    # Save the processed adjacency mapping directly
    output_adj = f"{GT_DIR}/csv_pair_adj_{rel_key}_processed.pkl"
    with open(output_adj, "wb") as f:
        pickle.dump(processed_csv_adj, f)
    print(f"✅ CSV adjacency mapping saved to {output_adj}")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument("--rel_key", type=str, default='direct_label', help="Exact key inside combined pickle (e.g., 'max_pr', 'direct_label_influential').")
    parser.add_argument("--overlap_rate_threshold", type=float, default=0.0, help=("Numeric threshold for similarity/overlap matrices; ignored for 'direct_label*' keys."))
    args = parser.parse_args()

    build_ground_truth(rel_key=args.rel_key, overlap_rate_threshold=args.overlap_rate_threshold)
