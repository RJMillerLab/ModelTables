"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-16
Description:
    Compute the overlap rate of citation and citing papers based on paperId
    with flexible “modes” controlled by a tiny factory layer.

    Supported relationship modes
    ----------------------------
    • overlap_rate   : uses a (pid → {related_pids}) mapping with weights
    • direct_label  : uses a (pid → {cited_by_pids}) mapping as GT

    Supported table source modes
    ----------------------------
    • step4_symlink  : use step‑4 parquet + *_sym columns
    • step3_dedup   : use step‑3 parquet + raw columns

    Outputs
    -------
    • pairwise overlap scores
    • thresholded related paper pairs
    • direct citation links (if paperId appears in references of another)
    • ground‑truth benchmark table paths (pickle)

Usage:
    python -m src.data_gt.step3_gt --rel_mode [overlap_rate, direct_label] --tbl_mode [step4_symlink, step3_dedup] (require step4.parquet by create_symlink.py)
    python -m src.data_gt.step3_gt --rel_mode direct_label --tbl_mode step4_symlink
    python -m src.data_gt.step3_gt --rel_mode overlap_rate --tbl_mode step4_symlink
"""

import os, json, gzip, pickle, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from enum import Enum
from collections import defaultdict, Counter
from typing import Dict, Iterable, Set, List, Tuple
from joblib import Parallel, delayed
from itertools import combinations, product
from scipy.sparse import csr_matrix, dok_matrix, save_npz, coo_matrix, vstack
import scipy.sparse as sp
from src.data_gt.modelcard_matrix import build_edges_sql, init_edge_db, insert_edges

# === Configuration ===
DATA_DIR = "data/processed"
GT_DIR = "data/gt"
GT_TMP_DIR = "data/gt_tmp"
os.makedirs(GT_TMP_DIR, exist_ok=True)
GT_MODEL_DB = "data/tmp/gt_mod_mod.db"

GT_COMBINED_PATH = f"{GT_DIR}/scilake_gt_all_matrices.pkl.gz"

# ---- default files (can be overridden by CLI/kwargs) ----
FILES = {
    "combined": f"{DATA_DIR}/modelcard_citation_all_matrices.pkl.gz",
    "step4_symlink": f"{DATA_DIR}/modelcard_step4.parquet",
    "step3_dedup": f"{DATA_DIR}/modelcard_step3_dedup.parquet",
    "integration": f"{DATA_DIR}/final_integration_with_paths.parquet",
    "title_list": f"{DATA_DIR}/modelcard_all_title_list.parquet",
    "symlink_mapping": f"{DATA_DIR}/symlink_mapping.pickle",
    "valid_title": f"{DATA_DIR}/all_title_list_valid.parquet"
}

MODEL_GT_PATH = f"{GT_DIR}/groundtruth_model_pairs.pickle"
MAPPED_GT_PATH = f"{GT_DIR}/groundtruth_mapped_pairs.pickle"
CSV_PAIR_FREQ_PATH = f"{GT_DIR}/csv_pair_frequency.pickle"

OUTPUT_GT_PATH = f"{GT_DIR}/scilakeUnionBenchmark_by_ids.pickle"
DISCOUNT_RATE = 0.5
IS_STRICT_MODE = True
THRESHOLD = 0.1 # for overlap_rate only

# ===== ENUMS =============================================================== #
class RelationshipMode(str, Enum):
    OVERLAP_RATE = "overlap_rate"
    DIRECTED_CITE = "direct_label"

class TableSourceMode(str, Enum):
    STEP4_SYMLINK = "step4_symlink"
    STEP3_MERGED = "step3_dedup"

# ===== FACTORIES =========================================================== #
def load_relationships(mode: RelationshipMode):
    """Factory loader for paperId‑level relationship graphs."""
    path = FILES["combined"]
    print(f"Loading relationships (combined) from: {path}")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    paper_index = data["paper_index"]
    # Select the proper score matrix from the new keys
    if mode == RelationshipMode.OVERLAP_RATE:
        score_matrix = data["max_pr"]                             
    elif mode == RelationshipMode.DIRECTED_CITE:
        score_matrix = data["direct_label"]
    else:
        raise ValueError(f"Unsupported relationship mode: {mode}")
    return paper_index, score_matrix

def load_table_source(mode: TableSourceMode):
    """Factory loader for table‑list metadata."""
    print(f"Loading table source ({mode.value}) from: {FILES[mode.value]}")
    # load valid_title, and merge by modelId
    df_valid_title = pd.read_parquet(FILES["valid_title"], columns=["modelId", "all_title_list_valid", "has_title", "has_valid_title"])
    df = pd.read_parquet(FILES[mode.value], columns=["modelId", "hugging_table_list_sym", "github_table_list_sym", "html_table_list_sym", "llm_table_list_sym", "hugging_table_list_dedup", "github_table_list_dedup", "html_table_list_mapped_dedup", "llm_table_list_mapped_dedup"])
    df_tables = pd.merge(df, df_valid_title, how="left", on="modelId")
    return df_tables.set_index("modelId", drop=False)

def load_symlink_mapping():
    path = FILES["symlink_mapping"]
    print(f"Loading symlink mapping from: {path}")
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    return mapping

# ===== HELPER FUNCTIONS FOR PARALLEL PROCESSING ============================
def process_model_pair(pair, df_tables, use_symlink=True):
    """Process a single model pair to extract CSV pair edges.
    Returns a tuple: ((m1, m2), set_of_csv_edges)
    """
    m1, m2 = pair
    csv_list_m1 = []
    csv_list_m2 = []
    if m1 in df_tables.index:
        csv_list_m1 = extract_csv_list(df_tables.loc[m1], use_symlink)
    if m2 in df_tables.index:
        csv_list_m2 = extract_csv_list(df_tables.loc[m2], use_symlink)
    union_csv = sorted(set(csv_list_m1).union(set(csv_list_m2)))
    csv_edges = set()
    for i in range(len(union_csv)):
        for j in range(i + 1, len(union_csv)):
            csv_edges.add((union_csv[i], union_csv[j]))
    return (pair, csv_edges)

def map_csv_edges(result, symlink_map):
    """Map CSV symlink edges back to their real paths.
    result: ((m1, m2), csv_edges)
    Returns: ((m1, m2), new_csv_edges)
    """
    pair, csv_edges = result
    new_csv_edges = set()
    for csv1, csv2 in csv_edges:
        new_csv_edges.add((symlink_map[csv1], symlink_map[csv2]))
    return (pair, new_csv_edges)

# ===== BUSINESS LOGIC ====================================================== #
def extract_csv_list(row: pd.Series, use_symlink: bool):
    if use_symlink:
        cols = ["hugging_table_list_sym", "github_table_list_sym", "html_table_list_sym", "llm_table_list_sym"]
    else:
        cols = ["hugging_table_list_dedup", "github_table_list_dedup", "html_table_list_mapped_dedup", "llm_table_list_mapped_dedup"]
    csv_list, seen = [], set()
    for col in cols:
        csv_list.extend(row.get(col, []))
    unique = []
    for csv in csv_list:
        if pd.isna(csv) or csv in seen:
            continue
        unique.append(csv)
        seen.add(csv)
    return unique

def build_csv_matrix(model_adj: csr_matrix, model_index: list, modelId_to_csvs: Dict[str, list], csv_index: list):
    csv_pos = {c: i for i, c in enumerate(csv_index)}
    # Build model→csv binary matrix (COO)
    m_rows, m_cols = [], []
    for m_idx, mid in enumerate(model_index):
        for c in modelId_to_csvs.get(mid, []):
            j = csv_pos.get(c)
            if j is not None:
                m_rows.append(m_idx)
                m_cols.append(j)
    data = np.ones(len(m_rows), dtype=np.bool_)
    M2C = coo_matrix((data, (m_rows, m_cols)), shape=(len(model_index), len(csv_index))).tocsr()
    csv_adj = (M2C.astype(bool).T @ (model_adj.astype(bool) @ M2C.astype(bool))).astype(bool)
    csv_adj.setdiag(False)
    csv_adj.eliminate_zeros()
    return csv_adj


def build_csv_matrix_from_blocks(
    block_files: List[str],
    model_index: List[str],
    model_to_csvs: Dict[str, List[str]],
) -> Tuple[csr_matrix, List[str]]:
    """Convert streamed MODEL pairs into CSV adjacency."""
    # 1) global CSV index
    all_csv = {c for lst in model_to_csvs.values() for c in lst}
    csv_index = sorted(all_csv)
    pos_csv = {c: i for i, c in enumerate(csv_index)}

    # 2) pre‑compute model → csv‑pos list
    model_csv_pos = [
        [pos_csv[c] for c in model_to_csvs.get(mid, []) if c in pos_csv]
        for mid in model_index
    ]

    rows_c, cols_c = [], []

    # 2a) inter‑model edges
    for r_models, c_models in iter_block_files(block_files):
        # guard against indices larger than model_csv_pos length
        max_valid = len(model_csv_pos) - 1
        mask = (r_models <= max_valid) & (c_models <= max_valid)
        if not np.any(mask):
            continue
        r_models = r_models[mask]
        c_models = c_models[mask]

        for u, v in zip(r_models, c_models):
            pos_u, pos_v = model_csv_pos[u], model_csv_pos[v]
            if not pos_u or not pos_v:
                continue
            for i in pos_u:
                for j in pos_v:
                    if i != j:
                        rows_c.append(i); cols_c.append(j)

    # 2b) intra‑model fully connected (excl self)            
    for pos_list in model_csv_pos:
        if len(pos_list) > 1:
            for i in pos_list:
                for j in pos_list:
                    if i != j:
                        rows_c.append(i); cols_c.append(j)

    if not rows_c:
        return csr_matrix((len(csv_index), len(csv_index)), dtype=bool), csv_index

    data = np.ones(len(rows_c), dtype=bool)
    adj = coo_matrix((data, (rows_c, cols_c)), shape=(len(csv_index), len(csv_index))).tocsr()
    adj.setdiag(False)
    adj.eliminate_zeros()
    return adj, csv_index


def map_csv_adj_to_real(csv_adj: csr_matrix, csv_index: list, symlink_map: Dict[str, str]):
    """
    map 0/1 symlink‑adjacency back to real‑path adjacency
    """
    real_paths = {symlink_map[p] for p in csv_index}
    real_index = sorted(real_paths)
    pos = {p: i for i, p in enumerate(real_index)}
    real_adj = dok_matrix((len(real_index), len(real_index)), dtype=np.int32)

    rows, cols = csv_adj.nonzero()
    for r, c in zip(rows, cols):
        real_r = pos[symlink_map[csv_index[r]]]
        real_c = pos[symlink_map[csv_index[c]]]
        if real_r == real_c:
            continue
        real_adj[real_r, real_c] += 1
    return real_adj.tocsr(), real_index

def query_related_ids(score_matrix, id_list, query_id, threshold=1.0):
    """
    Given an ID list and a sparse score matrix, return all IDs related to `query_id`
    with score >= threshold.

    Parameters:
    - score_matrix: csr_matrix of shape (n_items, n_items)
    - id_list: list of IDs (str)
    - query_id: the ID to query (str)
    - threshold: filter threshold (float)

    Returns:
    - List of related IDs
    """
    try:
        idx = id_list.index(query_id)
    except ValueError:
        raise ValueError(f"ID {query_id} not found in index list")
    row = score_matrix.getrow(idx)
    related_indices = row.indices[row.data >= threshold]
    return [id_list[i] for i in related_indices]

def build_paper_matrix(score_matrix: csr_matrix, paper_index: list, rel_mode: RelationshipMode, thr: float=THRESHOLD):
    thr = 1.0 if rel_mode == RelationshipMode.DIRECTED_CITE else thr
    paper_adj = (score_matrix >= thr).astype(np.bool_)
    paper_adj.setdiag(True)
    return paper_adj.tocsr()

def bool_matmul_csr(X: csr_matrix,
                    Y: csr_matrix,
                    block_rows: int = 2048) -> csr_matrix:
    """
    Memory-safe Boolean sparse matmul (C = X @ Y over OR/AND).
    • Only keep the product result of the current row-block, and extract coordinates immediately.
    • Build the sparse matrix with COO at the end to avoid high memory usage.
    """
    # --- empty quick-return --------------------------------------- ########
    if X.nnz == 0 or Y.nnz == 0:
        return csr_matrix((X.shape[0], Y.shape[1]), dtype=bool)

    X_int = X.astype(np.int8, copy=False)                           ########
    Y_int = Y.astype(np.int8, copy=False)                           ########

    m, p = X.shape[0], Y.shape[1]                                   ########
    row_acc, col_acc = [], []                                        ########

    for i0 in range(0, m, block_rows):                         ########
        i1 = min(i0 + block_rows, m)                                ########
        blk = (X_int[i0:i1] @ Y_int)                                ########
        r, c = blk.nonzero()                                        ########
        if r.size:                                                  ########
            row_acc.append(r + i0)                                  ########
            col_acc.append(c)                                       ########
        del blk                                                     ########

    if not row_acc:                                                 ########
        return csr_matrix((m, p), dtype=bool)                       ########

    rows = np.concatenate(row_acc)                                  ########
    cols = np.concatenate(col_acc)                                  ########
    data = np.ones_like(rows, dtype=bool)                           ########
    return coo_matrix((data, (rows, cols)), shape=(m, p)).tocsr()   ########

"""def naive_subset_adj(B: csr_matrix, A: csr_matrix):
    B_T = B.transpose().tocsr()
    C = (B_T @ A) @ B
    C.setdiag(True)
    return C"""

'''def build_model_matrix(comb_mat: csr_matrix,
                       paper_adj: csr_matrix,
                       block_size: int = 1000,
                       tmp_dir: str = "tmp_blocks"):
    """
    Build model-level adjacency via double-streaming blocks:
      1) Filter comb_mat (papers×models) rows and align paper_adj.
      2) Determine participating model indices (idx_C) using fast dot.
      3) For each block of idx_C:
         a) X_block = comb_sub.T[idx_block, :]
         b) C_block = bool_matmul_csr(X_block, pa_sub)
         c) sub_adj = bool_matmul_csr(C_block, comb_sub)
         d) Extract (r,c) coords, map to global, save block .npz
      4) Load all .npz, concatenate coords, assemble final adjacency matrix.
    """
    os.makedirs(tmp_dir, exist_ok=True)                         ########

    # 1) Filter comb_mat rows & align paper_adj                    ########
    paper_rows = np.array(comb_mat.sum(axis=1)).ravel() > 0     ########
    comb_sub = comb_mat[paper_rows, :].astype(bool)            ########
    adj_sub = paper_adj[paper_rows, :]                         ########

    # Filter adj_sub columns with any citation                     ########
    cols_sel = np.array(adj_sub.sum(axis=0)).ravel() > 0       ########
    pa_sub = adj_sub[:, cols_sel].astype(bool)                 ########

    M_models = comb_sub.shape[1]                               ########
    orig_model_idx = np.arange(M_models)                       ########

    # 2) Determine participating models                           ########
    mask = np.array(pa_sub.sum(axis=1)).ravel() > 0            ########
    counts = comb_sub.T.dot(mask.astype(np.int8))             ########
    idx_C = np.flatnonzero(counts)                             ########

    # 3) Block-wise compute adjacency                              ########
    for start in tqdm(range(0, len(idx_C), block_size), desc="streaming bool matmul"):             ########
        end = min(start + block_size, len(idx_C))              ########
        idx_block = idx_C[start:end]                           ########
        # a) extract block of models→papers                        ########
        X_block = comb_sub.transpose().tocsr()[idx_block, :]   ########
        # b) compute model_block→paper_block                       ########
        C_block = bool_matmul_csr(X_block, pa_sub)             ########
        # c) compute model_block→model_all                         ########
        sub_adj = bool_matmul_csr(C_block, comb_sub)           ########
        # d) extract non-zero coords and map to global indices     ########
        r, c = sub_adj.nonzero()                              ########
        rows_glob = orig_model_idx[idx_block[r]]              ########
        cols_glob = orig_model_idx[c]                         ########
        # save coords per block                                   ########
        np.savez(os.path.join(tmp_dir, f"block_{start}.npz"), rows=rows_glob, cols=cols_glob)########
        # free block memory                                       ########
        del X_block, C_block, sub_adj

    # 4) Load all blocks, concatenate coords                      ########
    row_list, col_list = [], []                                ########
    for fname in sorted(os.listdir(tmp_dir)):                  ########
        if fname.startswith("block_") and fname.endswith(".npz"):########
            data = np.load(os.path.join(tmp_dir, fname))      ########
            row_list.append(data['rows'])                     ########
            col_list.append(data['cols'])                     ########
    if not row_list:                                           ########
        return csr_matrix((M_models, M_models), dtype=bool)   ########
    rows = np.concatenate(row_list)                            ########
    cols = np.concatenate(col_list)                            ########

    # assemble final adjacency                                   ########
    model_adj = coo_matrix((np.ones_like(rows, dtype=bool),   ########
                            (rows, cols)),                    ########
                           shape=(M_models, M_models)).tocsr()########
    model_adj.setdiag(True)                                     ########
    model_adj.eliminate_zeros()                                 ########
    return model_adj'''

def build_model_matrix(comb_mat: csr_matrix,
                            paper_adj: csr_matrix,
                            db_path: str = GT_MODEL_DB,
                            block_size: int = 1000):
    """
    Write (model_i, model_j) edges directly into SQLite.
    comb_mat: papers×models bool   (pt_sub @ tm_sub)
    paper_adj: papers×papers bool  (paper-level adjacency)
    """
    paper_rows = np.array(comb_mat.sum(axis=1)).ravel() > 0
    comb_sub   = comb_mat[paper_rows, :].astype(bool)
    adj_sub    = paper_adj[paper_rows, :]
    cols_sel   = np.array(adj_sub.sum(axis=0)).ravel() > 0
    pa_sub     = adj_sub[:, cols_sel].astype(bool)
    M_models   = comb_sub.shape[1]
    orig_idx   = np.arange(M_models)
    mask       = np.array(pa_sub.sum(axis=1)).ravel() > 0
    counts     = comb_sub.T.dot(mask.astype(np.int8))
    idx_C      = np.flatnonzero(counts)
    conn, cur = init_edge_db(db_path)
    print(f"→ Streaming model edges into {db_path}")
    for start in tqdm(range(0, len(idx_C), block_size), desc="bool-matmul blocks"):
        end       = min(start + block_size, len(idx_C))
        idx_block = idx_C[start:end]
        X_block   = comb_sub.transpose().tocsr()[idx_block, :]
        C_block   = bool_matmul_csr(X_block, pa_sub)
        sub_adj   = bool_matmul_csr(C_block, comb_sub)
        r, c      = sub_adj.nonzero()
        if r.size == 0:
            continue
        def edge_iter():
            for i, j in zip(r, c):
                yield (orig_idx[idx_block[i]], orig_idx[j])
        insert_edges(cur, edge_iter())
        if start // block_size % 20 == 0:
            conn.commit()
        del X_block, C_block, sub_adj
    conn.commit()
    conn.close()
    print("✓ All model edges written to DB")

def iter_block_files(block_files):                                ########
    """Yield (rows, cols) numpy arrays from each .npz block."""     ########
    for path in block_files:                                       ########
        data = np.load(path)                                       ########
        yield data['rows'], data['cols']                           ########

def adjacency_to_dict(adj: csr_matrix, index_list: list):
    print("Converting adjacency matrix to ground-truth dictionary ...")
    gt_dict = defaultdict(list)
    rows, cols = adj.nonzero()
    for i, j in zip(rows, cols):
        if i == j:
            continue
        src = index_list[i]
        tgt = index_list[j]
        gt_dict[src].append(tgt)
    return dict(gt_dict)

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

def build_ground_truth(rel_mode: RelationshipMode = RelationshipMode.OVERLAP_RATE, tbl_mode: TableSourceMode = TableSourceMode.STEP4_SYMLINK, use_symlink = True):
    suffix = f"__{rel_mode.value}"
    """High‑level orchestration for building GT tables."""
    # ========== Step 1: Load paper-level info ==========
    paper_index, score_matrix = load_relationships(rel_mode)
    print(f"Step1: [Paper-level] Index shape: {len(paper_index)}. Score matrix shape: {score_matrix.shape}")
    # ========== Step 2: Build paper-level adjacency matrix ==========
    paper_paper_adj = build_paper_matrix(score_matrix, paper_index, rel_mode)
    with open(f"{GT_TMP_DIR}/paper_paper_adj_{suffix}.pkl", "wb") as f:
        pickle.dump(paper_paper_adj, f)
    with open(f"{GT_TMP_DIR}/paper_index_{suffix}.pkl", "wb") as f:
        pickle.dump(paper_index, f)
    del score_matrix, paper_index

    print(f"Step2: [Paper-level] Adjacency shape: {paper_paper_adj.shape}")
    print(paper_paper_adj[0])
    # ========== Step 3: Build paperId -> modelIds mapping ==========
    # Load metadata
    comb_mat, titles_list, model_list, paper_list = compute_subset_pt_tm(FILES)
    print(f"comb_mat: {comb_mat.shape}")
    # ========== Step 4: Build model-level adjacency matrix from the paper-level adjacency ==========

    time_start = time.time()
    cols = comb_mat.nonzero()[1]
    model_index = sorted({model_list[j] for j in cols})
    print('len(model_index): ', len(model_index))
    build_model_matrix(comb_mat, paper_paper_adj, db_path=GT_MODEL_DB)
    #model_model_adj = build_model_matrix(comb_mat, paper_paper_adj, db_path=GT_MODEL_DB)
    #with open(f"{GT_TMP_DIR}/model_model_adj_{suffix}.pkl", "wb") as f:
    #    pickle.dump(model_model_adj, f)
    del paper_paper_adj
    print(f"Time taken to build model-level adjacency matrix: {time.time() - time_start:.2f} seconds")
    #print(f"[Model-level] Adjacency shape: {model_model_adj.shape}")

    tmp_dir = "tmp_blocks"
    block_files = [os.path.join(tmp_dir, f) for f in sorted(os.listdir(tmp_dir)) if f.startswith("block_") and f.endswith(".npz")]
    # ---------- Step 4.5: modelId → csv mapping ----------
    df_tables = load_table_source(TableSourceMode.STEP4_SYMLINK) # fixed here! tbl_mode
    modelId_to_csvs_sym = {mid: extract_csv_list(row, use_symlink=True) for mid, row in df_tables.iterrows()}
    modelId_to_csvs_real = {mid: extract_csv_list(row, use_symlink=False) for mid, row in df_tables.iterrows()}
    print('shape of modelId_to_csvs_sym: ', len(modelId_to_csvs_sym))
    print('shape of modelId_to_csvs_real: ', len(modelId_to_csvs_real))
    del df_tables
    # ---------- Step 5: Build CSV‑level adjacency ----------
    csv_index = sorted({c for mid in model_index for c in modelId_to_csvs_real.get(mid, [])})
    #csv_csv_adj = build_csv_matrix(model_model_adj, model_index, modelId_to_csvs_real, csv_index)
    csv_csv_adj = build_csv_matrix_from_blocks(block_files, model_index, modelId_to_csvs_real)
    # save csv_index
    with open(f"{GT_TMP_DIR}/csv_index_{suffix}.pkl", "wb") as f:
        pickle.dump(csv_index, f)
    del csv_index
    # save csv_csv_adj
    with open(f"{GT_TMP_DIR}/csv_csv_adj_{suffix}.pkl", "wb") as f:
        pickle.dump(csv_csv_adj, f)
    del csv_csv_adj
    print(f"[CSV‑level] Adjacency shape: {csv_csv_adj.shape}")
     # ---------- Step 5a: symlink‑level (0/1, no self-loop) ----------
    csv_symlink_index = sorted({c for mid in model_index for c in modelId_to_csvs_sym.get(mid, [])})
    #csv_symlink_adj = build_csv_matrix(model_model_adj, model_index, modelId_to_csvs_sym, csv_symlink_index)
    csv_symlink_adj = build_csv_matrix_from_blocks(block_files, model_index, modelId_to_csvs_sym)
    # save model_model_adj, model_index
    
    with open(f"{GT_TMP_DIR}/model_index_{suffix}.pkl", "wb") as f:
        pickle.dump(model_index, f)
    del model_model_adj, model_index
    print(f"[CSV‑symlink] shape: {csv_symlink_adj.shape}")
    # ---------- Step 5b: real‑path level (count, no self-loop) ----------
    symlink_map = load_symlink_mapping()
    csv_real_adj, csv_real_index = map_csv_adj_to_real(csv_symlink_adj, csv_symlink_index, symlink_map)
    # save csv_symlink_adj, csv_symlink_index
    with open(f"{GT_TMP_DIR}/csv_symlink_adj_{suffix}.pkl", "wb") as f:
        pickle.dump(csv_symlink_adj, f)
    with open(f"{GT_TMP_DIR}/csv_symlink_index_{suffix}.pkl", "wb") as f:
        pickle.dump(csv_symlink_index, f)
    del csv_symlink_adj, csv_symlink_index
    print(f"[CSV‑real]    shape: {csv_real_adj.shape}")
    csv_real_gt = adjacency_to_dict(csv_real_adj, csv_real_index)
    # only record basename path
    csv_real_gt = {os.path.basename(k): [os.path.basename(vv) for vv in v] for k, v in csv_real_gt.items()}
    # save csv_real_gt
    with open(f"{GT_TMP_DIR}/csv_real_gt_{suffix}.pkl", "wb") as f:
        pickle.dump(csv_real_gt, f)
    del csv_real_gt
    # ========== Step 5: Save everything to disk ==========
    # ---- extra: save real‑path count adjacency (as dict) ----
    count_dict = defaultdict(dict)
    rows, cols = csv_real_adj.nonzero()
    for i, j in zip(rows, cols):
        if i == j:
            continue
        src = os.path.basename(csv_real_index[i])
        tgt = os.path.basename(csv_real_index[j])
        count_dict[src][tgt] = int(csv_real_adj[i, j])
    # save csv_real_count
    with open(f"{GT_TMP_DIR}/csv_real_count_{suffix}.pkl", "wb") as f:
        pickle.dump(dict(count_dict), f)
    del count_dict
    # save csv_real_adj
    with open(f"{GT_TMP_DIR}/csv_real_adj_{suffix}.pkl", "wb") as f:
        pickle.dump(csv_real_adj, f)
    del csv_real_adj
    # save csv_real_index
    with open(f"{GT_TMP_DIR}/csv_real_index_{suffix}.pkl", "wb") as f:
        pickle.dump(csv_real_index, f)
    del csv_real_index
    """combined = {
        "paper_adj":       paper_paper_adj,                        
        "paper_index":     paper_index,                            
        "model_adj":       model_model_adj,                        
        "model_index":     model_index,                            
        "csv_adj":         csv_csv_adj,                            
        "csv_index":       csv_index,                              
        "csv_symlink_adj": csv_symlink_adj,                        
        "csv_symlink_index": csv_symlink_index,                    
        "csv_real_adj":    csv_real_adj,                           
        "csv_real_index":  csv_real_index,                         
        "csv_real_gt":     csv_real_gt,                            
        "csv_real_count":  dict(count_dict),                       
    }
    with gzip.open(GT_COMBINED_PATH.replace(".pkl.gz", f"{suffix}.pkl.gz"), "wb") as f:
        pickle.dump(combined, f)
    print(f"✔️  All matrices & indices saved to {GT_COMBINED_PATH}{suffix}")"""                                   

    print("✅ Done building matrix-based groundtruth!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument("--rel_mode", choices=[m.value for m in RelationshipMode], default=RelationshipMode.OVERLAP_RATE.value, help="Relationship graph mode.")
    parser.add_argument("--tbl_mode", choices=[m.value for m in TableSourceMode], default=TableSourceMode.STEP4_SYMLINK.value, help="Table source mode.")
    args = parser.parse_args()

    build_ground_truth(
        rel_mode=RelationshipMode(args.rel_mode),
        tbl_mode=TableSourceMode(args.tbl_mode),
        use_symlink=True
    )
