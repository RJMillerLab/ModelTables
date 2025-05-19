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
from typing import Dict, Iterable, Set
from joblib import Parallel, delayed
from itertools import combinations, product
from scipy.sparse import csr_matrix, dok_matrix, save_npz, coo_matrix
import suitesparse_graphblas as gb
import scipy.sparse as sp

# === Configuration ===
DATA_DIR = "data/processed"
GT_DIR = "data/gt"

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
    path = FILES[mode.value]
    print(f"Loading table source ({mode.value}) from: {path}")
    # load valid_title, and merge by modelId
    df_valid_title = pd.read_parquet(FILES["valid_title"], columns=["modelId", "all_title_list_valid", "has_title", "has_valid_title"])
    df = pd.read_parquet(path)
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
    return csv_adj, csv_index

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

'''def naive_model_adj(B: csr_matrix, A: csr_matrix):
    C = (B.T @ A).astype(bool)
    adj = (C @ B).astype(bool).tocsr()
    adj.setdiag(True)
    return adj
'''

def compute_subset_adj_bit(B: csr_matrix, A: csr_matrix):
    """
    Boolean semiring multiplication via bit-packing:
      C = B.T @ A
      adj = C @ B
    but implemented with packbits + bitwise_and + any over bits,
    two for-loops (with tqdm) over models.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from tqdm import tqdm

    P, M = B.shape  # B: P papers × M models

    # 1) Pack B columns (each model → P-bit vector)
    B_arr = B.toarray().astype(np.bool_)         # P × M
    B_packed = np.packbits(B_arr, axis=0)        # (P/8) × M

    # 2) Pack A columns (each target paper column → P-bit vector)
    A_arr = A.toarray().astype(np.bool_)         # P × P
    A_packed = np.packbits(A_arr, axis=0)        # (P/8) × P

    # 3) Compute C = B.T @ A via bitwise
    C_arr = np.zeros((M, P), dtype=bool)         # to fill with boolean results
    nbytes = B_packed.shape[0]
    for i in tqdm(range(M), desc="Compute C = B.T @ A"):
        col_bits = B_packed[:, i:i+1]            # (nbytes × 1)
        common = np.bitwise_and(col_bits, A_packed)  # (nbytes × P)
        bits = np.unpackbits(common, axis=0, count=P)  # (P × P) in bits
        C_arr[i, :] = bits.any(axis=0)           # OR over papers

    # 4) Pack C rows for next multiply
    C_packed = np.packbits(C_arr.astype(np.uint8), axis=1)  # M × (P/8)

    # 5) Compute adj = C @ B via bitwise
    adj = np.zeros((M, M), dtype=bool)
    for i in tqdm(range(M), desc="Compute adj = C @ B"):
        row_bits = C_packed[i:i+1, :].T           # (nbytes × 1)
        common = np.bitwise_and(row_bits, B_packed)  # (nbytes × M)
        bits = np.unpackbits(common, axis=0, count=P) # (P × M)
        adj[i, :] = bits.any(axis=0)

    # 6) to sparse CSR
    return csr_matrix(adj)

def bool_matmul_csr(X: csr_matrix, Y: csr_matrix) -> csr_matrix:
    """
    Boolean sparse matrix multiplication: C = X @ Y over Boolean semiring.
    X: (m×n) csr, Y: (n×p) csr
    For each row i in X, C[i, j] = OR_k (X[i,k] AND Y[k,j]).
    """
    m, _ = X.shape
    _, p = Y.shape
    rows, cols = [], []
    # Pre-fetch Y row indices
    Y_indptr = Y.indptr
    Y_indices = Y.indices
    for i in tqdm(range(m), desc="Boolean matmul rows"):
        row_start, row_end = X.indptr[i], X.indptr[i+1]
        ks = X.indices[row_start:row_end]
        if ks.size == 0:
            continue
        neigh = set()
        for k in ks:
            y_start, y_end = Y_indptr[k], Y_indptr[k+1]
            neigh.update(Y_indices[y_start:y_end])
        for j in neigh:
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=bool)
    return csr_matrix((data, (rows, cols)), shape=(m, p), dtype=bool)
# corrected block partition over models axis
def compute_subset_adj(B: csr_matrix, A: csr_matrix):
    # 1) Compute C = B.T @ A
    C = bool_matmul_csr(B.transpose().tocsr(), A)
    print('shape of C: ', C.shape)
    # 2) Compute adj = C @ B
    adj = bool_matmul_csr(C, B)
    # 3) set diagonal True
    adj.setdiag(True)
    return adj

def build_model_matrix(comb_mat: csr_matrix, paper_adj: csr_matrix):   
    """
    Return a model-level Boolean adjacency (M × M) using either:
      • GraphBLAS Boolean semiring  —— memory-safe & fast
      • Fallback blocked SciPy path —— if GraphBLAS unavailable
    """
    print('shape of comb_mat: ', comb_mat.shape)
    print('shape of paper_adj: ', paper_adj.shape)
    # remove cols with all zeros
    row_nz = np.array(comb_mat.sum(axis=1)).ravel() > 0
    comb_mat = comb_mat[row_nz, :]
    comb_mat = comb_mat.astype(np.bool_)
    # remove rows with all zeros
    col_nz = np.array(paper_adj.sum(axis=0)).ravel() > 0
    paper_adj = paper_adj[:, col_nz]
    paper_adj = paper_adj.astype(np.bool_)
    print('after filtering, shape of comb_mat: ', comb_mat.shape)
    print('after filtering, shape of paper_adj: ', paper_adj.shape)
    """B_gb = gb.Matrix.from_scipy_sparse(comb_mat)
    A_gb = gb.Matrix.from_scipy_sparse(paper_adj)                 
    M_gb = (B_gb.T @ A_gb) @ B_gb                                  
    model_adj = M_gb.to_scipy_sparse().astype(np.bool_)            """
    model_adj = compute_subset_adj(comb_mat.astype(bool), paper_adj.astype(bool))    
    #model_adj = compute_subset_adj_bit(comb_mat.astype(bool), paper_adj.astype(bool))
    model_adj.setdiag(True)
    model_adj.eliminate_zeros()
    return model_adj

'''def build_model_matrix(comb_mat, paper_adj):
    """Edge‑scan implementation (memory‑light)."""
    paper2models = [set(comb_mat.getrow(i).indices) for i in range(comb_mat.shape[0])]
    links = defaultdict(set)
    # 1. cross‑paper edges
    rows, cols = paper_adj.nonzero()
    for p1, p2 in tqdm(zip(rows, cols), desc="Building model adjacency between papers"):
        if p1 > p2:   # include self later
            for m1 in paper2models[p1]:
                for m2 in paper2models[p2]:
                    if m1 == m2:
                        continue
                    links[m1].add(m2)
                    links[m2].add(m1)
    # 2. within‑paper edges (diagonal of A)
    for mods in tqdm(paper2models, desc="Building model adjacency within paper"):
        if len(mods) > 1:
            ms = list(mods)
            for i in range(len(ms)):
                for j in range(i+1, len(ms)):
                    m1, m2 = ms[i], ms[j]
                    links[m1].add(m2)
                    links[m2].add(m1)
    n_models = comb_mat.shape[1]
    if not links:
        adj = sp.identity(n_models, dtype=bool, format='csr')
        return adj
    r, c = [], []
    for m1, nbrs in tqdm(links.items(), desc="Building model adjacency between models"):
        for m2 in nbrs:
            r.append(m1); c.append(m2)
    data = np.ones(len(r), dtype=bool)
    adj = sp.csr_matrix((data, (r, c)), shape=(n_models, n_models), dtype=bool)
    adj.setdiag(True)
    adj.eliminate_zeros()
    return adj'''

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

def compute_subset_pt_tm(pt_mat, tm_mat):
    """
    Subset titles to only used ones, compute comb_sub full p x m.
    """
    used_pt = set(pt_mat.nonzero()[1])
    used_tm = set(tm_mat.nonzero()[0])
    used = sorted(used_pt.union(used_tm))
    # Subset matrices
    pt_sub = pt_mat[:, used]      # p x k
    tm_sub = tm_mat[used, :]      # k x m
    comb_sub = (pt_sub @ tm_sub).astype(bool).tocsr()  # p x m
    return comb_sub

def build_ground_truth(rel_mode: RelationshipMode = RelationshipMode.OVERLAP_RATE, tbl_mode: TableSourceMode = TableSourceMode.STEP4_SYMLINK, use_symlink = True):
    """High‑level orchestration for building GT tables."""
    # ========== Step 1: Load paper-level info ==========
    paper_index, score_matrix = load_relationships(rel_mode)
    print(f"Step1: [Paper-level] Index shape: {len(paper_index)}. Score matrix shape: {score_matrix.shape}")
    n = len(paper_index)

    # ========== Step 2: Build paper-level adjacency matrix ==========
    paper_paper_adj = build_paper_matrix(score_matrix, paper_index, rel_mode)
    print(f"Step2: [Paper-level] Adjacency shape: {paper_paper_adj.shape}")
    print(paper_paper_adj[0])

    # ========== Step 3: Build paperId -> modelIds mapping ==========
    # Load metadata
    df_metadata = pd.read_parquet(FILES["integration"], columns=["corpusid", "query"])
    df_titles = pd.read_parquet(FILES["title_list"], columns=["modelId", "all_title_list"])
    df_titles_exploded = df_titles.explode("all_title_list")

    time_start = time.time()
    tm_mat, titles_list, model_list = build_title_model_matrix(df_titles_exploded)
    pt_mat, paper_list = build_paper_title_matrix(df_metadata, titles_list)
    print(f"pt_mat: {pt_mat.shape}")
    print(f"tm_mat: {tm_mat.shape}")
    #comb_mat = (pt_mat @ tm_mat).tocsr()
    comb_mat = compute_subset_pt_tm(pt_mat, tm_mat)
    print(f"Time taken to compute subset approach of pt_sub @ tm_sub: {time.time() - time_start:.2f} seconds")
    print(f"comb_mat: {comb_mat.shape}")
    
    # ========== Step 4: Build model-level adjacency matrix from the paper-level adjacency ==========
    time_start = time.time()
    cols = comb_mat.nonzero()[1]
    model_index = sorted({ model_list[j] for j in cols })
    print('len(model_index): ', len(model_index))
    model_model_adj = build_model_matrix(comb_mat, paper_paper_adj)
    print(f"Time taken to build model-level adjacency matrix: {time.time() - time_start:.2f} seconds")
    print(f"[Model-level] Adjacency shape: {model_model_adj.shape}")

    # ---------- Step 4.5: modelId → csv mapping ----------
    df_tables = load_table_source(TableSourceMode.STEP4_SYMLINK) # fixed here! tbl_mode
    modelId_to_csvs_sym = {mid: extract_csv_list(row, use_symlink=True) for mid, row in df_tables.iterrows()}
    modelId_to_csvs_real = {mid: extract_csv_list(row, use_symlink=False) for mid, row in df_tables.iterrows()}
    # ---------- Step 5: Build CSV‑level adjacency ----------
    all_csvs_real = {c for mid in model_index for c in modelId_to_csvs_real.get(mid, [])}
    csv_index = sorted(all_csvs_real)
    csv_csv_adj, csv_index = build_csv_matrix(model_model_adj, model_index, modelId_to_csvs_real, csv_index)
    print(f"[CSV‑level] Adjacency shape: {csv_csv_adj.shape}")
     # ---------- Step 5a: symlink‑level (0/1, no self-loop) ----------
    all_csvs_sym = {c for mid in model_index for c in modelId_to_csvs_sym.get(mid, [])}
    csv_symlink_index = sorted(all_csvs_sym)
    csv_symlink_adj, csv_symlink_index = build_csv_matrix(model_model_adj, model_index, modelId_to_csvs_sym, csv_symlink_index)
    print(f"[CSV‑symlink] shape: {csv_symlink_adj.shape}")

    # ---------- Step 5b: real‑path level (count, no self-loop) ----------
    symlink_map = load_symlink_mapping()
    csv_real_adj, csv_real_index = map_csv_adj_to_real(csv_symlink_adj, csv_symlink_index, symlink_map)
    print(f"[CSV‑real]    shape: {csv_real_adj.shape}")

    csv_real_gt = adjacency_to_dict(csv_real_adj, csv_real_index)
    # only record basename path
    csv_real_gt = {
        os.path.basename(k): [os.path.basename(vv) for vv in v]
        for k, v in csv_real_gt.items()
    }

    # ========== Step 5: Save everything to disk ==========
    suffix = f"__{rel_mode.value}"
    
    # ---- extra: save real‑path count adjacency (as dict) ----
    count_dict = defaultdict(dict)
    rows, cols = csv_real_adj.nonzero()
    for i, j in zip(rows, cols):
        if i == j:
            continue
        src = os.path.basename(csv_real_index[i])
        tgt = os.path.basename(csv_real_index[j])
        count_dict[src][tgt] = int(csv_real_adj[i, j])
    combined = {
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
    print(f"✔️  All matrices & indices saved to {GT_COMBINED_PATH}{suffix}")

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
