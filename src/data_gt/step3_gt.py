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

import os, json, gzip, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from enum import Enum
from collections import defaultdict, Counter
from typing import Dict, Iterable, Set
from joblib import Parallel, delayed
from itertools import combinations, product
from scipy.sparse import csr_matrix, dok_matrix, save_npz, coo_matrix

# === Configuration ===
DATA_DIR = "data/processed"
GT_DIR = "data/gt"
CSV_LEVEL_ADJ_PATH  = f"{GT_DIR}/csv_level_adjacency.npz"
CSV_INDEX_PATH      = f"{GT_DIR}/csv_index_list.pickle"

PAPER_LEVEL_ADJ_PATH = f"{GT_DIR}/paper_level_adjacency.npz"
MODEL_LEVEL_ADJ_PATH = f"{GT_DIR}/model_level_adjacency.npz"
PAPER_INDEX_PATH      = f"{GT_DIR}/paper_index_list.pickle"
MODEL_INDEX_PATH      = f"{GT_DIR}/model_index_list.pickle"
CSV_SYMLINK_ADJ_PATH = f"{GT_DIR}/csv_symlink_adjacency.npz"
CSV_SYMLINK_INDEX_PATH = f"{GT_DIR}/csv_symlink_index.pickle"
CSV_REAL_ADJ_PATH   = f"{GT_DIR}/csv_real_adjacency.npz"
CSV_REAL_INDEX_PATH = f"{GT_DIR}/csv_real_index.pickle"

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
        score_matrix = data["max_pr"]                              ########  (use max‑PR by default)
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

def build_B_sparse(paper_index, model_index, paperId_to_modelIds):  ########
    """Build a CSR B (paper→model) with one pass COO construction."""
    paper_pos = {pid: i for i, pid in enumerate(paper_index)}
    model_pos = {mid: j for j, mid in enumerate(model_index)}
    rows, cols = [], []
    for pid, mids in paperId_to_modelIds.items():
        i = paper_pos.get(pid)
        if i is None:
            continue
        cols.extend(model_pos[m] for m in mids if m in model_pos)
        rows.extend([i] * len(mids))
    data = np.ones(len(rows), dtype=np.bool_)
    return coo_matrix((data, (rows, cols)), shape=(len(paper_index), len(model_index))).tocsr() ########

def build_model_matrix(paper_adj: csr_matrix, paper_index: list, paperId_to_modelIds: Dict[str, set], model_index: list):
    """Fast model‑level adjacency via Bᵀ A B with all CSR operands."""
    print("Building model-level adjacency matrix ...")
    B = build_B_sparse(paper_index, model_index, paperId_to_modelIds) ########
    B = B.astype(bool)
    paper_adj = paper_adj.astype(bool)
    print(f"B shape: {B.shape}")
    print(f"paper_adj shape: {paper_adj.shape}")
    print("Multiplying B^T @ A @ B ...")
    #model_adj = (B.T @ paper_adj @ B).astype(bool)              ########
    # split into twice multiplication
    model_adj_1 = (B.T @ paper_adj).astype(bool)
    model_adj = (model_adj_1 @ B).astype(bool)
    model_adj.setdiag(True)
    model_adj.eliminate_zeros()
    return model_adj, model_index

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

def build_ground_truth(rel_mode: RelationshipMode = RelationshipMode.OVERLAP_RATE, tbl_mode: TableSourceMode = TableSourceMode.STEP4_SYMLINK, use_symlink = True):
    """High‑level orchestration for building GT tables."""
    # ========== Step 1: Load paper-level info ==========
    paper_index, score_matrix = load_relationships(rel_mode)
    n = len(paper_index)

    # ========== Step 2: Build paper-level adjacency matrix ==========
    paper_paper_adj = build_paper_matrix(score_matrix, paper_index, rel_mode)
    print(f"[Paper-level] Adjacency shape: {paper_paper_adj.shape}")

    # ========== Step 3: Build paperId -> modelIds mapping ==========
    # Load metadata
    df_metadata = pd.read_parquet(FILES["integration"], columns=["corpusid", "query"])
    df_titles = pd.read_parquet(FILES["title_list"], columns=["modelId", "all_title_list"])
    df_titles_exploded = df_titles.explode("all_title_list")

    tm_mat, titles_list, model_list = build_title_model_matrix(df_titles_exploded)
    pt_mat, paper_list = build_paper_title_matrix(df_metadata, titles_list)
    comb_mat = (pt_mat @ tm_mat).tocsr()

    # convert back to mapping  (paperId -> {modelIds})                                  ########
    paperId_to_modelIds = defaultdict(set)                                              ########
    rows, cols = comb_mat.nonzero()                                                     ########
    for r, c in zip(rows, cols):                                                        ########
        paperId_to_modelIds[str(paper_list[r])].add(model_list[c]) 

    # ========== Step 4: Build model-level adjacency matrix from the paper-level adjacency ==========
    model_index = sorted({mid for mids in paperId_to_modelIds.values() for mid in mids})
    model_model_adj, model_index = build_model_matrix(paper_paper_adj, paper_index, paperId_to_modelIds, model_index)
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
