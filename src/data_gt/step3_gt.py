"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-04
Description:
    Compute the overlap rate of citation and citing papers based on paperId
    with flexible “modes” controlled by a tiny factory layer.

    Supported relationship modes
    ----------------------------
    • overlap_rate   : uses a (pid → {related_pids}) mapping with weights
    • direct_label  : uses a (pid → {cited_by_pids}) mapping as GT
    • overlap_label  : uses a (pid → {related_pids}) mapping with weights

    Supported table source modes
    ----------------------------
    • step4_symlink  : use step‑4 parquet + *_sym columns
    • step3_merged   : use step‑3 parquet + raw columns

    Outputs
    -------
    • pairwise overlap scores
    • thresholded related paper pairs
    • direct citation links (if paperId appears in references of another)
    • ground‑truth benchmark table paths (pickle)

Usage:
    python -m src.data_gt.step3_gt --rel_mode [overlap_rate, overlap_label, direct_label] --tbl_mode [step4_symlink, step3_merged] (require step4.parquet by create_symlink.py)
    python -m src.data_gt.step3_gt --rel_mode direct_label --tbl_mode step3_merged
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from enum import Enum
from collections import defaultdict, Counter
from typing import Dict, Iterable, Set
from joblib import Parallel, delayed
from itertools import combinations, product
from scipy.sparse import csr_matrix, dok_matrix, save_npz


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


# ---- default files (can be overridden by CLI/kwargs) ----
FILES = {
    "overlap_rate": f"{DATA_DIR}/modelcard_citation_overlap_rate.pickle",
    "overlap_label": f"{DATA_DIR}/modelcard_citation_overlap_label.pickle",
    "direct_label": f"{DATA_DIR}/modelcard_citation_direct_label.pickle",
    "step4_symlink": f"{DATA_DIR}/modelcard_step4.parquet",
    "step3_merged": f"{DATA_DIR}/modelcard_step3_merged.parquet",
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
THRESHOLD = 0.2 # for overlap_rate only


# ===== ENUMS =============================================================== #
class RelationshipMode(str, Enum):
    OVERLAP_RATE = "overlap_rate"
    DIRECTED_CITE = "direct_label"
    OVERLAP_LABEL = "overlap_label"


class TableSourceMode(str, Enum):
    STEP4_SYMLINK = "step4_symlink"
    STEP3_MERGED = "step3_merged"


# ===== FACTORIES =========================================================== #
def load_relationships(mode: RelationshipMode):
    """Factory loader for paperId‑level relationship graphs."""
    path = FILES[mode.value]
    print(f"Loading relationships ({mode.value}) from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    paper_index = data["paper_index"]
    score_matrix = data["score_matrix"]
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
            edge = (union_csv[i], union_csv[j])
            csv_edges.add(edge)
    return (pair, csv_edges)

def map_csv_edges(result, symlink_map):
    """Map CSV symlink edges back to their real paths.
    result: ((m1, m2), csv_edges)
    Returns: ((m1, m2), new_csv_edges)
    """
    pair, csv_edges = result
    new_csv_edges = set()
    for csv_edge in csv_edges:
        csv1, csv2 = csv_edge
        real_csv1 = symlink_map[csv1]
        real_csv2 = symlink_map[csv2]
        new_csv_edges.add((real_csv1, real_csv2))
    return (pair, new_csv_edges)

# ===== BUSINESS LOGIC ====================================================== #
def extract_csv_list(row: pd.Series, use_symlink: bool):
    if use_symlink:
        cols = ["hugging_table_list_sym", "github_table_list_sym", "html_table_list_sym", "llm_table_list_sym"]
    else:
        cols = ["hugging_table_list_dedup", "github_table_list_dedup", "html_table_list_mapped_dedup", "llm_table_list_mapped_dedup"]
    csv_list = []
    for col in cols:
        csv_list.extend(row.get(col, []))
    seen = set()
    return [csv for csv in csv_list if csv not in seen and not pd.isna(csv)]

def build_csv_matrix(model_adj: csr_matrix, model_index: list, modelId_to_csvs: Dict[str, list]):
    ######## gather all csv
    all_csvs = set()
    for csv_list in modelId_to_csvs.values():
        all_csvs.update(csv_list)
    csv_index = sorted(all_csvs)
    csv_pos = {c: i for i, c in enumerate(csv_index)}
    print(f"Total unique CSVs: {len(csv_index)}")
    csv_adj = dok_matrix((len(csv_index), len(csv_index)), dtype=np.int32)
    print("Building CSV‑level adjacency from model‑level adjacency …")
    for u_idx in tqdm(range(model_adj.shape[0]), desc="Model rows"):
        row_start = model_adj.indptr[u_idx]
        row_end   = model_adj.indptr[u_idx+1]
        v_indices = model_adj.indices[row_start:row_end]
        weights   = model_adj.data[row_start:row_end]
        u_model = model_index[u_idx]
        csv_u   = modelId_to_csvs.get(u_model, [])
        for v_idx, w in zip(v_indices, weights):
            v_model = model_index[v_idx]
            csv_v   = modelId_to_csvs.get(v_model, [])
            if not csv_u or not csv_v:
                continue
            ######## Use model‑edge weight on each (csv_u_i, csv_v_j)
            for cu in csv_u:
                for cv in csv_v:
                    i, j = csv_pos[cu], csv_pos[cv]
                    if i == j:
                        continue
                    csv_adj[i, j] = 1
    # self-loop
    # for i in range(len(csv_index)):
    #     csv_adj[i, i] = 1
    return csv_adj.tocsr(), csv_index

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

def build_paper_matrix(score_matrix: csr_matrix, paper_index: list, rel_mode: RelationshipMode):
    """
    Build a paper-level adjacency matrix from the raw score_matrix, using a threshold.
    - If rel_mode in [direct_label, overlap_label], threshold = 1.0
    - If rel_mode == overlap_rate, threshold = THRESHOLD (e.g. 0.2)

    The output adjacency matrix is square (n x n), with 1 if related, else 0.
    Includes a self-loop as 1 on the diagonal if you want that (just set it at the end).
    """
    n = len(paper_index)
    if rel_mode in [RelationshipMode.DIRECTED_CITE, RelationshipMode.OVERLAP_LABEL]:
        thr = 1.0
    else:
        thr = THRESHOLD
    print(f"Building paper-level adjacency with threshold={thr}")
    # Use a DOK for incremental building, then convert to CSR at the end
    paper_adj = dok_matrix((n, n), dtype=np.int8)
    for i in tqdm(range(n), desc="Filling adjacency"):
        row_start = score_matrix.indptr[i]
        row_end = score_matrix.indptr[i+1]
        indices = score_matrix.indices[row_start:row_end]
        data = score_matrix.data[row_start:row_end]
        for col_idx, val in zip(indices, data):
            if val >= thr:
                paper_adj[i, col_idx] = 1
    # Optionally add self-loops
    for i in range(n):
        paper_adj[i, i] = 1
    return paper_adj.tocsr()

def build_model_matrix(paper_adj: csr_matrix, paper_index: list, paperId_to_modelIds: Dict[str, set]):
    """
    Build model-level adjacency matrix from the paper-level adjacency matrix.
    Approach:
      1) Identify the full unique set of model IDs.
      2) For each paper i, get the set of models M_i. For each paper j in adjacency,
         get models M_j, then increment adjacency[u, v] for (u in M_i) × (v in M_j).
      3) By default, we 'count' how many paper→paper links connect each (u, v).
         If you want a binary adjacency, set 1 instead of += 1.
    Returns:
    - model_model_adj (csr_matrix)
    - model_index (list of all model IDs in sorted order)
    """
    print("Collecting all model IDs ...")
    all_model_ids = set()
    for mids in paperId_to_modelIds.values():
        all_model_ids.update(mids)
    model_index = sorted(all_model_ids)
    model_pos = {m: i for i, m in enumerate(model_index)}
    print(f"Total unique models: {len(model_index)}")
    model_adj = dok_matrix((len(model_index), len(model_index)), dtype=np.int32)
    n = paper_adj.shape[0]
    print("Building model-level adjacency from paper-level adjacency ...")
    for i in tqdm(range(n), desc="Paper-level adjacency => Model adjacency"):
        paper_i = paper_index[i]
        if paper_i not in paperId_to_modelIds:
            continue
        models_i = paperId_to_modelIds[paper_i]
        row_start = paper_adj.indptr[i]
        row_end = paper_adj.indptr[i + 1]
        neighbors = paper_adj.indices[row_start:row_end]  # all j for which adjacency[i, j] = 1
        for j in neighbors:
            paper_j = paper_index[j]
            if paper_j not in paperId_to_modelIds:
                continue
            models_j = paperId_to_modelIds[paper_j]
            # For each pair (u in models_i, v in models_j), increment adjacency
            for u in models_i:
                for v in models_j:
                    u_idx = model_pos[u]
                    v_idx = model_pos[v]
                    model_adj[u_idx, v_idx] = 1 # binary-based
                    # model_adj[u_idx, v_idx] += 1 # count-based
    # If you want self-loops for models, do:
    for i in range(len(model_index)):
        model_adj[i, i] = 1
    return model_adj.tocsr(), model_index

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

    # title -> modelIds
    title_to_modelIds = defaultdict(set)
    for _, row in tqdm(df_titles_exploded.iterrows(), total=len(df_titles_exploded), desc="Mapping titles->modelIds"):
        title = row["all_title_list"]
        if pd.notna(title):
            title_to_modelIds[title.strip()].add(row["modelId"])

    # paperId -> modelIds
    paperId_to_modelIds = defaultdict(set)
    for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc="Mapping paperId->modelIds"):
        pid = row["corpusid"]
        title = row["query"]
        if pd.notna(pid) and pd.notna(title):
            pid_str = str(pid)
            model_ids = title_to_modelIds[title.strip()]
            for mid in model_ids:
                paperId_to_modelIds[pid_str].add(mid)

    # ========== Step 4: Build model-level adjacency matrix from the paper-level adjacency ==========
    model_model_adj, model_index = build_model_matrix(paper_paper_adj, paper_index, paperId_to_modelIds)
    print(f"[Model-level] Adjacency shape: {model_model_adj.shape}")

    # ---------- Step 4.5: modelId → csv mapping ----------
    ######## modelId_to_csvs
    df_tables = load_table_source(TableSourceMode.STEP4_SYMLINK)
    modelId_to_csvs = {mid: extract_csv_list(row, use_symlink=True) for mid, row in df_tables.iterrows()}

    # ---------- Step 5: Build CSV‑level adjacency ----------
    csv_csv_adj, csv_index = build_csv_matrix(model_model_adj, model_index, modelId_to_csvs)
    print(f"[CSV‑level] Adjacency shape: {csv_csv_adj.shape}")

     # ---------- Step 5a: symlink‑level (0/1, no self-loop) ----------
    csv_symlink_adj, csv_symlink_index = build_csv_matrix(model_model_adj, model_index, modelId_to_csvs)
    print(f"[CSV‑symlink] shape: {csv_symlink_adj.shape}")

    # ---------- Step 5b: real‑path level (count, no self-loop) ----------
    symlink_map = load_symlink_mapping()
    csv_real_adj, csv_real_index = map_csv_adj_to_real(csv_symlink_adj, csv_symlink_index, symlink_map)
    print(f"[CSV‑real]    shape: {csv_real_adj.shape}")

    csv_real_gt = adjacency_to_dict(csv_real_adj, csv_real_index)

    # ========== Step 5: Save everything to disk ==========
    # 5.1 Paper adjacency
    save_npz(PAPER_LEVEL_ADJ_PATH, paper_paper_adj)
    with open(PAPER_INDEX_PATH, "wb") as f:
        pickle.dump(paper_index, f)
    print(f"Paper-level adjacency saved to {PAPER_LEVEL_ADJ_PATH}")
    print(f"Paper index list saved to {PAPER_INDEX_PATH}")

    # 5.2 Model adjacency
    save_npz(MODEL_LEVEL_ADJ_PATH, model_model_adj)
    with open(MODEL_INDEX_PATH, "wb") as f:
        pickle.dump(model_index, f)
    print(f"Model-level adjacency saved to {MODEL_LEVEL_ADJ_PATH}")
    print(f"Model index list saved to {MODEL_INDEX_PATH}")

    save_npz(CSV_LEVEL_ADJ_PATH, csv_csv_adj)
    with open(CSV_INDEX_PATH, "wb") as f:
        pickle.dump(csv_index, f)
    print(f"CSV‑level adjacency saved to {CSV_LEVEL_ADJ_PATH}")
    print(f"CSV index list saved to {CSV_INDEX_PATH}")

    # ---- save symlink‑level 0/1 adjacency ----
    save_npz(CSV_SYMLINK_ADJ_PATH, csv_symlink_adj)
    with open(CSV_SYMLINK_INDEX_PATH, "wb") as f:
        pickle.dump(csv_symlink_index, f)
    print(f"Symlink CSV adjacency → {CSV_SYMLINK_ADJ_PATH}")

    # ---- save real‑path count adjacency ----
    save_npz(CSV_REAL_ADJ_PATH, csv_real_adj)
    with open(CSV_REAL_INDEX_PATH, "wb") as f:
        pickle.dump(csv_real_index, f)
    print(f"Real CSV adjacency    → {CSV_REAL_ADJ_PATH}")

    with open(f"{GT_DIR}/scilake_large_gt.pickle", "wb") as f:
        pickle.dump(csv_real_gt, f)
    with open(f"{GT_DIR}/scilake_large_gt.json", "w") as f:
        json.dump(csv_real_gt, f, indent=2)

    print("✅ Done building matrix-based groundtruth!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument(
        "--rel_mode",
        choices=[m.value for m in RelationshipMode],
        default=RelationshipMode.OVERLAP_RATE.value,
        help="Relationship graph mode.",
    )
    parser.add_argument(
        "--tbl_mode",
        choices=[m.value for m in TableSourceMode],
        default=TableSourceMode.STEP4_SYMLINK.value,
        help="Table source mode.",
    )
    args = parser.parse_args()

    build_ground_truth(
        rel_mode=RelationshipMode(args.rel_mode),
        tbl_mode=TableSourceMode(args.tbl_mode),
        use_symlink=True
    )
