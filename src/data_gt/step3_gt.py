"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-16
Description: Build SciLake union benchmark tables.

Usage:
    python -m src.data_gt.step3_gt --rel_mode [overlap_rate, direct_label]
"""

import os, json, gzip, pickle, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from enum import Enum
from collections import defaultdict, Counter
from itertools import combinations
from scipy.sparse import csr_matrix

# === Configuration ===
DATA_DIR = "data/processed"
GT_DIR = "data/gt"

GT_COMBINED_PATH = f"{GT_DIR}/scilake_gt_all_matrices.pkl.gz"

# ---- default files (can be overridden by CLI/kwargs) ----
FILES = {
    "combined": f"{DATA_DIR}/modelcard_citation_all_matrices.pkl.gz",
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
    df_valid_title = pd.read_parquet(FILES["valid_title"], columns=["modelId", "all_title_list_valid"])
    df = pd.read_parquet(FILES[mode.value], columns=["modelId", "hugging_table_list_dedup", "github_table_list_dedup", "html_table_list_mapped_dedup", "llm_table_list_mapped_dedup"])
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

def build_paper_matrix(score_matrix: csr_matrix, paper_index: list, rel_mode: RelationshipMode, thr: float=THRESHOLD):
    thr = 1.0 if rel_mode == RelationshipMode.DIRECTED_CITE else thr
    paper_adj = (score_matrix >= thr).astype(np.bool_)
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

def build_ground_truth(rel_mode: RelationshipMode = RelationshipMode.OVERLAP_RATE):
    # modelId-paperList-csvList, Our aim is to use paper-paper matrix to build {csv:[csv1, csv2]} related json
    suffix = f"__{rel_mode.value}"
    """High‑level orchestration for building GT tables."""
    # ========== Step 1: Load paper-level info. Build paper-level adjacency matrix ==========
    paper_index, score_matrix = load_relationships(rel_mode)
    paper_paper_adj = build_paper_matrix(score_matrix, paper_index, rel_mode)
    print(f"Step1: [Paper-level] Adjacency shape: {paper_paper_adj.shape}")
    print(paper_paper_adj[0])

    # ---------- Step 2: modelId → csv mapping ----------
    df_tables = load_table_source(TableSourceMode.STEP3_MERGED) # fixed here!
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

    # Dictionary for flat tuple-key counts
    csv_pair_cnt = defaultdict(int)

    # 1) Intra-row counting
    for _, csvs in rows:
        for a, b in combinations(sorted(set(csvs)), 2):
            csv_pair_cnt[(a, b)] += 1  ######## marker for updated logic

    # 2) Inter-row counting for paper-paper edges
    pr, pc = paper_paper_adj.nonzero()
    for p_idx, q_idx in zip(pr, pc):
        if p_idx >= q_idx:
            continue
        for r in paper2rows.get(p_idx, []):
            for s in paper2rows.get(q_idx, []):
                if r == s:
                    continue
                for a in rows[r][1]:
                    for b in rows[s][1]:
                        if a == b:
                            continue
                        x, y = sorted((a, b))
                        csv_pair_cnt[(x, y)] += 1  ########
     # Save the counts
    output_path = f"{GT_DIR}/csv_pair_counts_{rel_mode.value}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dict(csv_pair_cnt), f)
    print(f"✅ CSV pair counts saved to {output_path}")
    # stats
    print('number of csv pairs: ', len(csv_pair_cnt))
    print('number of unique single csv: ',)
    print('sum of counts: ', sum(csv_pair_cnt.values()))

    csv_adj = defaultdict(list)
    for (a, b), cnt in csv_pair_cnt.items():
        if cnt > 0:
            csv_adj[a].append(b)
            csv_adj[b].append(a)
    csv_adj = {k: sorted(set(v)) for k, v in csv_adj.items()}
    processed_csv_adj = {                                                                            ########
        os.path.basename(k): [os.path.basename(x) for x in v]                                       ########
        for k, v in csv_adj.items()                                                              ########
    }
    output_adj = f"{GT_DIR}/csv_pair_adj_{rel_mode.value}_processed.pkl"
    with open(output_adj, "wb") as f:
        pickle.dump(processed_csv_adj, f)
    print(f"✅ CSV adjacency mapping saved to {output_adj}")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument("--rel_mode", choices=[m.value for m in RelationshipMode], default=RelationshipMode.OVERLAP_RATE.value, help="Relationship graph mode.")
    args = parser.parse_args()

    build_ground_truth( rel_mode=RelationshipMode(args.rel_mode))
