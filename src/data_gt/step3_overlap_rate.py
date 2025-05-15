"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-16
Description: Compute the overlap rate of citation and citing papers based on Id, and save:
- pairwise overlap scores
- thresholded related paper pairs
- direct citation links (if Id appears in references of another)
"""

import os, json, gzip, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import time

# === Configuration ===
#INPUT_PARQUET = "data/processed/modelcard_citation_enriched.parquet" # original query id from online
INPUT_PARQUET = "data/processed/extracted_annotations.parquet" # query title from online
SIMILARITY_MODES = ["max_pr", "jaccard", "dice"]
COMBINED_PATH = "data/processed/modelcard_citation_all_matrices.pkl.gz"

THRESHOLD = 0.1
SAVE_THRESHOLD_OVERLAP = True
MODE = "reference"  # or "citation"

def load_Id_lists(df, mode, influential=False):
    Id_to_ids = {}
    for index, row in df.iterrows():
        pid = row["corpusid"]
        if mode == "reference":
            if influential:
                Id_to_ids[pid] = set(row["ref_papers_overall_infl_ids"])
            else:
                Id_to_ids[pid] = set(row["ref_papers_overall_ids"])
        elif mode == "citation":
            if influential:
                Id_to_ids[pid] = set(row["cit_papers_overall_infl_ids"])
            else:
                Id_to_ids[pid] = set(row["cit_papers_overall_ids"])
    print(f"{'[Influential] ' if influential else ''}, Non-empty: {len(Id_to_ids)}")
    # turn to string list [1,2,3]->['1','2','3']
    Id_to_ids = {str(i):list(map(str, Id_to_ids[i])) for i in Id_to_ids}
    return Id_to_ids

def compute_overlap_matrices(Id_to_ref, paper_list):
    # Inverted index
    idx_map = {pid: i for i, pid in enumerate(paper_list)}
    # ref_id → [paper_indices]
    ref_to_papers = defaultdict(list)
    for pid, refs in Id_to_ref.items():
        i = idx_map[pid]
        for r in refs:
            ref_to_papers[r].append(i)
    # Compute intersections
    rows, cols, data = [], [], []
    for papers in ref_to_papers.values():
        for i, j in combinations(papers, 2):
            rows += [i, j]; cols += [j, i]; data += [1, 1]
    # Diagonal entries
    for pid, i in idx_map.items():
        rows.append(i); cols.append(i); data.append(len(Id_to_ref[pid]))
    n = len(paper_list)
    intersection = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    lens = np.array([len(Id_to_ref[pid]) for pid in paper_list], dtype=np.float32)
    # Jaccard
    union = lens[:, None] + lens[None, :] - intersection.toarray()
    jaccard = csr_matrix(intersection.toarray() / np.where(union == 0, 1, union))
    # Dice
    total = lens[:, None] + lens[None, :]
    dice = csr_matrix(2 * intersection.toarray() / np.where(total == 0, 1, total))
    # MaxPR
    pr_i = intersection.multiply(1.0 / lens[:, None])
    pr_j = intersection.multiply(1.0 / lens[None, :])
    max_pr = csr_matrix(pr_i.maximum(pr_j))
    return {
        "max_pr": max_pr,
        "jaccard": jaccard,
        "dice": dice
    }

"""def original_compute_overlap_matrices(df, influential):
    Id_to_ref  = load_Id_lists(df, mode="reference", influential=influential)
    paper_list = sorted(set(df['corpusid']))
    for pid in paper_list:                                    
        if pid not in Id_to_ref:                          
            Id_to_ref[pid] = set()                      
    paper_index = sorted(Id_to_ref.keys())

    n = len(paper_index)
    maxpr_matrix = np.zeros((n, n), dtype=np.float32)
    jaccard_matrix = np.zeros((n, n), dtype=np.float32)
    dice_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="Computing overlap matrices"):
        refs_i = set(Id_to_ref[paper_index[i]])
        len_i = len(refs_i)
        for j in range(i, n):
            refs_j = set(Id_to_ref[paper_index[j]])
            len_j = len(refs_j)
            intersection = len(refs_i & refs_j)
            # jaccard: intersection / (len_i + len_j - intersection)
            union = len_i + len_j - intersection
            score_jaccard = intersection / union if union else 0.0
            # dice: 2 * intersection / (len_i + len_j)
            total = len_i + len_j
            score_dice = 2 * intersection / total if total else 0.0
            # max_pr: max( intersection/len_i, intersection/len_j )
            prec = intersection / len_i if len_i else 0.0
            rec = intersection / len_j if len_j else 0.0
            score_maxpr = prec if prec >= rec else rec
            maxpr_matrix[i, j] = score_maxpr
            maxpr_matrix[j, i] = score_maxpr
            jaccard_matrix[i, j] = score_jaccard
            jaccard_matrix[j, i] = score_jaccard
            dice_matrix[i, j] = score_dice
            dice_matrix[j, i] = score_dice
    return {
        "paper_index": paper_index,
        "max_pr": csr_matrix(maxpr_matrix),
        "jaccard": csr_matrix(jaccard_matrix),
        "dice": csr_matrix(dice_matrix)
    }"""

"""def original_compute_direct_matrix(df, influential):  ######## added `influential` flag
    # load references & citations with optional isInfluential filter
    Id_to_ref  = load_Id_lists(df, mode="reference", influential=influential)
    Id_to_cite = load_Id_lists(df, mode="citation",  influential=influential)
    Id_to_all = {i:Id_to_ref[i] + Id_to_cite[i] for i in Id_to_ref}
    all_pids = sorted(set(df["corpusid"]))
    n = len(all_pids)
    mat = lil_matrix((n, n), dtype=np.bool_)
    for i in tqdm(range(n), desc=f"{'[Influential] ' if influential else ''}Computing direct matrix"):
        pid_i  = all_pids[i]
        refs_i = Id_to_all[pid_i]
        for j in range(i, n):
            pid_j  = all_pids[j]
            refs_j = Id_to_all[pid_j]
            if pid_j in refs_i or pid_i in refs_j:
                mat[i, j] = True
                mat[j, i] = True
    return {"paper_index": all_pids, "score_matrix": mat.tocsr()}"""

def compute_direct_matrix(Id_to_all, paper_list):
    idx_map = {pid: i for i, pid in enumerate(paper_list)}
    rows, cols = [], []
    for pid, neighbors in Id_to_all.items():
        i = idx_map.get(pid)
        for nbr in neighbors:
            j = idx_map.get(nbr)
            if j is not None:
                rows += [i, j]
                cols += [j, i]
    n = len(paper_list)
    mat = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n)).tocsr()
    return mat

def main():
    df = pd.read_parquet(INPUT_PARQUET)
    df['corpusid'] = df['corpusid'].astype(str)
    print(f"Loaded {len(df)} rows")

    Id_to_ref_inf  = load_Id_lists(df, mode="reference", influential=True)
    Id_to_cite_inf = load_Id_lists(df, mode="citation",  influential=True)
    Id_to_all_inf = {pid: Id_to_ref_inf.get(pid, []) + Id_to_cite_inf.get(pid, []) for pid in set(Id_to_ref_inf) | set(Id_to_cite_inf)}

    Id_to_ref  = load_Id_lists(df, mode="reference", influential=False)
    Id_to_cite = load_Id_lists(df, mode="citation",  influential=False)
    Id_to_all = {pid: Id_to_ref.get(pid, []) + Id_to_cite.get(pid, []) for pid in set(Id_to_ref) | set(Id_to_cite)}
    Id_list = sorted(set(df["corpusid"]))

    time_start = time.time()
    overlap_all = compute_overlap_matrices(Id_to_ref, Id_list)
    print(f"Time taken for normal: {time.time() - time_start} seconds")
    time_start = time.time()
    direct_data  = compute_direct_matrix(Id_to_all, Id_list)
    print(f"Time taken for normal: {time.time() - time_start} seconds")
        
    # Compute influential
    time_start = time.time()
    inf_overlap_all = compute_overlap_matrices(Id_to_ref_inf, Id_list)
    print(f"Time taken for normal: {time.time() - time_start} seconds")
    time_start = time.time()
    inf_direct_data  = compute_direct_matrix(Id_to_all_inf, Id_list)
    print(f"Time taken for normal: {time.time() - time_start} seconds")

    # Threshold normal overlap
    thresholds = {}
    for key in ["max_pr","jaccard","dice"]:
        m = overlap_all[key].copy()
        mask = m.data < THRESHOLD
        m.data[mask] = 0
        m.eliminate_zeros()
        thresholds[f"{key}_thresholded"] = m.astype(np.bool_)

    # Threshold influential overlap
    thresholds_inf = {}
    for key in ["max_pr","jaccard","dice"]:
        m = inf_overlap_all[key].copy()
        mask = m.data < THRESHOLD
        m.data[mask] = 0
        m.eliminate_zeros()
        thresholds_inf[f"{key}_influential_thresholded"] = m.astype(np.bool_)

    # Build combined dict
    combined = {
        "paper_index": Id_list,
        "direct_label": direct_data,
        "direct_label_influential": inf_direct_data,
        "max_pr":   overlap_all["max_pr"],
        "jaccard":  overlap_all["jaccard"],
        "dice":     overlap_all["dice"],
        "max_pr_influential":   inf_overlap_all["max_pr"],
        "jaccard_influential":  inf_overlap_all["jaccard"],
        "dice_influential":     inf_overlap_all["dice"],
        **thresholds,
        **thresholds_inf
    }

    # Save everything
    os.makedirs(os.path.dirname(COMBINED_PATH), exist_ok=True)
    with gzip.open(COMBINED_PATH, "wb") as f:
        pickle.dump(combined, f)
    print(f"Saved all matrices to {COMBINED_PATH}")

if __name__ == "__main__":
    main()
