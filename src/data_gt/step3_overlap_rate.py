"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-16
Description: Compute the overlap rate of citation and citing papers based on paperId, and save:
- pairwise overlap scores
- thresholded related paper pairs
- direct citation links (if paperId appears in references of another)
"""

import os, json, gzip, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
from scipy.sparse import csr_matrix, lil_matrix

# === Configuration ===
#INPUT_PARQUET = "data/processed/modelcard_citation_enriched.parquet" # original query id from online
INPUT_PARQUET = "data/processed/extracted_annotations.parquet" # query title from online
SIMILARITY_MODES = ["max_pr", "jaccard", "dice"]
COMBINED_PATH = "data/processed/modelcard_citation_all_matrices.pkl.gz"

THRESHOLD = 0.1
SAVE_THRESHOLD_OVERLAP = True
MODE = "reference"  # or "citation"

def load_paperId_lists(df, mode, influential=False):
    empty_id_list = []
    paperId_to_ids = {}
    if "parsed_response" in df.columns:
        col_name = "parsed_response"
    elif mode == "reference":
        col_name = "original_response_reference"
    else:
        col_name = "original_response_citation"
    for _, row in df.iterrows():
        pid = row["paperId"]
        assert not pd.isna(pid)
        try:
            parsed = json.loads(row[col_name])
        except Exception as e:
            print(f"Error parsing JSON for paper {pid}: {e}")
            parsed = {}
        paper_list = []
        if isinstance(parsed, dict) and "data" in parsed:
            for item in parsed["data"]:
                if influential and not item.get("isInfluential", False):  ######## only keep influential if flagged
                    continue
                if mode == "reference" and "citedPaper" in item:
                    paper_list.append(item["citedPaper"])
                elif mode == "citation" and "citingPaper" in item:
                    paper_list.append(item["citingPaper"])
        else: # old format
            key = "cited_papers" if mode == "reference" else "citing_papers"
            if key in parsed:
                paper_list = parsed[key]
        if not paper_list:
            #print(f"Warning: No papers found for {pid} in mode {mode}.")
            #paperId_to_ids[pid] = set()
            empty_id_list.append(pid)
        else:
            paperId_to_ids[pid] = {p["paperId"] for p in paper_list if "paperId" in p}
    print(f"{'[Influential] ' if influential else ''}Empty lists: {len(empty_id_list)}, Non-empty: {len(paperId_to_ids)}")
    return paperId_to_ids

def compute_overlap_matrices(df, influential):
    paperId_to_ref  = load_paperId_lists(df, mode="reference", influential=influential)
    paper_list = sorted(set(df['paperId']))
    for pid in paper_list:                                    
        if pid not in paperId_to_ref:                          
            paperId_to_ref[pid] = set()                      
    paper_index = sorted(paperId_to_ref.keys())

    n = len(paper_index)
    maxpr_matrix = np.zeros((n, n), dtype=np.float32)
    jaccard_matrix = np.zeros((n, n), dtype=np.float32)
    dice_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="Computing overlap matrices"):
        refs_i = set(paperId_to_ref[paper_index[i]])
        len_i = len(refs_i)
        for j in range(i, n):
            refs_j = set(paperId_to_ref[paper_index[j]])
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
    }

def compute_direct_matrix(df, influential):  ######## added `influential` flag
    # load references & citations with optional isInfluential filter
    paperId_to_ref  = load_paperId_lists(df, mode="reference", influential=influential)
    paperId_to_cite = load_paperId_lists(df, mode="citation",  influential=influential)
    #all_pids = sorted(set(paperId_to_ref) | set(paperId_to_cite))
    all_pids = sorted(set(df["paperId"]))
    n = len(all_pids)

    mat = lil_matrix((n, n), dtype=np.bool_)  ######## store as sparse boolean
    for i in tqdm(range(n), desc=f"{'[Influential] ' if influential else ''}Computing direct matrix"):
        pid_i  = all_pids[i]
        refs_i = paperId_to_ref.get(pid_i, set()) | paperId_to_cite.get(pid_i, set())
        for j in range(i, n):
            pid_j  = all_pids[j]
            refs_j = paperId_to_ref.get(pid_j, set()) | paperId_to_cite.get(pid_j, set())
            if pid_j in refs_i or pid_i in refs_j:
                mat[i, j] = True
                mat[j, i] = True                           ######## mirror in one step
    return {"paper_index": all_pids, "score_matrix": mat.tocsr()}

def main():
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} rows")

    # Compute normal
    overlap_all = compute_overlap_matrices(df, influential=False)
    direct_data  = compute_direct_matrix(df, influential=False)
    # Align direct to overlap index
    common_index = overlap_all["paper_index"]
    if common_index != direct_data["paper_index"]:
        direct_map = {pid:i for i,pid in enumerate(direct_data["paper_index"])}
        valid = [pid for pid in common_index if pid in direct_map]
        inds = [direct_map[pid] for pid in valid]
        aligned = direct_data["score_matrix"][inds,:][:,inds]
        direct_data["score_matrix"] = aligned
        
    # Compute influential
    inf_overlap_all = compute_overlap_matrices(df, influential=True)
    inf_direct_data  = compute_direct_matrix(df, influential=True)
    # Align inf_direct to overlap index
    if common_index != inf_direct_data["paper_index"]:
        inf_map = {pid:i for i,pid in enumerate(inf_direct_data["paper_index"])}
        valid_inf = [pid for pid in common_index if pid in inf_map]
        inds_inf = [inf_map[pid] for pid in valid_inf]
        inf_direct_data["score_matrix"] = inf_direct_data["score_matrix"][inds_inf,:][:,inds_inf]

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
        "paper_index": common_index,
        "direct_label": direct_data["score_matrix"],
        "direct_label_influential": inf_direct_data["score_matrix"],
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
