"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-11
Description: Compute the overlap rate of citation and citing papers based on paperId, and save:
- pairwise overlap scores
- thresholded related paper pairs
- direct citation links (if paperId appears in references of another)
"""

import os
import json
import pandas as pd
import pickle
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# === Configuration ===
INPUT_PARQUET = "data/processed/modelcard_citation_enriched.parquet"
SIMILARITY_MODES = ["max_pr", "jaccard", "dice"]
OVERLAP_MATRIX = "data/processed/modelcard_citation_overlap_rate.pickle"
DIRECT_MATRIX = "data/processed/modelcard_citation_direct_label.pickle"
OVERLAP_FILE_TEMPLATE = "data/processed/modelcard_citation_overlap_rate_{sim_mode}.pickle"
OVERLAP_THRESHOLD_FILE_TEMPLATE = "data/processed/modelcard_citation_overlap_rate_{sim_mode}_thresholded.pickle"

THRESHOLD = 0.2
SAVE_THRESHOLD_OVERLAP = True
MODE = "reference"  # or "citation"

def load_paperId_lists(df, mode):
    paperId_to_ids = {}
    for _, row in df.iterrows():
        pid = row["paperId"]
        assert not pd.isna(pid)
        parsed = json.loads(row["parsed_response"])
        paper_list = parsed["cited_papers"] if mode == "reference" else parsed["citing_papers"]
        if len(parsed["cited_papers"])==0:
            print(f"Warning: {len(parsed['cited_papers'])} cited papers found for {pid}.")
        id_list = [p["paperId"] for p in paper_list]
        paperId_to_ids[pid] = set(id_list)
    return paperId_to_ids

def compute_overlap_matrices(paperId_to_ids):
    paper_index = sorted(paperId_to_ids.keys())
    n = len(paper_index)
    maxpr_matrix = np.zeros((n, n), dtype=np.float32)
    jaccard_matrix = np.zeros((n, n), dtype=np.float32)
    dice_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="Computing overlap matrices"):
        refs_i = set(paperId_to_ids[paper_index[i]])
        len_i = len(refs_i)
        for j in range(i, n):
            refs_j = set(paperId_to_ids[paper_index[j]])
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

def compute_direct_matrix(df):
    paperId_to_reference = load_paperId_lists(df, mode="reference")
    paperId_to_citation = load_paperId_lists(df, mode="citation")
    paper_ids_set = set(paperId_to_reference.keys()).union(set(paperId_to_citation.keys()))
    paper_index = sorted(paper_ids_set)
    n = len(paper_index)
    direct_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="Computing direct matrix"):
        pid_i = paper_index[i]
        refs_i = paperId_to_reference.get(pid_i, set()) | paperId_to_citation.get(pid_i, set())
        for j in range(i, n):
            pid_j = paper_index[j]
            refs_j = paperId_to_reference.get(pid_j, set()) | paperId_to_citation.get(pid_j, set())
            if (pid_j in refs_i) or (pid_i in refs_j):
                score = 1.0
            else:
                score = 0.0
            direct_matrix[i, j] = direct_matrix[j, i] = score
    score_csr = csr_matrix(direct_matrix)
    return {
        "paper_index": paper_index,
        "score_matrix": score_csr
    }

def main():
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} rows from citation file.")
    paperId_to_ids = load_paperId_lists(df, MODE)
    
    overlap_all = compute_overlap_matrices(paperId_to_ids)
    common_index = overlap_all["paper_index"]
    direct_data = compute_direct_matrix(df)

    if common_index != direct_data["paper_index"]:
        print("Index mismatch detected! Aligning direct matrix to overlap matrix index...")
        overlap_map = {pid: i for i, pid in enumerate(common_index)}
        # create a mapping for overlap index to direct index
        n = len(common_index)
        new_direct_matrix = np.zeros((n, n), dtype=np.float32)
        direct_map = {pid: i for i, pid in enumerate(direct_data["paper_index"])}
        for pid in common_index:
            if pid in direct_map:
                i_new = overlap_map[pid]
                i_old = direct_map[pid]
                new_direct_matrix[i_new, :] = direct_data["score_matrix"][i_old, :]
        direct_data["score_matrix"] = new_direct_matrix
        direct_data["paper_index"] = common_index

    for mode in SIMILARITY_MODES:
        overlap_matrix = overlap_all[mode]
        file_path = OVERLAP_FILE_TEMPLATE.format(sim_mode=mode)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"paper_index": common_index, "score_matrix": overlap_matrix}, f)
        print(f"Saved full overlap matrix for mode {mode} to {file_path}")

        if SAVE_THRESHOLD_OVERLAP:
            thresholded_matrix = overlap_matrix.copy()
            mask = thresholded_matrix.data < THRESHOLD
            thresholded_matrix.data[mask] = 0.0
            thresholded_matrix.eliminate_zeros()
            threshold_file = OVERLAP_THRESHOLD_FILE_TEMPLATE.format(sim_mode=mode)
            with open(threshold_file, "wb") as f:
                pickle.dump({"paper_index": common_index, "score_matrix": thresholded_matrix}, f)
            print(f"Saved thresholded overlap matrix for mode {mode} (THRESHOLD={THRESHOLD}) to {threshold_file}")

    os.makedirs(os.path.dirname(OVERLAP_MATRIX), exist_ok=True)
    with open(DIRECT_MATRIX, "wb") as f:
        pickle.dump(direct_data, f)
    print(f"Saved direct matrix to {DIRECT_MATRIX}")

    print("✅ Done. All data saved:")

if __name__ == "__main__":
    main()

"""
Loaded 4547 rows from citation file.
Computing overlap matrix: 100%|███████████████████████████| 4544/4544 [01:31<00:00, 49.67it/s]
Computing direct matrix: 100%|████████████████████████████| 4544/4544 [02:07<00:00, 35.67it/s]
✅ Done. All data saved:
  - Overlap matrix and index: data/processed/modelcard_citation_overlap_rate.pickle
  - Direct matrix and index:  data/processed/modelcard_citation_direct_label.pickle
"""