"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-08
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
OVERLAP_MATRIX = "data/processed/modelcard_citation_overlap_rate.pickle"
DIRECT_MATRIX = "data/processed/modelcard_citation_direct_label.pickle"
#THRESHOLD = 0.6
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

def compute_overlap_matrix(paperId_to_ids):
    paper_index = sorted(paperId_to_ids.keys())
    n = len(paper_index)
    score_matrix = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="Computing overlap matrix"):
        refs_i = set(paperId_to_ids[paper_index[i]])
        for j in range(i, n):
            refs_j = set(paperId_to_ids[paper_index[j]])
            intersection = refs_i & refs_j
            union = refs_i | refs_j
            if union:
                score = len(intersection) / len(union)
            else:
                score = 0.0
            score_matrix[i, j] = score
            score_matrix[j, i] = score
    score_csr = csr_matrix(score_matrix)
    return {
        "paper_index": paper_index,
        "score_matrix": score_csr
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
    
    overlap_data = compute_overlap_matrix(paperId_to_ids)
    direct_data = compute_direct_matrix(df)

    overlap_index = overlap_data["paper_index"]
    direct_index = direct_data["paper_index"]

    if overlap_index != direct_index:
        print("Index mismatch detected! Aligning direct matrix to overlap matrix index...")
        overlap_map = {pid: i for i, pid in enumerate(overlap_index)}
        # create a mapping for overlap index to direct index
        n = len(overlap_index)
        new_direct_matrix = np.zeros((n, n), dtype=np.float32)
        direct_map = {pid: i for i, pid in enumerate(direct_index)}
        for pid in overlap_index:
            if pid in direct_map:
                i_new = overlap_map[pid]
                i_old = direct_map[pid]
                new_direct_matrix[i_new, :] = direct_data["score_matrix"][i_old, :]
        direct_data["score_matrix"] = new_direct_matrix
        direct_data["paper_index"] = overlap_index  ########

    os.makedirs(os.path.dirname(OVERLAP_MATRIX), exist_ok=True)
    with open(OVERLAP_MATRIX, "wb") as f:
        pickle.dump(overlap_data, f)
    with open(DIRECT_MATRIX, "wb") as f:
        pickle.dump(direct_data, f)

    print("✅ Done. All data saved:")
    print(f"  - Overlap matrix and index: {OVERLAP_MATRIX}")
    print(f"  - Direct matrix and index:  {DIRECT_MATRIX}")

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