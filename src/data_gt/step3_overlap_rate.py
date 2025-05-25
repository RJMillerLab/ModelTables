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
INTENTS = ["background", "methodology", "result", "methodology_or_result"]
COMBINED_PATH = "data/processed/modelcard_citation_all_matrices.pkl.gz"

THRESHOLD = 0.1
SAVE_THRESHOLD_OVERLAP = True
MODE = "reference"  # or "citation"

def load_Id_lists(df, mode, influential=False, intent=None):   ########
    """
    mode: 'reference' or 'citation'
    intent: None → overall  / 'methodology' / 'result'
    influential: bool
    """
    if intent == "methodology_or_result":                                                   ########
        prefix = "ref_papers" if mode=="reference" else "cit_papers"                        ########
        suffix = "_infl_ids" if influential else "_ids"                                    ########
        col1 = f"{prefix}_methodology{suffix}"                                             ########
        col2 = f"{prefix}_result{suffix}"                                                  ########
    else:                                                                                   ########
        prefix = "ref_papers" if mode == "reference" else "cit_papers"                     ########
        base   = f"{prefix}_{'overall' if intent is None else intent}"                     ########
        col    = f"{base}_{'infl_ids' if influential else 'ids'}"     

    Id_to_ids = {}
    for _, row in df.iterrows():
        pid = str(row["corpusid"])
        #ids = row[col]
        if intent == "methodology_or_result":                                               ########
            ids1 = row[col1]                                                       ########
            ids2 = row[col2]                                                       ########
            ids  = list(set(ids1) | set(ids2))                ########
        else:                                                                              ########
            ids = row[col] 
        if not isinstance(ids, (list, tuple, np.ndarray)):
            ids = []
        Id_to_ids[pid] = set(map(str, ids))
    return {pid: list(v) for pid, v in Id_to_ids.items()}

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

"""def original_compute_direct_matrix(Id_to_all, paper_list):
    # load references & citations with optional isInfluential filter
    n = len(paper_list)
    mat = lil_matrix((n, n), dtype=np.bool_)
    for i in tqdm(range(n), desc="Computing direct matrix"):
        pid_i  = paper_list[i]
        refs_i = Id_to_all[pid_i]
        for j in range(i, n):
            pid_j  = paper_list[j]
            refs_j = Id_to_all[pid_j]
            if pid_j in refs_i or pid_i in refs_j:
                mat[i, j] = True
                mat[j, i] = True
    return mat.tocsr()"""

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
    print('keys:', list(df.keys()))

    for intent in INTENTS + [None]:
        for mode, infl in [("reference", False), ("reference", True), ("citation", False), ("citation", True)]:
            d = load_Id_lists(df, mode, influential=infl, intent=intent)
            total_links = sum(len(v) for v in d.values())
            name = f"{mode}_{intent or 'overall'}_{'infl' if infl else 'norm'}"
        print(f"[DEBUG-LOAD] {name}: {len(d)} papers, total links = {total_links}")

    Id_to_ref        = load_Id_lists(df, "reference", influential=False)  ########
    Id_to_ref_infl   = load_Id_lists(df, "reference", influential=True)   ########
    Id_to_cite       = load_Id_lists(df, "citation",  influential=False)  ########
    Id_to_cite_infl  = load_Id_lists(df, "citation",  influential=True)   ########
    Id_to_all        = {pid: Id_to_ref.get(pid, []) + Id_to_cite.get(pid, [])            for pid in set(Id_to_ref)|set(Id_to_cite)}
    Id_to_all_infl   = {pid: Id_to_ref_infl.get(pid, []) + Id_to_cite_infl.get(pid, [])  for pid in set(Id_to_ref_infl)|set(Id_to_cite_infl)}
    Id_list = sorted(set(df["corpusid"]))

    time_start = time.time()
    overlap_all       = compute_overlap_matrices(Id_to_ref,       Id_list)
    overlap_all_infl  = compute_overlap_matrices(Id_to_ref_infl,  Id_list)
    direct            = compute_direct_matrix(Id_to_all,          Id_list)
    direct_infl       = compute_direct_matrix(Id_to_all_infl,     Id_list)
    print(f"[DEBUG] direct nnz={direct.nnz}, direct_infl nnz={direct_infl.nnz}")
    print(f"Time taken for normal: {time.time() - time_start} seconds")

    # --- intent loops
    intent_overlap       = {}
    intent_overlap_infl  = {}
    for intent in INTENTS:
        d_norm = load_Id_lists(df, "reference", False, intent)
        d_infl = load_Id_lists(df, "reference", True,  intent)
        print(f"[DEBUG-LOAD] methodology_or_result? intent={intent}, norm links={sum(len(v) for v in d_norm.values())}, infl links={sum(len(v) for v in d_infl.values())}")
        intent_overlap[intent]      = compute_overlap_matrices(d_norm, Id_list)
        intent_overlap_infl[intent] = compute_overlap_matrices(d_infl, Id_list)
        print(f"[DEBUG-OVR] {intent} non-zero per score: ", {k: mat.nnz for k,mat in intent_overlap[intent].items()})
        print(f"[DEBUG-OVR] {intent} non-zero per score: ", {k: mat.nnz for k,mat in intent_overlap_infl[intent].items()})

    # --- threshold helpers
    def thresh(m):  ########
        x = m.copy(); mask = (x.data < THRESHOLD); x.data[mask]=0; x.eliminate_zeros(); return x.astype(bool)

    combined = {
        "paper_index":             Id_list,
        "direct_label":            direct,
        "direct_label_influential":direct_infl
    }
    # overall
    for k in SIMILARITY_MODES:
        combined[k]                     = overlap_all[k]
        combined[f"{k}_influential"]    = overlap_all_infl[k]
        combined[f"{k}_thresholded"]    = thresh(overlap_all[k])
        combined[f"{k}_influential_thresholded"] = thresh(overlap_all_infl[k])
    # intents
    for intent in INTENTS:
        for k in SIMILARITY_MODES:
            combined[f"{k}_{intent}"]                     = intent_overlap[intent][k]            ########
            combined[f"{k}_{intent}_influential"]         = intent_overlap_infl[intent][k]       ########
            combined[f"{k}_{intent}_thresholded"]         = thresh(intent_overlap[intent][k])    ########
            combined[f"{k}_{intent}_influential_thresholded"] = thresh(intent_overlap_infl[intent][k]) ########

    # Save
    os.makedirs(os.path.dirname(COMBINED_PATH), exist_ok=True)
    with gzip.open(COMBINED_PATH, "wb") as f:
        pickle.dump(combined, f)
    print(f"Saved all matrices to {COMBINED_PATH}")

if __name__ == "__main__":
    main()
