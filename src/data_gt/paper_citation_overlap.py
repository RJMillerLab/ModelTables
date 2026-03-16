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
SIMILARITY_MODES = ["max_pr", "jaccard", "dice"]
INTENTS = ["methodology_or_result"] # "background", "methodology", "result", 

import argparse
from src.utils import load_config, is_list_like, to_list_safe

THRESHOLD = 0.1
SAVE_THRESHOLD_OVERLAP = True
MODE = "reference"  # or "citation"
PAPER_KEY = 'corpusId' # or "paperId"

def load_Id_lists(df, mode, influential=False, intent=None):
    """
    mode: 'reference' or 'citation'
    intent: None → overall  / 'methodology' / 'result'
    influential: bool
    """
    if intent=="methodology_or_result":
        prefix="ref_papers" if mode=="reference" else "cit_papers"; suffix="_infl_ids" if influential else "_ids"
        col1=f"{prefix}_methodology{suffix}"; col2=f"{prefix}_result{suffix}"
        pids=df[PAPER_KEY].astype(str).tolist(); ids_list1=df[col1].tolist(); ids_list2=df[col2].tolist()
        out={}
        for pid,ids1,ids2 in zip(pids,ids_list1,ids_list2):
            if not is_list_like(ids1): ids1=[]
            if not is_list_like(ids2): ids2=[]
            # Normalize all IDs to string so they live in the same space as PAPER_KEY / Id_list
            out[pid]=list(set(map(str, ids1))|set(map(str, ids2)))
        return out
    else:
        prefix="ref_papers" if mode=="reference" else "cit_papers"
        col=f"{prefix}_{'overall' if intent is None else intent}_{'infl_ids' if influential else 'ids'}"
        pids=df[PAPER_KEY].astype(str).tolist(); ids_list=df[col].tolist()
        out={}
        for pid,idt in zip(pids,ids_list):
            if not is_list_like(idt): idt=[]
            # Normalize all IDs to string so they live in the same space as PAPER_KEY / Id_list
            out[pid]=list(set(map(str, idt)))
        return out

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
    # intersection = intersection.maximum(intersection.T)
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
    mat = mat.maximum(mat.T) # if we only have reference, we need this to be symmetric. This work because citation is reversed of reference when filter out those papers don't in paper list.
    return mat

def test_index(INTENTS, df):
    for intent in INTENTS + [None]:
        for mode, infl in [("reference", False), ("reference", True)]: #, ("citation", False), ("citation", True)
            t = time.time()
            d = load_Id_lists(df, mode, influential=infl, intent=intent)
            total_links = sum(len(v) for v in d.values())
            name = f"{mode}_{intent or 'overall'}_{'infl' if infl else 'norm'}"
            print(f"[DEBUG-LOAD] {name}: {len(d)} papers, total links = {total_links}, time = {time.time() - t} seconds")


def parse_cit_papers(json_str, id_key = "citingcorpusid"):
    """
    Parse the input JSON string and extract cited paper details by intent.
    
    This function extracts lists of paper IDs and contexts for three intent types:
      - "methodology"
      - "background"
      - "result"
      
    It also produces 'overall' lists that aggregate all cited papers with any intent.
    
    Returns:
      tuple: (
            method_ids, method_contexts,
            background_ids, background_contexts,
            result_ids, result_contexts,
            overall_ids, overall_contexts
      )
    """
    cit_key = "data" # "cited_papers" or "citing_papers"
    method_ids = []
    method_contexts = []
    background_ids = []
    background_contexts = []
    result_ids = []
    result_contexts = []
    overall_ids = []
    none_ids = []
    
    method_infl_ids = []
    method_infl_ctxs = []
    background_infl_ids = []
    background_infl_ctxs = []
    result_infl_ids = []
    result_infl_ctxs = []
    overall_infl_ids = []

    if pd.isna(json_str) or not isinstance(json_str, str):
        # Keep return signature consistent with the main return below:
        # only *_ids lists, no contexts.
        return (method_ids,
                background_ids,
                result_ids,
                # method_contexts
                # background_contexts
                # result_contexts
                overall_ids,
                method_infl_ids,
                background_infl_ids,
                result_infl_ids,
                # method_infl_ctxs
                # background_infl_ctxs
                # result_infl_ctxs
                overall_infl_ids)
    #try:
    if True:
        data = json.loads(json_str)
        cit_papers = data[cit_key]
        for item in cit_papers:
            # Some records may miss the expected id_key; treat them as having no valid paper_id.
            '''paper_id = item[id_key]
            intents_nested = item["intents"]
            contexts = item["contexts"]'''
            paper_id = item.get(id_key)
            intents_nested = item.get("intents")
            contexts = item.get("contexts")
            if paper_id is None or not intents_nested:
                none_ids.append(paper_id)
                overall_ids.append(paper_id)
                #print(f"Missing paper_id or intents_nested or contexts: {item}")
                continue
            # Flatten: intents_nested = [['methodology'], ['result']] -> ['methodology', 'result']
            intents_flat = [i for sub in intents_nested for i in (sub if isinstance(sub, list) else [sub])]

            if len(intents_flat) == len(contexts):
                pairs = zip(intents_flat, contexts)
            else:
                # fallback: align all intents with a combined context string
                joined_context = " ".join(contexts)
                pairs = zip(intents_flat, [joined_context] * len(intents_flat))
            influential = item["isinfluential"]
            for intent, ctx in pairs:
                if intent == "methodology":
                    method_ids.append(paper_id)
                    method_contexts.append(ctx)
                    if influential:
                        method_infl_ids.append(paper_id)
                        method_infl_ctxs.append(ctx)
                elif intent == "background":
                    background_ids.append(paper_id)
                    background_contexts.append(ctx)
                    if influential:
                        background_infl_ids.append(paper_id)
                        background_infl_ctxs.append(ctx)
                elif intent == "result":
                    result_ids.append(paper_id)
                    result_contexts.append(ctx)
                    if influential:
                        result_infl_ids.append(paper_id)
                        result_infl_ctxs.append(ctx)
                elif intent in ["None", "none", None]:
                    none_ids.append(paper_id)
                else:
                    raise ValueError(f"Unknown intent: {intent}")
                # All intents contribute to overall
                overall_ids.append(paper_id)
                if influential:
                    overall_infl_ids.append(paper_id)
    # make them unique
    method_ids        = list(dict.fromkeys(method_ids))
    background_ids    = list(dict.fromkeys(background_ids))
    result_ids        = list(dict.fromkeys(result_ids))          
    overall_ids       = list(dict.fromkeys(overall_ids))         
    method_infl_ids   = list(dict.fromkeys(method_infl_ids))     
    background_infl_ids = list(dict.fromkeys(background_infl_ids)) 
    result_infl_ids   = list(dict.fromkeys(result_infl_ids))     
    overall_infl_ids  = list(dict.fromkeys(overall_infl_ids))    
    return (method_ids, background_ids, result_ids, overall_ids, method_infl_ids, background_infl_ids, result_infl_ids, overall_infl_ids)

def main(input_parquet, title2ids_parquet, combined_output_path, skip_threshold=True):
    t1 = time.time()
    # Load
    df = pd.read_parquet(input_parquet)
    df_title2ids = pd.read_parquet(title2ids_parquet, columns=[PAPER_KEY])
    print(f"Loaded {len(df)} rows from input_parquet")
    # 1) Drop rows where PAPER_KEY is null/NaN
    df = df[df[PAPER_KEY].notna()].copy()
    print("Columns:", list(df.keys()))
    print(f"Rows after dropping null {PAPER_KEY}: {len(df)}")
    # 2) Normalize PAPER_KEY in both dataframes to string without trailing '.0'
    df[PAPER_KEY] = df[PAPER_KEY].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df_title2ids[PAPER_KEY] = df_title2ids[PAPER_KEY].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    # 3) Build allowed id set from titles2ids and filter df by this set
    title_ids = set(df_title2ids[PAPER_KEY].dropna().unique())
    df = df[df[PAPER_KEY].isin(title_ids)].copy()
    print(f"Rows with {PAPER_KEY} present in titles2ids: {len(df)}")
    # 4) Deduplicate on PAPER_KEY after filtering
    df = df.drop_duplicates(subset=[PAPER_KEY]).copy()
    print(f"Rows after deduplicating by {PAPER_KEY}: {len(df)}")
    print(f"Time taken for deduplicating: {time.time() - t1} seconds")

    t2 = time.time()
    # apply
    #from src.data_preprocess.s2orc_merge import parse_cit_papers
    ref_new_cols = df["original_response"].apply(
        lambda x: pd.Series(parse_cit_papers(x, id_key="citedcorpusid"), index=["ref_papers_methodology_ids", "ref_papers_background_ids", "ref_papers_result_ids", "ref_papers_overall_ids", "ref_papers_methodology_infl_ids", "ref_papers_background_infl_ids", "ref_papers_result_infl_ids", "ref_papers_overall_infl_ids"])
    )
    df = pd.concat([df, ref_new_cols], axis=1) # cit_new_cols, 
    print(f"Time taken for parse_cit_papers: {time.time() - t2} seconds")
    #test_index(INTENTS, df)

    Id_to_ref        = load_Id_lists(df, "reference", influential=False)
    Id_to_ref_infl   = load_Id_lists(df, "reference", influential=True)
    #Id_to_cite       = load_Id_lists(df, "citation",  influential=False)
    #Id_to_cite_infl  = load_Id_lists(df, "citation",  influential=True)
    #Id_to_all        = {pid: Id_to_ref.get(pid, []) + Id_to_cite.get(pid, [])            for pid in set(Id_to_ref)|set(Id_to_cite)}
    #Id_to_all_infl   = {pid: Id_to_ref_infl.get(pid, []) + Id_to_cite_infl.get(pid, [])  for pid in set(Id_to_ref_infl)|set(Id_to_cite_infl)}
    Id_list = sorted(set(df[PAPER_KEY]))

    time_start = time.time()
    #direct            = compute_direct_matrix(Id_to_all,          Id_list)
    #direct_infl       = compute_direct_matrix(Id_to_all_infl,     Id_list)
    direct            = compute_direct_matrix(Id_to_ref,          Id_list)
    direct_infl       = compute_direct_matrix(Id_to_ref_infl,     Id_list)
    if not skip_threshold:
        overlap_all       = compute_overlap_matrices(Id_to_ref,       Id_list)
        overlap_all_infl  = compute_overlap_matrices(Id_to_ref_infl,  Id_list)
    print(f"[DEBUG] direct nnz={direct.nnz}, direct_infl nnz={direct_infl.nnz}")
    print(f"Time taken for computing matrix: {time.time() - time_start} seconds")

    # --- intent loops
    intent_overlap       = {}
    intent_overlap_infl  = {}
    intent_direct       = {}
    intent_direct_infl  = {}
    for intent in INTENTS:
        d_norm = load_Id_lists(df, "reference", False, intent)
        d_infl = load_Id_lists(df, "reference", True,  intent)
        print(f"[DEBUG-LOAD] methodology_or_result? intent={intent}, norm links={sum(len(v) for v in d_norm.values())}, infl links={sum(len(v) for v in d_infl.values())}")
        intent_direct[intent]       = compute_direct_matrix(d_norm, Id_list)
        intent_direct_infl[intent]  = compute_direct_matrix(d_infl, Id_list)
        if not skip_threshold:
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
    
    if not skip_threshold:
        print('Thresholding matrices...')
        # overall
        for k in SIMILARITY_MODES:
            combined[k]                     = overlap_all[k]
            combined[f"{k}_influential"]    = overlap_all_infl[k]
            combined[f"{k}_thresholded"]    = thresh(overlap_all[k])
            combined[f"{k}_influential_thresholded"] = thresh(overlap_all_infl[k])
            # intent
            for intent in INTENTS:
                combined[f"{k}_{intent}"]                           = intent_overlap[intent][k]
                combined[f"{k}_{intent}_influential"]               = intent_overlap_infl[intent][k]
                combined[f"{k}_{intent}_thresholded"]               = thresh(intent_overlap[intent][k])
                combined[f"{k}_{intent}_influential_thresholded"]   = thresh(intent_overlap_infl[intent][k])
    # intents
    print('Computing direct matrices...')
    for intent in INTENTS:
        combined[f"direct_label_{intent}"]                            = intent_direct[intent]
        combined[f"direct_label_{intent}_influential"]                = intent_direct_infl[intent]

    # Save
    print('Saving matrices...')
    os.makedirs(os.path.dirname(combined_output_path), exist_ok=True)
    with gzip.open(combined_output_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"Saved all matrices to {combined_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute paper-pair overlap scores for citation analysis")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""
    
    # Determine input/output paths based on tag
    #INPUT_PARQUET = "data/processed/modelcard_citation_enriched.parquet" # original query id from online
    #input_parquet = os.path.join(base_path, 'processed', f"s2orc_rerun{suffix}.parquet")
    input_parquet = os.path.join(base_path, 'processed', f"s2orc_references_cache{suffix}.parquet")
    title2ids_parquet = os.path.join(base_path, 'processed', f"s2orc_titles2ids{suffix}.parquet")
    combined_output_path = os.path.join(base_path, 'processed', f"modelcard_citation_all_matrices{suffix}.pkl.gz")
    
    print("📁 Paths in use:")
    print(f"   Input annotations:   {input_parquet}")
    print(f"   Title2ids parquet:    {title2ids_parquet}")
    print(f"   Output matrices:     {combined_output_path}")
    
    SKIP_THRESHOLD = True # don't run threshold mode
    main(input_parquet=input_parquet, title2ids_parquet=title2ids_parquet, combined_output_path=combined_output_path, skip_threshold=SKIP_THRESHOLD)
