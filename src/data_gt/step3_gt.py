"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-16
Description: Build SciLake union benchmark tables.

Usage:
python -m src.data_gt.step3_gt --overlap_rate_threshold 0.0 --rel_key [
    direct_label, 
    direct_label_influential, 
    direct_label_methodology_or_result, 
    direct_label_methodology_or_result_influential, 
    max_pr, 
    max_pr_influential, 
    max_pr_methodology_or_result, 
    max_pr_methodology_or_result_influential
]
"""
from multiprocessing import set_start_method
set_start_method("fork", force=True)

import os, json, gzip, pickle, time
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.sparse import csr_matrix, coo_matrix, save_npz


# === Configuration ===
DATA_DIR = "data/processed"
GT_DIR = "data/gt"

# ---- default files (can be overridden by CLI/kwargs) ----
# These will be updated based on tag if provided
FILES = {
    "combined": f"{DATA_DIR}/modelcard_citation_all_matrices.pkl.gz",
    "step3_dedup": f"{DATA_DIR}/modelcard_step3_dedup_v2.parquet",
    "integration": f"{DATA_DIR}/final_integration_with_paths_v2.parquet",
    "title_list": f"{DATA_DIR}/modelcard_all_title_list.parquet",
    "valid_title": f"{DATA_DIR}/all_title_list_valid.parquet"
}

def update_files_with_tag(tag=None):
    """Update FILES dictionary with tag suffix if provided."""
    global FILES
    suffix = f"_{tag}" if tag else ""
    FILES = {
        "combined": f"{DATA_DIR}/modelcard_citation_all_matrices{suffix}.pkl.gz",
        "step3_dedup": f"{DATA_DIR}/modelcard_step3_dedup_v2{suffix}.parquet",
        "integration": f"{DATA_DIR}/final_integration_with_paths_v2{suffix}.parquet",
        "title_list": f"{DATA_DIR}/modelcard_all_title_list{suffix}.parquet",
        "valid_title": f"{DATA_DIR}/all_title_list_valid{suffix}.parquet" if suffix else f"{DATA_DIR}/all_title_list_valid.parquet"
    }

# ===== FACTORIES =========================================================== #
def load_relationships(rel_key: str):
    """Factory loader for paperId‑level relationship graphs."""
    path = FILES["combined"]
    print(f"Loading relationships (combined) from: {path}")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    paper_index = data["paper_index"]
    print(f"[DEBUG] Loaded paper_index with length: {len(paper_index)}")
    if rel_key not in data:                           
        raise KeyError(
            f"Key '{rel_key}' not found. Available keys: {list(data.keys())[:10]} ...")  
    # Select the proper score matrix from the new keys
    score_matrix = data[rel_key]
    print(f"[DEBUG] Loaded score_matrix with shape: {score_matrix.shape}")
    return paper_index, score_matrix

def load_table_source():
    """Factory loader for table‑list metadata."""
    print(f"Loading table source from: step3_dedup")
    # load valid_title, and merge by modelId
    # Try tag version first, fallback to default if not found
    valid_title_path = FILES["valid_title"]
    if not os.path.exists(valid_title_path):
        # Fallback to default path (without tag)
        default_path = f"{DATA_DIR}/all_title_list_valid.parquet"
        if os.path.exists(default_path):
            print(f"⚠️  Tag version not found, using default: {default_path}")
            valid_title_path = default_path
        else:
            raise FileNotFoundError(f"Neither tag version ({FILES['valid_title']}) nor default version ({default_path}) found. Please run qc_stats.py first.")
    df_valid_title = pd.read_parquet(valid_title_path, columns=["modelId", "all_title_list_valid"])
    print(f"[DEBUG] Loaded df_valid_title with shape: {df_valid_title.shape}")
    df = pd.read_parquet(FILES["step3_dedup"], columns=["modelId", "hugging_table_list_dedup", "github_table_list_dedup", "html_table_list_mapped_dedup", "llm_table_list_mapped_dedup"])
    print(f"[DEBUG] Loaded df with shape: {df.shape}")
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
    print(f"[DEBUG] After merge, df_tables shape: {df_tables.shape}")
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

def build_paper_matrix(score_matrix: csr_matrix, rel_key: str, overlap_rate_threshold: float):
    if rel_key.startswith("direct_label"):
        paper_adj = (score_matrix >= 1.0).astype(np.bool_)
    else:
        paper_adj = (score_matrix > overlap_rate_threshold).astype(np.bool_)
    paper_adj.setdiag(True)
    print(f"[DEBUG] Built paper_adj matrix with shape: {paper_adj.shape}, nnz: {paper_adj.nnz}")
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

def build_ground_truth(rel_key, overlap_rate_threshold, save_matrix_flag=True, tag=None):
    # modelId-paperList-csvList, Our aim is to use paper-paper matrix to build {csv:[csv1, csv2]} related json
    """High‑level orchestration for building GT tables."""
    # ========== Step 1: Load paper-level info. Build paper-level adjacency matrix ==========
    paper_index, score_matrix = load_relationships(rel_key)
    paper_paper_adj = build_paper_matrix(score_matrix, rel_key, overlap_rate_threshold)
    paper_paper_adj = paper_paper_adj.astype(bool).tocsr()
    assert paper_paper_adj.data.dtype == np.bool_
    print(f"Step1: [Paper-level] Adjacency shape: {paper_paper_adj.shape}: ", paper_paper_adj.nnz)
    print(f"[DEBUG] Paper-paper adjacency matrix statistics:")
    print(f"  - Total papers: {len(paper_index)}")
    print(f"  - Non-zero elements: {paper_paper_adj.nnz}")
    print(f"  - Average connections per paper: {paper_paper_adj.nnz / len(paper_index):.2f}")
    
    # Check paper ID formats
    print("\n[DEBUG] Paper ID format check:")
    print(f"  - First 3 paper_index IDs: {list(paper_index)[:3]}")
    print(f"  - First 3 paper_index ID types: {[type(x) for x in list(paper_index)[:3]]}")

    title_df = pd.read_parquet(FILES["integration"], columns=["corpusid", "query"])
    print(f"[DEBUG] Loaded title_df with shape: {title_df.shape}")
    cid2titles = defaultdict(list)
    for cid, title in zip(title_df["corpusid"].astype(str), title_df["query"]):
        cid2titles[cid].append(title)
    print(f"[DEBUG] Built cid2titles with {len(cid2titles)} unique corpusids")
    print(f"[DEBUG] Sample of cid2titles (first 3 items): {dict(list(cid2titles.items())[:3])}")
    del title_df

    # ---------- Step 2: modelId → csv mapping ----------
    df_tables = load_table_source()
    # Filter rows with nonempty paper & csv lists
    rows = []
    for _, row in df_tables.iterrows():
        papers = row['all_title_list_valid']
        csvs   = row['all_table_list_dedup']  # updated col ########
        rows.append((papers, csvs))
    print(f"[DEBUG] Built rows list with length: {len(rows)}")
    del df_tables

    # Build reverse index: paper -> row IDs
    paper2rows = defaultdict(list)
    for rid, (papers, _) in enumerate(rows):
        for p in papers:
            paper2rows[p].append(rid)
    print(f"[DEBUG] Built paper2rows with {len(paper2rows)} unique papers")
    
    # Check paper2rows against paper_index
    valid_papers = set(paper_index)
    papers_in_rows = set(paper2rows.keys())
    print(f"[DEBUG] Paper mapping statistics:")
    print(f"  - Papers in paper_index: {len(valid_papers)}")
    print(f"  - Papers in paper2rows: {len(papers_in_rows)}")
    print(f"  - Papers in both: {len(valid_papers.intersection(papers_in_rows))}")
    print(f"  - Papers only in paper2rows: {len(papers_in_rows - valid_papers)}")
    print(f"  - Papers only in paper_index: {len(valid_papers - papers_in_rows)}")
    
    # Check paper ID formats in paper2rows
    print("\n[DEBUG] Paper2rows ID format check:")
    if len(paper2rows) == 0:
        print("  ⚠️  WARNING: paper2rows is empty! No valid tables found.")
        print("  This usually means:")
        print("    1. step2_dedup_tables.py filtered out all paths (check if directories exist)")
        print("    2. step2_merge_tables.py did not generate table lists correctly")
        print("    3. Paths in step3_merged do not match actual file locations")
        raise ValueError("Cannot proceed: paper2rows is empty. Please check step2_dedup_tables.py output and ensure table directories exist.")
    
    sample_papers = list(paper2rows.keys())[:3]
    print(f"  - First 3 paper2rows keys: {sample_papers}")
    print(f"  - First 3 paper2rows key types: {[type(x) for x in sample_papers]}")
    
    sample_pid = next(iter(paper2rows))
    print(f"[DEBUG] paper2rows first key → {sample_pid!r} , type={type(sample_pid)}")
    print(f"[DEBUG] idx2pid[0] → {paper_index[0]!r} , type={type(paper_index[0])}")
    print(f"[DEBUG] idx2pid[0] == sample?  {paper_index[0]==sample_pid}")

    # Dictionary for flat tuple-key counts
    csv_pair_cnt = defaultdict(int)

    # -- Vectorized CSV pair counting via sparse matrices --
    # build global CSV list & index
    flat = [c for _, cs in rows for c in cs]
    all_csvs = list(dict.fromkeys(flat))
    print(f"[DEBUG] Built all_csvs list with length: {len(all_csvs)}")
    csv2idx  = {c: i for i, c in enumerate(all_csvs)}
    print(f"[DEBUG] Built csv2idx with length: {len(csv2idx)}")

    # 1) intra-row: construct B for same-model CSV pairs
    row_b, col_b = [], []
    for _, cs in rows:
        for a, b in combinations(sorted(set(cs)), 2):
            ia, ib = csv2idx[a], csv2idx[b]
            row_b += [ia, ib]; col_b += [ib, ia]
    B = coo_matrix((np.ones(len(row_b), int), (row_b, col_b)),
                   shape=(len(all_csvs), len(all_csvs))).tocsr()
    print(f"Step2: [Intra-row] Adjacency shape: {B.shape}: ", B.nnz)
    del row_b, col_b

    # 2) inter-row: build A (paper→CSV) and compute C = Aᵀ·P·A
    corpus2pidx = {cid:i for i,cid in enumerate(paper_index)}
    print(f"[DEBUG] Built corpus2pidx with length: {len(corpus2pidx)}")
    title2cid   = {t:cid for cid,titles in cid2titles.items() for t in titles}
    print(f"[DEBUG] Built title2cid with length: {len(title2cid)}")
    print(f"[DEBUG] Sample of title2cid (first 3 items): {dict(list(title2cid.items())[:3])}")
    print('length of title2cid: ', len(title2cid))
    print('length of cid2titles: ', len(cid2titles))
    
    # Track get() operations
    missing_title_count = 0
    missing_cid_count = 0
    missing_p_count = 0
    total_titles = 0
    successful_mappings = 0
    
    # Track sample of missing mappings
    missing_title_samples = set()
    missing_cid_samples = set()
    
    row_i, col_i = [], []
    for titles, cs in rows:
        for t in titles:
            total_titles += 1
            cid = title2cid.get(t)
            if cid is None:
                missing_title_count += 1
                if len(missing_title_samples) < 3:
                    missing_title_samples.add(t)
                continue
            p = corpus2pidx.get(cid)
            if p is None:
                missing_cid_count += 1
                if len(missing_cid_samples) < 3:
                    missing_cid_samples.add(f"{t} -> {cid}")
                continue
            successful_mappings += 1
            for c in cs:
                row_i.append(p)
                col_i.append(csv2idx[c])
    
    print(f"\n[DEBUG] Detailed mapping statistics:")
    print(f"  - Total titles processed: {total_titles}")
    print(f"  - Titles not found in title2cid: {missing_title_count} ({missing_title_count/total_titles*100:.2f}%)")
    print(f"  - CIDs not found in corpus2pidx: {missing_cid_count} ({missing_cid_count/total_titles*100:.2f}%)")
    print(f"  - Successfully mapped titles: {successful_mappings} ({successful_mappings/total_titles*100:.2f}%)")
    print(f"\n[DEBUG] Sample of missing mappings:")
    print(f"  - Missing title samples: {list(missing_title_samples)}")
    print(f"  - Missing CID samples: {list(missing_cid_samples)}")
    
    print(f"[DEBUG] Built row_i and col_i with lengths: {len(row_i)}, {len(col_i)}")
    print('finished building A')
    A = coo_matrix((np.ones(len(row_i), bool),(row_i,col_i)), shape=(len(paper_index), len(all_csvs))).astype(bool).tocsr()
    print(f"[DEBUG] Built A matrix with shape: {A.shape}, nnz: {A.nnz}")
    C = A.T.dot(paper_paper_adj).dot(A).tocsr()
    print(f"Step2: [Inter-row] Adjacency shape: {C.shape}: ", C.nnz)
    #del A, corpus2pidx, title2cid, row_i, col_i, paper_paper_adj

    # 3) sum and extract
    M = (B + C).astype(bool).tocsr() # we didn't care count
    M.setdiag(False)
    print(f"[DEBUG] Final M matrix shape: {M.shape}, nnz: {M.nnz}")
    del B, C

    # Add detailed mapping checks
    print("\n[DEBUG] Detailed mapping integrity check:")
    
    # Check title2cid mapping
    print("\n1. Title to CID mapping check:")
    sample_titles = list(title2cid.keys())[:3]
    print(f"  - Sample titles: {sample_titles}")
    print(f"  - Their CIDs: {[title2cid[t] for t in sample_titles]}")
    print(f"  - CIDs in paper_index: {[cid in paper_index for cid in [title2cid[t] for t in sample_titles]]}")
    
    # Check corpus2pidx mapping
    print("\n2. CID to matrix index mapping check:")
    sample_cids = list(corpus2pidx.keys())[:3]
    print(f"  - Sample CIDs: {sample_cids}")
    print(f"  - Their matrix indices: {[corpus2pidx[cid] for cid in sample_cids]}")
    
    # Check csv2idx mapping
    print("\n3. CSV to index mapping check:")
    sample_csvs = list(csv2idx.keys())[:3]
    print(f"  - Sample CSVs: {sample_csvs}")
    print(f"  - Their indices: {[csv2idx[csv] for csv in sample_csvs]}")
    
    # Check row_i and col_i construction
    print("\n4. Row and column index construction check:")
    print(f"  - First 5 row_i values: {row_i[:5]}")
    print(f"  - First 5 col_i values: {col_i[:5]}")
    print(f"  - Corresponding papers: {[paper_index[i] for i in row_i[:5]]}")
    print(f"  - Corresponding CSVs: {[all_csvs[i] for i in col_i[:5]]}")
    
    if save_matrix_flag:
        time1 = time.time()
        print('saving matrix and csv list')
        suffix = f"_{tag}" if tag else ""
        matrix_path = f"{GT_DIR}/csv_pair_matrix_{rel_key}{suffix}.npz"
        save_npz(matrix_path, M, compressed=True)
        print(f"✅ Sparse matrix saved to {matrix_path}")
        csv_list_path = f"{GT_DIR}/csv_list_{rel_key}{suffix}.pkl"
        # all_csvs (set) get basename
        all_csvs = [os.path.basename(csv) for csv in all_csvs]
        with open(csv_list_path, "wb") as f:
            pickle.dump(all_csvs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ CSV list saved (order matches matrix rows/cols) to {csv_list_path}")
        time2 = time.time()
        print(f"time cost: {time2 - time1} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument("--rel_key", type=str, default='direct_label', help="Exact key inside combined pickle (e.g., 'max_pr', 'direct_label_influential').")
    parser.add_argument("--overlap_rate_threshold", type=float, default=0.0, help=("Numeric threshold for similarity/overlap matrices; ignored for 'direct_label*' keys."))
    parser.add_argument("--tag", dest="tag", default=None, help="Tag suffix for versioning (e.g., 251117). Enables versioning mode for input files.")
    args = parser.parse_args()

    # Update file paths based on tag
    if args.tag:
        update_files_with_tag(args.tag)
        print("📁 Using tag-based input files:")
        for key, path in FILES.items():
            print(f"   {key}: {path}")
        print(f"   GT output directory: {GT_DIR}/ (with tag suffix)")

    build_ground_truth(rel_key=args.rel_key, overlap_rate_threshold=args.overlap_rate_threshold, tag=args.tag)
