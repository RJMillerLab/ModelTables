"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-16
Description: Build SciLake union benchmark tables.

Usage:
python -m src.data_gt.step3_gt --tag 251117 --v2_mode
"""
from multiprocessing import set_start_method
set_start_method("fork", force=True)

import os, json, gzip, pickle, time
import duckdb
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
from src.utils import is_list_like, to_list_safe


############## Multiplication ####################

def fast_multiply(A: csr_matrix, B: csr_matrix):
    """
    Fast multiply C = A @ B with two layers of simple pruning

    First layer of pruning:
    - Drop rows of A that are all zero
    - Drop columns of B that are all zero

    Second layer of pruning:
    - Drop columns of A that are all zero
    - Drop rows of B that are all zero

    Finally, restore the dropped rows/columns with zeros, so the returned C is still the original size.
    Also print the pruning ratios and a rough "dense computation ratio".
    """
    # Ensure A and B are CSR matrices for stable getnnz / slicing
    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    if not isinstance(B, csr_matrix):
        B = B.tocsr()

    n_rows_A, n_cols_A = A.shape
    n_rows_B, n_cols_B = B.shape
    assert n_cols_A == n_rows_B, "Shape mismatch for A @ B"

    print("[fast-matmul] === A @ B start ===")
    print("[fast-matmul] A:", A.shape, "nnz", A.nnz,
          "| B:", B.shape, "nnz", B.nnz)

    # 1) Outer layer: non-zero rows of A / non-zero columns of B
    row_mask_A = np.diff(A.indptr) > 0
    col_nnz_B = np.asarray(B.getnnz(axis=0)).ravel()
    col_mask_B = col_nnz_B > 0

    kept_rows_A = int(row_mask_A.sum())
    kept_cols_B = int(col_mask_B.sum())
    print(f"[fast-matmul] keep A rows: {kept_rows_A}/{n_rows_A} "
          f"(dropped {n_rows_A - kept_rows_A} zero rows)")
    print(f"[fast-matmul] keep B cols: {kept_cols_B}/{n_cols_B} "
          f"(dropped {n_cols_B - kept_cols_B} zero cols)")

    A_outer = A[row_mask_A]
    B_outer = B[:, col_mask_B]

    # 2) Shared dimension: only keep k where both A_outer[:, k] and B_outer[k, :] are non-zero
    col_nnz_A = np.asarray(A_outer.getnnz(axis=0)).ravel()
    row_nnz_B = np.asarray(B_outer.getnnz(axis=1)).ravel()
    active = (col_nnz_A > 0) & (row_nnz_B > 0)
    idx = np.where(active)[0]

    print(f"[fast-matmul] active shared indices: {len(idx)}/{A_outer.shape[1]} "
          f"(dropped {A_outer.shape[1] - len(idx)} columns(A)/rows(B))")

    # Rough dense work estimate: original ~ n_rows_A * n_cols_A * n_cols_B
    dense_before = float(n_rows_A) * float(n_cols_A) * float(n_cols_B)
    dense_after = float(kept_rows_A) * float(len(idx)) * float(kept_cols_B)
    ratio = dense_after / dense_before if dense_before > 0 else 1.0
    print(f"[fast-matmul] approx dense-work ratio after pruning: {ratio:.4e} "
          f"(~{ratio*100:.2f}% of original)")

    # 3) Perform multiplication on the pruned submatrices
    A_inner = A_outer[:, idx]
    B_inner = B_outer[idx, :]
    C_inner = A_inner.dot(B_inner)
    print("[fast-matmul] inner result shape:", C_inner.shape, "nnz:", C_inner.nnz)

    # 4) Restore rows/columns to original size (dropped positions are all zeros)
    if kept_rows_A == n_rows_A:
        C_rows_restored = C_inner
    else:
        C_rows_restored = csr_matrix((n_rows_A, kept_cols_B), dtype=C_inner.dtype)
        C_rows_restored[row_mask_A] = C_inner

    if kept_cols_B == n_cols_B:
        C_full = C_rows_restored
    else:
        C_full = csr_matrix((n_rows_A, n_cols_B), dtype=C_inner.dtype)
        C_full[:, col_mask_B] = C_rows_restored

    print("[fast-matmul] full result shape:", C_full.shape, "nnz:", C_full.nnz)
    print("[fast-matmul] === A @ B done ===")
    return C_full

def get_final_csv_csv_adj_by_multiplication(paper_csv_adj, paper_paper_adj):
    paper_csv_adj = paper_csv_adj.astype(np.int8).tocsr()
    paper_paper_adj = paper_paper_adj.astype(np.int8).tocsr()
    t1 = time.time()
    tmp = paper_csv_adj.T.dot(paper_paper_adj)
    print(f"dot1: A^T @ P done, time={time.time() - t1:.4f}s, shape={tmp.shape}, nnz={tmp.nnz}")

    t1 = time.time()
    # Prune zero rows/cols to reduce matmul workload
    csv_csv_adj_outside_model = fast_multiply(A=tmp, B=paper_csv_adj)
    #csv_csv_adj_outside_model = tmp.dot(paper_csv_adj).astype(bool).tocsr()
    print(f"dot2: C=A^T P A via fast_multiply done, time={time.time() - t1:.4f}s, shape={csv_csv_adj_outside_model.shape}, nnz={csv_csv_adj_outside_model.nnz}")
    return csv_csv_adj_outside_model

############## Local CSR <-> Parquet helpers ####################

def local_csrnpz_to_parquet(npz_path: str, parquet_path: str, row_col_names=("row_idx", "col_idx")) -> None:
    """
    Convert a local CSR .npz file into a Parquet edge list.

    - npz_path: path to CSR matrix saved via scipy.sparse.save_npz
    - parquet_path: output Parquet path
    - row_col_names: (row_name, col_name) to use as column names in the Parquet file
    """
    row_name, col_name = row_col_names
    mat = load_npz(npz_path).tocsr()
    rows, cols = mat.nonzero()
    df = pd.DataFrame(
        {
            row_name: rows.astype(np.int64),
            col_name: cols.astype(np.int64),
        }
    )
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df.to_parquet(parquet_path)
    print(f"[local] csrnpz→parquet: npz={npz_path}, parquet={parquet_path}, edges={len(df)}")


def local_parquet_to_csrnpz(
    parquet_path: str,
    npz_path: str,
    shape: tuple[int, int],
    row_col_names=("row_idx", "col_idx"),
) -> None:
    """
    Convert a local Parquet edge list into a CSR .npz file.

    - parquet_path: input Parquet with (row, col) index columns
    - npz_path: output .npz path to save CSR matrix
    - shape: (n_rows, n_cols) of the target matrix
    - row_col_names: (row_name, col_name) used in the Parquet file
    """
    row_name, col_name = row_col_names
    con = duckdb.connect()
    try:
        t_load = time.time()
        df = con.execute(
            f"SELECT {row_name}, {col_name} FROM read_parquet('{parquet_path}')"
        ).df()
        print(f"[local] parquet→csrnpz load: parquet={parquet_path}, rows={len(df)}, time={time.time() - t_load:.4f}s")
    finally:
        con.close()

    n_rows, n_cols = shape
    if df.empty:
        M = csr_matrix((n_rows, n_cols), dtype=bool)
    else:
        rows = df[row_name].to_numpy(dtype=np.int64)
        cols = df[col_name].to_numpy(dtype=np.int64)
        data = np.ones(len(rows), dtype=np.int8)
        t_mat = time.time()
        M = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).astype(bool).tocsr()
        print(f"[local] parquet→csrnpz materialize: shape={M.shape}, nnz={M.nnz}, build_time={time.time() - t_mat:.4f}s")

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    save_npz(npz_path, M, compressed=True)
    print(f"[local] parquet→csrnpz saved: npz={npz_path}")


def get_n_rows_from_csrnpz(npz_path: str) -> int:
    """
    Lightweight helper to read the number of rows from a CSR .npz file
    without loading the full sparse matrix into memory.
    """
    with np.load(npz_path, allow_pickle=False) as f:
        shape = f["shape"]
    return int(shape[0])

def get_final_csv_csv_adj_by_join(
    A_parquet_path: str,
    P_npz_path: str,
    path_outside_parquet: str,
) -> None:
    """
    Step2: compute csv‑csv adjacency edges (Aᵀ·P·A) and write them directly to a Parquet file.

    - Input files:
      - A_parquet_path: global edge-list Parquet for A (paper → csv), built once
      - P_npz_path:     CSR .npz for P (paper → paper) for this rel_key
    - Output file:
      - path_outside_parquet: Parquet with columns (src_csv, dst_csv), deduplicated
    """
    # 1) write temporary Parquet file for P edges for DuckDB to consume
    base_dir = os.path.dirname(path_outside_parquet)
    os.makedirs(base_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path_outside_parquet))[0]
    pathP = os.path.join(base_dir, f"{stem}_P_edges_tmp.parquet")

    local_csrnpz_to_parquet(P_npz_path, pathP, row_col_names=("paper_i", "paper_j"))
    print(f"[join] Step2 wrote P edge Parquet for {stem} in {base_dir}")

    # 2) Chunked DuckDB join (split by paper_i range) to keep temp usage bounded.
    n_paper = get_n_rows_from_csrnpz(P_npz_path)
    chunk_size = 10_000
    out_parts_glob = os.path.join(base_dir, f"{stem}_outside_part_*.parquet")

    temp_dir = os.path.join("data", "tmp", "duckdb_temp")
    os.makedirs(temp_dir, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA temp_directory='{temp_dir}';")
        con.execute("PRAGMA enable_object_cache=false;")

        t_all = time.time()
        part_paths = []
        part_idx = 0
        for start in range(0, n_paper, chunk_size):
            end = min(start + chunk_size, n_paper)
            part_path = os.path.join(base_dir, f"{stem}_outside_part_{part_idx:05d}.parquet")
            part_paths.append(part_path)

            t1 = time.time()
            query = f"""
            COPY (
                SELECT DISTINCT a.csv AS src_csv, b.csv AS dst_csv
                FROM read_parquet('{A_parquet_path}') a
                JOIN read_parquet('{pathP}') p
                  ON a.paper = p.paper_i
                JOIN read_parquet('{A_parquet_path}') b
                  ON p.paper_j = b.paper
                WHERE p.paper_i >= {start} AND p.paper_i < {end}
            ) TO '{part_path}' (FORMAT PARQUET);
            """
            con.execute(query)
            print(f"[join] Step2 chunk {part_idx}: paper_i=[{start},{end}) time={time.time() - t1:.4f}s out={part_path}")
            part_idx += 1

        # 3) Merge all part Parquets into the final outside Parquet (dedup on (src_csv, dst_csv)).
        t1 = time.time()
        merge_query = f"""
        COPY (
            SELECT DISTINCT src_csv, dst_csv
            FROM read_parquet('{out_parts_glob}')
        ) TO '{path_outside_parquet}' (FORMAT PARQUET);
        """
        con.execute(merge_query)
        print(f"[join] Step2 merge parts: time={time.time() - t1:.4f}s out={path_outside_parquet}")
        print(f"[join] Step2 total time={time.time() - t_all:.4f}s chunks={part_idx} n_paper={n_paper} chunk_size={chunk_size}")
    finally:
        con.close()

    # 4) Cleanup chunk outputs and temp P edges to reduce disk usage.
    for p in part_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.remove(pathP)
    except OSError:
        pass

def concat_csv_csv_adj_by_addition(csv_csv_adj_within_model, csv_csv_adj_outside_model):
    M = (csv_csv_adj_within_model + csv_csv_adj_outside_model).astype(bool).tocsr() # we didn't care count
    M.setdiag(False)# remove self-loop if any
    return M

def concat_csv_csv_adj_by_join(
    within_npz_path: str,
    path_outside_parquet: str,
    path_final_parquet: str,
) -> None:
    """
    Step3: merge within‑model and outside‑model csv‑csv adjacencies on the edge level, fully in DuckDB/Parquet.

    This is logically equivalent to:
        boolean OR of the two matrices, followed by removing self-loops.

    Implementation:
    - Convert within‑model adjacency .npz to an edge list Parquet (src_csv, dst_csv).
    - Use DuckDB to concatenate with outside‑model edges Parquet.
    - Drop self-loops (src_csv == dst_csv).
    - Drop duplicate edges.
    - Write the merged edges as a Parquet file; no DataFrame is returned.
    """
    base_dir = os.path.dirname(path_final_parquet)
    os.makedirs(base_dir, exist_ok=True)
    path_within_parquet = os.path.join(base_dir, "csv_csv_within_tmp.parquet")

    # 1) within-model npz → edge list Parquet
    local_csrnpz_to_parquet(within_npz_path, path_within_parquet, row_col_names=("src_csv", "dst_csv"))

    # 2) DuckDB concat + dedup + remove self-loops → write final Parquet
    con = duckdb.connect()
    try:
        t1 = time.time()
        query = f"""
        COPY (
            SELECT DISTINCT src_csv, dst_csv
            FROM (
                SELECT src_csv, dst_csv FROM read_parquet('{path_within_parquet}')
                UNION ALL
                SELECT src_csv, dst_csv FROM read_parquet('{path_outside_parquet}')
            )
            WHERE src_csv <> dst_csv
        ) TO '{path_final_parquet}' (FORMAT PARQUET);
        """
        con.execute(query)
        print(f"[join] Step3 concat via DuckDB→Parquet: time={time.time() - t1:.4f}s, out={path_final_parquet}")
    finally:
        con.close()

############## defined path ####################
def get_npz_path(v2_suffix, suffix, root_dir):
    LEVEL_NPZ = {
        "direct": os.path.join(root_dir, 'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_direct_label{v2_suffix}{suffix}.npz"),
        "direct_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_direct_label_influential{v2_suffix}{suffix}.npz"),
        "direct_methodology_or_result": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_direct_label_methodology_or_result{v2_suffix}{suffix}.npz"),
        "direct_methodology_or_result_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_direct_label_methodology_or_result_influential{v2_suffix}{suffix}.npz"),
        "max_pr": os.path.join('data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_max_pr{v2_suffix}{suffix}.npz"),
        "max_pr_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_max_pr_influential{v2_suffix}{suffix}.npz"),
        "max_pr_methodology_or_result": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_max_pr_methodology_or_result{v2_suffix}{suffix}.npz"),
        "max_pr_methodology_or_result_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_max_pr_methodology_or_result_influential{v2_suffix}{suffix}.npz"),
        "union": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_union_direct_processed{v2_suffix}{suffix}.npz"),
        "model": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"scilake_gt_modellink_model_adj_processed{v2_suffix}{suffix}.npz"),
        "dataset": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"scilake_gt_modellink_dataset_adj_processed{v2_suffix}{suffix}.npz"),
    }

    # Mapping of level names to CSV list pickle filenames
    CANONICAL_CSVLIST = f"csv_list{v2_suffix}{suffix}.pkl"
    LEVEL_CSVLIST = {
        "direct": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"{CANONICAL_CSVLIST}"),
        "direct_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"{CANONICAL_CSVLIST}"),
        "direct_methodology_or_result": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"{CANONICAL_CSVLIST}"),
        "direct_methodology_or_result_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"{CANONICAL_CSVLIST}"),
        "max_pr": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"{CANONICAL_CSVLIST}"),
        "max_pr_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"{CANONICAL_CSVLIST}"),
        "max_pr_methodology_or_result": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_max_pr_methodology_or_result{v2_suffix}{suffix}.pkl"),
        "max_pr_methodology_or_result_influential": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_matrix_max_pr_methodology_or_result_influential{v2_suffix}{suffix}.pkl"),
        "union": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"csv_pair_union_direct_processed_csv_list{v2_suffix}{suffix}.pkl"),
        "model": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"scilake_gt_modellink_model_adj_csv_list{v2_suffix}{suffix}_processed.pkl"),
        "dataset": os.path.join(root_dir,'data', f'gt{v2_suffix}{suffix}', f"scilake_gt_modellink_dataset_adj_csv_list{v2_suffix}{suffix}_processed.pkl"),
    }
    return LEVEL_NPZ, LEVEL_CSVLIST


# ===== FACTORIES =========================================================== #
def load_score_matrix(rel_key: str):
    """Factory loader for paperId‑level relationship graphs."""
    path = FILES["combined"]
    print(f"Loading relationships (combined) from: {path}")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    if rel_key not in data:                           
        raise KeyError("Key '{rel_key}' not found. Available keys: {list(data.keys())[:10]} ...")  
    # Select the proper score matrix from the new keys
    score_matrix = data[rel_key]
    print(f"[DEBUG] Loaded score_matrix with shape: {score_matrix.shape}")
    return score_matrix

def load_paper_index():
    """Load paper_index from combined pickle."""
    with gzip.open(FILES["combined"], "rb") as f:
        data = pickle.load(f)
    paper_index = data["paper_index"]
    print(f"[DEBUG] Loaded paper_index with length: {len(paper_index)}")
    return paper_index

def load_titles_to_tables_with_modelid():
    """Load titles_to_tables_with_modelid (rows) via DuckDB. Dedup by (titles, csvs) in SQL; returns (rows, count_before_dedup)."""
    print(f"Loading table source from: step3_dedup (DuckDB)")
    valid_title_path = os.path.abspath(FILES["valid_title"])
    step3_path = os.path.abspath(FILES["step3_dedup"])
    step3_path_sql = step3_path.replace("\\", "/")
    valid_title_path_sql = valid_title_path.replace("\\", "/")

    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE VIEW step3 AS
            SELECT
                modelId,
                list_concat(
                    coalesce(hugging_table_list_dedup, []),
                    coalesce(github_table_list_dedup, []),
                    coalesce(html_table_list_mapped_dedup, []),
                    coalesce(llm_table_list_mapped_dedup, [])
                ) AS all_table_list_dedup
            FROM read_parquet(?)
            """.replace("read_parquet(?)", f"read_parquet('{step3_path_sql}')")
        )
        con.execute(
            """
            CREATE VIEW titles AS
            SELECT modelId, all_title_list_valid
            FROM read_parquet(?)
            """.replace("read_parquet(?)", f"read_parquet('{valid_title_path_sql}')")
        )
        con.execute(
            """
            CREATE VIEW joined AS
            SELECT s.all_table_list_dedup, t.all_title_list_valid
            FROM step3 AS s
            JOIN titles AS t USING (modelId)
            WHERE all_title_list_valid IS NOT NULL
              AND array_length(all_title_list_valid, 1) > 0
              AND all_table_list_dedup IS NOT NULL
              AND array_length(all_table_list_dedup, 1) > 0
            """
        )
        (count_before_dedup,) = con.execute("SELECT count(*) FROM joined").fetchone()
        df = con.execute(
            """
            SELECT DISTINCT all_table_list_dedup, all_title_list_valid
            FROM joined
            """
        ).fetchdf()
    finally:
        con.close()

    # Normalize CSV paths to basenames once here so downstream code only sees basenames.
    rows = []
    for titles, csvs in zip(df["all_title_list_valid"], df["all_table_list_dedup"]):
        titles = list(titles)
        csvs = [os.path.basename(c) for c in csvs]
        rows.append((titles, csvs))
    count_after_dedup = len(rows)
    print(f"[DEBUG] Dedup (by titles+csvs): before {count_before_dedup}, after {count_after_dedup}")
    print(f"[DEBUG] Loaded titles_to_tables_with_modelid with length: {len(rows)}")
    return rows

def build_paper_matrix(rel_key: str, overlap_rate_threshold: float):
    score_matrix = load_score_matrix(rel_key)
    if rel_key.startswith("direct_label"):
        paper_adj = (score_matrix >= 1.0).astype(np.bool_)
    else:
        paper_adj = (score_matrix > overlap_rate_threshold).astype(np.bool_)
    paper_adj.setdiag(True)
    print(f"[DEBUG] Built paper_adj matrix with shape: {paper_adj.shape}, nnz: {paper_adj.nnz}")
    return paper_adj.tocsr().astype(bool).tocsr()

def build_A_matrix_paper2csv(titles_to_tables_with_modelid, csv2idx):
    paper_index = load_paper_index() 
    title2cid = build_titles2cid()
    # 2) inter-row: build A (paper→CSV) and compute C = Aᵀ·P·A
    corpus2pidx = {cid:i for i,cid in enumerate(paper_index)}
    
    # Build A: per row get paper indices and csv indices; Cartesian product via numpy (repeat/tile + concatenate)
    all_row_ps, all_row_cs = [], []
    for titles, csvs in titles_to_tables_with_modelid:
        row_ps = []
        for t in titles: # for each paper
            cid = title2cid.get(t) # get corpusid
            if cid is None:
                continue
            p = corpus2pidx.get(cid) # index
            if p is None:
                continue
            row_ps.append(p) # paper index
        all_row_ps.append(row_ps)
        all_row_cs.append([csv2idx[c] for c in csvs]) # paper related csvs' index
    parts = [(np.repeat(ps, len(cs)), np.tile(cs, len(ps))) for ps, cs in zip(all_row_ps, all_row_cs) if len(ps) > 0 and len(cs) > 0]
    row_i = np.concatenate([p[0] for p in parts]) if parts else np.array([], dtype=np.int64)
    col_i = np.concatenate([p[1] for p in parts]) if parts else np.array([], dtype=np.int64)
    A = coo_matrix((np.ones(len(row_i), bool), (row_i, col_i)), shape=(len(paper_index), len(csv2idx))).astype(bool).tocsr()
    # Save A as sparse matrix and also as a global edge-list Parquet (paper, csv).
    A_npz_path = f"data/gt{v2_suffix}{suffix}/A_matrix{v2_suffix}{suffix}.npz"
    A_parquet_path = f"data/gt{v2_suffix}{suffix}/A_edges{v2_suffix}{suffix}.parquet"
    save_npz(A_npz_path, A, compressed=True)
    local_csrnpz_to_parquet(A_npz_path, A_parquet_path, row_col_names=("paper", "csv"))
    return A

def build_B_matrix_csv2csv_within_model(titles_to_tables_with_modelid, csv2idx):
    # 1) intra-row: construct B for same-model CSV pairs
    row_b, col_b = [], []
    for _, cs in titles_to_tables_with_modelid:
        inds = sorted(set(csv2idx[c] for c in cs))
        for i, j in combinations(inds, 2):
            row_b.extend([i, j])
            col_b.extend([j, i])
    csv_csv_adj_within_model = coo_matrix((np.ones(len(row_b), int), (row_b, col_b)), shape=(len(csv2idx), len(csv2idx))).tocsr() # B
    print(f"Step2: [Intra-row] Adjacency shape: {csv_csv_adj_within_model.shape}: ", csv_csv_adj_within_model.nnz)
    save_npz(f"data/gt{v2_suffix}{suffix}/B_matrix{v2_suffix}{suffix}.npz", csv_csv_adj_within_model, compressed=True)

def build_titles2cid():
    # Use titles2ids as canonical source of (corpusId, query_title)
    title_df = pd.read_parquet(FILES["titles2ids"],columns=["corpusId", "query_title"])
    # Keep only rows where both corpusId and query_title are non-null, and normalize corpusId
    title_df = title_df[title_df["corpusId"].notna() & title_df["query_title"].notna()].copy()
    title_df["corpusId"] = title_df["corpusId"].astype(str).str.strip().str.replace(".0", "", regex=False)
    print(f"[DEBUG] Loaded title_df with shape: {title_df.shape}")
    cid2titles = defaultdict(list)
    for cid, title in zip(title_df["corpusId"], title_df["query_title"]):
        cid2titles[cid].append(title) # corpusid -> title
    title2cid   = {t:cid for cid,titles in cid2titles.items() for t in titles} # title -> corpusid
    print(f"[DEBUG] Built cid2titles with {len(cid2titles)} unique corpusids")
    print(f"[DEBUG] Sample of cid2titles (first 3 items): {dict(list(cid2titles.items())[:3])}")
    return title2cid

def build_element_matrix_at_one_time():
    # modelId-paperList-csvList, Our aim is to use paper-paper matrix to build {csv:[csv1, csv2]} related json
    """High‑level orchestration for building GT tables."""
    t1 = time.time()
    # ---------- modelId → csv mapping (DuckDB SQL → titles_to_tables_with_modelid only) ----------
    titles_to_tables_with_modelid = load_titles_to_tables_with_modelid()
    print(f"Step0: Loaded titles_to_tables_with_modelid with length: {len(titles_to_tables_with_modelid)} in {time.time() - t1:.2f} seconds")
    t1 = time.time()
    # build global CSV list & index
    flat = [c for _, cs in titles_to_tables_with_modelid for c in cs]
    all_csvs = list(dict.fromkeys(flat))# already basename in titles_to_tables_with_modelid
    #all_csvs = [os.path.basename(csv) for csv in all_csvs]
    #csv_list_path = f"data/gt{v2_suffix}{suffix}/csv_list{v2_suffix}{suffix}.pkl"
    LEVEL_NPZ, LEVEL_CSVLIST = get_npz_path(v2_suffix, suffix, "/Users/doradong/Repo/ModelTables")
    csv_list_path = LEVEL_CSVLIST["direct"]
    with open(csv_list_path, "wb") as f:
        pickle.dump(all_csvs, f)
    print(f"✅ CSV list saved (order matches matrix rows/cols) to {csv_list_path}")
    print('time to build csv2idx and save csv list: ', time.time() - t1)
    csv2idx  = {c: i for i, c in enumerate(all_csvs)}
    print(f"[DEBUG] Built csv2idx with length: {len(csv2idx)}")
    t1 = time.time()
    build_B_matrix_csv2csv_within_model(titles_to_tables_with_modelid, csv2idx)
    print(f"time to build B matrix: ", time.time() - t1)
    t1 = time.time()
    build_A_matrix_paper2csv(titles_to_tables_with_modelid, csv2idx)
    print(f"time to build A matrix: ", time.time() - t1)

def build_ground_truth(rel_key, overlap_rate_threshold, suffix, v2_suffix):
    t1 = time.time()
    paper_paper_adj = build_paper_matrix(rel_key, overlap_rate_threshold)
    print(f"time to build paper-paper adjacency matrix: ", time.time() - t1)
    assert paper_paper_adj.data.dtype == np.bool_
    print(f"Step1: [Paper-level] Adjacency shape: {paper_paper_adj.shape}: ", paper_paper_adj.nnz)
    print(f"[DEBUG] Paper-paper adjacency matrix statistics:")
    print(f"  - Non-zero elements: {paper_paper_adj.nnz}")

    within_npz_path = f"data/gt{v2_suffix}{suffix}/B_matrix{v2_suffix}{suffix}.npz"
    A_npz_path = f"data/gt{v2_suffix}{suffix}/A_matrix{v2_suffix}{suffix}.npz"
    A_parquet_path = f"data/gt{v2_suffix}{suffix}/A_edges{v2_suffix}{suffix}.parquet"
    P_npz_path = f"data/gt{v2_suffix}{suffix}/P_matrix_{rel_key}{v2_suffix}{suffix}.npz"
    n_csv = get_n_rows_from_csrnpz(within_npz_path)
    # Persist paper-paper adjacency once so Step2 can work purely from disk.
    save_npz(P_npz_path, paper_paper_adj, compressed=True)

    # Derive Parquet paths for outside edges and final merged edges.
    gt_dir = f"data/gt{v2_suffix}{suffix}"
    os.makedirs(gt_dir, exist_ok=True)
    outside_parquet = os.path.join(gt_dir, f"csv_csv_outside_{rel_key}{v2_suffix}{suffix}.parquet")
    final_parquet = os.path.join(gt_dir, f"csv_csv_final_{rel_key}{v2_suffix}{suffix}.parquet")

    '''
    # by multiplication: old implementation, deprecated for memory 
    usage
    csv_csv_adj_within_model = load_npz(f"data/gt{v2_suffix}
    {suffix}/B_matrix{v2_suffix}{suffix}.npz")
    paper_csv_adj = load_npz(f"data/gt{v2_suffix}{suffix}/A_matrix
    {v2_suffix}{suffix}.npz")
    csv_csv_adj_within_model = csv_csv_adj_within_model.astype(np.
    int8).tocsr()
    paper_csv_adj = load_npz(A_npz_path)

    outside_edges = get_final_csv_csv_adj_by_multiplication
    (paper_csv_adj, paper_paper_adj)
    M = concat_csv_csv_adj_by_addition(csv_csv_adj_within_model, 
    outside_edges)
    matrix_path = f"data/gt{v2_suffix}{suffix}/csv_pair_matrix_
    {rel_key}{v2_suffix}{suffix}.npz"
    save_npz(matrix_path, M, compressed=True)
    print(f"✅ Sparse matrix saved to {matrix_path}")
    print(f"[DEBUG] Final M matrix shape: {M.shape}, nnz: {M.nnz}")
    print(f"time to build final GT matrix: ", time.time() - t1)
    '''

    # Step 2: compute outside-model csv‑csv edges via DuckDB join (Aᵀ·P·A semantics) → Parquet only.
    get_final_csv_csv_adj_by_join(A_parquet_path, P_npz_path, outside_parquet)
    # Step 3: merge within-model adjacency with outside edges, fully in DuckDB/Parquet.
    concat_csv_csv_adj_by_join(within_npz_path, outside_parquet, final_parquet)

    '''# Materialize final boolean csr matrix from the merged Parquet edges into .npz.
    matrix_path = f"data/gt{v2_suffix}{suffix}/csv_pair_matrix_{rel_key}{v2_suffix}{suffix}.npz"
    local_parquet_to_csrnpz(
        parquet_path=final_parquet,
        npz_path=matrix_path,
        shape=(n_csv, n_csv),
        row_col_names=("src_csv", "dst_csv"),
    )'''

    print(f"time to build final GT matrix: ", time.time() - t1)
    print(f"✅ Sparse matrix saved to {matrix_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument("--tag", dest="tag", default=None, help="Tag suffix for versioning (e.g., 251117). Enables versioning mode for input files.")
    parser.add_argument("--v2_mode", dest="v2_mode", action="store_true", help="Use v2 mode.")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    FILES = {
        "combined": f"data/processed/modelcard_citation_all_matrices{suffix}.pkl.gz",
        "titles2ids": f"data/processed/s2orc_titles2ids{suffix}.parquet",
        "title_list": f"data/processed/modelcard_all_title_list{suffix}.parquet",
        "step3_dedup": f"data/processed/modelcard_step3_dedup{v2_suffix}{suffix}.parquet",
        "valid_title": f"data/processed/all_title_list_valid{v2_suffix}{suffix}.parquet"
    }

    # Always print paths in use so you can verify in the log
    print("📁 Paths in use:")
    print(f"   tag: {args.tag!r}")
    print(f"   v2_mode: {args.v2_mode!r}")
    for key, path in FILES.items():
        print(f"   {key}: {path}")
    print(f"   GT output directory: data/gt{v2_suffix}{suffix}/")
    os.makedirs(f"data/gt{v2_suffix}{suffix}/", exist_ok=True)

    REL_KEY_LIST = [
        'direct_label', 
        'direct_label_influential', 
        'direct_label_methodology_or_result', 
        'direct_label_methodology_or_result_influential', 
        'max_pr', 
        'max_pr_influential', 
        'max_pr_methodology_or_result', 
        'max_pr_methodology_or_result_influential'
    ]
    OVERLAP_RATE_THRESHOLD = 0.0
    build_element_matrix_at_one_time()
    for rel_key in REL_KEY_LIST:
        build_ground_truth(rel_key=rel_key, overlap_rate_threshold=OVERLAP_RATE_THRESHOLD, suffix=suffix, v2_suffix=v2_suffix)
