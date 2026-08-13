"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2026-08-12
Description: Build BibTeX-anchored SciLake union benchmark tables as CSR matrices.

Only successfully parsed primary model-card BibTeX titles that resolve to a
Semantic Scholar paper are used as paper anchors. The table-validity list still
defines which linked tables may become GT nodes.

Usage:
python -m src.data_gt.step3_gt --tag 251117 --v2_mode
"""
from multiprocessing import set_start_method
set_start_method("fork", force=True)

import os, gzip, pickle, re, time
import shutil
import tempfile
import zipfile
import duckdb
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.sparse import coo_matrix, save_npz, load_npz


GENERIC_NAV_TITLES = {
    "quick start",
    "table of contents",
    "tables of contents",
}
PAPER_CHUNK_SIZE = 2500


def is_generic_navigation_title(title: str) -> bool:
    """Return whether a resolved paper title is actually a README navigation label."""
    normalized = re.sub(r"\s+", " ", str(title).lower()).strip()
    normalized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", normalized)
    return normalized in GENERIC_NAV_TITLES


def normalize_title_key(title: str) -> str:
    """Match title variants using the existing intra-row title-dedup convention."""
    if not isinstance(title, str):
        return ""
    return title.replace("{", "").replace("}", "").replace("-", "").replace(" ", "").replace(".", "").lower().strip()


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


def parquet_to_csrnpz(
    parquet_path: str,
    npz_path: str,
    shape: tuple[int, int],
    row_col_names=("row_idx", "col_idx"),
    batch_size: int = 1_000_000,
) -> None:
    """Write the final GT matrix without loading all edges into memory."""
    row_name, col_name = row_col_names
    n_rows, n_cols = shape
    parquet_sql = os.path.abspath(parquet_path).replace("'", "''")
    output_dir = os.path.dirname(npz_path)
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="csr_export_", dir=output_dir)
    try:
        counts = np.zeros(n_rows, dtype=np.int64)
        con = duckdb.connect()
        try:
            reader = con.execute(
                f"SELECT {row_name}, {col_name} FROM read_parquet('{parquet_sql}')"
            ).to_arrow_reader(batch_size)
            total = 0
            for batch in reader:
                rows = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                counts += np.bincount(rows, minlength=n_rows)
                total += len(rows)
        finally:
            con.close()

        indptr = np.empty(n_rows + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        if total > np.iinfo(np.int32).max:
            raise ValueError(f"CSR export has {total} edges, exceeding int32 index capacity")

        indices_path = os.path.join(temp_dir, "indices.npy")
        data_path = os.path.join(temp_dir, "data.npy")
        indptr_path = os.path.join(temp_dir, "indptr.npy")
        indices = np.lib.format.open_memmap(indices_path, mode="w+", dtype=np.int32, shape=(total,))
        data = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.bool_, shape=(total,))
        positions = indptr[:-1].copy()
        con = duckdb.connect()
        try:
            reader = con.execute(
                f"SELECT {row_name}, {col_name} FROM read_parquet('{parquet_sql}')"
            ).to_arrow_reader(batch_size)
            for batch in reader:
                rows = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                cols = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
                order = np.argsort(rows, kind="stable")
                rows, cols = rows[order], cols[order]
                starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
                stops = np.r_[starts[1:], len(rows)]
                for start, stop in zip(starts, stops):
                    row = rows[start]
                    write_start = positions[row]
                    write_stop = write_start + stop - start
                    indices[write_start:write_stop] = cols[start:stop]
                    data[write_start:write_stop] = True
                    positions[row] = write_stop
        finally:
            con.close()

        np.save(indptr_path, indptr)
        del indices, data
        format_path = os.path.join(temp_dir, "format.npy")
        shape_path = os.path.join(temp_dir, "shape.npy")
        np.save(format_path, np.asarray("csr"))
        np.save(shape_path, np.asarray((n_rows, n_cols), dtype=np.int64))
        with zipfile.ZipFile(npz_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in ("format.npy", "shape.npy", "data.npy", "indices.npy", "indptr.npy"):
                archive.write(os.path.join(temp_dir, name), arcname=name)
        print(f"[stream] parquet→csrnpz saved: npz={npz_path}, shape={shape}, edges={total}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    chunk_size = PAPER_CHUNK_SIZE
    if chunk_size < 1:
        raise ValueError(f"paper chunk size must be positive, got {chunk_size}")
    out_parts_glob = os.path.join(base_dir, f"{stem}_outside_part_*.parquet")
    dedup_parts_glob = os.path.join(base_dir, f"{stem}_dedup_part_*.parquet")

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
            ) TO '{part_path}' (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE);
            """
            con.execute(query)
            print(f"[join] Step2 chunk {part_idx}: paper_i=[{start},{end}) time={time.time() - t1:.4f}s out={part_path}")
            part_idx += 1

        # 3) Deduplicate bounded src_csv ranges independently.
        for path in os.listdir(base_dir):
            if path.startswith(f"{stem}_dedup_part_") and path.endswith(".parquet"):
                os.remove(os.path.join(base_dir, path))

        (max_src_csv,) = con.execute(
            f"SELECT max(src_csv) FROM read_parquet('{out_parts_glob}')"
        ).fetchone()
        src_bucket_size = 5_000
        src_starts = list(range(0, int(max_src_csv) + 1, src_bucket_size))
        dedup_paths = []
        for bucket_idx, src_start in enumerate(src_starts):
            src_end = src_start + src_bucket_size
            dedup_path = os.path.join(base_dir, f"{stem}_dedup_part_{bucket_idx:05d}.parquet")
            dedup_paths.append(dedup_path)
            t_bucket = time.time()
            con.execute(
                f"""
                COPY (
                    SELECT DISTINCT src_csv, dst_csv
                    FROM read_parquet('{out_parts_glob}')
                    WHERE src_csv >= {src_start} AND src_csv < {src_end}
                ) TO '{dedup_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """
            )
            print(
                f"[join] Step2 dedup src bucket {bucket_idx + 1}/{len(src_starts)} "
                f"[{src_start},{src_end}): time={time.time() - t_bucket:.4f}s"
            )

        # 5) Concatenate already-disjoint source partitions without another DISTINCT.
        t1 = time.time()
        con.execute(
            f"""
            COPY (
                SELECT src_csv, dst_csv
                FROM read_parquet('{dedup_parts_glob}')
            ) TO '{path_outside_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE);
            """
        )
        print(f"[join] Step2 merge dedup partitions: time={time.time() - t1:.4f}s out={path_outside_parquet}")
        print(f"[join] Step2 total time={time.time() - t_all:.4f}s chunks={part_idx} n_paper={n_paper} chunk_size={chunk_size}")
    finally:
        con.close()

    # 5) Cleanup chunk outputs and temporary dedup files.
    for p in part_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    for p in dedup_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.remove(pathP)
    except OSError:
        pass

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
    final_stem = os.path.splitext(os.path.basename(path_final_parquet))[0]
    final_parts_glob = os.path.join(base_dir, f"{final_stem}_part_*.parquet")

    # 1) within-model npz → edge list Parquet
    local_csrnpz_to_parquet(within_npz_path, path_within_parquet, row_col_names=("src_csv", "dst_csv"))

    # 2) Deduplicate bounded source ranges across within/outside edges.
    con = duckdb.connect()
    try:
        for path in os.listdir(base_dir):
            if path.startswith(f"{final_stem}_part_") and path.endswith(".parquet"):
                os.remove(os.path.join(base_dir, path))

        (max_src_csv,) = con.execute(
            f"""
            SELECT max(src_csv)
            FROM (
                SELECT src_csv FROM read_parquet('{path_within_parquet}')
                UNION ALL
                SELECT src_csv FROM read_parquet('{path_outside_parquet}')
            )
            """
        ).fetchone()
        src_bucket_size = 5_000
        src_starts = list(range(0, int(max_src_csv) + 1, src_bucket_size))
        final_parts = []
        for bucket_idx, src_start in enumerate(src_starts):
            src_end = src_start + src_bucket_size
            part_path = os.path.join(base_dir, f"{final_stem}_part_{bucket_idx:05d}.parquet")
            final_parts.append(part_path)
            t_bucket = time.time()
            con.execute(
                f"""
                COPY (
                    SELECT DISTINCT src_csv, dst_csv
                    FROM (
                        SELECT src_csv, dst_csv FROM read_parquet('{path_within_parquet}')
                        UNION ALL
                        SELECT src_csv, dst_csv FROM read_parquet('{path_outside_parquet}')
                    )
                    WHERE src_csv >= {src_start}
                      AND src_csv < {src_end}
                      AND src_csv <> dst_csv
                ) TO '{part_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """
            )
            print(
                f"[join] Step3 dedup src bucket {bucket_idx + 1}/{len(src_starts)} "
                f"[{src_start},{src_end}): time={time.time() - t_bucket:.4f}s"
            )

        t1 = time.time()
        con.execute(
            f"""
            COPY (
                SELECT src_csv, dst_csv
                FROM read_parquet('{final_parts_glob}')
            ) TO '{path_final_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE);
            """
        )
        print(f"[join] Step3 merge source partitions: time={time.time() - t1:.4f}s, out={path_final_parquet}")
    finally:
        con.close()

    for path in final_parts:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.remove(path_within_parquet)
    except OSError:
        pass


def load_valid_table_names(valid_tables_path: str) -> set[str]:
    """Load the QC-valid table list as CSV basenames."""
    if not os.path.exists(valid_tables_path):
        raise FileNotFoundError(
            f"Valid-table list not found: {valid_tables_path}. "
            "Run src.data_analysis.qc_stats first."
        )

    with open(valid_tables_path, encoding="utf-8") as f:
        valid_tables = {
            os.path.basename(line.strip())
            for line in f
            if line.strip()
        }
    print(f"[valid-tables] Loaded {len(valid_tables)} tables from {valid_tables_path}")
    return valid_tables


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

def load_titles_to_tables_with_modelid(valid_table_names=None):
    """Load BibTeX-resolved paper titles and their QC-valid model-card tables."""
    print(f"Loading table source from: step3_dedup (DuckDB)")
    valid_title_path = os.path.abspath(FILES["valid_title"])
    title_list_path = os.path.abspath(FILES["title_list"])
    step3_path = os.path.abspath(FILES["step3_dedup"])
    step3_path_sql = step3_path.replace("\\", "/")
    valid_title_path_sql = valid_title_path.replace("\\", "/")
    title_list_path_sql = title_list_path.replace("\\", "/")

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
            CREATE VIEW bibtex AS
            SELECT
                modelId,
                list_transform(parsed_bibtex_tuple_list, entry -> entry.title) AS parsed_bibtex_titles
            FROM read_parquet(?)
            """.replace("read_parquet(?)", f"read_parquet('{title_list_path_sql}')")
        )
        con.execute(
            """
            CREATE VIEW joined AS
            SELECT s.all_table_list_dedup, t.all_title_list_valid, b.parsed_bibtex_titles
            FROM step3 AS s
            JOIN titles AS t USING (modelId)
            JOIN bibtex AS b USING (modelId)
            WHERE all_title_list_valid IS NOT NULL
              AND array_length(all_title_list_valid, 1) > 0
              AND parsed_bibtex_titles IS NOT NULL
              AND array_length(parsed_bibtex_titles, 1) > 0
              AND all_table_list_dedup IS NOT NULL
              AND array_length(all_table_list_dedup, 1) > 0
            """
        )
        (count_before_dedup,) = con.execute("SELECT count(*) FROM joined").fetchone()
        df = con.execute(
            """
            SELECT DISTINCT all_table_list_dedup, all_title_list_valid, parsed_bibtex_titles
            FROM joined
            """
        ).fetchdf()
    finally:
        con.close()

    # Normalize CSV paths to basenames once here so downstream code only sees basenames.
    rows = []
    filtered_table_refs = 0
    filtered_rows = 0
    filtered_title_refs = 0
    unmatched_bibtex_title_refs = 0
    for titles, csvs, bibtex_titles in zip(
        df["all_title_list_valid"],
        df["all_table_list_dedup"],
        df["parsed_bibtex_titles"],
    ):
        titles = list(titles)
        bibtex_title_keys = {normalize_title_key(title) for title in bibtex_titles}
        kept_titles = [
            title
            for title in titles
            if normalize_title_key(title) in bibtex_title_keys and not is_generic_navigation_title(title)
        ]
        filtered_title_refs += len(titles) - len(kept_titles)
        titles = kept_titles
        unmatched_bibtex_title_refs += len(bibtex_titles) - len(bibtex_title_keys & {normalize_title_key(title) for title in titles})
        if not titles:
            continue
        csvs = [os.path.basename(c) for c in csvs]
        if valid_table_names is not None:
            kept_csvs = [csv_name for csv_name in csvs if csv_name in valid_table_names]
            filtered_table_refs += len(csvs) - len(kept_csvs)
            csvs = kept_csvs
        if not csvs:
            filtered_rows += 1
            continue
        rows.append((titles, csvs))
    count_after_dedup = len(rows)
    print(f"[DEBUG] Dedup (by titles+csvs): before {count_before_dedup}, after {count_after_dedup}")
    if valid_table_names is not None:
        print(
            f"[valid-tables] Removed {filtered_table_refs} invalid table references "
            f"and {filtered_rows} rows with no valid tables"
        )
    print(f"[bibtex-titles] Kept {sum(len(titles) for titles, _ in rows)} resolved primary-BibTeX title references")
    print(f"[bibtex-titles] Removed {filtered_title_refs} non-BibTeX, unresolved, or navigation title references")
    print(f"[bibtex-titles] Parsed BibTeX titles without a resolved valid-title match: {unmatched_bibtex_title_refs}")
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

def build_element_matrix_at_one_time(valid_table_names):
    # modelId-paperList-csvList, Our aim is to use paper-paper matrix to build {csv:[csv1, csv2]} related json
    """High‑level orchestration for building GT tables."""
    t1 = time.time()
    # ---------- modelId → csv mapping (DuckDB SQL → titles_to_tables_with_modelid only) ----------
    titles_to_tables_with_modelid = load_titles_to_tables_with_modelid(valid_table_names)
    print(f"Step0: Loaded titles_to_tables_with_modelid with length: {len(titles_to_tables_with_modelid)} in {time.time() - t1:.2f} seconds")
    t1 = time.time()
    # build global CSV list & index
    flat = [c for _, cs in titles_to_tables_with_modelid for c in cs]
    all_csvs = list(dict.fromkeys(flat))# already basename in titles_to_tables_with_modelid
    print(f"[valid-tables] GT tables after valid-table filtering: {len(all_csvs)}")
    csv_list_path = f"data/gt{v2_suffix}{suffix}/csv_list{v2_suffix}{suffix}.pkl"
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
    return all_csvs

def build_ground_truth(
    rel_key,
    overlap_rate_threshold,
    suffix,
    v2_suffix,
    csv_list,
):
    t1 = time.time()
    gt_dir = f"data/gt{v2_suffix}{suffix}"
    os.makedirs(gt_dir, exist_ok=True)
    matrix_path = os.path.join(gt_dir, f"csv_pair_matrix_{rel_key}{v2_suffix}{suffix}.npz")
    if os.path.exists(matrix_path):
        print(f"[skip] Final GT matrix already exists: {matrix_path}")
        return
    outside_parquet = os.path.join(gt_dir, f"csv_csv_outside_{rel_key}{v2_suffix}{suffix}.parquet")
    final_parquet = os.path.join(gt_dir, f"csv_csv_final_{rel_key}{v2_suffix}{suffix}.parquet")
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

    get_final_csv_csv_adj_by_join(A_parquet_path, P_npz_path, outside_parquet)
    concat_csv_csv_adj_by_join(within_npz_path, outside_parquet, final_parquet)
    parquet_to_csrnpz(final_parquet, matrix_path, (n_csv, n_csv), ("src_csv", "dst_csv"))
    for path in (P_npz_path, outside_parquet, final_parquet): os.remove(path)

    print(f"time to build final GT matrix: ", time.time() - t1)
    print(f"✅ GT matrix saved to {matrix_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument("--tag", dest="tag", default=None, help="Tag suffix for versioning (e.g., 251117). Enables versioning mode for input files.")
    parser.add_argument("--v2_mode", dest="v2_mode", action="store_true", help="Use v2 mode.")
    parser.add_argument(
        "--valid-tables",
        default=None,
        help="QC-valid table TXT. Defaults to data/analysis/all_valid_title_valid[_v2][_tag].txt.",
    )
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    FILES = {
        "combined": f"data/processed/modelcard_citation_all_matrices{suffix}.pkl.gz",
        "titles2ids": f"data/processed/s2orc_titles2ids{suffix}.parquet",
        "step3_dedup": f"data/processed/modelcard_step3_dedup{v2_suffix}{suffix}.parquet",
        "title_list": f"data/processed/modelcard_all_title_list{suffix}.parquet",
        "valid_title": f"data/processed/all_title_list_valid{v2_suffix}{suffix}.parquet",
        "valid_tables": args.valid_tables
        or f"data/analysis/all_valid_title_valid{v2_suffix}{suffix}.txt",
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
    ]
    if all(os.path.exists(f"data/gt{v2_suffix}{suffix}/csv_pair_matrix_{rel_key}{v2_suffix}{suffix}.npz") for rel_key in REL_KEY_LIST):
        raise SystemExit("[skip] All final GT matrices already exist.")
    OVERLAP_RATE_THRESHOLD = 0.0
    valid_table_names = load_valid_table_names(FILES["valid_tables"])
    csv_list = build_element_matrix_at_one_time(valid_table_names)
    for rel_key in REL_KEY_LIST:
        build_ground_truth(
            rel_key=rel_key,
            overlap_rate_threshold=OVERLAP_RATE_THRESHOLD,
            suffix=suffix,
            v2_suffix=v2_suffix,
            csv_list=csv_list,
        )

    # A/B are shared construction intermediates and are unnecessary after all requested relations finish.
    for intermediate_path in (
        f"data/gt{v2_suffix}{suffix}/A_matrix{v2_suffix}{suffix}.npz",
        f"data/gt{v2_suffix}{suffix}/A_edges{v2_suffix}{suffix}.parquet",
        f"data/gt{v2_suffix}{suffix}/B_matrix{v2_suffix}{suffix}.npz",
    ):
        try:
            os.remove(intermediate_path)
        except OSError:
            pass
