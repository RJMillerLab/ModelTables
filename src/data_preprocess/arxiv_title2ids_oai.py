# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-03-28
Last Modified: 2026-03-10
Description: This script is used to get the arXiv ID for the title extracted from the PDF file.

arXiv API rate limit (Terms of Use, https://info.arxiv.org/help/api/tou.html):
  - At most one request every 3 seconds; single connection at a time.
  - Exceeding this leads to 429 and possible blocking. We enforce 3s delay between each query.

To resume oai querying, it only accept resume token
"""
import os, re
import json
import time
import argparse
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import duckdb
from src.data_preprocess.step2_arxiv_github_title import extract_arxiv_id
from src.utils import load_config

FINAL_QUERY_STATUSES = {"found", "not_found"}  # treated as "successfully validated"

# OAI-PMH constants (mirrored from build_arxiv_oai_index.py)
OAI_BASE = "https://oaipmh.arxiv.org/oai"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
OAI_DELAY_SEC = 25  # polite delay between OAI requests (harvester guideline)

def normalize_title(title):
    """
    Normalize the title by converting to lower-case and reducing whitespace.
    """
    return " ".join(title.lower().split())

def preprocess_title(title):
    title = re.sub(r"[-:_*@&'\"]+", " ", title)
    return " ".join(title.split())

def load_json_cache(file_path):
    """
    Load a JSON file (expected format: {key: value}) in UTF-8.
    """
    if not os.path.isfile(file_path):
        print(f"[WARN] JSON cache file not found: {file_path}")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] Loaded JSON cache from {file_path} with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"[ERROR] Could not load JSON cache: {e}")
        return {}

def save_json_cache(data, file_path):
    """
    Save a dict to a JSON file, using UTF-8 encoding.
    """
    try:
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        #print(f"[INFO] Saved JSON cache to {file_path} with {len(data)} entries.")
    except Exception as e:
        print(f"[ERROR] Could not save JSON cache: {e}")

######## OAI-PMH helpers (simplified from build_arxiv_oai_index.py) ########
def extract_arxiv_id_from_identifier(identifier_text):
    """
    Extract bare arxiv_id from OAI identifier, e.g.:
      - 'oai:arXiv.org:2101.12345'
      - 'http://arxiv.org/abs/2101.12345'
    """
    if not identifier_text:
        return None
    s = identifier_text.strip()
    if "arXiv.org:" in s:
        return s.split("arXiv.org:")[-1].strip()
    if "arxiv.org/abs/" in s:
        return s.split("arxiv.org/abs/")[-1].split("/")[0].strip()
    return None


def harvest_oai(metadata_prefix="oai_dc", set_spec=None, limit=None, start_token=None):
    """
    Yield (norm_title, arxiv_id) from OAI-PMH ListRecords.
    Uses resumptionToken for paging.
    limit: stop after this many records (for quick testing).
    """
    params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
    if set_spec:
        params["set"] = set_spec
    # If start_token is provided, resume from that token; otherwise start from initial params.
    if start_token:
        token = start_token.strip()
        next_url = f"{OAI_BASE}?verb=ListRecords&resumptionToken={requests.utils.quote(token)}"
        next_params = None
        print(f"[OAI] Resuming from provided resumptionToken (truncated): {token[:80]}...")
    else:
        next_params = params
        next_url = None  # first request uses params; later use resumptionToken URL

    while True:
        try:
            if next_url is not None:
                resp = requests.get(next_url, timeout=120)
            else:
                resp = requests.get(OAI_BASE, params=next_params, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] OAI request failed: {e}")
            return
        root = ET.fromstring(resp.content)

        for rec in root.findall(f".//{OAI_NS}record"):
            header = rec.find(f"{OAI_NS}header")
            metadata = rec.find(f"{OAI_NS}metadata")
            if header is None:
                continue
            ident_el = header.find(f"{OAI_NS}identifier")
            arxiv_id = extract_arxiv_id_from_identifier(ident_el.text if ident_el is not None else None)
            if not arxiv_id:
                continue
            title = None
            if metadata is not None:
                # oai_dc: <dc:title>...</dc:title>
                t_el = metadata.find(f".//{DC_NS}title")
                if t_el is not None and t_el.text:
                    title = t_el.text.strip()
            if not title:
                continue
            norm = normalize_title(title)
            if norm:
                yield (norm, arxiv_id)
                if limit is not None:
                    limit -= 1
                    if limit <= 0:
                        return

        # Next page
        resumption = root.find(f".//{OAI_NS}resumptionToken")
        if resumption is None or not (resumption.text or "").strip():
            break
        token = (resumption.text or "").strip()
        print(f"[OAI] resumptionToken (truncated): {token[:80]}...")
        next_url = f"{OAI_BASE}?verb=ListRecords&resumptionToken={requests.utils.quote(token)}"
        next_params = None
        time.sleep(OAI_DELAY_SEC)


def build_or_update_oai_index(oai_index_path: str, oai_set: str | None = None, oai_limit: int | None = None, oai_start_token: str | None = None) -> None:
    """
    Build or update the local OAI index parquet at `oai_index_path`.

    Parameters
    ----------
    oai_index_path : str
        Target Parquet path for the index (norm_title, arxiv_id).
    oai_set, oai_limit, oai_start_token :
        Parameters forwarded to harvest_oai().

    This function is intentionally side-effect limited:
      - It only reads/writes the OAI index parquet.
      - It does NOT touch the main cache parquet or any other state.
      - It does NOT return anything; callers read the parquet separately if needed.
    """
    existing_rows = []
    if os.path.isfile(oai_index_path):
        df_existing_oai = pd.read_parquet(oai_index_path, columns=["norm_title", "arxiv_id"])
        existing_rows = df_existing_oai.to_dict("records")
        print(f"[INFO] Loaded existing OAI index with {len(df_existing_oai)} rows from {oai_index_path}")

    print(f"[INFO] Updating OAI index via OAI-PMH...")
    new_rows = []
    for norm_title, arxiv_id in harvest_oai(
        metadata_prefix="oai_dc",
        set_spec=oai_set,
        limit=oai_limit,
        start_token=oai_start_token,
    ):
        new_rows.append({"norm_title": norm_title, "arxiv_id": arxiv_id})
        if len(new_rows) % 5000 == 0:
            print(f"[INFO] OAI harvest so far: {len(new_rows)} records...")

    if not new_rows and not existing_rows:
        print("[WARN] OAI harvest returned no records; nothing to write to OAI index.")
        return

    df_oai = pd.DataFrame(existing_rows + new_rows)
    df_oai = df_oai.drop_duplicates(subset=["norm_title"], keep="first")
    os.makedirs(os.path.dirname(oai_index_path), exist_ok=True)
    df_oai.to_parquet(oai_index_path, index=False)
    print(f"[INFO] Wrote updated OAI index with {len(df_oai)} rows to {oai_index_path}")

def print_cache_stats_sql(cache_parquet_path: str) -> None:
    """SQL-based cache stats using DuckDB over the Parquet file (read-only)."""
    if not os.path.isfile(cache_parquet_path):
        print(f"[WARN] Cache parquet not found: {cache_parquet_path}")
        return
    con = duckdb.connect(database=":memory:")
    try:
        total_rows = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [cache_parquet_path]).fetchone()[0]
    except Exception as e:
        print(f"[ERROR] Failed to read cache parquet in DuckDB: {e}")
        return
    if total_rows == 0:
        print(f"[STATS] Cache parquet is empty: {cache_parquet_path}")
        return
    try:
        unique_titles = con.execute("SELECT COUNT(DISTINCT title) FROM read_parquet(?)", [cache_parquet_path]).fetchone()[0]
    except Exception as e:
        print(f"[WARN] Could not compute unique_titles (likely missing 'title' column): {e}")
        unique_titles = 0
    try:
        rows_with_arxiv_id = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?) WHERE arxiv_id IS NOT NULL AND TRIM(CAST(arxiv_id AS VARCHAR)) <> ''",
            [cache_parquet_path],
        ).fetchone()[0]
    except Exception as e:
        print(f"[WARN] Could not compute rows_with_arxiv_id: {e}")
        rows_with_arxiv_id = 0
    try:
        searched_but_not_found_rows = con.execute("SELECT COUNT(*) FROM read_parquet(?) WHERE query_status = 'not_found'", [cache_parquet_path]).fetchone()[0]
    except Exception as e:
        print(f"[WARN] Could not compute searched_but_not_found_rows: {e}")
        searched_but_not_found_rows = 0
    final_status_list = ",".join(f"'{s}'" for s in FINAL_QUERY_STATUSES)
    try:
        remaining_to_search_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet(?) WHERE (arxiv_id IS NULL OR TRIM(CAST(arxiv_id AS VARCHAR)) = '') AND (query_status IS NULL OR query_status NOT IN ({final_status_list}))",
            [cache_parquet_path],
        ).fetchone()[0]
    except Exception as e:
        print(f"[WARN] Could not compute remaining_to_search_rows: {e}")
        remaining_to_search_rows = 0
    print(f"[STATS] Cache parquet (SQL): {cache_parquet_path}")
    print(f"        total_rows: {total_rows}")
    print(f"        unique_titles: {unique_titles}")
    print(f"        rows_with_arxiv_id: {rows_with_arxiv_id}")
    print(f"        searched_but_not_found_rows (query_status=='not_found'): {searched_but_not_found_rows}")
    print(
        "        remaining_to_search_rows (arxiv_id empty & query_status not in FINAL_QUERY_STATUSES): "
        f"{remaining_to_search_rows}"
    )


def write_final_missing_titles_from_cache_sql(cache_parquet_path: str, output_txt_path: str) -> None:
    """SQL-based export of final missing titles from cache parquet."""
    if not os.path.isfile(cache_parquet_path):
        print(f"[WARN] Cache parquet not found: {cache_parquet_path}")
        return
    con = duckdb.connect(database=":memory:")
    try:
        rows = con.execute(
            "SELECT DISTINCT TRIM(CAST(title AS VARCHAR)) AS title FROM read_parquet(?) "
            "WHERE (arxiv_id IS NULL OR TRIM(CAST(arxiv_id AS VARCHAR)) = '') "
            "AND query_status = 'not_found' "
            "AND title IS NOT NULL "
            "AND TRIM(CAST(title AS VARCHAR)) <> '' "
            "ORDER BY title",
            [cache_parquet_path],
        ).fetchall()
    except Exception as e:
        print(
            "[WARN] Could not select final missing titles "
            "(likely missing columns 'title', 'arxiv_id', or 'query_status'): "
            f"{e}"
        )
        return
    titles = [r[0] for r in rows]
    if not titles:
        print("[INFO] No final missing titles found in cache parquet (SQL); nothing written.")
        return
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for t in titles:
            f.write(t + "\n")
    print(f"[INFO] (SQL) Wrote {len(titles)} unique final missing titles to {output_txt_path}")


def rescue_cache_from_oai_index_sql(cache_parquet_path: str, oai_index_parquet_path: str) -> None:
    """
    SQL-based "dry-run" OAI rescue:

    - Uses DuckDB to JOIN cache (missing arxiv_id) with OAI index by normalized title.
    - Prints how many rows *would* be matched, and how many would remain missing,
      but does NOT modify the parquet files.

    Normalization (approx SQL version of normalize_title + preprocess_title):
      norm1 = lower(regexp_replace(trim(title), '\\s+', ' ', 'g'))
      norm2 = lower(
                regexp_replace(
                  trim(
                    regexp_replace(title, '[-:_*@&''\"]+', ' ', 'g')
                  ),
                  '\\s+', ' ', 'g'
                )
              )
    """
    if not os.path.isfile(cache_parquet_path):
        print(f"[WARN] Cache parquet not found: {cache_parquet_path}")
        return
    if not os.path.isfile(oai_index_parquet_path):
        print(f"[WARN] OAI index parquet not found: {oai_index_parquet_path}")
        return
    con = duckdb.connect(database=":memory:")
    start_time = time.time()
    try:
        total_missing_before = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?) WHERE (arxiv_id IS NULL OR TRIM(CAST(arxiv_id AS VARCHAR)) = '')",
            [cache_parquet_path],
        ).fetchone()[0]
    except Exception as e:
        print(f"[ERROR] Could not compute total missing rows in cache (SQL): {e}")
        return
    if total_missing_before == 0:
        print("[INFO] (SQL) No rows without arxiv_id in cache; nothing for OAI rescue to do.")
        return
    try:
        rows = con.execute(
            "WITH cache_missing AS ("
            "  SELECT TRIM(CAST(title AS VARCHAR)) AS title,"
            "         LOWER(REGEXP_REPLACE(TRIM(CAST(title AS VARCHAR)), '\\\\s+', ' ', 'g')) AS norm1,"
            "         LOWER(REGEXP_REPLACE(TRIM(REGEXP_REPLACE(CAST(title AS VARCHAR), '[-:_*@&''\"]+', ' ', 'g')), '\\\\s+', ' ', 'g')) AS norm2 "
            "  FROM read_parquet(?) "
            "  WHERE (arxiv_id IS NULL OR TRIM(CAST(arxiv_id AS VARCHAR)) = '') "
            "    AND title IS NOT NULL "
            "    AND TRIM(CAST(title AS VARCHAR)) <> ''"
            "), "
            "oai AS ("
            "  SELECT "
            "    LOWER(TRIM(CAST(norm_title AS VARCHAR))) AS norm_title,"
            "    LOWER(REGEXP_REPLACE(TRIM(REGEXP_REPLACE(CAST(norm_title AS VARCHAR), '[-:_*@&''\"]+', ' ', 'g')), '\\\\s+', ' ', 'g')) AS norm_title2,"
            "    TRIM(CAST(arxiv_id AS VARCHAR)) AS arxiv_id "
            "  FROM read_parquet(?) "
            "  WHERE norm_title IS NOT NULL "
            "    AND arxiv_id IS NOT NULL "
            "    AND TRIM(CAST(norm_title AS VARCHAR)) <> '' "
            "    AND TRIM(CAST(arxiv_id AS VARCHAR)) <> ''"
            "), "
            "joined AS ("
            "  SELECT DISTINCT c.title, c.norm1, o.arxiv_id "
            "  FROM cache_missing c "
            "  JOIN oai o ON "
            "       c.norm1 = o.norm_title "
            "    OR c.norm2 = o.norm_title "
            "    OR c.norm1 = o.norm_title2 "
            "    OR c.norm2 = o.norm_title2"
            ") "
            "SELECT title, norm1, arxiv_id FROM joined",
            [cache_parquet_path, oai_index_parquet_path],
        ).fetchall()
    except Exception as e:
        print(f"[ERROR] (SQL) Failed to compute OAI rescue matches via JOIN: {e}")
        return

    matched_rows = len(rows)
    remaining_after = max(total_missing_before - matched_rows, 0)
    elapsed = time.time() - start_time
    print("[INFO] (SQL OAI rescue)")
    print(f"        total_missing_before: {total_missing_before}")
    print(f"        matched_via_OAI_index (now applied): {matched_rows}")
    print(f"        remaining_missing_after: {remaining_after}")
    print(f"        time_spent_seconds: {elapsed:.3f}")

    if matched_rows == 0:
        return

    try:
        con.execute("CREATE TEMP TABLE oai_updates (title VARCHAR, norm_title_new VARCHAR, arxiv_id_new VARCHAR)")
        con.executemany("INSERT INTO oai_updates (title, norm_title_new, arxiv_id_new) VALUES (?, ?, ?)", rows)
        # DuckDB COPY TO does not support ? placeholder for output path; use escaped path literal.
        path_escaped = cache_parquet_path.replace("'", "''")
        con.execute(
            "COPY (WITH cache AS (SELECT * FROM read_parquet(?)), joined AS (SELECT c.*, u.norm_title_new, u.arxiv_id_new FROM cache c LEFT JOIN oai_updates u ON TRIM(CAST(c.title AS VARCHAR)) = TRIM(CAST(u.title AS VARCHAR))) SELECT * EXCLUDE (norm_title, arxiv_id, query_status), CASE WHEN (arxiv_id IS NULL OR TRIM(CAST(arxiv_id AS VARCHAR)) = '') AND arxiv_id_new IS NOT NULL THEN norm_title_new ELSE norm_title END AS norm_title, CASE WHEN (arxiv_id IS NULL OR TRIM(CAST(arxiv_id AS VARCHAR)) = '') AND arxiv_id_new IS NOT NULL THEN TRIM(CAST(arxiv_id_new AS VARCHAR)) ELSE arxiv_id END AS arxiv_id, CASE WHEN (arxiv_id IS NULL OR TRIM(CAST(arxiv_id AS VARCHAR)) = '') AND arxiv_id_new IS NOT NULL THEN 'found' ELSE query_status END AS query_status FROM joined) TO '"
            + path_escaped
            + "' (FORMAT PARQUET)",
            [cache_parquet_path],
        )
    except Exception as e:
        print(f"[ERROR] (SQL) Failed to apply OAI rescue updates back to cache parquet: {e}")


def sync_html_paths_from_folder(cache_parquet_path: str, html_folder: str, project_root: str | None = None) -> int:
    """
    Scan html_folder for *.html files, match filename (stem) to arxiv_id in cache parquet,
    and fill html_path for rows where html_path is empty. Supports symlinks (ln).

    Stored paths are relative to project_root (e.g. data/arxiv_fulltext_html_251117/xxx.html),
    not absolute (/Users/...).

    Returns number of rows updated.
    """
    if not os.path.isdir(html_folder):
        print(f"[WARN] HTML folder not found: {html_folder}; skipping sync.")
        return 0
    if not os.path.isfile(cache_parquet_path):
        print(f"[WARN] Cache parquet not found: {cache_parquet_path}; skipping sync.")
        return 0

    html_folder_abs = os.path.abspath(html_folder)
    if project_root is None:
        project_root = os.getcwd()
    project_root_abs = os.path.abspath(project_root)

    # Build arxiv_id -> stored path (relative to project_root)
    file_id_to_path: dict[str, str] = {}
    for f in os.listdir(html_folder):
        if not f.endswith(".html"):
            continue
        stem = f[:-5]  # strip .html
        if not stem:
            continue
        full_path = os.path.join(html_folder, f)
        if not os.path.isfile(full_path):  # skip dirs; symlinks to files are fine
            continue
        try:
            rel = os.path.relpath(os.path.abspath(full_path), project_root_abs)
        except ValueError:
            rel = os.path.join(os.path.basename(html_folder), f)
        file_id_to_path[stem] = rel

    if not file_id_to_path:
        print(f"[INFO] No *.html files in {html_folder}; nothing to sync.")
        return 0
    print(f"[INFO] Found {len(file_id_to_path)} HTML files in folder; syncing to cache.")

    df = pd.read_parquet(cache_parquet_path)
    if "html_path" not in df.columns:
        df["html_path"] = ""
    if "arxiv_id" not in df.columns:
        print("[WARN] Cache has no arxiv_id column; skipping sync.")
        return 0

    empty_mask = (df["html_path"].isna()) | (df["html_path"].astype(str).str.strip().eq(""))
    has_id_mask = df["arxiv_id"].notna() & (df["arxiv_id"].astype(str).str.strip().ne(""))
    to_update_mask = empty_mask & has_id_mask

    n_updated = 0
    for idx in df.index[to_update_mask]:
        aid = str(df.at[idx, "arxiv_id"]).strip()
        if not aid:
            continue
        # Match base id (e.g. 2101.12345 matches 2101.12345.html or 2101.12345v2.html)
        base_id = re.sub(r"v\d+$", "", aid)
        path_val = file_id_to_path.get(aid) or file_id_to_path.get(base_id)
        if path_val:
            df.at[idx, "html_path"] = path_val
            n_updated += 1

    if n_updated > 0:
        os.makedirs(os.path.dirname(cache_parquet_path), exist_ok=True)
        df.to_parquet(cache_parquet_path, index=False)
        print(f"[STATS] Synced {n_updated} html_path from folder to cache.")
    return n_updated


def rescue_cache_from_bibtex(cache_parquet_path: str, bibtex_parquet_path: str) -> None:
    """
    Use bibtex (title, arxiv_id) to fill missing arxiv_ids in cache. Updates df_cache in place and saves to file.
    """
    if not os.path.isfile(cache_parquet_path):
        print(f"[INFO] Cache parquet not found; skipping bibtex rescue.")
        return
    if not os.path.isfile(bibtex_parquet_path):
        print(f"[INFO] Bibtex parquet not found: {bibtex_parquet_path}; skipping rescue.")
        return

    df_cache = pd.read_parquet(cache_parquet_path)

    df_bib = pd.read_parquet(bibtex_parquet_path, columns=["title", "arxiv_id"])
    df_bib = df_bib[df_bib["title"].notna() & df_bib["arxiv_id"].notna()]
    df_bib = df_bib[df_bib["title"].astype(str).str.strip().ne("") & df_bib["arxiv_id"].astype(str).str.strip().ne("")]
    df_bib = df_bib.assign(norm_title=df_bib["title"].apply(normalize_title))
    df_bib = df_bib.drop_duplicates(subset=["norm_title"], keep="first")
    bib_map = dict(zip(df_bib["norm_title"], df_bib["arxiv_id"]))

    missing_mask = (df_cache["arxiv_id"].isna()) | (df_cache["arxiv_id"].astype(str).str.strip().eq(""))
    if not missing_mask.any():
        print("[INFO] No rows without arxiv_id in cache; nothing for bibtex rescue.")
        return

    df_missing = df_cache.loc[missing_mask].copy()
    df_missing["_norm"] = df_missing["norm_title"].fillna(df_missing["title"].apply(normalize_title))
    df_missing["_norm2"] = df_missing["title"].apply(lambda t: normalize_title(preprocess_title(str(t))) if pd.notna(t) else "")
    df_missing["_arxiv_id"] = df_missing["_norm"].map(bib_map).fillna(df_missing["_norm2"].map(bib_map))

    to_update = df_missing["_arxiv_id"].notna()
    if not to_update.any():
        print("[INFO] bibtex had no new titles to rescue.")
        return

    idxs = df_missing.index[to_update]
    df_cache.loc[idxs, "arxiv_id"] = df_missing.loc[idxs, "_arxiv_id"].values
    df_cache.loc[idxs, "norm_title"] = df_missing.loc[idxs, "_norm"].values
    df_cache.loc[idxs, "query_status"] = "found"

    n = to_update.sum()
    os.makedirs(os.path.dirname(cache_parquet_path), exist_ok=True)
    df_cache.to_parquet(cache_parquet_path, index=False)
    print(f"[STATS] Rescued {n} titles from bibtex; saved cache to {cache_parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="Resolve titles to arXiv IDs using legacy caches + OAI-PMH")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--oai-set', dest='oai_set', default=None, help='OAI set (e.g., cs, physics:hep-th). Omit for full harvest.')
    parser.add_argument('--oai-limit', dest='oai_limit', type=int, default=None, help='Stop OAI harvest after N records (for quick testing).')
    parser.add_argument('--oai-start-token', dest='oai_start_token', default=None, help='Optional OAI resumptionToken to resume harvesting manually.')
    args = parser.parse_args()

    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""

    # parquet_path = os.path.join(processed_base_path, f"extracted_annotations{suffix}.parquet")
    parquet_path = os.path.join(processed_base_path, f"s2orc_titles2ids{suffix}.parquet")
    # legacy JSON cache paths (used only as optional one‑time seed)
    #HTML_CACHE_FILE = os.path.join(processed_base_path, f"arxiv_html_cache{suffix}.json")
    #NEW_CACHE_PATH = os.path.join(processed_base_path, f"title2arxiv_new_cache{suffix}.json")
    # unified schema cache (primary storage)
    CACHE_PARQUET_PATH = os.path.join(processed_base_path, f"title2arxiv_cache{suffix}.parquet")
    bibtex_parquet_path = os.path.join(processed_base_path, f"bibtex_title_arxiv{suffix}.parquet")
    print(f"📁 Input titles parquet: {parquet_path}")
    print(f"📁 Unified title/id/HTML cache (Parquet, primary): {CACHE_PARQUET_PATH}")

    if os.path.isfile(CACHE_PARQUET_PATH):
        print(f"[INFO] Unified cache parquet found.")
    else:
        # Initialize: s2orc titles + arxiv_titles_cache (url->title, extract arxiv_id). Concat, dedup, save.
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"s2orc_titles2ids parquet not found: {parquet_path}")
        os.makedirs(os.path.dirname(CACHE_PARQUET_PATH), exist_ok=True)

        # 1) df from s2orc_titles2ids: title=query_title (raw), norm_title=normalize(retrieved_title)
        #    Same as original SQL: norm_title = LOWER(REGEXP_REPLACE(TRIM(retrieved_title), '\s+', ' ', 'g'))
        df_s2orc = pd.read_parquet(parquet_path, columns=["query_title", "retrieved_title"])
        df_s2orc = df_s2orc[df_s2orc["retrieved_title"].notna() & (df_s2orc["retrieved_title"].astype(str).str.strip() != "")]
        df_s2orc = df_s2orc[~df_s2orc["retrieved_title"].astype(str).str.lower().isin(["nan", "none"])]
        df_s2orc["title"] = df_s2orc["query_title"].fillna(df_s2orc["retrieved_title"]).astype(str).str.strip()
        df_s2orc["norm_title"] = df_s2orc["retrieved_title"].astype(str).str.strip().apply(normalize_title)
        df_s2orc["_norm_key"] = df_s2orc["norm_title"]  # for cross-source dedup
        df_s2orc = df_s2orc.drop_duplicates(subset=["_norm_key"])
        df_s2orc["arxiv_id"] = ""
        df_s2orc["html_path"] = ""
        df_s2orc["query_status"] = ""

        # 2) df from arxiv_titles_cache: title=t (extracted), norm_title=normalize(t)
        titles_cache_path = os.path.join(processed_base_path, f"arxiv_titles_cache{suffix}.json")
        rows_titles = []
        if os.path.isfile(titles_cache_path):
            url_to_title = load_json_cache(titles_cache_path)
            for url, extracted_title in url_to_title.items():
                aid = extract_arxiv_id(url)
                if aid and extracted_title and str(extracted_title).strip():
                    t = str(extracted_title).strip()
                    norm_t = normalize_title(t)
                    rows_titles.append({
                        "title": t,
                        "norm_title": norm_t,
                        "_norm_key": norm_t,
                        "arxiv_id": aid,
                        "html_path": "",
                        "query_status": "found",
                    })
        df_titles = pd.DataFrame(rows_titles) if rows_titles else pd.DataFrame(columns=["title", "norm_title", "_norm_key", "arxiv_id", "html_path", "query_status"])
        if not df_titles.empty:
            df_titles = df_titles.drop_duplicates(subset=["_norm_key"], keep="first")

        # 3) Concat, dedup by _norm_key (keep row with arxiv_id when duplicate)
        df_s2orc_cols = ["title", "norm_title", "_norm_key", "arxiv_id", "html_path", "query_status"]
        df_s2orc = df_s2orc[df_s2orc_cols]
        df_init = pd.concat([df_s2orc, df_titles], ignore_index=True)
        df_init["_has_id"] = df_init["arxiv_id"].notna() & (df_init["arxiv_id"].astype(str).str.strip() != "")
        df_init = df_init.sort_values("_has_id", ascending=False).drop_duplicates(subset=["_norm_key"], keep="first")
        df_init = df_init.drop(columns=["_has_id", "_norm_key"])
        df_init.to_parquet(CACHE_PARQUET_PATH, index=False)
        print(f"[INFO] Initialized cache: s2orc={len(df_s2orc)}, arxiv_titles_cache={len(df_titles)}, merged={len(df_init)} rows.")

    print_cache_stats_sql(CACHE_PARQUET_PATH)
    rescue_cache_from_bibtex(CACHE_PARQUET_PATH, bibtex_parquet_path)

    ######## 4) OAI-PMH index: resolve missing titles from bulk metadata########
    SKIP_QUERY_OAI = True  # OAI harvest is slow; default to using existing local index only.
    oai_index_path = os.path.join(processed_base_path, f"title2arxiv_oai_index{suffix}.parquet")
    if not SKIP_QUERY_OAI:
        print(f"[INFO] Building OAI index...")
        build_or_update_oai_index(oai_index_path=oai_index_path, oai_set=getattr(args, "oai_set", None), oai_limit=getattr(args, "oai_limit", None), oai_start_token=getattr(args, "oai_start_token", None))
    else:
        print(f"[INFO] SKIP_QUERY_OAI=True; using existing OAI index only.")

    rescue_cache_from_oai_index_sql(CACHE_PARQUET_PATH, oai_index_path)

    ######## 5) Sync html_path from existing HTML folder (only fill empty) ########
    html_folder = os.path.join(base_path, f"arxiv_fulltext_html{suffix}")
    base_abs = os.path.abspath(base_path)
    project_root = os.path.dirname(base_abs) if os.path.isdir(base_abs) else os.getcwd()
    sync_html_paths_from_folder(CACHE_PARQUET_PATH, html_folder, project_root=project_root)

    print_cache_stats_sql(CACHE_PARQUET_PATH)

    final_missing_txt = os.path.join(processed_base_path, f"final_missing_titles_from_cache{suffix}.txt")
    write_final_missing_titles_from_cache_sql(CACHE_PARQUET_PATH, final_missing_txt)

if __name__ == "__main__":
    main()