# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-03-28
Last Modified: 2025-03-29
Description: This script is used to get the arXiv ID for the title extracted from the PDF file.

arXiv API rate limit (Terms of Use, https://info.arxiv.org/help/api/tou.html):
  - At most one request every 3 seconds; single connection at a time.
  - Exceeding this leads to 429 and possible blocking. We enforce 3s delay between each query.
"""
import os, re
import json
import time
import argparse
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from src.data_preprocess.step2_arxiv_github_title import extract_arxiv_id
from src.utils import load_config, extract_non_empty_column_list_sql

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


def harvest_oai(metadata_prefix="oai_dc", set_spec=None, limit=None):
    """
    Yield (norm_title, arxiv_id) from OAI-PMH ListRecords.
    Uses resumptionToken for paging.
    limit: stop after this many records (for quick testing).
    """
    params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
    if set_spec:
        params["set"] = set_spec
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
        next_url = f"{OAI_BASE}?verb=ListRecords&resumptionToken={requests.utils.quote(token)}"
        next_params = None
        time.sleep(OAI_DELAY_SEC)

def main():
    parser = argparse.ArgumentParser(description="Resolve titles to arXiv IDs using legacy caches + OAI-PMH")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--oai-set', dest='oai_set', default=None, help='OAI set (e.g., cs, physics:hep-th). Omit for full harvest.')
    parser.add_argument('--oai-limit', dest='oai_limit', type=int, default=None, help='Stop OAI harvest after N records (for quick testing).')
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
    print(f"📁 Input titles parquet: {parquet_path}")
    print(f"📁 Unified title/id/HTML cache (Parquet, primary): {CACHE_PARQUET_PATH}")
    #print(f"📁 Legacy HTML cache JSON (optional seed): {HTML_CACHE_FILE}")
    #print(f"📁 Legacy title→arxiv cache JSON (optional seed): {NEW_CACHE_PATH}")

    ######## 1) Extract all titles from the main parquet ########
    all_titles_list = extract_non_empty_column_list_sql(parquet_path, "retrieved_title")
    all_titles_list = [t for t in all_titles_list if str(t).lower() not in {"nan", "none"}]
    all_titles = set(all_titles_list)
    print(f"[INFO] Loaded {len(all_titles_list)} non-empty rows from {parquet_path}, found {len(all_titles)} unique 'retrieved_title'.")

    ######## 2) Load / initialize the unified cache parquet (schema‑based cache) ########
    if os.path.isfile(CACHE_PARQUET_PATH):
        df_cache = pd.read_parquet(CACHE_PARQUET_PATH)
        print(f"[INFO] Loaded unified cache parquet with {len(df_cache)} rows from {CACHE_PARQUET_PATH}")
        # Ensure required schema columns exist
        if "title" not in df_cache.columns:
            df_cache["title"] = ""
        if "arxiv_id" not in df_cache.columns:
            df_cache["arxiv_id"] = ""
        if "html_path" not in df_cache.columns:
            df_cache["html_path"] = ""
        if "norm_title" not in df_cache.columns:
            df_cache["norm_title"] = df_cache["title"].astype(str).map(normalize_title)
        if "query_status" not in df_cache.columns:
            # Backward compat: infer "found" only when we actually have a non-empty arxiv_id.
            has_id = df_cache["arxiv_id"].notna() & (df_cache["arxiv_id"].astype(str).str.strip() != "")
            df_cache["query_status"] = has_id.map(lambda x: "found" if x else "unknown")
    else:
        # JSON caches are deprecated; start from an empty unified cache and let OAI/bibtex fill it.
        df_cache = pd.DataFrame(columns=["title", "norm_title", "arxiv_id", "html_path", "query_status"])
        print(f"[INFO] Unified cache parquet not found; starting from empty cache (no legacy JSON).")

    ######## 3) Set of titles we have in cache vs missing (need to query) ########
    # Only treat "found" / "not_found" as "successfully validated" cache entries.
    ok_mask = df_cache["query_status"].isin(FINAL_QUERY_STATUSES)
    cache_norm_set = set(df_cache.loc[ok_mask, "norm_title"].dropna())
    need_norm_set = set(normalize_title(t) for t in all_titles)
    missing_norm_set = need_norm_set - cache_norm_set

    ######## 3.1) Rescue: bibtex_title_arxiv parquet has (title, arxiv_id); title is already normalized ########
    bibtex_parquet_path = os.path.join(processed_base_path, f"bibtex_title_arxiv{suffix}.parquet")
    if os.path.isfile(bibtex_parquet_path):
        df_bib = pd.read_parquet(bibtex_parquet_path, columns=["title", "arxiv_id"])
        df_bib = df_bib[df_bib["title"].notna() & df_bib["arxiv_id"].notna()]
        df_bib = df_bib[df_bib["title"].ne("") & df_bib["arxiv_id"].ne("")]
        # Only keep rows whose title is in the missing set (still need to query)
        to_rescue = df_bib[df_bib["title"].isin(missing_norm_set)]
        to_rescue = to_rescue[~to_rescue["title"].isin(cache_norm_set)].drop_duplicates(subset=["title"])
        if not to_rescue.empty:
            to_rescue = to_rescue.assign(norm_title=to_rescue["title"], html_path="", query_status="found")
            df_cache = pd.concat([df_cache, to_rescue[["title", "norm_title", "arxiv_id", "html_path", "query_status"]]], ignore_index=True)
            cache_norm_set = set(df_cache["norm_title"].dropna())
            missing_norm_set = need_norm_set - cache_norm_set
            print(f"[STATS] Rescued {len(to_rescue)} titles from bibtex_title_arxiv (added to cache); they will not be queried.")
        else:
            print(f"[INFO] bibtex_title_arxiv had no new titles to rescue (all already in cache or not in source list).")
    else:
        print(f"[INFO] No bibtex_title_arxiv{suffix}.parquet found; skipping rescue.")

    ######## 4) OAI-PMH index: resolve missing titles from bulk metadata (avoids 5000 API calls) ########
    oai_index_path = os.path.join(processed_base_path, f"title2arxiv_oai_index{suffix}.parquet")
    if os.path.isfile(oai_index_path):
        df_oai = pd.read_parquet(oai_index_path, columns=["norm_title", "arxiv_id"])
    else:
        print(f"[INFO] No title2arxiv_oai_index{suffix}.parquet found; harvesting OAI index on the fly (this may take a long time).")
        rows = []
        for norm_title, arxiv_id in harvest_oai(metadata_prefix="oai_dc", set_spec=args.oai_set, limit=args.oai_limit):
            rows.append({"norm_title": norm_title, "arxiv_id": arxiv_id})
            if len(rows) % 5000 == 0:
                print(f"[INFO] OAI harvest so far: {len(rows)} records...")
        if not rows:
            print("[WARN] OAI harvest returned no records; skipping OAI rescue.")
            df_oai = None
        else:
            df_oai = pd.DataFrame(rows)
            df_oai = df_oai.drop_duplicates(subset=["norm_title"], keep="first")
            os.makedirs(processed_base_path, exist_ok=True)
            df_oai.to_parquet(oai_index_path, index=False)
            print(f"[INFO] Wrote OAI index with {len(df_oai)} rows to {oai_index_path}")

    if df_oai is not None:
        df_oai = df_oai[df_oai["norm_title"].notna() & df_oai["arxiv_id"].notna()]
        df_oai = df_oai[df_oai["norm_title"].astype(str).str.strip().ne("") & df_oai["arxiv_id"].astype(str).str.strip().ne("")]
        oai_lookup = df_oai.set_index("norm_title")["arxiv_id"].to_dict()
        to_rescue_oai = []
        for norm_t in list(missing_norm_set):
            aid = oai_lookup.get(norm_t)
            if aid and str(aid).strip():
                to_rescue_oai.append((norm_t, norm_t, str(aid).strip(), ""))
        if to_rescue_oai:
            df_rescue = pd.DataFrame(to_rescue_oai, columns=["title", "norm_title", "arxiv_id", "html_path"])
            df_rescue["query_status"] = "found"
            df_cache = pd.concat([df_cache, df_rescue], ignore_index=True)
            cache_norm_set = set(df_cache["norm_title"].dropna())
            missing_norm_set = need_norm_set - cache_norm_set
            print(f"[STATS] Rescued {len(to_rescue_oai)} titles from OAI-PMH index; they will not be queried via other means.")
        else:
            print(f"[INFO] OAI index had no new titles to rescue for current missing set.")

    ######## 3) in_cache / missing (for reporting and downstream) ########
    in_cache = need_norm_set & cache_norm_set
    missing = missing_norm_set
    print(f"[INFO] Titles in Parquet: {len(all_titles)}")
    print(f"[INFO] Already in unified cache: {len(in_cache)}")
    print(f"[INFO] Missing (need fetch): {len(missing)}")
    print(f"[STATS] Remaining to query (after rescue): {len(missing)}")

    ######## 4) Save missing titles to a txt file for manual inspection ########
    tmp_missing_file = os.path.join(processed_base_path, f"missing_titles_tmp{suffix}.txt")
    with open(tmp_missing_file, "w", encoding="utf-8") as f:
        for title in sorted(missing):
            f.write(title + "\n")
    print(f"[INFO] Saved {len(missing)} missing titles to {tmp_missing_file}")

    ######## 5) Save final unified schema parquet and coverage stats (OAI/legacy only) ########
    df_cache.to_parquet(CACHE_PARQUET_PATH, index=False)
    print(f"[INFO] Unified Parquet cache written with {len(df_cache)} rows to {CACHE_PARQUET_PATH}")

    cache_norm_set = set(df_cache["norm_title"])
    covered = sum(1 for t in all_titles if normalize_title(t) in cache_norm_set)
    print(f"[INFO] Coverage: {covered}/{len(all_titles)} titles present in unified cache.")