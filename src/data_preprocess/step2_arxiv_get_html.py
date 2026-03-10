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
from src.data_preprocess.step2_arxiv_github_title import extract_arxiv_id
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET
from src.utils import load_config, extract_non_empty_column_list_sql

FINAL_QUERY_STATUSES = {"found", "not_found"}  # treated as "successfully validated"

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

def search_arxiv_title(title_query, max_results=5):
    base_url = "http://export.arxiv.org/api/query"
    title_query = preprocess_title(title_query)
    encoded_query = quote(title_query) 
    params = {
        "search_query": f"ti:{encoded_query}", 
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    # print(f"[DEBUG] arXiv URL: {base_url}?{'&'.join(f'{k}={v}' for k,v in params.items())}")  # verbose per-query
    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.text, "ok"
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 429:
            print(f"[WARN] arXiv API rate‑limited (429) for title '{title_query}'.")
            return None, "rate_limited_429"
        else:
            print(f"[ERROR] arXiv API request failed (status={status}): {e}")
            if status is not None:
                return None, f"http_{status}"
            return None, "http_error"
    except requests.exceptions.Timeout:
        print(f"[WARN] arXiv API request timed out for title '{title_query}'.")
        return None, "timeout"
    except requests.exceptions.ConnectionError:
        print(f"[WARN] arXiv API connection error for title '{title_query}'.")
        return None, "connection_error"
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] arXiv API request failed: {e}")
        return None, "request_error"
    except Exception as e:
        print(f"[ERROR] arXiv API unexpected error: {e}")
        return None, "unexpected_error"

def parse_arxiv_atom(xml_text):
    """
    Parse the XML (Atom feed) from arXiv to extract a list of entries.
    Each entry is a dict with keys: 'title', 'id', 'summary', 'updated', 'published'.
    """
    root = ET.fromstring(xml_text)
    ns = "{http://www.w3.org/2005/Atom}"
    entries_info = []
    for entry in root.findall(ns + 'entry'):
        entry_title = entry.find(ns + 'title')
        entry_id = entry.find(ns + 'id')
        entry_summary = entry.find(ns + 'summary')
        entry_updated = entry.find(ns + 'updated')
        entry_published = entry.find(ns + 'published')
        if entry_title is not None and entry_id is not None:
            info = {
                "title": entry_title.text.strip() if entry_title.text else "",
                "id": entry_id.text.strip() if entry_id.text else "",
                "summary": entry_summary.text.strip() if (entry_summary is not None and entry_summary.text) else "",
                "updated": entry_updated.text.strip() if (entry_updated is not None and entry_updated.text) else "",
                "published": entry_published.text.strip() if (entry_published is not None and entry_published.text) else ""
            }
            entries_info.append(info)
    return entries_info

def fetch_ar5iv_html(arxiv_id, html_folder):
    """
    Fetch HTML from ar5iv (ar5iv.labs.arxiv.org) given an arXiv ID.
    Save the HTML to a local file in folder html_folder with filename '{arxiv_id}.html'
    and return the file path, or None on failure.
    """
    file_path = os.path.join(html_folder, f"{arxiv_id}.html")  
    if os.path.exists(file_path):
        return file_path
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            html_text = resp.text
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_text)
            return file_path
        elif resp.status_code == 404:
            print(f"[WARN] HTML not exist: {arxiv_id}")
            return None
        else:
            print(f"[WARN] ar5iv HTML not found for {arxiv_id}, status={resp.status_code}")
            ######## NEW: if HTML not found, try to fetch the base arXiv ID (without version) ########
            base_arxiv_id = re.sub(r'v\d+$', '', arxiv_id)  
            if base_arxiv_id != arxiv_id:
                print(f"[INFO] Fallback: trying base arXiv ID '{base_arxiv_id}' for {arxiv_id}.")  
                base_file_path = os.path.join(html_folder, f"{base_arxiv_id}.html")  
                if os.path.exists(base_file_path):
                    return base_file_path  
                base_url = f"https://ar5iv.labs.arxiv.org/html/{base_arxiv_id}"  
                try:
                    base_resp = requests.get(base_url, timeout=15)  
                    if base_resp.status_code == 200:
                        html_text = base_resp.text  
                        with open(base_file_path, "w", encoding="utf-8") as f:  
                            f.write(html_text)  
                        return base_file_path  
                    else:
                        print(f"[WARN] Fallback ar5iv HTML not found for {base_arxiv_id}, status={base_resp.status_code}")  
                        return None  
                except Exception as ex:
                    print(f"[ERROR] Fallback ar5iv HTML fetch error for {base_arxiv_id}: {ex}")  
                    return None  
            return None
    except Exception as e:
        print(f"[ERROR] ar5iv HTML fetch error for {arxiv_id}: {e}")
        return None

######## NEW: Single function that (1) searches by title, (2) picks ID, (3) fetches HTML ########
def fetch_id_and_html_for_title(title, html_folder, max_results=3, html_cache=None):
    """
    Given a title, query arXiv, parse the Atom feed, pick the first result's ID,
    and then fetch HTML from ar5iv.
    Immediately update the provided html_cache dict with the result (arxiv_id -> local HTML file path).
    Returns (arxiv_id, html_file_path, query_status, need_wait).
    query_status:
      - found: validated (Atom feed had entries and we parsed an arXiv id)
      - not_found: validated (Atom feed returned 200 but had no entries)
      - other values: retryable / error states (429/5xx/timeout/etc.); treated as missing next time
    """
    # 1) Search
    try:
        xml_text, qstatus = search_arxiv_title(title, max_results=max_results)
        if xml_text is None:
            # API returned 429 / timeout / error — do not parse; treat as retryable
            return None, None, qstatus, True
        entries = parse_arxiv_atom(xml_text)
        if not entries:
            print(f"[INFO] No Atom entries found for title: {title}")
            return None, None, "not_found", False
    except Exception as e:
        print(f"[ERROR] Atom feed error for '{title}': {e}")
        return None, None, "atom_parse_error", False

    # 2) Take the first entry as best guess
    arxiv_url = entries[0]["id"]  # e.g., "http://arxiv.org/abs/2101.12345"
    arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else None
    if not arxiv_id:
        print(f"[INFO] Could not parse ID from feed for title: {title}")
        return None, None, "id_parse_error", False

    # 3) Fetch HTML from ar5iv
    if arxiv_id in html_cache:
        html_file_path = html_cache[arxiv_id]
        if html_file_path and os.path.isfile(html_file_path):
            print(f"[INFO] HTML already cached for {arxiv_id}: {html_file_path}")
            return arxiv_id, html_file_path, "found", False
            #file_size = os.path.getsize(html_file_path)
            #else:
            #file_size = 0
        else:
            # Path is invalid or empty, attempt to re-fetch
            print(f"[INFO] Cached HTML missing for {arxiv_id}, re-fetching...")
            html_file_path = fetch_ar5iv_html(arxiv_id, html_folder)
            if html_file_path:
                html_cache[arxiv_id] = html_file_path
            else:
                html_cache[arxiv_id] = ""  # Mark as failed
            return arxiv_id, html_file_path, "found", True
        print(f"[INFO] Already in HTML cache: {arxiv_id} (file: {html_file_path})")
    else:
        html_file_path = fetch_ar5iv_html(arxiv_id, html_folder)
        if html_file_path:
            html_cache[arxiv_id] = html_file_path
            #file_size = os.path.getsize(html_file_path)
            #print(f"[INFO] Fetched HTML for {arxiv_id}, saved to {html_file_path}.")
        else:
            html_cache[arxiv_id] = ""
            print(f"[INFO] No valid HTML for {arxiv_id}. Marked empty in cache.")
    return arxiv_id, html_file_path, "found", True

def real_batch_title_to_arxiv_id(titles, html_folder, html_cache):
    """
    For each title in 'titles':
      1) Query arXiv for the ID.
      2) Fetch HTML from ar5iv for that ID and save to local file.
      3) Save ID to DataFrame row, and update HTML cache with file path.
    Returns:
      - df_new: DataFrame with columns ["title", "arxiv_id", "query_status"]
      - html_cache: updated {arxiv_id -> html_path} dict (in‑memory only).
    """
    new_rows = []
    for t in titles:
        t_stripped = t.strip()
        arxiv_id, html_file_path, query_status, need_wait = fetch_id_and_html_for_title(
            t_stripped, html_folder, max_results=3, html_cache=html_cache
        )
        new_rows.append((t_stripped, arxiv_id, query_status))
        ######## Log with file size if available ########
        if html_file_path and os.path.isfile(html_file_path):
            size_info = os.path.getsize(html_file_path)
        else:
            size_info = 0
        print(f"[INFO] Title='{t_stripped}' -> ID='{arxiv_id}', status='{query_status}', HTML file='{html_file_path}', size='{size_info}'")
        # arXiv ToU: at most 1 request every 3 seconds, single connection. Always wait after each API use.
        if arxiv_id is None and need_wait:
            time.sleep(3.1)  # Rate-limited (429) or timeout; back off before next request
        else:
            time.sleep(3.1)   # Normal delay between requests (required by arXiv API Terms of Use)
    df_new = pd.DataFrame(new_rows, columns=["title", "arxiv_id", "query_status"])
    return df_new, html_cache

def main():
    parser = argparse.ArgumentParser(description="Download arXiv HTML pages for retrieved titles")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    args = parser.parse_args()

    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""

    # parquet_path = os.path.join(processed_base_path, f"extracted_annotations{suffix}.parquet")
    parquet_path = os.path.join(processed_base_path, f"s2orc_titles2ids{suffix}.parquet")
    # legacy JSON cache paths (used only as optional one‑time seed)
    HTML_CACHE_FILE = os.path.join(processed_base_path, f"arxiv_html_cache{suffix}.json")
    NEW_CACHE_PATH = os.path.join(processed_base_path, f"title2arxiv_new_cache{suffix}.json")
    # unified schema cache (primary storage)
    CACHE_PARQUET_PATH = os.path.join(processed_base_path, f"title2arxiv_cache{suffix}.parquet")
    HTML_FOLDER = os.path.join(base_path, f"arxiv_fulltext_html{suffix}")
    os.makedirs(HTML_FOLDER, exist_ok=True)
    print(f"📁 Input titles parquet: {parquet_path}")
    print(f"📁 Unified title/id/HTML cache (Parquet, primary): {CACHE_PARQUET_PATH}")
    print(f"📁 Legacy HTML cache JSON (optional seed): {HTML_CACHE_FILE}")
    print(f"📁 Legacy title→arxiv cache JSON (optional seed): {NEW_CACHE_PATH}")

    ######## 1) Extract all titles from the main parquet ########
    all_titles_list = extract_non_empty_column_list_sql(parquet_path, "retrieved_title")
    all_titles_list = [t for t in all_titles_list if str(t).lower() not in {"nan", "none"}]
    all_titles = set(all_titles_list)
    print(f"[INFO] Loaded {len(all_titles_list)} non-empty rows from {parquet_path}, found {len(all_titles)} unique 'retrieved_title'.")

    ######## 2) Load / initialize the unified cache parquet (schema‑based cache) ########
    if os.path.isfile(CACHE_PARQUET_PATH):
        df_cache = pd.read_parquet(CACHE_PARQUET_PATH)
        print(f"[INFO] Loaded unified cache parquet with {len(df_cache)} rows from {CACHE_PARQUET_PATH}")
    else:
        print(f"[INFO] Unified cache parquet not found, building initial cache from legacy JSON (if any).")
        # 2.a) Use the newer title->id JSON as initial seed
        new_cache = load_json_cache(NEW_CACHE_PATH)
        if new_cache:
            print("[INFO] Using title2arxiv_new_cache JSON as seed.")
            combined_dict = dict(new_cache)
        else:
            combined_dict = {}
            print("[INFO] No title2arxiv_new_cache JSON; starting with empty title/id map.")

        # 2.b) Also try to extract IDs from the old {url -> title} JSON and merge them
        json_cache_path = os.path.join(processed_base_path, f"arxiv_titles_cache{suffix}.json")
        old_cache = load_json_cache(json_cache_path)  # Format: {url: title}
        if old_cache:
            old_title_id_dict = {}
            for url, extracted_title in old_cache.items():
                aid = extract_arxiv_id(url)  # e.g., '2101.12345'
                if aid and extracted_title:
                    old_title_id_dict[extracted_title] = aid
            print(f"[INFO] Converted old cache into {len(old_title_id_dict)} (title -> arxiv_id) pairs.")
            # Use normalize_title to deduplicate and merge
            for title, aid in old_title_id_dict.items():
                norm_title = normalize_title(title)
                if not any(normalize_title(k) == norm_title for k in combined_dict.keys()):
                    combined_dict[title] = aid
            print(f"[INFO] Combined legacy JSON caches -> {len(combined_dict)} (title -> arxiv_id) pairs.")

        # 2.c) Legacy HTML cache is only used as an initial value for html_path
        html_cache_legacy = load_json_cache(HTML_CACHE_FILE)

        cache_rows = []
        for title, aid in combined_dict.items():
            norm_t = normalize_title(title)
            if not aid:
                html_path = ""
            else:
                html_path = html_cache_legacy.get(aid, "")
            query_status = "found" if (aid and str(aid).strip()) else "unknown"
            cache_rows.append((title, norm_t, aid, html_path, query_status))

        if cache_rows:
            df_cache = pd.DataFrame(cache_rows, columns=["title", "norm_title", "arxiv_id", "html_path", "query_status"])
        else:
            df_cache = pd.DataFrame(columns=["title", "norm_title", "arxiv_id", "html_path", "query_status"])
        print(f"[INFO] Initial unified cache built with {len(df_cache)} rows from legacy JSON.")

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

    ######## 3) Set of titles we have in cache vs missing (need to query) ########
    # Only treat "found" / "not_found" as "successfully validated" cache entries.
    ok_mask = df_cache["query_status"].isin(FINAL_QUERY_STATUSES)
    cache_norm_set = set(df_cache.loc[ok_mask, "norm_title"].dropna())
    need_norm_set = set(normalize_title(t) for t in all_titles)
    missing_norm_set = need_norm_set - cache_norm_set

    ######## 2.5) Rescue: bibtex_title_arxiv parquet has (title, arxiv_id); title is already normalized ########
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

    ######## 2.6) OAI-PMH index: resolve missing titles from bulk metadata (avoids 5000 API calls) ########
    oai_index_path = os.path.join(processed_base_path, f"title2arxiv_oai_index{suffix}.parquet")
    if os.path.isfile(oai_index_path):
        df_oai = pd.read_parquet(oai_index_path, columns=["norm_title", "arxiv_id"])
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
            print(f"[STATS] Rescued {len(to_rescue_oai)} titles from OAI-PMH index (title2arxiv_oai_index); they will not be queried via API.")
        else:
            print(f"[INFO] OAI index had no new titles to rescue for current missing set.")
    else:
        print(f"[INFO] No title2arxiv_oai_index{suffix}.parquet found. Run build_arxiv_oai_index.py (with same --tag if you use one) to build (recommended for 5000+ titles).")

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

    ######## 5) For existing arxiv_ids in the cache, fill / correct html_path ########
    # Use a dict as the in‑memory html_cache
    html_cache = {}
    for _, row in df_cache.iterrows():
        aid = row.get("arxiv_id")
        path = row.get("html_path") or ""
        if aid:
            # Only keep one entry; later we will re‑map html_path for all rows
            html_cache[aid] = path

    all_known_ids = {aid for aid in df_cache["arxiv_id"].dropna() if aid}
    print(f"[INFO] Checking local HTML files for {len(all_known_ids)} arXiv IDs from unified cache...")
    for aid in all_known_ids:
        cached_path = html_cache.get(aid, "")
        if cached_path and os.path.isfile(cached_path):
            #print(f"[INFO] HTML exists: {aid} -> {cached_path}")
            continue
        # Attempt to fetch HTML
        html_file_path = fetch_ar5iv_html(aid, HTML_FOLDER)
        if html_file_path:
            html_cache[aid] = html_file_path
            #print(f"[INFO] Downloaded HTML for {aid} to {html_file_path}")
        else:
            html_cache[aid] = ""  # Mark failure
            print(f"[WARN] Failed to fetch HTML for {aid}")

    ######## 6) For missing titles: query arxiv_id + download HTML in batches, persisting after each batch ########
    if missing:
        missing_list = list(missing)
        batch_size = 50  # adjust this if needed
        total_missing = len(missing_list)
        print(f"[INFO] Fetching IDs + HTML for {total_missing} missing titles in batches of {batch_size}...")

        for start in range(0, total_missing, batch_size):
            end = min(start + batch_size, total_missing)
            batch_titles = missing_list[start:end]
            print(f"[INFO] Processing batch {start}-{end} / {total_missing} missing titles...")

            df_new, html_cache = real_batch_title_to_arxiv_id(batch_titles, HTML_FOLDER, html_cache)
            print(f"[INFO] real_batch_title_to_arxiv_id returned {len(df_new)} rows for this batch.")

            if not df_new.empty:
                df_new["norm_title"] = df_new["title"].astype(str).map(normalize_title)
                df_new["html_path"] = df_new["arxiv_id"].map(lambda aid: html_cache.get(aid, "") if aid else "")

                # Persist both "found" and "not_found" as validated results.
                # Persist other statuses too (for auditing), but they will still be treated as missing next run.
                # Upsert by norm_title: replace any existing rows for those titles.
                df_cache = df_cache[~df_cache["norm_title"].isin(set(df_new["norm_title"]))].copy()

                if not df_new.empty:
                    df_cache = pd.concat(
                        [df_cache, df_new[["title", "norm_title", "arxiv_id", "html_path", "query_status"]]],
                        ignore_index=True,
                    )

                    # Persist unified cache after each batch to avoid losing progress on long runs
                    df_cache.to_parquet(CACHE_PARQUET_PATH, index=False)
                    print(f"[INFO] Persisted unified cache after batch {start}-{end}; total rows now {len(df_cache)}.")
    else:
        print("[INFO] No missing titles; no additional fetch needed.")

    ######## 7) Recompute html_path column from html_cache to ensure consistency ########
    df_cache["html_path"] = df_cache["arxiv_id"].map(lambda aid: html_cache.get(aid, "") if aid else "")

    ######## 8) Save final unified schema parquet ########
    df_cache.to_parquet(CACHE_PARQUET_PATH, index=False)
    print(f"[INFO] Unified Parquet cache written with {len(df_cache)} rows to {CACHE_PARQUET_PATH}")

    # Coverage overview: how many source titles are covered by the cache
    cache_norm_set = set(df_cache["norm_title"])
    covered = sum(1 for t in all_titles if normalize_title(t) in cache_norm_set)
    print(f"[INFO] Coverage: {covered}/{len(all_titles)} titles present in unified cache.")

if __name__ == "__main__":
    main()
