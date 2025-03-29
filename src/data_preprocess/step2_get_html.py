# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-03-28
Last Modified: 2025-03-29 
Description: This script is used to get the arXiv ID for the title extracted from the PDF file.
"""
import os
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.data_preprocess.step2_se_url_title import extract_arxiv_id
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET

######## Normalization helper function ########
def normalize_title(title):
    """
    Normalize the title by converting to lower-case and reducing whitespace.
    """
    return " ".join(title.lower().split())

######## JSON cache load/save for new mapping ########
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
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[INFO] Saved JSON cache to {file_path} with {len(data)} entries.")
    except Exception as e:
        print(f"[ERROR] Could not save JSON cache: {e}")

######## Searching arXiv using the Atom API ########
def search_arxiv_title(title_query, max_results=5):
    """
    Query the arXiv Atom API using a broad query with the 'all' field.
    Uses URL encoding for safety.
    Returns the raw XML text from the feed.
    """
    base_url = "http://export.arxiv.org/api/query"
    # URL encode the query string
    encoded_query = quote(title_query)
    # Use 'all:' to search more broadly
    query = f"all:{encoded_query}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    print(f"[DEBUG] Querying arXiv with: {params['search_query']}")
    resp = requests.get(base_url, params=params, timeout=15)
    print(f"[DEBUG] Request URL: {resp.url}")
    resp.raise_for_status()
    return resp.text

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
    print(f"[DEBUG] Parsed {len(entries_info)} entries from the Atom feed.")
    return entries_info

######## Real batch function that queries arXiv with retries ########
def real_batch_title_to_arxiv_id(titles, max_results=2, sleep_time=3, retries=3):
    """
    For each title in 'titles', query arXiv using the Atom API (with retries)
    and return a DataFrame with columns ["title", "arxiv_id"].
    For each title, a single summary line is printed with the following format:
    ------------ Title: '...'  |  Answer: ...  |  Comment: ...  ------------
    """
    new_rows = []
    for t in titles:
        t_stripped = t.strip()
        success = False
        attempt = 0
        aid = None
        comment = ""
        while attempt < retries and not success:
            try:
                xml_text = search_arxiv_title(t_stripped, max_results=max_results)
                entries = parse_arxiv_atom(xml_text)
                if entries:
                    # Take the first entry as the best match
                    aid = entries[0]["id"].split('/')[-1]
                    comment = "Success"
                    success = True
                else:
                    comment = "No results"
            except Exception as e:
                comment = f"Error: {e}"
            attempt += 1
            if not success and attempt < retries:
                time.sleep(sleep_time)
        if not success:
            comment = f"Failed after {retries} attempts"
        # Print a single summary line per title using separators
        print("------------ Title: '{}'  |  Answer: {}  |  Comment: {} ------------".format(t_stripped, aid, comment))
        new_rows.append((t_stripped, aid))
    return pd.DataFrame(new_rows, columns=["title", "arxiv_id"])

######## Main Script ########
def main():
    ######## 1) Load a local Parquet file with "retrieved_title" column ########
    parquet_path = "extracted_annotations.parquet"
    if not os.path.isfile(parquet_path):
        print(f"[ERROR] Parquet file not found: {parquet_path}")
        return
    df_parquet = pd.read_parquet(parquet_path)
    if "retrieved_title" not in df_parquet.columns:
        print("[ERROR] 'retrieved_title' column not found in the parquet file.")
        return
    all_titles = set(df_parquet["retrieved_title"].dropna().unique())
    print(f"[INFO] Loaded {len(df_parquet)} rows from {parquet_path}, found {len(all_titles)} unique 'retrieved_title'.")

    ######## 2) Load the original old JSON cache {url -> extracted_title} ########
    json_cache_path = "data/processed/arxiv_titles_cache.json"
    old_cache = load_json_cache(json_cache_path)  # Format: {url: title}
    
    ######## 3) Convert old cache to {title -> arxiv_id} using extract_arxiv_id ########
    old_title_id_dict = {}
    for url, extracted_title in old_cache.items():
        aid = extract_arxiv_id(url)  # e.g., '2101.12345'
        if aid and extracted_title:
            old_title_id_dict[extracted_title] = aid
    print(f"[INFO] Converted old cache into {len(old_title_id_dict)} (title -> arxiv_id) pairs.")

    ######## 4) Load the new JSON-based cache for {title -> arxiv_id} ########
    new_cache_path = "title2arxiv_new_cache.json"
    new_cache = load_json_cache(new_cache_path)  # This is your new cache file
    
    ######## 5) Combine the two caches (old converted + new) using normalization ########
    combined_dict = dict(new_cache)  # start with new cache if exists
    for title, aid in old_title_id_dict.items():
        norm_title = normalize_title(title)
        if not any(normalize_title(k) == norm_title for k in combined_dict.keys()):
            combined_dict[title] = aid
    print(f"[INFO] Combined old and new caches, now have {len(combined_dict)} (title -> arxiv_id) pairs in memory.")

    ######## 6) Determine which titles from the Parquet are already in the combined cache ########
    in_cache = set()
    missing = set()
    normalized_combined = {normalize_title(k) for k in combined_dict.keys()}
    for t in all_titles:
        norm_t = normalize_title(t)
        if norm_t in normalized_combined:
            in_cache.add(t)
        else:
            missing.add(t)
    print(f"[INFO] Titles in Parquet:       {len(all_titles)}")
    print(f"[INFO] Already in combined cache: {len(in_cache)}")
    print(f"[INFO] Missing (need fetch):     {len(missing)}")
    
    ######## 7) Save missing titles to a temporary file for manual inspection ########
    tmp_missing_file = "missing_titles_tmp.txt"
    with open(tmp_missing_file, "w", encoding="utf-8") as f:
        for title in sorted(missing):
            f.write(title + "\n")
    print(f"[INFO] Saved {len(missing)} missing titles to {tmp_missing_file}")

    ######## 8) For missing titles, run real_batch_title_to_arxiv_id with a 3s delay per title ########
    if missing:
        print(f"[INFO] Fetching IDs for {len(missing)} missing titles, 3s delay per title...")
        df_new = real_batch_title_to_arxiv_id(missing, max_results=2, sleep_time=3, retries=3)
        print(f"[INFO] real_batch_title_to_arxiv_id returned {len(df_new)} rows.")
        new_fetched = dict(zip(df_new["title"], df_new["arxiv_id"]))
        for t, aid in new_fetched.items():
            norm_t = normalize_title(t)
            if not any(normalize_title(k) == norm_t for k in combined_dict.keys()):
                combined_dict[t] = aid
    else:
        print("[INFO] No missing titles; no additional fetch needed.")

    ######## 9) Finally, save the combined dictionary to the new JSON-based cache ########
    save_json_cache(combined_dict, new_cache_path)
    print(f"[INFO] New cache now contains {len(combined_dict)} entries total.")

if __name__ == "__main__":
    main()
