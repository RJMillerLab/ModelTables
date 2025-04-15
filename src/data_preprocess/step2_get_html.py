# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-03-28
Last Modified: 2025-03-29 
Description: This script is used to get the arXiv ID for the title extracted from the PDF file.
"""
import os, re
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.data_preprocess.step2_se_url_title import extract_arxiv_id
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET

HTML_CACHE_FILE = "data/processed/arxiv_html_cache.json"  ########
HTML_FOLDER = "arxiv_fulltext_html"  ########
NEW_CACHE_PATH = "data/processed/title2arxiv_new_cache.json"

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
    print(f"[DEBUG] Final arXiv API URL: {base_url}?{'&'.join(f'{k}={v}' for k,v in params.items())}")
    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[ERROR] arXiv API request failed: {e}")
        return None

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

def fetch_ar5iv_html(arxiv_id):
    """
    Fetch HTML from ar5iv (ar5iv.labs.arxiv.org) given an arXiv ID.
    Save the HTML to a local file in folder HTML_FOLDER with filename '{arxiv_id}.html'
    and return the file path, or None on failure.
    """
    file_path = os.path.join(HTML_FOLDER, f"{arxiv_id}.html")  ########
    if os.path.exists(file_path):
        return file_path
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            html_text = resp.text
            os.makedirs(HTML_FOLDER, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_text)
            return file_path
        elif resp.status_code == 404:
            print(f"[WARN] HTML not exist: {arxiv_id}")
            return None
        else:
            print(f"[WARN] ar5iv HTML not found for {arxiv_id}, status={resp.status_code}")
            ######## NEW: if HTML not found, try to fetch the base arXiv ID (without version) ########
            base_arxiv_id = re.sub(r'v\d+$', '', arxiv_id)  ########
            if base_arxiv_id != arxiv_id:
                print(f"[INFO] Fallback: trying base arXiv ID '{base_arxiv_id}' for {arxiv_id}.")  ########
                base_file_path = os.path.join(HTML_FOLDER, f"{base_arxiv_id}.html")  ########
                if os.path.exists(base_file_path):
                    return base_file_path  ########
                base_url = f"https://ar5iv.labs.arxiv.org/html/{base_arxiv_id}"  ########
                try:
                    base_resp = requests.get(base_url, timeout=15)  ########
                    if base_resp.status_code == 200:
                        html_text = base_resp.text  ########
                        os.makedirs(HTML_FOLDER, exist_ok=True)  ########
                        with open(base_file_path, "w", encoding="utf-8") as f:  ########
                            f.write(html_text)  ########
                        return base_file_path  ########
                    else:
                        print(f"[WARN] Fallback ar5iv HTML not found for {base_arxiv_id}, status={base_resp.status_code}")  ########
                        return None  ########
                except Exception as ex:
                    print(f"[ERROR] Fallback ar5iv HTML fetch error for {base_arxiv_id}: {ex}")  ########
                    return None  ########
            return None
    except Exception as e:
        print(f"[ERROR] ar5iv HTML fetch error for {arxiv_id}: {e}")
        return None

######## NEW: Single function that (1) searches by title, (2) picks ID, (3) fetches HTML ########
def fetch_id_and_html_for_title(title, max_results=3, html_cache=None):
    """
    Given a title, query arXiv, parse the Atom feed, pick the first result's ID,
    and then fetch HTML from ar5iv.
    Immediately update the provided html_cache dict with the result (arxiv_id -> local HTML file path).
    Returns (arxiv_id, html_file_path) or (None, None) if not found.
    """
    # 1) Search
    try:
        xml_text = search_arxiv_title(title, max_results=max_results)
        entries = parse_arxiv_atom(xml_text)
        if not entries:
            print(f"[INFO] No Atom entries found for title: {title}")
            return None, None, False
    except Exception as e:
        print(f"[ERROR] Atom feed error for '{title}': {e}")
        return None, None, False

    # 2) Take the first entry as best guess
    arxiv_url = entries[0]["id"]  # e.g., "http://arxiv.org/abs/2101.12345"
    arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else None
    if not arxiv_id:
        print(f"[INFO] Could not parse ID from feed for title: {title}")
        return None, None, False

    # 3) Fetch HTML from ar5iv
    if arxiv_id in html_cache:
        html_file_path = html_cache[arxiv_id]
        if html_file_path and os.path.isfile(html_file_path):
            print(f"[INFO] HTML already cached for {arxiv_id}: {html_file_path}")
            return arxiv_id, html_file_path, False
            #file_size = os.path.getsize(html_file_path)
            #else:
            #file_size = 0
        else:
            # Path is invalid or empty, attempt to re-fetch
            print(f"[INFO] Cached HTML missing for {arxiv_id}, re-fetching...")
            html_file_path = fetch_ar5iv_html(arxiv_id)
            if html_file_path:
                html_cache[arxiv_id] = html_file_path
            else:
                html_cache[arxiv_id] = ""  # Mark as failed
            return arxiv_id, html_file_path, True
        print(f"[INFO] Already in HTML cache: {arxiv_id} (file: {html_file_path})")
    else:
        html_file_path = fetch_ar5iv_html(arxiv_id)
        if html_file_path:
            html_cache[arxiv_id] = html_file_path
            #file_size = os.path.getsize(html_file_path)
            #print(f"[INFO] Fetched HTML for {arxiv_id}, saved to {html_file_path}.")
        else:
            html_cache[arxiv_id] = ""
            print(f"[INFO] No valid HTML for {arxiv_id}. Marked empty in cache.")
    return arxiv_id, html_file_path, True

def real_batch_title_to_arxiv_id(titles, html_cache_path=HTML_CACHE_FILE):
    """
    For each title in 'titles':
      1) Query arXiv for the ID.
      2) Fetch HTML from ar5iv for that ID and save to local file.
      3) Save ID to DataFrame row, and update HTML cache with file path.
    Returns a DataFrame with columns ["title", "arxiv_id"].
    """
    html_cache = load_json_cache(html_cache_path)

    new_rows = []
    for t in titles:
        t_stripped = t.strip()
        arxiv_id, html_file_path, need_wait = fetch_id_and_html_for_title(t_stripped, max_results=3, html_cache=html_cache)
        new_rows.append((t_stripped, arxiv_id))
        ######## Save HTML cache after each fetch to avoid losing progress ########
        save_json_cache(html_cache, html_cache_path)

        ######## Log with file size if available ########
        if html_file_path and os.path.isfile(html_file_path):
            size_info = os.path.getsize(html_file_path)
        else:
            size_info = 0
        print(f"[INFO] Title='{t_stripped}' -> ID='{arxiv_id}', HTML file='{html_file_path}', size='{size_info}'")
        if need_wait:
            time.sleep(3)
    return pd.DataFrame(new_rows, columns=["title", "arxiv_id"])

def main():
    ######## 1) Load a local Parquet file with "retrieved_title" column ########
    parquet_path = "data/processed/extracted_annotations.parquet"
    df_parquet = pd.read_parquet(parquet_path)
    all_titles = set(df_parquet["retrieved_title"].dropna().unique())
    print(f"[INFO] Loaded {len(df_parquet)} rows from {parquet_path}, found {len(all_titles)} unique 'retrieved_title'.")
    ######## 2) Load the new JSON-based cache for {title -> arxiv_id} ########
    new_cache = load_json_cache(NEW_CACHE_PATH)
    if new_cache:
        print('Use the new cache instead of old parquet')
        combined_dict = new_cache
        pass
    else:
        print('No new cache found, use the old parquet')
        ######## 3) Load the original old JSON cache {url -> extracted_title} ########
        json_cache_path = "data/processed/arxiv_titles_cache.json"
        old_cache = load_json_cache(json_cache_path)  # Format: {url: title}
        ######## 4) Convert old cache to {title -> arxiv_id} using extract_arxiv_id ########
        old_title_id_dict = {}
        for url, extracted_title in old_cache.items():
            aid = extract_arxiv_id(url)  # e.g., '2101.12345'
            if aid and extracted_title:
                old_title_id_dict[extracted_title] = aid
        print(f"[INFO] Converted old cache into {len(old_title_id_dict)} (title -> arxiv_id) pairs.")
        ######## 5) Combine the two caches (old converted + new) using normalization ########
        combined_dict = dict(old_title_id_dict)  # start with new cache if exists
        for title, aid in new_cache.items():
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
    print(f"[INFO] Titles in Parquet: {len(all_titles)}")
    print(f"[INFO] Already in combined cache: {len(in_cache)}")
    print(f"[INFO] Missing (need fetch): {len(missing)}")
    #pause
    ######## 7) Save missing titles to a temporary file for manual inspection ########
    tmp_missing_file = "missing_titles_tmp.txt"
    with open(tmp_missing_file, "w", encoding="utf-8") as f:
        for title in sorted(missing):
            f.write(title + "\n")
    print(f"[INFO] Saved {len(missing)} missing titles to {tmp_missing_file}")

    ######## 8) For missing titles, run our updated real_batch_title_to_arxiv_id with NO retry ########
    html_cache = load_json_cache(HTML_CACHE_FILE)  ########
    all_known_ids = {aid for aid in combined_dict.values() if aid}
    print(f"[INFO] Checking HTML for {len(all_known_ids)} arXiv IDs...")
    for aid in all_known_ids:
        cached_path = html_cache.get(aid, "")
        if cached_path and os.path.isfile(cached_path):
            print(f"[INFO] HTML exists: {aid} -> {cached_path}")
            continue
        # Attempt to fetch HTML
        html_file_path = fetch_ar5iv_html(aid)
        if html_file_path:
            html_cache[aid] = html_file_path
            print(f"[INFO] Downloaded HTML for {aid} to {html_file_path}")
        else:
            html_cache[aid] = ""  # Mark failure
            print(f"[WARN] Failed to fetch HTML for {aid}")
    save_json_cache(html_cache, HTML_CACHE_FILE)

    if missing:
        print(f"[INFO] Fetching IDs + HTML for {len(missing)} missing titles...")
        df_new = real_batch_title_to_arxiv_id(missing)  ########
        print(f"[INFO] real_batch_title_to_arxiv_id returned {len(df_new)} rows.")
        new_fetched = dict(zip(df_new["title"], df_new["arxiv_id"]))
        for t, aid in new_fetched.items():
            norm_t = normalize_title(t)
            if aid and not any(normalize_title(k) == norm_t for k in combined_dict.keys()):
                combined_dict[t] = aid
    else:
        print("[INFO] No missing titles; no additional fetch needed.")
    ######## 9) Finally, save the combined dictionary to the new JSON-based cache ########
    save_json_cache(combined_dict, NEW_CACHE_PATH)
    print(f"[INFO] New cache now contains {len(combined_dict)} entries total.")

if __name__ == "__main__":
    main()
