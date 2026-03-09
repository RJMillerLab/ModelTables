"""
Author: Zhengyuan Dong
Created: 2025-04-10
Last Modified: 2026-03-09
Description:
    1. Query Semantic Scholar API to get paper details, citations, and references. (This might be slower than local database query)
    2. Save the results to parquet files.
    3. Merge the results into a final output file.
    4. Handle API rate limits and errors.
TODO: add tqdm for following steps (already tqdm for step1)
"""

import os
import json
import time
import argparse
import re
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from src.utils import to_parquet

######## HYPER PATHS & FILES (in uppercase for clarity)
prefix = "" #"_429"

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"  ######## API endpoint for search/match
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"  ######## API endpoint for batch query
CITATION_URL_TEMPLATE = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"  ######## Citation endpoint template
REFERENCE_URL_TEMPLATE = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"  ######## Reference endpoint template
BATCH_FIELDS = "corpusId,paperId,title,authors,year,venue,citations,references"  ######## Fields for batch query


load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["x-api-key"] = API_KEY

CACHE_COLUMNS = ["query_title", "retrieved_title", "paperId", "corpusId", "paper_identifier", "query_status"]
# Statuses that we will NOT retry on next run (404 = not found, success = done)
SKIP_RETRY_STATUSES = {"success", "404"}

def query_titles_to_paper(new_titles, sleep_time=1, cache_file=""):
    """
    Query /paper/search/match for each title. ALL outcomes (success or failure) are written to cache
    with a query_status column: "success", "404", "429", "no_results", "no_paper_id", "timeout",
    "request_error", "exceeded_retries".
    On next run: only retry titles with status 429, no_results, etc. Skip 404 (not found) and success.
    """
    if os.path.exists(cache_file):
        print(f"🔄 Loading cached title mapping from {cache_file}")
        df_cache = pd.read_parquet(cache_file)
        if "query_status" not in df_cache.columns:
            df_cache["query_status"] = "success"  # backward compat: old rows had paperId
    else:
        df_cache = pd.DataFrame(columns=CACHE_COLUMNS)

    # Only skip titles that are done: success or 404 (permanent failure)
    cached_done = set()
    if "query_status" in df_cache.columns:
        done_mask = df_cache["query_status"].isin(SKIP_RETRY_STATUSES)
        cached_done = set(df_cache.loc[done_mask, "query_title"].dropna().astype(str).tolist())
    else:
        cached_done = set(df_cache["query_title"].dropna().astype(str).tolist())

    titles_set = set(new_titles)
    titles_to_query = list(titles_set - cached_done)
    pending_rows = []  # Accumulate rows, flush every 100 and at end

    def _flush_pending():
        nonlocal df_cache
        if pending_rows:
            df_cache = pd.concat([df_cache, pd.DataFrame(pending_rows)], ignore_index=True)
            to_parquet(df_cache, cache_file)
            pending_rows.clear()
            print(f"💾 Saved {len(df_cache)} rows to {cache_file}")

    success_count = 0  ######## Counter for successful queries
    failure_count = 0  ######## Counter for failed queries
    if titles_to_query:
        print(f"🔍 {len(titles_to_query)} new titles to be queried.")
        # Use tqdm to create a progress bar for titles_to_query
        for i, query_title in enumerate(tqdm(titles_to_query, desc="Processing Titles")):
            if i > 0 and i % 100 == 0:
                _flush_pending()
            print(f"🔍 Searching for title: {query_title}")
            params = {
                "query": query_title,
                "fields": "paperId,corpusId,title",
                "limit": 1
            }
            max_retries = 5
            backoff = sleep_time
            retries = 0
            response = None
            while True:
                time.sleep(sleep_time)  # Space out requests to reduce 429 rate limits
                try:
                    response = requests.get(SEARCH_URL, headers=HEADERS, params=params, timeout=60)
                except requests.exceptions.Timeout:
                    print(f"❌ Timeout while searching for: {query_title}")
                    fail_row = {"query_title": query_title, "retrieved_title": None, "paperId": None, "corpusId": None, "paper_identifier": None, "query_status": "timeout"}
                    pending_rows.append(fail_row)
                    failure_count += 1
                    break
                except requests.exceptions.RequestException as e:
                    print(f"❌ Request exception while searching for: {query_title} ({e})")
                    fail_row = {"query_title": query_title, "retrieved_title": None, "paperId": None, "corpusId": None, "paper_identifier": None, "query_status": "request_error"}
                    pending_rows.append(fail_row)
                    failure_count += 1
                    break
                if response.status_code == 429:
                    if retries < max_retries:
                        print(f"⚠️ Rate limited (429), retry {retries+1}/{max_retries} after {backoff}s for: {query_title}")
                        time.sleep(backoff)
                        backoff *= 2
                        retries += 1
                        continue
                    else:
                        print(f"❌ Exceeded max retries for: {query_title}")
                        fail_row = {"query_title": query_title, "retrieved_title": None, "paperId": None, "corpusId": None, "paper_identifier": None, "query_status": "exceeded_retries"}
                        pending_rows.append(fail_row)
                        failure_count += 1
                        break
                break
            if response is None or response.status_code != 200:
                if response is not None and response.status_code not in (429,):
                    print(f"❌ HTTP error {response.status_code} while searching for: {query_title}")
                    status = "404" if response.status_code == 404 else f"http_{response.status_code}"
                    fail_row = {"query_title": query_title, "retrieved_title": None, "paperId": None, "corpusId": None, "paper_identifier": None, "query_status": status}
                    pending_rows.append(fail_row)
                    failure_count += 1
                continue
            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])
                if papers:
                    paper = papers[0]
                    paperId = paper.get("paperId")
                    corpusId = paper.get("corpusId")
                    retrieved_title = paper.get("title")
                    if paperId is not None:
                        pid = f"CorpusID:{corpusId}" if corpusId is not None else paperId
                        new_row = {
                            "query_title": query_title,
                            "retrieved_title": retrieved_title,
                            "paperId": paperId,
                            "corpusId": corpusId,
                            "paper_identifier": pid,
                            "query_status": "success"
                        }
                        pending_rows.append(new_row)
                        success_count += 1  ######## Increment success count
                        print(f"✅ For '{query_title}': paperId={paperId}, corpusId={corpusId}, retrieved_title='{retrieved_title}'")
                    else:
                        print(f"⚠️ No paperId found for title: {query_title}")
                        fail_row = {"query_title": query_title, "retrieved_title": retrieved_title, "paperId": None, "corpusId": None, "paper_identifier": None, "query_status": "no_paper_id"}
                        pending_rows.append(fail_row)
                        failure_count += 1
                else:
                    print(f"⚠️ No results for title: {query_title}")
                    fail_row = {"query_title": query_title, "retrieved_title": None, "paperId": None, "corpusId": None, "paper_identifier": None, "query_status": "no_results"}
                    pending_rows.append(fail_row)
                    failure_count += 1
        _flush_pending()  # Final save for remaining rows
        print(f"\n📊 Processing Complete: {len(titles_to_query)} titles processed, {success_count} successful, {failure_count} failed.")
    else:
        print("🔄 All titles are already in cache.")

def batch_get_details_for_ids(mapping_df, batch_size=500, sleep_time=1, timeout=60, cache_file=""):
    """
    We deprecate because this can not get full citations and references, being truncated.
    
    Use the paper_identifier column from mapping_df to batch query the /paper/batch endpoint for paper details.
    Merge the batch results with mapping_df to include the original query_title and retrieved_title.
    Each record (one per paper) will have:
      query_title, retrieved_title, paperId, corpusId, year, venue, original_response, parsed_response.
    Save the result to a parquet file.
    """
    if os.path.exists(cache_file):
        print(f"🔄 Loading cached batch results from {cache_file}")
        return pd.read_parquet(cache_file)
    
    paper_ids = mapping_df["paper_identifier"].tolist()
    results = []
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i+batch_size]
        payload = {"ids": batch}
        params = {"fields": BATCH_FIELDS}
        print(f"🔍 Batch querying {len(batch)} paper IDs (batch {i//batch_size+1})...")
        try:
            response = requests.post(BATCH_URL, headers=HEADERS, params=params, json=payload, timeout=timeout)
        except requests.exceptions.Timeout:
            print(f"❌ Timeout error on batch starting at index {i}")
            continue
        if response.status_code == 200:
            batch_data = response.json()
            results.extend(batch_data)
            print(f"✅ Batch query returned {len(batch_data)} papers.")
        else:
            print(f"❌ HTTP error {response.status_code} on batch: {response.text}")
        time.sleep(sleep_time)
    
    processed = []
    for res in results:
        if res is None or not isinstance(res, dict):
            continue
        original_response = json.dumps(res)
        citing_papers = res.get("citations", [])
        cited_papers = res.get("references", [])
        parsed_response = json.dumps({
            "citing_papers": citing_papers,
            "cited_papers": cited_papers
        })
        processed.append({
            "paperId": res.get("paperId", ""),
            "corpusId": res.get("corpusId", ""),
            "retrieved_title": res.get("title", ""),
            "year": res.get("year", ""),
            "venue": res.get("venue", ""),
            "original_response": original_response,
            "parsed_response": parsed_response
        })
    df_batch = pd.DataFrame(processed)
    merge_df = pd.merge(mapping_df, df_batch, on="paperId", how="left", suffixes=("_query", ""))
    # Keep the following columns:
    cols = ["query_title", "retrieved_title", "paperId", "corpusId", "year", "venue", "original_response", "parsed_response"]
    merge_df = merge_df[cols]
    to_parquet(merge_df, cache_file)
    print(f"💾 Batch results saved to {cache_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Semantic Scholar API for S2ORC metadata")
    parser.add_argument("--tag", dest="tag", default=None, help="Tag suffix for versioning (e.g., 251117).")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    # Tag-aware file paths (do not overwrite old global caches)
    titles_json_file = f"data/processed/modelcard_dedup_titles{suffix}.json"
    titles_cache_file = f"data/processed/s2orc_titles2ids{suffix}{prefix}.parquet"
    batch_cache_file = f"data/processed/s2orc_batch_results{suffix}{prefix}.parquet"
    citations_cache_file = f"data/processed/s2orc_citations_cache{suffix}{prefix}.parquet"
    references_cache_file = f"data/processed/s2orc_references_cache{suffix}{prefix}.parquet"

    ######## Load titles from the JSON file (tag-aware)
    print(f"🔄 Loading titles from {titles_json_file}")
    with open(titles_json_file, "r", encoding="utf-8") as f:
        TITLES = json.load(f)
    # 1. Update titles mapping and cache to parquet.
    #    Skip: success, 404. Retry: 429, no_results, etc.
    total_titles = len(TITLES)
    titles_set = set(TITLES)

    cache_exists = os.path.exists(titles_cache_file)
    if cache_exists:
        _df_cache = pd.read_parquet(titles_cache_file)
        cache_total = len(_df_cache)
        cache_valid = (_df_cache["paperId"].notna() & (_df_cache["paperId"].astype(str) != "")).sum()
        if "query_status" in _df_cache.columns:
            done_mask = _df_cache["query_status"].isin(SKIP_RETRY_STATUSES)
            cached_done = set(_df_cache.loc[done_mask, "query_title"].dropna().astype(str).tolist())
        else:
            cached_done = set(_df_cache["query_title"].dropna().astype(str).tolist())
        n_skip = len(titles_set & cached_done)
        n_to_query = len(titles_set - cached_done)
    else:
        cache_total = cache_valid = n_skip = 0
        n_to_query = total_titles

    print("\n" + "=" * 60)
    print("📋 INCREMENTAL TITLE SEARCH SUMMARY")
    print("=" * 60)
    print(f"  Cache file:        {titles_cache_file}")
    print(f"  Cache exists:      {cache_exists}")
    if cache_exists:
        print(f"  Cache total rows:  {cache_total}")
        print(f"  Cache valid rows: {cache_valid} (with paperId)")
    print(f"  Titles (JSON):     {total_titles} total")
    print(f"  Skip (success/404): {n_skip} (will not retry)")
    print(f"  To query/retry:    {n_to_query} (429, no_results, or new)")
    print("=" * 60 + "\n")

    query_titles_to_paper(TITLES, sleep_time=1.01, cache_file=titles_cache_file)
    print(f"\n💾 Titles mapping updated and saved. Total mapped rows: {len(TITLES)}")
    
    # 2. Batch query paper details and merge with titles mapping.
    #batch_df = batch_get_details_for_ids(mapping_df, batch_size=500, sleep_time=1, timeout=60, cache_file=batch_cache_file)
    #print("\n💾 Batch query results saved.")
