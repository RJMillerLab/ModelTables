"""
Author: Zhengyuan Dong
Created: 2025-04-10
Last Modified: 2026-03-09
Description:
    1. Query Semantic Scholar API to get paper details, citations, and references. (This might be slower than local database query)
    2. Save the results to parquet files.
    3. Merge the results into a final output file.
    4. Handle API rate limits and errors.
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
from src.utils import to_parquet, extract_non_empty_column_list_sql

######## HYPER PATHS & FILES (in uppercase for clarity)
prefix = "" #"_429"

CITATION_URL_TEMPLATE = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"  ######## Citation endpoint template
REFERENCE_URL_TEMPLATE = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"  ######## Reference endpoint template

# Semantic Scholar API cap: offset + limit must be < 10000. We use limit=100, so stop at offset 9900.
S2ORC_CITATION_MAX_OFFSET = 9900


load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["x-api-key"] = API_KEY

def get_single_citations_row(paper_id, sleep_time=1.5, timeout=60, merge_key="corpusId"):
    """
    Query the /paper/{paper_id}/citations endpoint for all citations using pagination.
    Each page retrieves up to 100 records. All citations are merged into a single response.
    """
    id_for_cache = str(paper_id).strip()
    if merge_key == "corpusId":
        id_for_cache = re.sub(r"\.0+$", "", id_for_cache)
        id_for_query = f"CorpusID:{id_for_cache}"
    else:
        id_for_query = id_for_cache

    url = CITATION_URL_TEMPLATE.format(paper_id=id_for_query)
    all_data = []
    offset = 0
    limit = 100
    max_retries = 5
    backoff     = sleep_time
    retries     = 0

    while True:
        if offset >= S2ORC_CITATION_MAX_OFFSET:
            print(f"⚠️ Citations cap reached for {id_for_cache} (offset {offset} >= {S2ORC_CITATION_MAX_OFFSET}), keeping {len(all_data)} items.")
            break
        params = {
            "fields": "citingPaper.title,citingPaper.abstract,contexts,intents,isInfluential",
            "limit": limit,
            "offset": offset
        }
        print(
            f"🔍 Querying citations for {merge_key}: {id_for_cache} "
            f"(api_id={id_for_query}, offset={offset}) ..."
        )
        time.sleep(sleep_time)
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        except requests.exceptions.Timeout:
            print(f"❌ Timeout error on citations query for paper_id: {paper_id}")
            return {}
        if response.status_code == 429:
            if retries < max_retries:
                print(f"⚠️ Rate limited (429), retry {retries+1}/{max_retries} after {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                retries += 1
                continue
            else:
                print("❌ Exceeded max retries, skipping this paper")
                break

        if response.status_code == 400 and "10000" in (response.text or ""):
            print(f"⚠️ API cap (offset+limit<10000) for {id_for_cache}, keeping {len(all_data)} citations.")
            break
        if response.status_code != 200:
            print(f"❌ HTTP error {response.status_code} on citations query: {response.text}")
            return {}
        # Reset retry state on successful response
        retries = 0
        backoff = sleep_time

        page = response.json().get("data", [])
        if not page:
            break
        all_data.extend(page)
        offset += limit
    return {                                                     
        merge_key: id_for_cache,
        "original_response": json.dumps({"data": all_data}),
        "parsed_response" : json.dumps({"citing_papers": all_data})
    }

def get_single_references_row(paper_id, sleep_time=1, timeout=60, merge_key="corpusId"):
    """
    Query the /paper/{paper_id}/references endpoint for all references using pagination.
    Each page retrieves up to 100 records. All references are merged into a single response.
    """
    id_for_cache = str(paper_id).strip()
    if merge_key == "corpusId":
        id_for_cache = re.sub(r"\.0+$", "", id_for_cache)
        id_for_query = f"CorpusID:{id_for_cache}"
    else:
        id_for_query = id_for_cache

    url = REFERENCE_URL_TEMPLATE.format(paper_id=id_for_query)
    all_data = []
    offset = 0
    limit = 100
    max_retries = 5
    backoff     = sleep_time
    retries     = 0
    while True:
        if offset >= S2ORC_CITATION_MAX_OFFSET:
            print(f"⚠️ References cap reached for {id_for_cache} (offset {offset} >= {S2ORC_CITATION_MAX_OFFSET}), keeping {len(all_data)} items.")
            break
        params = {
            "fields": "citedPaper.title,citedPaper.abstract,contexts,intents,isInfluential",
            "limit": limit,
            "offset": offset
        }
        print(
            f"🔍 Querying references for {merge_key}: {id_for_cache} "
            f"(api_id={id_for_query}, offset={offset}) ..."
        )
        time.sleep(sleep_time)
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        except requests.exceptions.Timeout:
            print(f"❌ Timeout error on references query for paper_id: {paper_id}")
            return {}
        if response.status_code == 429:
            if retries < max_retries:
                print(f"⚠️ Rate limited (429), retry {retries+1}/{max_retries} after {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                retries += 1
                continue
            else:
                print("❌ Exceeded max retries, skipping this paper")
                break

        if response.status_code == 400 and "10000" in (response.text or ""):
            print(f"⚠️ API cap (offset+limit<10000) for {id_for_cache}, keeping {len(all_data)} references.")
            break
        if response.status_code != 200:
            print(f"❌ HTTP error {response.status_code} on references query: {response.text}")
            return {}
        # Reset retry state on successful response
        retries = 0
        backoff = sleep_time

        page = response.json().get("data", [])
        if not page:
            break
        all_data.extend(page)
        offset += limit
    return {                                                     
        merge_key: id_for_cache,
        "original_response": json.dumps({"data": all_data}),
        "parsed_response" : json.dumps({"cited_papers": all_data})
    }

def update_all_single_citations(query_ids, sleep_time=1, timeout=60, force_refresh=False, cache_file="", merge_key="corpusId"):
    """
    For each paper_id in the list, call get_single_citations_row to update the citations cache.
    All records are saved to a single parquet file.
    
    Returns:
        A dictionary mapping paper_id to its citations record.
    """
    if not force_refresh and os.path.exists(cache_file):
        df_cache = pd.read_parquet(cache_file)
        cached_series = df_cache[merge_key].astype(str)
        if merge_key == "corpusId":
            cached_series = cached_series.str.replace(r"\.0+$", "", regex=True)
        cached_ids = set(cached_series)
    else:
        df_cache = pd.DataFrame(columns=[merge_key, "original_response", "parsed_response"])
        cached_ids = set()

    print('Paper/Corpus IDs in total:', len(query_ids))
    to_process = [pid for pid in query_ids if str(pid) not in cached_ids]
    print('Paper/Corpus IDs to process:', len(to_process))

    results = {}
    pending = 0
    for pid in tqdm(to_process, desc="Citations"):
        rec = get_single_citations_row(pid, sleep_time=sleep_time, timeout=timeout, merge_key=merge_key)
        if rec:
            df_cache = pd.concat([df_cache, pd.DataFrame([rec])], ignore_index=True)
            pending += 1
            if pending >= 100:
                to_parquet(df_cache, cache_file, verbose=True)
                pending = 0
            results[pid] = rec
    if pending > 0:
        to_parquet(df_cache, cache_file, verbose=True)

def update_all_single_references(query_ids, sleep_time=1, timeout=60, force_refresh=False, cache_file="", merge_key="corpusId"):
    """
    For each paper_id in the list, call get_single_references_row to update the references cache.
    All records are saved to a single parquet file.
    
    Returns:
        A dictionary mapping paper_id to its references record.
    """
    if not force_refresh and os.path.exists(cache_file):
        df_cache = pd.read_parquet(cache_file)
        cached_series = df_cache[merge_key].astype(str)
        if merge_key == "corpusId":
            cached_series = cached_series.str.replace(r"\.0+$", "", regex=True)
        cached_ids = set(cached_series)
    else:
        df_cache = pd.DataFrame(columns=[merge_key, "original_response", "parsed_response"])
        cached_ids = set()
    
    print('Paper/Corpus IDs in total:', len(query_ids))
    to_process = [pid for pid in query_ids if str(pid) not in cached_ids]
    print('Paper/Corpus IDs to process:', len(to_process))

    results = {}
    pending = 0
    for pid in tqdm(to_process, desc="References"):
        rec = get_single_references_row(pid, sleep_time=sleep_time, timeout=timeout, merge_key=merge_key)
        if rec:
            df_cache = pd.concat([df_cache, pd.DataFrame([rec])], ignore_index=True)
            pending += 1
            if pending >= 100:
                to_parquet(df_cache, cache_file, verbose=False)
                pending = 0
            results[pid] = rec
    if pending > 0:
        to_parquet(df_cache, cache_file, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Semantic Scholar API for S2ORC metadata")
    parser.add_argument("--tag", dest="tag", default=None, help="Tag suffix for versioning (e.g., 251117).")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    # Tag-aware file paths (do not overwrite old global caches)
    titles_cache_file = f"data/processed/s2orc_titles2ids{suffix}{prefix}.parquet"
    citations_cache_file = f"data/processed/s2orc_citations_cache{suffix}{prefix}.parquet"
    references_cache_file = f"data/processed/s2orc_references_cache{suffix}{prefix}.parquet"


    force_refresh = False
    MERGE_KEY = "corpusId"
    # 3. Update all single citations for all paperIds from mapping (only valid paperIds).
    #query_ids = mapping_df[mapping_df[MERGE_KEY].notna() & (mapping_df[MERGE_KEY].astype(str) != "")][MERGE_KEY].tolist()
    query_ids = extract_non_empty_column_list_sql(titles_cache_file, MERGE_KEY)
    print(f"🔍 Found {len(query_ids)} paper/corpus IDs in {titles_cache_file}")
    query_ids = [str(x).strip() for x in query_ids if str(x).lower() not in {"nan", "none"}]
    if MERGE_KEY == "corpusId":
        query_ids = [re.sub(r"\.0+$", "", x) for x in query_ids]
    query_ids = list(dict.fromkeys(query_ids))
    print(f"🔍 Found {len(query_ids)} valid paper/corpus IDs in {titles_cache_file}")

    # this is fixed after paper is released
    update_all_single_references(query_ids, sleep_time=1.01, timeout=60, force_refresh=force_refresh, cache_file=references_cache_file, merge_key=MERGE_KEY)
    print(f"\n💾 All single references queries have been processed and saved to {references_cache_file}.")
    # this might be updated as new papers come out
    update_all_single_citations(query_ids, sleep_time=1.01, timeout=60, force_refresh=force_refresh, cache_file=citations_cache_file, merge_key=MERGE_KEY)
    print(f"\n💾 All single citations queries have been processed and saved to {citations_cache_file}.")
    