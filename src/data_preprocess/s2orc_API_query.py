"""
Author: Zhengyuan Dong
Created: 2025-04-10
Last Modified: 2025-04-10
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
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm  ######## Imported tqdm for progress bar display

######## HYPER PATHS & FILES (in uppercase for clarity)
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"  ######## API endpoint for search/match
BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"  ######## API endpoint for batch query
CITATION_URL_TEMPLATE = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"  ######## Citation endpoint template
REFERENCE_URL_TEMPLATE = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"  ######## Reference endpoint template
BATCH_FIELDS = "corpusId,paperId,title,authors,year,venue,citations,references"  ######## Fields for batch query

# File paths
prefix = "" #"_429"
TITLES_JSON_FILE = f"data/processed/modelcard_dedup_titles{prefix}.json"  ######## Input file for titles
TITLES_CACHE_FILE = f"data/processed/s2orc_titles2ids{prefix}.parquet"  ######## Cache file for titles mapping
BATCH_CACHE_FILE = f"data/processed/s2orc_batch_results{prefix}.parquet"  ######## Cache file for batch results
CITATIONS_CACHE_FILE = f"data/processed/s2orc_citations_cache{prefix}.parquet"  ######## Cache file for citations
REFERENCES_CACHE_FILE = f"data/processed/s2orc_references_cache{prefix}.parquet"  ######## Cache file for references
MERGED_RESULTS_FILE = f"data/processed/s2orc_query_results{prefix}.parquet"  ######## Output file for merged results

load_dotenv()
API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["x-api-key"] = API_KEY

def update_titles_to_paper_ids(new_titles, sleep_time=1, cache_file=TITLES_CACHE_FILE):
    """
    Given a list of paper titles (new_titles), query the /paper/search/match endpoint to retrieve
    details for each paper. Save the following fields for each record:
      - query_title: the original input title
      - retrieved_title: the title returned by the API
      - paperId, corpusId, and paper_identifier (if corpusId is available, format as "CorpusID:<corpusId>")
    
    This function loads an existing cache (if available) from cache_file, and only queries titles that
    are not already in the cache. The new results are appended (not replacing existing records) and then
    the updated DataFrame is saved to a parquet file.
    
    Returns:
        A DataFrame containing the mapping.
    """
    if os.path.exists(cache_file):
        print(f"🔄 Loading cached title mapping from {cache_file}")
        df_cache = pd.read_parquet(cache_file)
    else:
        df_cache = pd.DataFrame(columns=["query_title", "retrieved_title", "paperId", "corpusId", "paper_identifier"])
    
    cached_titles = set(df_cache["query_title"].tolist())
    titles_to_query = list(set(new_titles) - cached_titles)
    
    new_rows = []
    success_count = 0  ######## Counter for successful queries
    failure_count = 0  ######## Counter for failed queries
    if titles_to_query:
        print(f"🔍 {len(titles_to_query)} new titles to be queried.")
        # Use tqdm to create a progress bar for titles_to_query
        for query_title in tqdm(titles_to_query, desc="Processing Titles"):
            print(f"🔍 Searching for title: {query_title}")
            params = {
                "query": query_title,
                "fields": "paperId,corpusId,title",
                "limit": 1
            }
            response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
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
                        new_rows.append({
                            "query_title": query_title,
                            "retrieved_title": retrieved_title,
                            "paperId": paperId,
                            "corpusId": corpusId,
                            "paper_identifier": pid
                        })
                        success_count += 1  ######## Increment success count
                        print(f"✅ For '{query_title}': paperId={paperId}, corpusId={corpusId}, retrieved_title='{retrieved_title}'")
                    else:
                        print(f"⚠️ No paperId found for title: {query_title}")
                        failure_count += 1  ######## Increment failure count if no paperId
                else:
                    print(f"⚠️ No results for title: {query_title}")
                    failure_count += 1  ######## Increment failure count if no results
            else:
                print(f"❌ HTTP error {response.status_code} while searching for: {query_title}")
                failure_count += 1  ######## Increment failure count for HTTP errors
            time.sleep(sleep_time)
        print(f"\n📊 Processing Complete: {len(titles_to_query)} titles processed, {success_count} successful, {failure_count} failed.")
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_cache = pd.concat([df_cache, df_new], ignore_index=True)
            df_cache.to_parquet(cache_file, index=False)
            print(f"💾 Updated title mapping saved to {cache_file} (total {len(df_cache)} records)")
    else:
        print("🔄 All titles are already in cache.")
    return df_cache

def batch_get_details_for_ids(mapping_df, batch_size=500, sleep_time=1, timeout=60, cache_file=BATCH_CACHE_FILE):
    """
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
    merge_df.to_parquet(cache_file, index=False)
    print(f"💾 Batch results saved to {cache_file}")
    return merge_df

def get_single_citations_row(paper_id, sleep_time=1, timeout=60, cache_file=CITATIONS_CACHE_FILE):
    """
    Query the /paper/{paper_id}/citations endpoint for a single paper and save the result as a single row record.
    The record includes paperId, original_response (full JSON string), and parsed_response (JSON string containing "citing_papers": [...]).
    All single citations are saved in one parquet file.
    """
    if os.path.exists(cache_file):
        df_cache = pd.read_parquet(cache_file)
        if paper_id in df_cache["paperId"].astype(str).tolist():
            print(f"🔄 Cached citations for paper_id {paper_id} found in {cache_file}")
            record = df_cache[df_cache["paperId"].astype(str) == paper_id].to_dict(orient="records")[0]
            return record
    else:
        df_cache = pd.DataFrame(columns=["paperId", "original_response", "parsed_response"])
    
    url = CITATION_URL_TEMPLATE.format(paper_id=paper_id)
    params = {
        "fields": "citingPaper.title,citingPaper.abstract,contexts,intents,isInfluential",
        "limit": 100
    }
    print(f"🔍 Querying citations for paper_id: {paper_id} ...")
    time.sleep(sleep_time)
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
    except requests.exceptions.Timeout:
        print(f"❌ Timeout error on citations query for paper_id: {paper_id}")
        return {}
    if response.status_code == 200:
        data = response.json()
        original_response = json.dumps(data)
        parsed_response = json.dumps({"citing_papers": data.get("data", [])})
        new_record = {
            "paperId": paper_id,
            "original_response": original_response,
            "parsed_response": parsed_response
        }
        df_new = pd.DataFrame([new_record])
        df_cache = pd.concat([df_cache, df_new], ignore_index=True)
        df_cache.to_parquet(cache_file, index=False)
        print(f"💾 Updated citations cache saved to {cache_file}")
        return new_record
    else:
        print(f"❌ HTTP error {response.status_code} on citations query: {response.text}")
        return {}

def get_single_references_row(paper_id, sleep_time=1, timeout=60, cache_file=REFERENCES_CACHE_FILE):
    """
    Query the /paper/{paper_id}/references endpoint for a single paper and save the result as a single row record.
    The record includes paperId, original_response, and parsed_response (JSON string containing "cited_papers": [...]).
    All single references are saved in one parquet file.
    """
    if os.path.exists(cache_file):
        df_cache = pd.read_parquet(cache_file)
        if paper_id in df_cache["paperId"].astype(str).tolist():
            print(f"🔄 Cached references for paper_id {paper_id} found in {cache_file}")
            record = df_cache[df_cache["paperId"].astype(str) == paper_id].to_dict(orient="records")[0]
            return record
    else:
        df_cache = pd.DataFrame(columns=["paperId", "original_response", "parsed_response"])
    
    url = REFERENCE_URL_TEMPLATE.format(paper_id=paper_id)
    params = {
        "fields": "citedPaper.title,citedPaper.abstract,contexts,intents,isInfluential",
        "limit": 100
    }
    print(f"🔍 Querying references for paper_id: {paper_id} ...")
    time.sleep(sleep_time)
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
    except requests.exceptions.Timeout:
        print(f"❌ Timeout error on references query for paper_id: {paper_id}")
        return {}
    if response.status_code == 200:
        data = response.json()
        original_response = json.dumps(data)
        parsed_response = json.dumps({"cited_papers": data.get("data", [])})
        new_record = {
            "paperId": paper_id,
            "original_response": original_response,
            "parsed_response": parsed_response
        }
        df_new = pd.DataFrame([new_record])
        df_cache = pd.concat([df_cache, df_new], ignore_index=True)
        df_cache.to_parquet(cache_file, index=False)
        print(f"💾 Updated references cache saved to {cache_file}")
        return new_record
    else:
        print(f"❌ HTTP error {response.status_code} on references query: {response.text}")
        return {}

def update_all_single_citations(paper_ids, sleep_time=1, timeout=60):
    """
    For each paper_id in the list, call get_single_citations_row to update the citations cache.
    All records are saved to a single parquet file (CITATIONS_CACHE_FILE).
    
    Returns:
        A dictionary mapping paper_id to its citations record.
    """
    results = {}
    for pid in paper_ids:
        record = get_single_citations_row(pid, sleep_time=sleep_time, timeout=timeout)
        results[pid] = record
    return results

def update_all_single_references(paper_ids, sleep_time=1, timeout=60):
    """
    For each paper_id in the list, call get_single_references_row to update the references cache.
    All records are saved to a single parquet file (REFERENCES_CACHE_FILE).
    
    Returns:
        A dictionary mapping paper_id to its references record.
    """
    results = {}
    for pid in paper_ids:
        record = get_single_references_row(pid, sleep_time=sleep_time, timeout=timeout)
        results[pid] = record
    return results

def merge_all_results(titles_cache=TITLES_CACHE_FILE,
                      citations_cache=CITATIONS_CACHE_FILE,
                      references_cache=REFERENCES_CACHE_FILE,
                      output_file=MERGED_RESULTS_FILE):
    """
    Merge the titles mapping, single citations, and single references parquet files into one consolidated parquet.
    The merge is performed by paperId. The columns from the citations data are renamed with suffix _citations,
    and those from references are renamed with suffix _references.
    
    The final merged DataFrame contains:
      - query_title, retrieved_title, paperId, corpusId (from titles mapping)
      - original_response and parsed_response from citations (with suffix _citations)
      - original_response and parsed_response from references (with suffix _references)
    The merged result is saved to output_file.
    """
    if not os.path.exists(titles_cache):
        print("Titles cache not found.")
        return
    df_titles = pd.read_parquet(titles_cache)
    
    if os.path.exists(citations_cache):
        df_citations = pd.read_parquet(citations_cache)
        # Rename columns with _citations suffix (except paperId)
        df_citations = df_citations.rename(columns={
            "original_response": "original_response_citations",
            "parsed_response": "parsed_response_citations"
        })
    else:
        df_citations = pd.DataFrame(columns=["paperId", "original_response_citations", "parsed_response_citations"])
    
    if os.path.exists(references_cache):
        df_references = pd.read_parquet(references_cache)
        # Rename columns with _references suffix (except paperId)
        df_references = df_references.rename(columns={
            "original_response": "original_response_references",
            "parsed_response": "parsed_response_references"
        })
    else:
        df_references = pd.DataFrame(columns=["paperId", "original_response_references", "parsed_response_references"])
    
    # Merge titles with citations and references using left join on paperId
    df_merged = pd.merge(df_titles, df_citations, on="paperId", how="left")
    df_merged = pd.merge(df_merged, df_references, on="paperId", how="left")
    
    df_merged.to_parquet(output_file, index=False)
    print(f"💾 Merged results saved to {output_file}")
    return df_merged

if __name__ == "__main__":
    ######## Load titles from the JSON file defined in TITLES_JSON_FILE
    if os.path.exists(TITLES_JSON_FILE):
        with open(TITLES_JSON_FILE, "r", encoding="utf-8") as f:
            TITLES = json.load(f)
    else:
        print(f"❌ Titles file {TITLES_JSON_FILE} does not exist.")
        TITLES = []
    ######## End of loading titles
    
    # 1. Update titles mapping and cache to parquet.
    mapping_df = update_titles_to_paper_ids(TITLES, sleep_time=1, cache_file=TITLES_CACHE_FILE)
    print("\n💾 Titles mapping updated and saved.")
    
    # 2. Batch query paper details and merge with titles mapping.
    #batch_df = batch_get_details_for_ids(mapping_df, batch_size=500, sleep_time=1, timeout=60, cache_file=BATCH_CACHE_FILE)
    #print("\n💾 Batch query results saved.")
    
    # 3. Update all single citations for all paperIds from mapping.
    paper_ids = mapping_df["paperId"].tolist()
    update_all_single_citations(paper_ids, sleep_time=1, timeout=60)
    print(f"\n💾 All single citations queries have been processed and saved to {CITATIONS_CACHE_FILE}.")
    
    # 4. Update all single references for all paperIds from mapping.
    update_all_single_references(paper_ids, sleep_time=1, timeout=60)
    print(f"\n💾 All single references queries have been processed and saved to {REFERENCES_CACHE_FILE}.")
    
    # 5. Merge all caches into one consolidated parquet file.
    merged_df = merge_all_results(titles_cache=TITLES_CACHE_FILE,
                                  citations_cache=CITATIONS_CACHE_FILE,
                                  references_cache=REFERENCES_CACHE_FILE,
                                  output_file=MERGED_RESULTS_FILE)
    print("\n💾 Merge process complete.")
