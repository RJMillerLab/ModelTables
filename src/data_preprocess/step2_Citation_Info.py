import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os, re, time, json, asyncio
from src.data_ingestion.readme_parser import BibTeXExtractor, MarkdownHandler
from src.data_ingestion.bibtex_parser import BibTeXFactory
from src.data_ingestion.citation_fetcher import search_and_fetch_info
from src.utils import load_config, get_statistics_table, clean_title, to_parquet
from src.data_preprocess.step2_arxiv_github_title import load_cache, save_cache

tqdm.pandas()

CITATION_CACHE_PATH = "data/processed/citation_cache.json"

def extract_json_titles(json_input):
    if isinstance(json_input, str):
        try:
            parsed = json.loads(json_input)
        except Exception:
            return []
    elif isinstance(json_input, dict):
        parsed = json_input
    else:
        return []
    if not (isinstance(parsed, dict) and "data" in parsed):
        return []
    titles = []
    for item in parsed["data"]:
        if isinstance(item, dict):
            for key in ["citedPaper", "citingPaper"]:
                if key in item and isinstance(item[key], dict):
                    title = item[key].get("title")
                    if isinstance(title, str):
                        titles.append(
                            title.replace("{", "")
                                 .replace("}", "")
                                 .lower()
                                 .strip()
                        )
    return titles

def get_cache_key_from_title(title: str) -> str:
    cleaned = clean_title(title) if title else ""
    return cleaned

def empty_result():
    return {"paper_id": "", "info": {}, "references": [], "citations": []}

def format_result(info_dict):
    return {
        "paper_id": info_dict.get("paper_id", ""),
        "info": info_dict.get("info", {}),
        "references": info_dict.get("references", []),
        "citations": info_dict.get("citations", []),
    }

def gather_unique_titles(df, col="all_bibtex_titles"):
    title_map = {}
    for idx, row in df.iterrows():
        titles = row.get(col, [])
        if not titles:
            continue
        for t in titles:
            if not t.strip():
                continue
            ckey = get_cache_key_from_title(t)
            if ckey:
                if ckey not in title_map:
                    title_map[ckey] = []
                title_map[ckey].append(idx)
    return title_map

def map_citations_back_all(df, title_map, cache):
    row_results = [[] for _ in range(len(df))]
    for title_str, indices in title_map.items():
        citation_info = cache.get(title_str, empty_result())
        for idx in indices:
            row_results[idx].append(citation_info)
    df["citation_results"] = row_results

######## In fetch_citations_for_titles: add logging for each query and its result
async def fetch_citations_for_titles(title_list, cache, rate_limit=True, delay_seconds=1.0):  ######## updated
    """
    Asynchronously query deduplicated titles and update the cache with results.
    Optional rate limiting (default 1s per query) for API compliance.
    """
    for t in title_list:
        if t in cache:
            continue
        print(f"######## Querying title: {t}")
        try:
            result = await asyncio.to_thread(search_and_fetch_info, doi=None, title=t)
            print(f"######## Query result for title '{t}': {result}")
            if result:
                cache[t] = format_result(result)
            else:
                cache[t] = empty_result()
        except Exception as e:
            print(f"[fetch_citations_for_titles] Error fetching title='{t}' => {e}")
            cache[t] = empty_result()
        if rate_limit:
            await asyncio.sleep(delay_seconds)  ######## control delay between queries

def map_citations_back(df, title_map, cache, key="all_bibtex_titles"):
    row_results = [[] for _ in range(len(df))]
    for title_str, positions in title_map.items():
        citation_info = cache.get(title_str, empty_result())
        for (ridx, eidx) in positions:
            row_results[ridx].append(citation_info)
    paper_ids_list = []
    infos_list = []
    refs_list = []
    cits_list = []
    for per_row in row_results:
        pids = [x["paper_id"] for x in per_row]
        infs = [x["info"] for x in per_row]
        refs = [x["references"] for x in per_row]
        cits = [x["citations"] for x in per_row]
        paper_ids_list.append(pids)
        infos_list.append(json.dumps(infs))
        refs_list.append(json.dumps(refs))
        cits_list.append(json.dumps(cits))
    df["paper_ids"] = paper_ids_list
    df["infos"] = infos_list
    df["references_within_dataset"] = refs_list
    df["citations_within_dataset"] = cits_list

def citation_retrieve_async(df, key="all_bibtex_titles"):
    title_map = gather_unique_titles(df, col=key)
    print(f"[citation_retrieve_async] There are {len(title_map)} unique titles to process.")
    cache = load_cache(CITATION_CACHE_PATH)
    for k, v in list(cache.items()):
        if not v:
            cache.pop(k, None)
    new_titles = [t for t in title_map.keys() if t not in cache]
    if new_titles:
        print(f"[citation_retrieve_async] Preparing to query {len(new_titles)} titles...")
        asyncio.run(fetch_citations_for_titles(new_titles, cache))
        save_cache(cache, CITATION_CACHE_PATH, mode="update")
    else:
        print("[citation_retrieve_async] All titles are cached; skipping queries.")
    map_citations_back(df, title_map, cache, key=key)

def citation_retrieve_process(df, key="all_bibtex_titles"):
    assert key in df.columns, f"Column '{key}' does not exist in DataFrame"
    valid_mask = df[key].apply(lambda x: bool(x) and x is not None)
    valid_rows = df[valid_mask].copy()
    print('Number of valid rows:', len(valid_rows))
    asyncio.run(citation_retrieve_async(valid_rows, key=key))
    df.update(valid_rows)
    print('New attributes added: ["paper_ids", "infos", "references_within_dataset", "citations_within_dataset"].')

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    df = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_all_title_list.parquet"))
    ######## Step 3.1: Gather all deduplicated titles from df["all_title_list"] ########
    start_time = time.time()
    t1 = start_time
    print("⚠️ Step 3.1: Gathering deduplicated titles from 'all_title_list' ...")
    title_map_all = gather_unique_titles(df, col="all_title_list")
    dedup_titles = list(title_map_all.keys())
    print(f"    --> Found {len(dedup_titles)} unique titles in total.")
    ######## In main(): after gathering deduplicated titles (Step 3.1), save intermediate results for inspection
    print("    --> Found {} unique titles in total.".format(len(dedup_titles)))
    with open("tmp_dedup_titles.json", "w", encoding="utf-8") as f:
        json.dump(dedup_titles, f, ensure_ascii=False, indent=2)
    with open("tmp_title_map_all.json", "w", encoding="utf-8") as f:
        json.dump(title_map_all, f, ensure_ascii=False, indent=2)
    ######## Step 3.2: Query new titles and update cache ########
    print("⚠️ Step 3.2: Querying new titles (if not in cache)...")
    cache = load_cache(CITATION_CACHE_PATH)
    for k, v in list(cache.items()):
        if not v:
            cache.pop(k, None)
    new_titles = [t for t in dedup_titles if t not in cache]
    print(f"    --> Number of new titles to query: {len(new_titles)}")
    if new_titles:
        use_rate_limit = True       # Set to False if querying local DB
        delay_between_queries = 1.0  # 1 second per query (adjust as needed)
        asyncio.run(fetch_citations_for_titles(new_titles, cache, rate_limit=use_rate_limit, delay_seconds=delay_between_queries))
        save_cache(cache, CITATION_CACHE_PATH, mode="update")
    else:
        print("    --> All titles already in cache, skip querying.")
    print("    --> Done fetching. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    ######## In main(): after querying new titles (Step 3.2), save the updated cache for inspection
    print("    --> Done fetching. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    with open("tmp_cache_after_query.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    ######## Step 3.3: Map results back to each row in df ########
    print("⚠️ Step 3.3: Mapping query results back to df ...")
    map_citations_back_all(df, title_map_all, cache)
    print("    --> Done mapping. Created column 'citation_results'.")
    ########
    print("⚠️ Step 4: Parsing references & citations JSON -> 'cited_paper_list', 'citing_paper_list'")
    if "references_within_dataset" in df.columns and "citations_within_dataset" in df.columns:
        df["cited_paper_list"] = df["references_within_dataset"].progress_apply(extract_json_titles)
        df["citing_paper_list"] = df["citations_within_dataset"].progress_apply(extract_json_titles)
    out_path = os.path.join(processed_base_path, f"{data_type}_citation_API.parquet")
    df.drop(columns=['card_tags', 'downloads'], inplace=True, errors='ignore')
    to_parquet(df, out_path)
    print(f"✨ Final output saved to: {out_path}")
    print("Total time cost: {:.2f} seconds.".format(time.time() - t1))

if __name__ == "__main__":
    main()

