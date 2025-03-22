import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
######## Added new import for os, re, time, json, asyncio
import os, re, time, json, asyncio
########
from src.data_ingestion.readme_parser import BibTeXExtractor, MarkdownHandler
from src.data_ingestion.bibtex_parser import BibTeXFactory
from src.data_ingestion.citation_fetcher import search_and_fetch_info
######## The original imports of os, re, time, json, asyncio are now merged above
from src.utils import load_config, load_data, get_statistics_table, clean_title

tqdm.pandas()

CITATION_CACHE_PATH = "data/processed/citation_cache.json"

def load_cache(file_path):
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_cache(data, file_path, mode="update"):
    if mode == "overwrite":
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif mode == "update":
        existing = load_cache(file_path)
        existing.update(data)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

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
        "citations": info_dict.get("citations", [])
    }

def gather_all_titles(df, key="parsed_bibtex_tuple_list"):
    """
    Extract titles from the specified column in the DataFrame (the column may be a string or list),
    and return a mapping: {cleaned_title: [(row_idx, entry_idx), ...]} for later batch query and mapping.
    """
    title_map = {}
    for idx, row in df.iterrows():
        entries = row.get(key, None)
        if not entries:
            continue
        if isinstance(entries, str):
            entries = [entries]
        for e_i, parsed_data in enumerate(entries):
            if not parsed_data or not isinstance(parsed_data, dict):
                continue
            raw_title = parsed_data.get("title", "")
            if not raw_title.strip():
                continue
            ckey = get_cache_key_from_title(raw_title)
            if ckey:
                if ckey not in title_map:
                    title_map[ckey] = []
                title_map[ckey].append((idx, e_i))
    return title_map

async def fetch_citations_for_titles(title_list, cache):
    """
    For a given batch of deduplicated titles, call search_and_fetch_info for each title.
    Skip titles that already exist in the cache.
    """
    tasks = []
    for t in title_list:
        if t in cache:
            continue
        tasks.append((t, asyncio.to_thread(search_and_fetch_info, doi=None, title=t)))
    results = await asyncio.gather(*(fut for (_, fut) in tasks), return_exceptions=True)

    for i, (title_str, _) in enumerate(tasks):
        r = results[i]
        if isinstance(r, Exception):
            print(f"[fetch_citations_for_titles] Error fetching title='{title_str}' => {r}")
            cache[title_str] = empty_result()
        elif not r:
            cache[title_str] = empty_result()
        else:
            cache[title_str] = format_result(r)

def map_citations_back(df, title_map, cache, key="parsed_bibtex_tuple_list"):
    """
    For each row in the DataFrame and each bibtex entry, retrieve the corresponding cached citation info,
    then update the DataFrame with lists of citation results.
    """
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

async def citation_retrieve_async(df, key="parsed_bibtex_tuple_list"):
    title_map = gather_all_titles(df, key=key)
    print(f"[citation_retrieve_async] There are {len(title_map)} unique titles to process.")
    cache = load_cache(CITATION_CACHE_PATH)
    for k, v in list(cache.items()):
        if not v:
            cache.pop(k, None)
    new_titles = [t for t in title_map.keys() if t not in cache]
    if new_titles:
        print(f"[citation_retrieve_async] Preparing to query {len(new_titles)} titles...")
        await fetch_citations_for_titles(new_titles, cache)
        save_cache(cache, CITATION_CACHE_PATH, mode="update")
    else:
        print("[citation_retrieve_async] All titles are cached; skipping queries.")
    map_citations_back(df, title_map, cache, key=key)

def citation_retrieve_process(df, key="parsed_bibtex_tuple_list"):
    assert key in df.columns
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

    start_time = time.time()
    t1 = start_time
    print("⚠️Step 1: Loading data...")
    load_path = os.path.join(processed_base_path, f"{data_type}_step1.parquet")
    print('load_path:', load_path)
    df_step1 = load_data(load_path, columns=['modelId', 'parsed_bibtex_tuple_list', 'downloads'])
    ext_title_path = os.path.join(processed_base_path, f"{data_type}_ext_title.parquet")
    df_ext = pd.read_parquet(ext_title_path)
    df = pd.merge(df_step1, df_ext, on="modelId", how="left")
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️Step 2: Retrieving citations (with caching, deduplicated titles)...")
    start_time = time.time()
    def merge_bibtex(row):
        list1 = row.get("parsed_bibtex_tuple_list", [])
        list2 = row.get("parsed_bibtex_tuple_list_github", [])
        if not isinstance(list1, list):
            list1 = []
        if not isinstance(list2, list):
            list2 = []
        return list1 + list2
    df["parsed_bibtex_tuple_list_all"] = df.apply(merge_bibtex, axis=1)
    citation_retrieve_process(df, key="parsed_bibtex_tuple_list_all")
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    df.to_parquet(os.path.join(processed_base_path, f"{data_type}_citation_API.parquet"), index=False)
    print("Final time cost: {:.2f} seconds.".format(time.time() - t1))

if __name__ == "__main__":
    main()
