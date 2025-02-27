
import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from src.data_ingestion.readme_parser import BibTeXExtractor, MarkdownHandler
from src.data_ingestion.bibtex_parser import BibTeXFactory
from src.data_ingestion.citation_fetcher import search_and_fetch_info
import os, re, time, json
import asyncio
from src.utils import load_config, load_data, get_statistics_table, clean_title

tqdm.pandas()

def get_cache_key(parsed_data):
    #doi = parsed_data.get("doi", "").lower().strip()
    #if doi:
    #    return f"doi:{doi}"
    title = parsed_data.get("title", "")
    cleaned_title = clean_title(title) if title else ""
    if cleaned_title:
        return f"title:{clean_title}"
    return None
      
def empty_result():
    return {"paper_id": "", "info": {}, "references": [], "citations": []}
def format_result(info_dict):
    return {
        "paper_id": info_dict.get("paper_id", ""),
        "info": info_dict.get("info", {}),
        "references": info_dict.get("references", []),
        "citations": info_dict.get("citations", [])
    }

async def citation_retrieve(df, key="parsed_bibtex_tuple_list"):
    async def process_row(idx, row, cache):
        model_id = row["modelId"]
        parsed_bibtex_entries = row[key]
        all_results = []
        if not isinstance(parsed_bibtex_entries, (list, np.ndarray, tuple)):
            print(f"[Row {idx}] ❌ Invalid BibTeX type for model {model_id}")
            return all_results
        for i, parsed_data in enumerate(parsed_bibtex_entries):
            if not parsed_data:
                all_results.append({"paper_id": "", "info": {}, "references": [], "citations": []})
                continue
            cache_key = get_cache_key(parsed_data)
            doi = parsed_data.get("doi", "").lower().strip()
            title = clean_title(parsed_data.get("title", "")) if parsed_data.get("title") else ""

            if cache_key and cache_key in cache:
                result = cache[cache_key]
                #result = format_result(info_dict)
                all_results.append(result)
                continue
            try:
                info_dict = await asyncio.to_thread(search_and_fetch_info, doi=doi, title=title)
                if not info_dict:
                    print(f"[Row {idx}] ⚠️ No results for model {model_id} | BibTeX {i+1} | Title: {title}")
                    result = empty_result()
                else:
                    result = format_result(info_dict)
                if cache_key:
                    cache[cache_key] = info_dict
                all_results.append(result)
            except Exception as e:
                print(f"[Row {idx}] ❌ Error in model {model_id} | BibTeX {i+1} | Title: {title} | Error: {str(e)}")
                all_results.append(empty_result())
        return all_results
    cache = {}
    tasks = [process_row(idx, row, cache) for idx, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    total_success = sum(1 for _, _, success in results if success)
    total_failed = len(results) - total_success
    print(f"\n✅ Total Success: {total_success}, ❌ Total Failed: {total_failed}")
    return results

def citation_retrieve_process(df, key="parsed_bibtex_tuple_list"):
    assert key in df.columns
    valid_mask = df[key].apply(lambda x: isinstance(x, (list, np.ndarray, tuple)) and len(x) > 0)
    valid_rows = df[valid_mask].copy()
    print('length of valid rows:', len(valid_rows))
    processed_results = asyncio.run(citation_retrieve(valid_rows, key=key))

    valid_rows = valid_rows.assign(
        paper_ids=[x["paper_id"] for res in processed_results for x in res],
        infos=[json.dumps(x["info"]) for res in processed_results for x in res],
        references_within_dataset=[json.dumps(x["references"]) for res in processed_results for x in res],
        citations_within_dataset=[json.dumps(x["citations"]) for res in processed_results for x in res]
    )
    df.update(valid_rows)
    print('New attributes added: ["paper_id", "info", "references_within_dataset", "citations_within_dataset"].')

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'

    start_time = time.time()
    t1 = start_time
    print("⚠️Step 1: Loading data...")
    load_path=os.path.join(processed_base_path, f"{data_type}_step3.parquet")
    print('load_path:', load_path)
    df_new = load_data(load_path, columns=['modelId', 'parsed_bibtex_tuple_list'])
    df_makeup = load_data(os.path.join(processed_base_path, f"{data_type}_step2.parquet"), columns=['modelId', 'downloads'])
    df = df_new.merge(df_makeup, on='modelId')
    del df_new, df_makeup
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 2: Retrieving citations...")
    start_time = time.time()
    citation_retrieve_process(df, key="parsed_bibtex_tuple_list")
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    df.to_parquet(os.path.join(processed_base_path, f"{data_type}_step4.parquet"), index=False)
    print("Final time cost: {:.2f} seconds.".format(time.time() - t1))

if __name__ == "__main__":
    main()