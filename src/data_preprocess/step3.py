
import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from data_ingestion.readme_parser import BibTeXExtractor, MarkdownHandler
from data_ingestion.bibtex_parser import BibTeXFactory
from data_ingestion.citation_fetcher import search_and_fetch_info
import os, re, time, json
from utils import load_data, get_statistics_table, clean_title
import asyncio
tqdm.pandas()

async def citation_retrieve(df, key="parsed_bibtex_tuple_list"):
    async def process_row(idx, row):
        parsed_bibtex_entries = row[key]
        if not ((isinstance(parsed_bibtex_entries, (list, tuple)) and parsed_bibtex_entries) or
                (isinstance(parsed_bibtex_entries, np.ndarray) and parsed_bibtex_entries.size)):
            print(f"[Row {idx}] ❌ No valid BibTeX entries.")
            return (json.dumps([]), json.dumps([]), False)
        for parsed_data in parsed_bibtex_entries:
            if isinstance(parsed_data, (list, np.ndarray, tuple)) and not parsed_data:
                continue
            if not parsed_data:
                continue
            doi = parsed_data.get("doi")
            title = parsed_data.get("title")
            title = clean_title(title)
            if doi:
                doi = doi.lower().strip()
            print(f"[Row {idx}] 🔎 Searching for: DOI={doi}, Title={title}")
            try:
                # Run sync search_and_fetch_info in a thread
                info_dict = await asyncio.to_thread(search_and_fetch_info, doi=doi, title=title)
                if info_dict is not None:
                    info = info_dict.get("info", {})
                    references = info_dict.get("references", [])
                    citations = info_dict.get("citations", [])
                    if len(references) < 3 or len(citations) < 3:
                        print(f"[Row {idx}] ⚠️ Low results! Only {len(references)} references & {len(citations)} citations.")
                    print(f"[Row {idx}] ✅ Found {len(references)} references and {len(citations)} citations.")
                    return (json.dumps(references), json.dumps(citations), True)
                else:
                    print(f"[Row {idx}] ⚠️ No results found.")
            except Exception as e:
                print(f"[Row {idx}] ❌ Error: {e}")
        return (json.dumps([]), json.dumps([]), False)

    tasks = [process_row(idx, row) for idx, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    total_success = sum(1 for _, _, success in results if success)
    total_failed = len(results) - total_success
    print(f"\n✅ Total Success: {total_success}, ❌ Total Failed: {total_failed}")
    return results

def citation_retrieve_process(df, key="parsed_bibtex_tuple_list"):
    assert key in df.columns
    valid_bibtex_indices = df[
        df[key].apply(lambda x: isinstance(x, (list, np.ndarray, tuple)) and len(x) > 0)
    ].index
    valid_rows = df.loc[valid_bibtex_indices].copy()
    print('length of valid rows:', len(valid_rows))
    processed_results = asyncio.run(citation_retrieve(valid_rows, key=key))
    references_list, citations_list, success_flags = zip(*processed_results)
    valid_rows["references_within_dataset"] = references_list
    valid_rows["citations_within_dataset"] = citations_list
    valid_rows["success_flag"] = success_flags
    df.loc[valid_rows.index, "references_within_dataset"] = valid_rows["references_within_dataset"]
    df.loc[valid_rows.index, "citations_within_dataset"] = valid_rows["citations_within_dataset"]
    df.loc[valid_rows.index, "success_flag"] = valid_rows["success_flag"]

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    data_type = 'modelcard'
    # Load data
    start_time = time.time()
    t1 = start_time
    print("⚠️Step 1: Loading data...")
    df_new = load_data(f"{output_dir}/{data_type}_step2.parquet", columns=['modelId', 'extracted_markdown_table_tuple', 'extracted_bibtex_tuple', 'extracted_bibtex', 'csv_path', 'parsed_bibtex_tuple_list'])
    df_makeup = load_data(f"{output_dir}/{data_type}_step3.parquet", columns=['modelId', 'downloads'])
    df = df_new.merge(df_makeup, on='modelId')
    del df_new, df_makeup
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 2: Retrieving citations...")
    start_time = time.time()
    citation_retrieve_process(df, key="parsed_bibtex_tuple_list")
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    df.to_parquet(f"{output_dir}/{data_type}_step3.parquet", index=False)
    print("Final time cost: {:.2f} seconds.".format(time.time() - t1))

if __name__ == "__main__":
    main()