
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
tqdm.pandas()

def citation_retrieve(df, doi_col="doi", title_col="title", key="parsed_bibtex_tuple_list"):
    """
    Synchronously processes each row with search_and_fetch_info().
    Returns references, citations, and success flags for each row.
    """
    results = []
    total_success = 0
    total_failed = 0
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        parsed_bibtex_entries = row[key]
        if not ((isinstance(parsed_bibtex_entries, (list, tuple)) and parsed_bibtex_entries) or (isinstance(parsed_bibtex_entries, np.ndarray) and parsed_bibtex_entries.size)):
            print(f"[Row {idx}] ❌ No valid BibTeX entries.")
            results.append((json.dumps([]), json.dumps([]), False))
            total_failed += 1
            continue
        references, citations = [], []
        success = False
        for parsed_data in parsed_bibtex_entries:
            if isinstance(parsed_data, (list, np.ndarray, tuple)) and not parsed_data:
                continue
            if not parsed_data:
                continue
            doi = parsed_data.get("doi")
            title = parsed_data.get("title")
            # ✅ Fix: Clean title and lowercase DOI
            title = clean_title(title)
            if doi:
                doi = doi.lower().strip()
            print(f"[Row {idx}] 🔎 Searching for: DOI={doi}, Title={title}")
            try:
                info_dict = search_and_fetch_info(doi=doi, title=title)
                if info_dict is not None:
                    references = info_dict.get("references", [])
                    citations = info_dict.get("citations", [])
                    # ✅ Debugging: Print when few references/citations are found
                    if len(references) < 3 or len(citations) < 3:
                        print(f"[Row {idx}] ⚠️ Low results! Only {len(references)} references & {len(citations)} citations.")
                    print(f"[Row {idx}] ✅ Found {len(references)} references and {len(citations)} citations.")
                    success = True
                    total_success += 1
                    break
                else:
                    print(f"[Row {idx}] ⚠️ No results found.")
                    total_failed += 1
            except Exception as e:
                print(f"[Row {idx}] ❌ Error: {e}")
                total_failed += 1
        results.append((json.dumps(references), json.dumps(citations), success))
    print(f"\n✅ Total Success: {total_success}, ❌ Total Failed: {total_failed}")
    return results

def citation_retrieve_process(df, key="parsed_bibtex_tuple_list"):
    assert key in df.columns
    valid_bibtex_indices = df[
        df[key].apply(lambda x: (isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, tuple)) and len(x) > 0)
    ].index
    valid_rows = df.loc[valid_bibtex_indices].copy()
    print('length of valid rows:', len(valid_rows))
    processed_results = citation_retrieve(valid_rows, key=key)
    references_list, citations_list, success_flags = zip(*processed_results)
    valid_rows["references_within_dataset"] = references_list
    valid_rows["citations_within_dataset"] = citations_list
    valid_rows["success_flag"] = success_flags
    df.loc[valid_rows.index, "references_within_dataset"] = valid_rows["references_within_dataset"]
    df.loc[valid_rows.index, "citations_within_dataset"] = valid_rows["citations_within_dataset"]
    df.loc[valid_rows.index, "success_flag"] = valid_rows["success_flag"]
    #df.to_csv("statistics/annotated_groundtruth.csv", index=False)
    #valid_rows.to_csv("statistics/debug_results.csv", index=False)  # Save debug file
    #print("\n🔹 Annotated ground truth saved to: 'statistics/annotated_groundtruth.csv'")
    #print("🔹 Debug results saved to: 'statistics/debug_results.csv'")

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    data_type = 'modelcard'
    # Load data
    start_time = time.time()
    t1 = start_time
    print("⚠️Step 1: Loading data...")
    df_new = load_data(f"{output_dir}/{data_type}_step4.parquet", columns=['modelId', 'extracted_markdown_table_tuple', 'extracted_bibtex_tuple', 'extracted_bibtex', 'csv_path', 'parsed_bibtex_tuple_list'])
    df_makeup = load_data(f"{output_dir}/{data_type}_step3.parquet", columns=['modelId', 'downloads'])
    df = df_new.merge(df_makeup, on='modelId')
    del df_new, df_makeup
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 2: Retrieving citations...")
    start_time = time.time()
    citation_retrieve_process(df, key="parsed_bibtex_tuple_list")
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    df.to_parquet(f"{output_dir}/{data_type}_step5.parquet", index=False)
    print("Final time cost: {:.2f} seconds.".format(time.time() - t1))

if __name__ == "__main__":
    main()