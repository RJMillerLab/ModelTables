
# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from joblib import Memory

memory = Memory("./cachedir", verbose=0)

def extract_titles(bibtex_list):
    if not isinstance(bibtex_list, (list, tuple, np.ndarray)):
        return []
    return [d.get("title", "").replace("{", "").replace("}", "").lower().strip()
            for d in bibtex_list if isinstance(d, dict) and d.get("title")]

def extract_json_titles(json_input):
    if isinstance(json_input, str):
        try:
            parsed = json.loads(json_input)
        except Exception as e:
            print(f"[extract_json_titles] ❌ JSON parse error: {e}")
            return []
    elif isinstance(json_input, dict):
        parsed = json_input
    else:
        return []
    if not (isinstance(parsed, dict) and "data" in parsed):
        return []
    titles = []
    for item in parsed.get("data", []):
        if isinstance(item, dict):
            for key in ["citedPaper", "citingPaper"]:
                if key in item and isinstance(item[key], dict):
                    title = item[key].get("title")
                    if isinstance(title, str):
                        titles.append(title)
    return titles

def process_row_info(row):
    cp = row.csv_path
    titles = set(row.title) if isinstance(row.title, list) else set()
    refs = extract_json_titles(row.references_within_dataset)
    cites = extract_json_titles(row.citations_within_dataset)
    return cp, titles, set(refs+cites)

def main():
    data_type = "modelcard"
    input_file = f"data/{data_type}_step3.parquet"
    output_file = "groundtruth_associations.json"
    
    print("⚠️ Step 1: Loading data...")
    start_time = time.time()
    df = pd.read_parquet(input_file)
    print(f"✅ Data loaded. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 2: Extracting titles...")
    start_time = time.time()
    tqdm.pandas()
    df["title"] = df["parsed_bibtex_tuple_list"].apply(extract_titles)

    valid_df = df[df["csv_path"].notnull()].copy()
    keys = valid_df["csv_path"].tolist()
    res = Parallel(n_jobs=-1)(delayed(process_row_info)(row) for row in valid_df.itertuples())
    cp_list, titles_list, ref_cite_list = zip(*res)
    titles_dict = dict(zip(cp_list, titles_list))
    ref_cite_dict = dict(zip(cp_list, ref_cite_list))

    """
    titles_dict = {}
    ref_cite_dict = {}
    for _, row in df.iterrows():
        csv_path = row["csv_path"]
        if pd.isnull(csv_path):
            continue
        titles_dict[csv_path] = set([t.lower().strip() for t in row["title"]] if isinstance(row["title"], list) else [])
        refs = extract_json_titles(row["references_within_dataset"])
        cites = extract_json_titles(row["citations_within_dataset"])
        ref_cite_dict[csv_path] = set([t.lower().strip() for t in refs + cites])"""
    print(f"Step 2 Done. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 3: Processing associations in parallel...")
    #groundtruth = {row["csv_path"]: [] for _, row in df.iterrows() if pd.notnull(row["csv_path"])}
    #keys = list(groundtruth.keys())

    @memory.cache
    def process_row(i): #, titles_dict, ref_cite_dict
        cp = keys[i]
        cp_titles = titles_dict[cp]
        cp_refs = ref_cite_dict[cp]
        associations = []
        for k in keys[i + 1:]:
            if titles_dict[k] & cp_refs or cp_titles & ref_cite_dict[k]:
                associations.append((cp, k))
        return associations

    chunk_size = 1000
    all_associations = []
    with tqdm_joblib(tqdm(total=len(keys), desc="Processing associations")):
        for start in range(0, len(keys), chunk_size):
            end = min(start + chunk_size, len(keys))
            chunk_associations = Parallel(n_jobs=-1)(
                delayed(process_row)(i) for i in range(start, end)
            )
            all_associations.extend(chunk_associations)

    for associations in all_associations:
        for cp_i, cp_j in associations:
            groundtruth[cp_i].append(cp_j)
            groundtruth[cp_j].append(cp_i)
    
    print("⚠️ Step 4: Saving results...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(groundtruth, f, indent=4, ensure_ascii=False)
    print("✅ Groundtruth associations saved successfully!")

if __name__ == "__main__":
    main()
