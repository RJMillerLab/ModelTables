import os
import json
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_titles(bibtex_list):
    """Extract clean titles from a list of BibTeX entries."""
    if not isinstance(bibtex_list, (list, tuple, np.ndarray)):
        return []
    titles = []
    for entry in bibtex_list:
        if isinstance(entry, dict):
            raw_title = entry.get("title")
            if raw_title:
                clean_title = raw_title.replace("{", "").replace("}", "")
                titles.append(clean_title)
    return titles

def extract_json_titles(json_input):
    """Extract titles from references and citations stored in JSON format."""
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

def process_row(i, keys, titles_dict, ref_cite_dict):
    """Check for citation associations between different csv_path entries."""
    cp_i = keys[i]
    associations = []
    print(f"[Process Row] 🔍 Checking index {i}: {cp_i}")
    for j in range(i + 1, len(keys)):
        cp_j = keys[j]
        found = any(t in ref_cite_dict.get(cp_j, []) for t in titles_dict.get(cp_i, []))
        if not found:
            found = any(t in ref_cite_dict.get(cp_i, []) for t in titles_dict.get(cp_j, []))
        if found:
            associations.append((cp_i, cp_j))
    print(f"[Process Row] ✅ Completed index {i}: Found {len(associations)} associations.")
    return associations

def main():
    data_type = "modelcard"
    input_file = f"data/{data_type}_step3.parquet"
    output_file = "groundtruth_associations.json"
    
    print("⚠️ Step 1: Loading data...")
    start_time = time.time()
    df = pd.read_parquet(input_file)
    print(f"✅ Data loaded. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 2: Extracting titles...")
    df["title"] = df["parsed_bibtex_tuple_list"].apply(extract_titles)
    
    groundtruth = {row["csv_path"]: [] for _, row in df.iterrows() if pd.notnull(row["csv_path"])}
    
    titles_dict = {}
    ref_cite_dict = {}
    for _, row in df.iterrows():
        csv_path = row["csv_path"]
        if pd.isnull(csv_path):
            continue
        titles_dict[csv_path] = [t.lower().strip() for t in row["title"]] if isinstance(row["title"], list) else []
        refs = extract_json_titles(row["references_within_dataset"])
        cites = extract_json_titles(row["citations_within_dataset"])
        ref_cite_dict[csv_path] = [t.lower().strip() for t in refs + cites]
    
    print("⚠️ Step 3: Processing associations in parallel...")
    all_associations = []
    keys = list(groundtruth.keys())
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_row, i, keys, titles_dict, ref_cite_dict) for i in range(len(keys))]
        for completed, fut in enumerate(as_completed(futures), 1):
            all_associations.extend(fut.result())
            print(f"[Parallel] Completed {completed}/{len(keys)} rows.")
    
    for cp_i, cp_j in all_associations:
        groundtruth[cp_i].append(cp_j)
        groundtruth[cp_j].append(cp_i)
    
    print("⚠️ Step 4: Saving results...")
    with open(output_file, "w") as f:
        json.dump(groundtruth, f, indent=4, ensure_ascii=False)
    print("✅ Groundtruth associations saved successfully!")

if __name__ == "__main__":
    main()
