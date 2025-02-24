# -*- coding: utf-8 -*-
import time, os, json, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

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
                        titles.append(title.replace("{", "").replace("}", "").lower().strip())
    return titles

def main():
    data_type = "modelcard"
    input_file = f"data/{data_type}_step3.parquet"
    
    print("⚠️ Step 1: Loading data...")
    start_time = time.time()
    df = pd.read_parquet(input_file)
    print(f"✅ Data loaded. Time cost: {time.time() - start_time:.2f} seconds.")
    
    print("⚠️ Step 2: Extracting titles and processing new columns...")
    start_time = time.time()
    tqdm.pandas()
    df["title_list"] = df["parsed_bibtex_tuple_list"].progress_apply(extract_titles)
    df["cited_paper_list"] = df["references_within_dataset"].progress_apply(extract_json_titles)
    df["citing_paper_list"] = df["citations_within_dataset"].progress_apply(extract_json_titles)

    print("⚠️ Step 3: Building title to paper index mapping...")
    # Replaced iteritems() with items() since iteritems() is no longer available in newer Pandas versions.
    title_to_paper_indices = defaultdict(set)
    for idx, titles in df["title_list"].items():
        for title in titles:
            title_to_paper_indices[title].add(idx)
    print(f"✅ Title mapping built. Time cost: {time.time() - start_time:.2f} seconds.")

    print("⚠️ Step 4: Finding associations...")
    groundtruth = defaultdict(list)
    valid_rows = df[(df['title_list'].str.len() > 0) | 
                (df['cited_paper_list'].str.len() > 0) | 
                (df['citing_paper_list'].str.len() > 0)]
    print('total: ', len(valid_rows[valid_rows["csv_path"].notnull()]))
    #print(len(valid_rows & df["csv_path"].notnull()))

    for idx, row in tqdm(valid_rows.iterrows(), total=valid_rows.shape[0]):
        overlaps = defaultdict(int)
        ref_cite_titles = set(row.get('cited_paper_list', []) + row.get('citing_paper_list', []))
        for title in ref_cite_titles: # for each related paper
            candidate_indices = title_to_paper_indices.get(title, set()) # rows indices with the same title of related papers
            for candidate_idx in candidate_indices:
                if candidate_idx == idx: # skip the same row
                    continue
                candidate_csv_path = df.at[candidate_idx, "csv_path"] # get the csv path of the candidate
                if candidate_csv_path is not None:
                    overlaps[candidate_idx] += 1 # overlapped found!
        current_csv_path = row["csv_path"]
        if current_csv_path is not None:
            for candidate_idx, count in overlaps.items():
                candidate_csv_path = df.at[candidate_idx, "csv_path"]
                if candidate_csv_path is not None:
                    groundtruth[current_csv_path].append(candidate_csv_path)  # Record the current paper citing the candidate
                    groundtruth[candidate_csv_path].append(current_csv_path)  # Record the candidate citing the current paper
                #groundtruth[row["csv_path"]].append(df.at[candidate_idx, "csv_path"])
    print('groundtruth length: ', len(groundtruth))
    print(f"✅ Associations found. Time cost: {time.time() - start_time:.2f} seconds.")
        
    print("⚠️ Step 5: Saving results...")
    filtered_groundtruth = {
        os.path.basename(key): [os.path.basename(v) for v in values]
        for key, values in groundtruth.items()
    }
    with open("./scilakeUnionBenchmark.pickle", "wb") as f:
        pickle.dump(filtered_groundtruth, f)
    print("✅ Groundtruth associations saved successfully!")

if __name__ == "__main__":
    main()
