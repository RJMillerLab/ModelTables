import pandas as pd
import numpy as np


data_type = "modelcard"

df = pd.read_parquet(f"data/{data_type}_step5.parquet")
#res = df.dropna(subset=['references_within_dataset', 'citations_within_dataset'])[['references_within_dataset', 'citations_within_dataset']].head(10)


def extract_titles(bibtex_list):
    # Accept list, tuple, or numpy.ndarray; otherwise return empty list.
    if not isinstance(bibtex_list, (list, tuple, np.ndarray)):
        return []
    
    titles = []
    for entry in bibtex_list:
        # Ensure the entry is a dict before processing.
        if isinstance(entry, dict):
            raw_title = entry.get('title')
            if raw_title:
                clean_title = raw_title.replace('{', '').replace('}', '')
                titles.append(clean_title)
    return titles

# Apply the function to each row of the DataFrame.
df['title'] = df['parsed_bibtex_tuple_list'].apply(extract_titles)
# Example: Print the titles from row 3 (if available).
#print(df['title'].iloc[4])

import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Helper function to extract titles from JSON strings/dicts ---
def extract_json_titles(json_input):
    print("[extract_json_titles] Input:", json_input)  # Debug print
    if isinstance(json_input, str):
        try:
            parsed = json.loads(json_input)
        except Exception as e:
            print("[extract_json_titles] JSON parse error:", e)
            return []
    elif isinstance(json_input, dict):
        parsed = json_input
    else:
        return []
    if not (isinstance(parsed, dict) and 'data' in parsed):
        return []
    titles = []
    for item in parsed.get("data", []):
        if isinstance(item, dict):
            # Check for reference paper title.
            if "citedPaper" in item and isinstance(item["citedPaper"], dict):
                t = item["citedPaper"].get("title")
                if t and isinstance(t, str):
                    titles.append(t)
            # Also check for citation paper title.
            if "citingPaper" in item and isinstance(item["citingPaper"], dict):
                t = item["citingPaper"].get("title")
                if t and isinstance(t, str):
                    titles.append(t)
    print("[extract_json_titles] Extracted titles:", titles)
    return titles

# --- Prepare the groundtruth dictionary ---
groundtruth = {}
# Initialize for each row with a valid csv_path.
for idx, row in df.iterrows():
    csv_path = row["csv_path"]
    if pd.isnull(csv_path):
        continue
    groundtruth[csv_path] = []

# --- Precompute candidate titles (from the "title" column)
# and the reference/citation titles (from the json fields) ---
titles_dict = {}   # key: csv_path, value: list of candidate titles (lowercase & stripped)
ref_cite_dict = {} # key: csv_path, value: list of JSON-parsed titles (lowercase & stripped)

for idx, row in df.iterrows():
    csv_path = row["csv_path"]
    if pd.isnull(csv_path):
        continue
    # Get candidate titles (already computed in df['title'])
    cand_titles = row["title"] if isinstance(row["title"], list) else []
    cand_titles = [t.lower().strip() for t in cand_titles if isinstance(t, str)]
    titles_dict[csv_path] = cand_titles
    # Combine titles from references and citations.
    refs = extract_json_titles(row["references_within_dataset"])
    cites = extract_json_titles(row["citations_within_dataset"])
    combined = refs + cites
    combined = [t.lower().strip() for t in combined if isinstance(t, str)]
    ref_cite_dict[csv_path] = combined
    print(f"[Setup] csv_path: {csv_path}\n  Candidate titles: {titles_dict[csv_path]}\n  Ref/Cite titles: {ref_cite_dict[csv_path]}")
# List of valid csv_path keys.
keys = list(groundtruth.keys())

# --- Define a function to process the association for a given row index ---
def process_row(i, keys, titles_dict, ref_cite_dict):
    cp_i = keys[i]
    associations = []
    print(f"[Process Row] Starting index {i} (csv_path: {cp_i})")
    for j in range(i+1, len(keys)):
        cp_j = keys[j]
        found = False
        # First: check if any candidate title in row i appears in row j’s ref/cite titles.
        for t in titles_dict.get(cp_i, []):
            if t in ref_cite_dict.get(cp_j, []):
                found = True
                break
        # Second: if not found yet, check vice versa.
        if not found:
            for t in titles_dict.get(cp_j, []):
                if t in ref_cite_dict.get(cp_i, []):
                    found = True
                    break
        if found:
            associations.append((cp_i, cp_j))
    print(f"[Process Row] Finished index {i} (csv_path: {cp_i}). Associations found: {associations}")
    return associations

# --- Process all rows in parallel ---
all_associations = []
with ProcessPoolExecutor() as executor:
    # Submit tasks: each process_id will handle the associations for one row.
    futures = [executor.submit(process_row, i, keys, titles_dict, ref_cite_dict) for i in range(len(keys))]
    completed = 0
    for fut in as_completed(futures):
        result = fut.result()  # This is a list of (csv_path_i, csv_path_j) tuples.
        all_associations.extend(result)
        completed += 1
        print(f"[Parallel] Completed processing of {completed}/{len(keys)} rows.")
# --- Update the groundtruth dictionary using the associations ---
for cp_i, cp_j in all_associations:
    groundtruth[cp_i].append(cp_j)
    groundtruth[cp_j].append(cp_i)

print("\n[Final Groundtruth Mapping]")

# --- Save the groundtruth dictionary to a JSON file ---
with open("groundtruth_associations.json", "w") as f:
    json.dump(groundtruth, f, indent=4, ensure_ascii=False)