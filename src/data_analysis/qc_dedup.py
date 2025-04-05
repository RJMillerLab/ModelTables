"""
Author: Zhengyuan Dong
Created: 2025-04-04
Last Modified: 2025-04-05
Description: Directly load CSV files from specified directories, deduplicate based on their content hash and resource priority,
             update the original parquet file's file paths accordingly with cross-resource deduplication (i.e., if a higher-priority
             resource already contains the canonical file, remove duplicates from lower-priority resources and add the canonical file
             to the higher-priority resource column if missing), and save duplicate mapping details, a unique file list,
             as well as cross-resource duplicate overlap details.
Tips: Better save a copy of the four folders, to avoid QC control will affect the original files.
"""

import os, shutil, json
import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict, Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from datetime import datetime

# ---------------- QC CONFIG ----------------
QC_BACKUP_ROOT = "data/qc_backup"
os.makedirs(QC_BACKUP_ROOT, exist_ok=True)

def is_placeholder(cell):
    s = str(cell).strip().lower()
    return s == "" or s == "nan" or all(ch in " :-" for ch in s)

# Global set to record invalid file paths
INVALID_FILES = set()  ########

########
# Hyperparameters / configuration
INPUT_DIR = "data/processed"
INPUT_PARQUET = os.path.join(INPUT_DIR, "modelcard_step4.parquet")
OUTPUT_DIR = "data/deduped"
OUTPUT_PARQUET = os.path.join(INPUT_DIR, "modelcard_step4_dedup.parquet")
DUPLICATE_MAPPING_JSON = os.path.join(OUTPUT_DIR, "duplicate_mapping.json")
UNIQUE_FILES_TXT = os.path.join(OUTPUT_DIR, "unique_files.txt")
DUPLICATE_GROUPS_JSON = os.path.join(OUTPUT_DIR, "duplicate_groups.json")
# Directories containing CSV files with their resource labels and priorities.
DIRS = [
    {"path": "data/processed/deduped_hugging_csvs", "resource": "hugging", "priority": 1},
    {"path": "data/processed/deduped_github_csvs", "resource": "github", "priority": 2},
    {"path": "data/processed/tables_output", "resource": "html", "priority": 3},
    {"path": "data/processed/llm_tables", "resource": "llm", "priority": 4}
]
# Resource priority dictionary (for comparing cross-resource priority)
RESOURCE_PRIORITY = {
    "hugging": 1,
    "github": 2,
    "html": 3,
    "llm": 4
}

# Ensure the output directory exists.
os.makedirs(OUTPUT_DIR, exist_ok=True)
# makeidrs for qc_backup/{resources}
for resource in RESOURCE_PRIORITY.keys():
    os.makedirs(os.path.join(QC_BACKUP_ROOT, resource), exist_ok=True)

# ---------------- QC FUNCTIONS ----------------

def backup_and_remove(file_path, resource):
    """
    Backup the file to QC_BACKUP_ROOT/{timestamp}/{resource}/ and remove the original file.
    """
    backup_path = os.path.join(QC_BACKUP_ROOT, resource, os.path.basename(file_path))
    try:
        shutil.copy2(file_path, backup_path)
    except Exception as e:
        print(f"[QC] Error backing up {file_path}: {e}")
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"[QC] Error removing {file_path}: {e}")

def qc_csv_file(file_path, resource, allow_one_row=True):
    """
    Perform quality control on a CSV file with a single read.
    Checks include:
      - If the file is unreadable, empty (zero rows or zero columns), or has only one row (if not allowed),
        backup and remove the file.
      - If the first data row is entirely placeholders, backup the original file, remove the row,
        and overwrite with the cleaned data.
    Returns:
      "valid" if the file passes QC (and is cleaned if needed),
      Otherwise returns an error status.
    """
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[QC] Error reading {file_path}: {e}")
        backup_and_remove(file_path, resource)
        return "error"
    if df.shape[1] == 0:
        print(f"[QC] File {file_path} has zero columns.")
        backup_and_remove(file_path, resource)
        return "zero_col"
    if df.shape[0] == 0:
        print(f"[QC] File {file_path} has zero rows.")
        backup_and_remove(file_path, resource)
        return "zero_row"
    if df.shape[0] == 1 and not allow_one_row:
        print(f"[QC] File {file_path} has only one row and one row is not allowed.")
        backup_and_remove(file_path, resource)
        return "one_row"
    
    invalid_rows = []
    for idx in df.index:
        # If all cells in the row are placeholders, mark this row as invalid.
        if all(is_placeholder(cell) for cell in df.loc[idx]):
            invalid_rows.append(idx)
    if invalid_rows:
        print(f"[QC] Removing invalid data rows in {file_path}: {invalid_rows}")
        # Backup the original file before modifying.
        backup_and_remove(file_path, resource)
        df_clean = df.drop(index=invalid_rows).reset_index(drop=True)
        if df_clean.empty:
            print(f"[QC] After cleaning, file {file_path} is empty.")
            return "empty_after_clean"
        df_clean.to_csv(file_path, index=False)
        print(f"[QC] Cleaned file saved (removed invalid rows): {file_path}")
    return "valid"

def compute_file_hash(file_path):
    """
    Compute the SHA256 hash for the file content.
    If the file cannot be read, return None.
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
def process_file_in_dir(file_path, resource, order, priority, allow_one_row):
    status = qc_csv_file(file_path, resource, allow_one_row=allow_one_row)
    if status != "valid":
        print(f"[QC] File {file_path} is invalid due to status: {status}. It has been removed.")
        return {"valid": False, "file_path": file_path}
    else:
        return {"valid": True, "file_info": {"file_path": file_path, "resource": resource, "priority": priority, "order": order}}

def list_files_from_directories(directories):
    """
    Given a list of directory info dictionaries (each with 'path', 'resource', and 'priority'),
    return a list of dictionaries containing file_path, resource, priority, and order (the sequence in the directory).
    Skips directories that do not exist.
    """
    files_info = []
    for dir_info in directories:
        directory = dir_info["path"]
        resource = dir_info["resource"]
        ALLOW_ONE_ROW = resource != 'html'
        priority = dir_info["priority"]
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist. Skipping.")
            continue
        # List CSV files and sort them to maintain sequence order
        file_names = sorted([f for f in os.listdir(directory) if f.lower().endswith('.csv')])

        results = Parallel(n_jobs=-1)(
            delayed(process_file_in_dir)(os.path.join(directory, file_name), resource, order, priority, ALLOW_ONE_ROW)
            for order, file_name in enumerate(file_names)
        )
        for res in results:
            if res["valid"]:
                files_info.append(res["file_info"])
            else:
                INVALID_FILES.add(res["file_path"])
    return files_info

def remove_invalid_paths_from_list(path_list, invalid_set):
    """
    Remove file paths that are present in the invalid_set.
    """
    return [p for p in path_list if p not in invalid_set]

def update_row(row, duplicate_mapping, resource_priority):
    """
    For a single row, update the file lists across resources.
    Steps:
      - Collect all file paths from the four resource columns.
      - Group files by their canonical value (using duplicate_mapping).
      - For each group, determine the resource present in the row and choose the resource with the highest priority (lowest numeric value)
        as the designated resource for that group.
      - Update each column: only retain the canonical file in the designated resource's column; remove the group files from other resource columns.
    Returns a dictionary mapping updated column names (original column name + "_dedup") to the new file lists.
    """
    # Define mapping between column names and their resource labels.
    resource_of_col = {
       "hugging_table_list": "hugging",
       "github_table_list": "github",
       "html_table_list_mapped": "html",
       "llm_table_list_mapped": "llm"
    }
    # Collect files in each resource column (as a set).
    row_files = {}
    for col, resource in resource_of_col.items():
        lst = row[col]
        if isinstance(lst, (list, tuple, np.ndarray)):
            #valid_list = remove_invalid_paths_from_list(lst, INVALID_FILES)
            #row_files[resource] = set(valid_list)
            row_files[resource] = set(lst)
        else:
            row_files[resource] = set()
    
    # Get the union of all files in this row.
    all_files = set()
    for files in row_files.values():
        all_files.update(files)
    
    # Group files by canonical value using duplicate_mapping.
    canonical_to_files = defaultdict(set)
    for f in all_files:
        canonical = duplicate_mapping.get(f, f)
        canonical_to_files[canonical].add(f)
    
    # For each group, determine which resources are present and choose the one with the highest priority.
    canonical_designation = {}
    for canonical, files in canonical_to_files.items():
        resources_present = set()
        for resource, files_set in row_files.items():
            if files_set & files:
                resources_present.add(resource)
        if resources_present:
            designated = min(resources_present, key=lambda r: resource_priority[r])
            canonical_designation[canonical] = designated
    
    # Build updated sets for each resource (only keep the canonical file in the designated resource).
    updated = {r: set() for r in resource_of_col.values()}
    for canonical, designated in canonical_designation.items():
        updated[designated].add(canonical)
    
    # Construct new lists for each column preserving as much of the original order as possible.
    result = {}
    for col, resource in resource_of_col.items():
        original_list = row[col] if isinstance(row[col], (list, tuple, np.ndarray)) else []
        new_list = []
        # Add in original order: if the canonical of the file belongs to the current resource's updated set, add it.
        for f in original_list:
            canonical = duplicate_mapping.get(f, f)
            if canonical in updated[resource] and canonical not in new_list:
                new_list.append(canonical)
        # Append any canonical files in the updated set not already present.
        for f in updated[resource]:
            if f not in new_list:
                new_list.append(f)
        new_list = remove_invalid_paths_from_list(new_list, INVALID_FILES)
        result[col + "_dedup"] = new_list
    return result

def main():
    # --- Step 1: Load file information from specified directories ---
    print("Listing files from directories...")
    files_info = list_files_from_directories(DIRS)  ########

    print(f"Total files found: {len(files_info)}")
    
    # --- Step 2: Compute the SHA256 hash for each file in parallel ---
    print("Computing file hashes...")
    with tqdm_joblib(tqdm(desc="Hashing files", total=len(files_info))):  ########
        hashes = Parallel(n_jobs=-1)(
            delayed(compute_file_hash)(fi["file_path"]) for fi in files_info
        )
    for i, h in enumerate(hashes):
        files_info[i]["hash"] = h
    
    # --- Step 3: Group files by hash ---
    hash_groups = defaultdict(list)
    for fi in files_info:
        if fi["hash"] is not None:
            hash_groups[fi["hash"]].append(fi)
    
    # --- Step 4: Determine the canonical file for each hash group and build duplicate_mapping ---
    duplicate_mapping = {}  # Key: duplicate file path, Value: canonical file path
    group_stats = []  # Store details for each hash group
    for h, group in hash_groups.items():
        # Sort group by priority and order to preserve the original sequence.
        group_sorted = sorted(group, key=lambda x: (x["priority"], x["order"]))
        canonical = group_sorted[0]
        canonical_path = canonical["file_path"]
        duplicates = []
        for item in group_sorted:
            if item["file_path"] != canonical_path:
                duplicate_mapping[item["file_path"]] = canonical_path
                duplicates.append(item["file_path"])
        group_stats.append({
            "hash": h,
            "canonical": canonical_path,
            "duplicates": duplicates,
            "resources": [fi["resource"] for fi in group]
        })
    
    # --- Step 5: Compute internal‑resource duplicates and a symmetric cross‑resource matrix ---
    import itertools                                          
    resources = ["hugging", "github", "html", "llm"]          
    dup_overlap = {r: {s: 0 for s in resources} for r in resources}
    dedup_unique = Counter()      # final unique‑file count after priority dedup
    for h, group in hash_groups.items():
        # how many files from each resource are in this hash group?
        res_counts = Counter(fi["resource"] for fi in group if fi["resource"] in resources)
        # pick the canonical file (highest‑priority, earliest order)
        canonical_resource = sorted(group, key=lambda x: (x["priority"], x["order"]))[0]["resource"]
        dedup_unique[canonical_resource] += 1
        # 1) internal duplicates (diagonal)
        for r, cnt in res_counts.items():
            if cnt > 1:
                dup_overlap[r][r] += cnt - 1
        # 2) cross‑resource duplicates (off‑diagonal, keep matrix symmetric)
        for r, s in itertools.combinations(res_counts.keys(), 2):
            cross = min(res_counts[r], res_counts[s])
            dup_overlap[r][s] += cross
            dup_overlap[s][r] += cross

    # --- Step 5‑B: pretty‑print the results + double‑check unique counts ---
    total_files = Counter(fi["resource"] for fi in files_info if fi["resource"] in resources)
    summary_line = " ".join([f"{r} ({total_files[r]})" for r in resources])
    print("\nDuplicate Matrix (duplicate file counts):")
    print(summary_line)
    dup_matrix = pd.DataFrame(dup_overlap).T
    print("\nCross‑Resource Duplicate Overlap Matrix (detailed):")
    print(dup_matrix, "\n")
    print("Deduplicated unique‑file counts (after priority rules):")
    for r in resources:
        print(f"  {r}: {dedup_unique[r]}")
    ###################################
    
    # --- Step 6: Update the original parquet file with deduplicated file paths across resources ---
    print("Loading original parquet file...")
    df = pd.read_parquet(INPUT_PARQUET)  ########
    cols = ["hugging_table_list", "github_table_list", "html_table_list_mapped", "llm_table_list_mapped"]
    
    print("Updating file paths in DataFrame using cross-resource duplicate mapping...")
    new_cols = {col + "_dedup": [] for col in cols}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        updated = update_row(row, duplicate_mapping, RESOURCE_PRIORITY)
        for col in updated:
            new_cols[col].append(updated[col])
    for col in new_cols:
        df[col] = new_cols[col]
    
    df.to_parquet(OUTPUT_PARQUET, index=False)  ########
    print(f"Updated parquet saved as {OUTPUT_PARQUET}")
    
    # --- Step 7: Save duplicate mapping, unique file list, and duplicate group details ---
    with open(DUPLICATE_MAPPING_JSON, "w") as f:  ########
        json.dump(duplicate_mapping, f, indent=2)
    print(f"Duplicate mapping saved to {DUPLICATE_MAPPING_JSON}")
    
    unique_file_paths = set()
    for group in hash_groups.values():
        if group:
            group_sorted = sorted(group, key=lambda x: (x["priority"], x["order"]))
            unique_file_paths.add(group_sorted[0]["file_path"])
    unique_file_paths = list(unique_file_paths)
    with open(UNIQUE_FILES_TXT, "w") as f:  ########
        for file_path in unique_file_paths:
            f.write(file_path + "\n")
    print(f"Unique file list saved to {UNIQUE_FILES_TXT}")
    
    with open(DUPLICATE_GROUPS_JSON, "w") as f:  ########
        json.dump(group_stats, f, indent=2)
    print(f"Duplicate group details saved to {DUPLICATE_GROUPS_JSON}")

if __name__ == "__main__":
    main()

"""
Example output:

[QC] Removing invalid data rows in data/processed/llm_tables/725590_table3.csv: [5]
[QC] Cleaned file saved (removed invalid rows): data/processed/llm_tables/725590_table3.csv
[QC] Removing invalid data rows in data/processed/llm_tables/91184391_table4.csv: [3]
[QC] Cleaned file saved (removed invalid rows): data/processed/llm_tables/91184391_table4.csv
Total files found: 250631
Computing file hashes...
100%|████████████████| 250631/250631 [00:27<00:00, 9142.13it/s]
Hashing files:   0%|                | 0/250631 [00:27<?, ?it/s]

Duplicate Matrix (duplicate file counts):
hugging (174209) github (4085) html (48904) llm (23433)

Cross‑Resource Duplicate Overlap Matrix (detailed):
         hugging  github  html  llm
hugging    27939    1375     0    1
github      1375     284     0    0
html           0       0  1527    0
llm            1       0     0   19

Deduplicated unique‑file counts (after priority rules):
  hugging: 146270
  github: 2457
  html: 47377
  llm: 23413
Loading original parquet file...
Updating file paths in DataFrame using cross-resource duplicate mapping...
Processing rows: 100%|█| 1108759/1108759 [03:19<00:00, 5561.22i
Updated parquet saved as data/processed/modelcard_step4_dedup.parquet
Duplicate mapping saved to data/deduped/duplicate_mapping.json
Unique file list saved to data/deduped/unique_files.txt
Duplicate group details saved to data/deduped/duplicate_groups.json
"""