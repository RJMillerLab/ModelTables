"""
Author: Zhengyuan Dong
Created: 2025-04-04
Last Modified: 2025-04-04
Description: Directly load CSV files from specified directories, deduplicate based on their content hash and resource priority,
             update the original parquet file's file paths accordingly with cross-resource deduplication (i.e., if a higher-priority
             resource already contains the canonical file, remove duplicates from lower-priority resources and add the canonical file
             to the higher-priority resource column if missing), and save duplicate mapping details, a unique file list,
             as well as cross-resource duplicate overlap details.
"""

import os
import pandas as pd
import numpy as np
import hashlib
from collections import defaultdict, Counter
from tqdm import tqdm  ########
from joblib import Parallel, delayed  ########
from tqdm_joblib import tqdm_joblib  ########
import json

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
########

########
# Ensure the output directory exists.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
########

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

########
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
        priority = dir_info["priority"]
        if not os.path.exists(directory):  ########
            print(f"Warning: Directory {directory} does not exist. Skipping.")  ########
            continue  ########
        # List CSV files and sort them to maintain sequence order
        file_names = sorted([f for f in os.listdir(directory) if f.lower().endswith('.csv')])
        for order, file_name in enumerate(file_names):
            file_path = os.path.join(directory, file_name)
            files_info.append({
                "file_path": file_path,
                "resource": resource,
                "priority": priority,
                "order": order
            })
    return files_info
########

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
    
    # --- Step 5: Compute duplicate file counts and cross-resource duplicate overlap matrix  ########
    resources = ["hugging", "github", "html", "llm"]
    total_files = Counter(fi["resource"] for fi in files_info if fi["resource"] in resources)
    dup_counts = Counter()  # Total duplicate file counts per resource
    dup_overlap = {r: {s: 0 for s in resources} for r in resources}
    
    for group in hash_groups.values():
        if len(group) > 1:
            group_counter = Counter(fi["resource"] for fi in group if fi["resource"] in resources)
            canonical_resource = sorted(group, key=lambda x: (x["priority"], x["order"]))[0]["resource"]
            dup_in_group = {}
            for r, cnt in group_counter.items():
                if r == canonical_resource:
                    dup_in_group[r] = max(cnt - 1, 0)
                else:
                    dup_in_group[r] = cnt
                dup_counts[r] += dup_in_group[r]
            present_resources = list(dup_in_group.keys())
            for r in present_resources:
                for s in present_resources:
                    dup_overlap[r][s] += dup_in_group[r]
    
    unique_files = {r: total_files[r] - dup_counts[r] for r in resources}
    
    ########
    # Print a summary line: each resource with its total file count (e.g., "hugging (174218) github (4090) ...")
    summary_line = " ".join([f"{r} ({total_files[r]})" for r in resources])
    print("\nDuplicate Matrix (duplicate file counts):")
    print(summary_line)
    ########
    
    dup_matrix = pd.DataFrame(dup_overlap).T  ########
    print("\nCross-Resource Duplicate Overlap Matrix (detailed):")
    print(dup_matrix, "\n")
    
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
