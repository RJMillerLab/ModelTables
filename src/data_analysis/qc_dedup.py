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

import os, shutil, json, time
import pandas as pd
import numpy as np
import hashlib
import itertools
from collections import defaultdict, Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from datetime import datetime
from src.utils import to_parquet

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# ---------------- QC CONFIG ----------------
QC_BACKUP_ROOT = "data/qc_backup"
os.makedirs(QC_BACKUP_ROOT, exist_ok=True)

# Global set to record invalid file paths
INVALID_FILES = set()

# ---------------- Hyperparameters / configuration ----------------
INPUT_DIR = "data/processed"
INPUT_PARQUET = os.path.join(INPUT_DIR, "modelcard_step3_merged.parquet")
OUTPUT_DIR = "data/deduped"
FIG_DIR = "data/analysis"

# ====== Skip generic CSV sets ======
GENERIC_TABLE_PATTERNS = [
    "1910.09700_table",
    "204823751_table"
]
# Ensure the output directory exists.
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PARQUET = os.path.join(INPUT_DIR, "modelcard_step3_dedup.parquet")
DUPLICATE_MAPPING_JSON = os.path.join(OUTPUT_DIR, "duplicate_mapping.json")
UNIQUE_FILES_TXT = os.path.join(OUTPUT_DIR, "unique_files.txt")
DUPLICATE_GROUPS_JSON = os.path.join(OUTPUT_DIR, "duplicate_groups.json")
STATS_PATH = os.path.join(OUTPUT_DIR, "stats.json")
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
# makeidrs for qc_backup/{resources}
for resource in RESOURCE_PRIORITY.keys():
    os.makedirs(os.path.join(QC_BACKUP_ROOT, resource), exist_ok=True)


def is_placeholder(cell):
    s = str(cell).strip().lower()
    return s == "" or s == "nan" or all(ch in " :-" for ch in s)

def is_generic_table(path):
    """Check if file should be filtered as generic/too-general table."""
    filename = os.path.basename(path)
    return any(pattern in filename for pattern in GENERIC_TABLE_PATTERNS)

def get_linked_set_from_parquet(df, cols):
    linked_set = []
    for col in cols:
        if col in df.columns:
            for paths in df[col]:
                if isinstance(paths, (list, tuple, np.ndarray)):
                    linked_set.extend([p for p in paths if isinstance(p, str)])
                elif isinstance(paths, str):
                    linked_set.append(paths)
    return linked_set

def infer_resource_from_path(path: str):
    """Infer the resource label from the canonical file path."""
    if "/deduped_hugging_csvs/" in path or "/hugging" in path:
        return "hugging"
    if "/deduped_github_csvs/" in path or "/github" in path:
        return "github"
    if "/tables_output/" in path or "/html" in path:
        return "html"
    if "/llm_tables/" in path or "/llm" in path:
        return "llm"
    return None

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

def find_v2_csv_path(original_path):
    """Find v2 version of CSV file if it exists, otherwise return original path."""
    # Check if file exists
    if not os.path.exists(original_path):
        return original_path
    
    # Get directory and filename
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    
    # Look for v2 directory
    v2_dir = dir_path.replace('deduped_hugging_csvs', 'deduped_hugging_csvs_v2')
    v2_dir = v2_dir.replace('deduped_github_csvs', 'deduped_github_csvs_v2')
    v2_dir = v2_dir.replace('tables_output', 'tables_output_v2')
    
    # Check if v2 directory exists
    if not os.path.exists(v2_dir):
        return original_path
    
    # Look for v2 file
    v2_path = os.path.join(v2_dir, filename)
    if os.path.exists(v2_path):
        return v2_path
    
    return original_path

def qc_csv_file(file_path, resource, allow_one_row=True, use_v2=False):
    """
    Perform quality control on a CSV file with a single read.
    Checks include:
      - If the file is unreadable, empty (zero rows or zero columns), or has only one row (if not allowed),
        backup and remove the file.
      - If the first data row is entirely placeholders, backup the original file, remove the row,
        and overwrite with the cleaned data.
    
    Args:
        file_path: Path to the CSV file
        resource: Resource type (hugging, github, etc.)
        allow_one_row: Whether to allow files with only one row
        use_v2: Whether to use v2 version of the file if available
    
    Returns:
      "valid" if the file passes QC (and is cleaned if needed),
      Otherwise returns an error status.
    """
    try:
        # Use v2 version if requested and available
        actual_csv_file = find_v2_csv_path(file_path) if use_v2 else file_path
        df = pd.read_csv(actual_csv_file, dtype=str, keep_default_na=False)
        # Normalize whitespace-only cells and drop fully empty rows on read
        try:
            df = df.replace(r'^\s*$', pd.NA, regex=True).dropna(axis=0, how='all')
        except Exception:
            pass
    except Exception as e:
        print(f"[QC] Error reading {file_path}: {e}")
        backup_and_remove(file_path, resource)
        return "error", None
    if df.shape[1] == 0:
        print(f"[QC] File {file_path} has zero columns.")
        backup_and_remove(file_path, resource)
        return "zero_col", None
    if df.shape[0] == 0:
        print(f"[QC] File {file_path} has zero rows.")
        backup_and_remove(file_path, resource)
        return "zero_row", None
    
    invalid_rows = []
    for idx in df.index:
        # If all cells in the row are placeholders, mark this row as invalid.
        if all(is_placeholder(cell) for cell in df.loc[idx]):
            invalid_rows.append(idx)
    if invalid_rows:
        print(f"[QC] Removing invalid data rows in {file_path}: {invalid_rows}")
        # Backup the original file before modifying.
        backup_and_remove(file_path, resource)
        df = df.drop(index=invalid_rows).reset_index(drop=True)
        if df.empty:
            print(f"[QC] After cleaning, file {file_path} is empty.")
            return "empty_after_clean", None
        df.to_csv(file_path, index=False)
        print(f"[QC] Cleaned file saved (removed invalid rows): {file_path}")
    
    if df.shape[0] == 1 and not allow_one_row:
        print(f"[QC] File {file_path} has only one row and one row is not allowed.")
        backup_and_remove(file_path, resource)
        return "one_row", None
    # ----- compute hash using DataFrame's CSV representation -----
    sha256 = None
    try:
        csv_string = df.to_csv(index=False)
        sha256 = hashlib.sha256(csv_string.encode("utf-8")).hexdigest()
    except Exception as e:
        print(f"[QC] Hashing failed for {file_path}: {e}")
    return "valid", sha256
    
def process_file_in_dir(file_path, resource, order, priority, allow_one_row):
    status, file_hash = qc_csv_file(file_path, resource, allow_one_row=allow_one_row)
    if status != "valid":
        print(f"[QC] File {file_path} is invalid due to status: {status}. It has been removed.")
        return {"valid": False, "file_path": file_path}
    else:
        return {"valid": True, "file_info": {"file_path": file_path, "resource": resource, "priority": priority, "order": order, "hash": file_hash}}

def valid_filelist_with_qc_from_local(directories):
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
        # List CSV files and sort them to maintain sequence order
        file_names = sorted([f for f in os.listdir(directory) if f.lower().endswith('.csv')])

        with tqdm_joblib(tqdm(desc="Processing files", total=len(file_names))):
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
    For a single row, update the file lists across resources using the following process:
      1. Gather all file paths from all resource columns, regardless of their original resource.
      2. Map each file to its canonical value using duplicate_mapping.
      3. Record the set of resources in which each canonical file appears.
      4. Determine the designated resource for each canonical file based on the highest priority (lowest numeric value).
      5. Build the new deduped lists such that each canonical file is placed only in the dedup column of its designated resource.
         (The updated resource is taken from the canonical file's designated resource.)
    Returns a dictionary mapping updated column names (original column name + "_dedup") to the new file lists.
    """
    # Mapping from column names to their resource labels.
    resource_of_col = {
       "hugging_table_list": "hugging",
       "github_table_list": "github",
       "html_table_list_mapped": "html",
       "llm_table_list_mapped": "llm"
    }
    # Step 1 & 2: Gather all file paths and convert to canonical, preserving order.
    ordered_canonical = []
    seen = set()
    for col in resource_of_col:
        lst = row[col] if isinstance(row[col], (list, tuple, np.ndarray)) else []
        for f in lst:
            canonical = duplicate_mapping.get(f, f)
            if canonical not in seen:
                seen.add(canonical)
                ordered_canonical.append(canonical)
    # ---- Step 2: decide the target resource for each canonical file ------
    designated = {}
    for canonical in ordered_canonical:
        # 2a. If the canonical path already appears in one of the row's lists,
        #     we keep it under that column's resource.
        target_resource = None
        for col, res in resource_of_col.items():
            lst = row[col] if isinstance(row[col], (list, tuple, np.ndarray)) else []
            if canonical in lst:
                target_resource = res
                break
        # 2b. Otherwise, infer the resource from the path itself.
        if target_resource is None:
            target_resource = infer_resource_from_path(canonical)
        # 2c. Fallback: if still unknown, choose the highest‑priority resource
        #     among duplicates present in this row; if none, default to 'hugging'.
        if target_resource is None:
            dup_resources = {resource_of_col[c]
                             for c in resource_of_col
                             if any(duplicate_mapping.get(p, p) == canonical
                                    for p in (row[c] if isinstance(row[c], (list, tuple, np.ndarray)) else []))}
            target_resource = (min(dup_resources, key=lambda r: resource_priority[r])
                               if dup_resources else "hugging")
        designated[canonical] = target_resource
    # ---- Step 3: construct the new deduped lists -------------------------
    result = {col + "_dedup": [] for col in resource_of_col}
    for canonical in ordered_canonical:
        tgt_res = designated[canonical]
        for col, res in resource_of_col.items():
            if res == tgt_res:
                result[col + "_dedup"].append(canonical)
                break
    return result

def compute_dup_matrix_from_sha(files_info):
    keys = ["hugging", "github", "html", "llm"]
    resource_sha = {r: [fi["hash"] for fi in files_info if fi["resource"] == r] for r in keys}
    total_files = {r: len(resource_sha[r]) for r in keys}
    resource_sha_set = {r: set(sha_list) for r, sha_list in resource_sha.items()}
    unique_files = {r: len(resource_sha_set[r]) for r in keys}
    internal_duplicates = {r: total_files[r]-unique_files[r] for r in keys}
    dup_overlap = {r: {s: 0 for s in keys} for r in keys}
    for i in range(len(keys)):
        for j in range(len(keys)):
            r = keys[i]
            s = keys[j]
            if r == s:
                dup_overlap[r][s] = internal_duplicates[r]
            else:
                overlap = len(resource_sha_set[r].intersection(resource_sha_set[s]))
                dup_overlap[r][s] = overlap
                dup_overlap[s][r] = overlap
    dup_matrix = pd.DataFrame(dup_overlap).T
    # group by hash
    hash_groups = defaultdict(list)
    for fi in files_info:
        h = fi["hash"]
        if h is not None:
            hash_groups[h].append(fi)
    # sort hash group by priority
    cross_unique_counts = {r: 0 for r in keys}
    cross_unique_files = {r: [] for r in keys}
    overall_unique = []
    for h, group_sorted in hash_groups.items():
        group_sorted = sorted(group_sorted, key=lambda x: (x["priority"], x["order"]))
        canonical = group_sorted[0]
        res = canonical["resource"]
        cross_unique_counts[res] += 1
        cross_unique_files[res].append(canonical["file_path"])
        hash_groups[h] = group_sorted # update the group to sorted order
        overall_unique.append(canonical["file_path"])
    
    stats = {
        "total_files": total_files,
        "internal_duplicates": internal_duplicates,
        "unique_files": unique_files,
        "cross_unique_counts": cross_unique_counts,
        "cross_unique_files": cross_unique_files,
        "overall_unique": overall_unique
    }
    return dup_matrix, stats, hash_groups

# draw
class BiasedLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, bias=0.3, **kwargs):
        super().__init__(vmin=vmin, vmax=vmax, **kwargs)
        self.bias = bias

    def __call__(self, value, clip=None):
        scaled = super().__call__(value, clip)
        return np.power(scaled, self.bias)

def save_heatmap(dup_matrix, unique_counts, output_dir, is_percentage=False):
    fontsize = 18
    plt.rcParams.update({
        'font.size': 18,           
        'axes.titlesize': 18,      
        'axes.labelsize': 18,   
        'xtick.labelsize': 17,    
        'ytick.labelsize': 17,     
        'legend.fontsize': 18,     
        'figure.titlesize': 18     
    })
    figsize = (12, 6)

    # ---- Rename resources for better labels ----
    name_map = {
        "hugging": "Hugging",
        "github": "GitHub",
        "html": "HTML",
        "llm": "S2ORC"
    }
    dup_matrix = dup_matrix.rename(index=name_map, columns=name_map)
    unique_counts = {name_map[k]: v for k, v in unique_counts.items()}

    # Step 1: prepare plotting matrix
    if is_percentage:
        # Calculate total files per resource from parquet
        df = pd.read_parquet(INPUT_PARQUET, columns=['modelId', 'hugging_table_list', 'github_table_list', 'html_table_list_mapped', 'llm_table_list_mapped'])
        total_files = {}
        for res, col in {
            "Hugging": "hugging_table_list",
            "GitHub": "github_table_list",
            "HTML": "html_table_list_mapped",
            "S2ORC": "llm_table_list_mapped"
        }.items():
            total_files[res] = df[col].apply(lambda x: len(x) if isinstance(x, (list, tuple, np.ndarray)) else 0).sum()
        
        # Calculate percentages based on total files
        dup_matrix_plot = dup_matrix.copy()
        print("\nPercentage calculation details:")
        print("Total files per resource (denominator):")
        for res, count in total_files.items():
            print(f"{res}: {count}")
        print("\nOriginal overlap values (numerator):")
        print(dup_matrix_plot)
        # Convert to percentages using total files as denominator
        for idx in dup_matrix_plot.index:
            denominator = total_files[idx]
            dup_matrix_plot.loc[idx] = (dup_matrix_plot.loc[idx] / denominator) * 100
        # Round to 1 decimal place for display
        dup_matrix_plot = dup_matrix_plot.round(1)
        print("\nCalculated percentages:")
        print(dup_matrix_plot)
        vmin, vmax = 0, 100
        fmt = ".1f"
    else:
        dup_matrix_plot = dup_matrix.copy()
        dup_matrix_plot[dup_matrix_plot < 10] = 10
        vmin, vmax = 10, 1000
        fmt = ".0f"

    # Step 2: define color map - use same teal colors for both
    colors = ["#a5d2bc", "#50a89d", "#4e8094", "#486f90"]
    cmap = LinearSegmentedColormap.from_list("teal_gradient", colors)
    
    # Step 3: use appropriate normalization
    if is_percentage:
        norm = None  # Linear normalization for percentages
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)

    # Step 4: plot
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        dup_matrix_plot,
        annot=dup_matrix if not is_percentage else dup_matrix_plot,
        cmap=cmap,
        fmt=fmt,
        square=True,
        cbar=True,
        xticklabels=False,
        norm=norm
    )
    
    # Add percentage sign to colorbar if showing percentages
    if is_percentage:
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Percentage (%)', fontsize=fontsize)
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_yticklabels())
    
    # Step 5: add top labels
    xticks = np.arange(len(unique_counts))
    for idx, res in enumerate(unique_counts.keys()):
        ax.text(
            xticks[idx] + 0.5,
            -0.05,
            res,
            ha='center',
            va='bottom',
            fontsize=fontsize
        )
    plt.tight_layout()
    
    # Save with appropriate filename
    suffix = "percentage" if is_percentage else "overlap"
    plt.savefig(os.path.join(output_dir, f"heatmap_{suffix}.pdf"))
    print(f"Heatmap saved to {os.path.join(output_dir, f'heatmap_{suffix}.pdf')}")

def save_heatmap_percentage(dup_matrix, unique_counts, output_dir):
    """
    Save a heatmap showing the percentage of overlap between resources.
    The percentages are calculated based on the row totals.
    """
    save_heatmap(dup_matrix, unique_counts, output_dir, is_percentage=True)

def main():
    time_start = time.time()
    # --- Step 1: get the linked set (some files exist in local but not linked to model) ---
    df = pd.read_parquet(INPUT_PARQUET, columns=['modelId', 'hugging_table_list', 'github_table_list', 'html_table_list_mapped', 'llm_table_list_mapped'])
    cols = ["hugging_table_list", "github_table_list", "html_table_list_mapped", "llm_table_list_mapped"]
    file_paths = [item['path'] for item in DIRS]

    linked_set = get_linked_set_from_parquet(df, cols)
    linked_set = set(linked_set)
    print(f"Linked set size from parquet: {len(linked_set)}")
    # intersection
    """existing_set = []
    for dir in file_paths:
        # I want path starts with dir, directly modify based on above
        existing_set.extend([os.path.join(dir, f) for f in os.listdir(dir)])
    print('existing_set size:', len(existing_set))
    linked_set = set(linked_set) & set(existing_set)
    print(f"Linked set size from parquet: {len(linked_set)}")"""
    print(f"time taken: {time.time() - time_start} seconds")
    
    time_start = time.time()
    # --- Step 2: QC and sha256 hash ---
    # we don't care what's stats before qc. However, we retain all the csv in data/qc_backup for future reference.
    files_info = valid_filelist_with_qc_from_local(DIRS)
    # Filter local files that unlinked to modelcard and generic tables
    resource_totals = {res: 0 for res in RESOURCE_PRIORITY.keys()}
    resource_filtered = {res: 0 for res in RESOURCE_PRIORITY.keys()}
    resource_generic_filtered = {res: 0 for res in RESOURCE_PRIORITY.keys()}
    filtered_files_info = []
    for fi in tqdm(files_info, desc="Filtering files"):
        res = fi["resource"]
        resource_totals[res] += 1
        
        # Filter 1: Must be in linked_set
        # Filter 2: Must not be a generic table
        if fi["file_path"] in linked_set:
            if not is_generic_table(fi["file_path"]):
                resource_filtered[res] += 1
                filtered_files_info.append(fi)
            else:
                resource_generic_filtered[res] += 1
    
    for res in RESOURCE_PRIORITY.keys():
        total = resource_totals[res]
        kept = resource_filtered[res]
        generic_removed = resource_generic_filtered[res]
        filtered_out = total - kept - generic_removed
        print(f"Resource {res}: total {total}, kept {kept}, generic removed {generic_removed}, filtered out {filtered_out}")

    # check duplicate stats
    dup_matrix, stats, hash_groups = compute_dup_matrix_from_sha(filtered_files_info)
    overall_unique = stats["overall_unique"]
    # save overall_unique
    with open(os.path.join(OUTPUT_DIR, "overall_unique.txt"), "w") as f:
        for file_path in overall_unique:
            f.write(file_path + "\n")
    print(f"Overall unique file count: {len(overall_unique)}")
    # save cross_unique_files
    cross_unique_files = stats["cross_unique_files"]
    for res, files in cross_unique_files.items():
        with open(os.path.join(OUTPUT_DIR, f"{res}_unique.txt"), "w") as f:
            for file_path in files:
                f.write(file_path + "\n")
        print(f"{res} unique file count: {len(files)}")

    print("Duplicate Overlap Matrix (across resources):")
    print(dup_matrix)
    print("\nStatistics:")
    print("Total files per resource:", stats["total_files"])
    print("Interal Unique files per resource:", stats["unique_files"])
    print("Cross-resource unique counts:", stats["cross_unique_counts"])
    print(f"Time taken: {time.time() - time_start} seconds")

    time_start = time.time()
    # --- Step 4: Determine the canonical file for each hash group and build duplicate_mapping ---
    duplicate_mapping = {}  # Key: duplicate file path, Value: canonical file path
    group_stats = []  # Store details for each hash group
    for h, group_sorted in hash_groups.items():
        # Sort group by priority and order to preserve the original sequence.
        canonical = group_sorted[0]
        duplicates = []
        for item in group_sorted:
            if item["file_path"] != canonical["file_path"]:
                duplicate_mapping[item["file_path"]] = canonical["file_path"]
                duplicates.append(item["file_path"])
        group_stats.append({
            "hash": h,
            "canonical": canonical["file_path"],
            "duplicates": duplicates,
            "resources": [fi["resource"] for fi in group_sorted]
        })
    print(f"Time taken: {time.time() - time_start} seconds")
    
    time_start = time.time()
    # --- Step 6: Update the original parquet file with deduplicated file paths across resources ---
    print("Updating file paths in DataFrame using cross-resource duplicate mapping...")
    # filter out invalid paths (qc remove)
    VALID_PATHS = set(fi['file_path'] for fi in filtered_files_info)
    for col in cols:
        total_before = df[col].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0).sum()
        print(f"Filtering {col}... Before: {total_before}")
        df[col] = df[col].apply(lambda x: [p for p in x if p in VALID_PATHS])
        total_after = df[col].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0).sum()
        print(f"After: {total_after}")
    # map the file path to the canonical file path
    new_cols = {col + "_dedup": [] for col in cols}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        updated = update_row(row, duplicate_mapping, RESOURCE_PRIORITY)
        for col in updated:
            new_cols[col].append(updated[col])
    for col in new_cols:
        df[col] = new_cols[col]
    print('add new cols:', new_cols.keys())
    
    df.drop(columns=['card_tags', 'downloads', 'github_link', 'pdf_link', 'hugging_table_list', 'github_table_list', 'html_table_list_mapped', 'llm_table_list_mapped'], inplace=True, errors='ignore')
    to_parquet(df, OUTPUT_PARQUET)
    print(f"Updated parquet saved as {OUTPUT_PARQUET}")
    print(f"Time taken: {time.time() - time_start} seconds")

    time_start = time.time()
    # --- Step 7: Save duplicate mapping, unique file list, and duplicate group details ---
    with open(DUPLICATE_MAPPING_JSON, "w") as f:
        json.dump(duplicate_mapping, f, indent=2)
    print(f"Duplicate mapping saved to {DUPLICATE_MAPPING_JSON}")
    
    unique_file_paths = set()
    for group_sorted in hash_groups.values():
        if group_sorted:
            unique_file_paths.add(group_sorted[0]["file_path"])
    unique_file_paths = list(unique_file_paths)
    assert len(unique_file_paths)==len(hash_groups)
    # please assert the len of unique_file_paths == the unique files above
    # save stats
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Unique file count: {len(unique_file_paths)}")
    with open(UNIQUE_FILES_TXT, "w") as f:
        for file_path in unique_file_paths:
            f.write(file_path + "\n")
    print(f"Unique file list saved to {UNIQUE_FILES_TXT}")
    
    with open(DUPLICATE_GROUPS_JSON, "w") as f:
        json.dump(group_stats, f, indent=2)
    print(f"Duplicate group details saved to {DUPLICATE_GROUPS_JSON}")
    print(f"Time taken: {time.time() - time_start} seconds")

    time_start = time.time()
    # --- Step 4.5: Save dup_matrix and stats for later reuse
    dup_matrix_file = os.path.join(OUTPUT_DIR, "dup_matrix.pkl")
    dup_matrix.to_pickle(dup_matrix_file)
    print(f"Dup matrix saved to {dup_matrix_file}")

    # Save both absolute and percentage heatmaps
    save_heatmap(dup_matrix, stats["cross_unique_counts"], FIG_DIR)
    save_heatmap_percentage(dup_matrix, stats["cross_unique_counts"], FIG_DIR)
    print(f"Time taken: {time.time() - time_start} seconds")

if __name__ == "__main__":
    main()
