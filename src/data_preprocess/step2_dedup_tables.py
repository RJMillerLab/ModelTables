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

import os, shutil, json, time, re
import argparse
import pandas as pd
import numpy as np
import hashlib
import itertools
from collections import defaultdict, Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from datetime import datetime
from src.utils import to_parquet, load_config, is_list_like, to_list_safe

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# ---------------- QC CONFIG ----------------
QC_BACKUP_ROOT = "data/qc_backup"
os.makedirs(QC_BACKUP_ROOT, exist_ok=True)

# Global set to record invalid file paths
INVALID_FILES = set()

# ====== Skip generic CSV sets ======
GENERIC_TABLE_PATTERNS = [
    "1910.09700_table",
    "204823751_table"
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
    """
    Extract file paths from parquet columns (as-is; matching is done by (resource, basename) later).
    """
    linked_set = []
    for col in cols:
        if col in df.columns:
            for paths in df[col]:
                if is_list_like(paths):
                    for p in to_list_safe(paths):
                        if isinstance(p, str):
                            linked_set.append(p)
                elif isinstance(paths, str):
                    linked_set.append(paths)
    return linked_set


def parquet_path_to_local_path(p, path_by_key):
    """
    Map a parquet path (any prefix: CitationLake, ModelTables, relative) to local canonical path
    by (resource, basename). Returns None if not found.
    """
    if not isinstance(p, str):
        return None
    res = infer_resource_from_path(p)
    if res is None:
        return None
    key = (res, os.path.basename(p))
    return path_by_key.get(key)


def normalize_path_to_relative(path):
    """
    Normalize a path to relative form (data/processed/...) for duplicate_mapping lookup.
    Used in update_row when resolving canonical path. Handles absolute paths under
    any repo (ModelTables, CitationLake) by stripping to data/processed/...
    """
    if not isinstance(path, str):
        return path
    if os.path.isabs(path):
        try:
            data_processed_abs = os.path.abspath('data/processed')
            if path.startswith(data_processed_abs):
                return os.path.relpath(path, os.getcwd())
            if "data/processed/" in path:
                return path[path.find("data/processed/"):]
        except Exception:
            pass
    return path

def infer_resource_from_path(path: str):
    """Infer the resource label from the canonical file path.
    Uses prefix match so tagged dirs are recognized (e.g. deduped_hugging_csvs_v2_251117, llm_tables_251117).
    """
    if "/deduped_hugging_csvs" in path or "/hugging" in path:
        return "hugging"
    if "/deduped_github_csvs" in path or "/github" in path:
        return "github"
    if "/tables_output" in path or "/html" in path:
        return "html"
    if "/llm_tables" in path or "/llm" in path:
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

def normalize_path_for_mapping(path):
    """
    Normalize path to match duplicate_mapping format.
    Handles tag differences (e.g., deduped_hugging_csvs_v2_251117 -> deduped_hugging_csvs_v2)
    """
    if not isinstance(path, str):
        return path
    # Remove tag suffix from directory names if present
    # Pattern: deduped_*_csvs_v2_<tag> -> deduped_*_csvs_v2
    # Match patterns like: deduped_hugging_csvs_v2_251117, deduped_github_csvs_v2_251117, etc.
    normalized = re.sub(r'(deduped_(?:hugging|github)_csvs_v2)_\d+', r'\1', path)
    normalized = re.sub(r'(tables_output_v2)_\d+', r'\1', normalized)
    normalized = re.sub(r'(llm_tables)_\d+', r'\1', normalized)
    return normalized

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
        lst = to_list_safe(row[col]) if is_list_like(row[col]) else []
        for f in lst:
            # Normalize the path first (relative/absolute conversion)
            normalized_f = normalize_path_to_relative(f)
            # Try multiple lookup strategies to find canonical path:
            canonical = None
            # 1. Direct lookup with original path
            if f in duplicate_mapping:
                canonical = duplicate_mapping[f]
            # 2. Try with normalized (relative) path
            elif normalized_f in duplicate_mapping:
                canonical = duplicate_mapping[normalized_f]
            # 3. Try with absolute path (if normalized is relative)
            elif not os.path.isabs(normalized_f):
                abs_f = os.path.abspath(normalized_f)
                if abs_f in duplicate_mapping:
                    canonical = duplicate_mapping[abs_f]
            # 4. Try original path as absolute (if it's relative)
            elif not os.path.isabs(f):
                abs_f_orig = os.path.abspath(f)
                if abs_f_orig in duplicate_mapping:
                    canonical = duplicate_mapping[abs_f_orig]
            # 5. Try normalized path for tag differences
            if canonical is None:
                tag_normalized = normalize_path_for_mapping(f)
                if tag_normalized in duplicate_mapping:
                    canonical = duplicate_mapping[tag_normalized]
                elif not os.path.isabs(tag_normalized):
                    abs_tag_normalized = os.path.abspath(tag_normalized)
                    if abs_tag_normalized in duplicate_mapping:
                        canonical = duplicate_mapping[abs_tag_normalized]
            # If still not found, use normalized path as canonical (keep original if it exists)
            if canonical is None:
                canonical = normalized_f
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
            lst = to_list_safe(row[col]) if is_list_like(row[col]) else []
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
                             if any(
                                 (duplicate_mapping.get(p) or duplicate_mapping.get(normalize_path_for_mapping(p)) or p) == canonical
                                 for p in (to_list_safe(row[c]) if is_list_like(row[c]) else [])
                             )}
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

def save_heatmap(dup_matrix, unique_counts, table_parquet_path, output_dir, is_percentage=False, file_suffix="", v2_suffix=""):
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
        df = pd.read_parquet(table_parquet_path, columns=['modelId', 'hugging_table_list', 'github_table_list', 'html_table_list_mapped', 'llm_table_list_mapped'])
        total_files = {}
        for res, col in {
            "Hugging": "hugging_table_list",
            "GitHub": "github_table_list",
            "HTML": "html_table_list_mapped",
            "S2ORC": "llm_table_list_mapped"
        }.items():
            total_files[res] = df[col].apply(lambda x: len(to_list_safe(x)) if is_list_like(x) else 0).sum()
        
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
    ax = sns.heatmap(dup_matrix_plot, annot=dup_matrix if not is_percentage else dup_matrix_plot, cmap=cmap, fmt=fmt, square=True, cbar=True, xticklabels=False, norm=norm)
    
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
        ax.text(xticks[idx] + 0.5, -0.05, res, ha='center', va='bottom', fontsize=fontsize)
    plt.tight_layout()
    
    # Save with appropriate filename (file_suffix for versioning: e.g. _251117 for tag, empty for v2-only)
    kind = "percentage" if is_percentage else "overlap"
    outpath = os.path.join(output_dir, f"heatmap_{kind}{v2_suffix}{file_suffix}.pdf")
    plt.savefig(outpath)
    print(f"Heatmap saved to {outpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate raw tables, prioritizing Hugging Face > GitHub > HTML > LLM")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    # Determine input/output paths with optional override from CLI.
    table_parquet_path = os.path.join(base_path, 'processed', f"modelcard_step3_merged{v2_suffix}{suffix}.parquet")
    output_parquet = os.path.join(base_path, 'processed', f"modelcard_step3_dedup{v2_suffix}{suffix}.parquet")
    output_dir = os.path.join(base_path, f"deduped{v2_suffix}{suffix}")
    fig_dir = os.path.join(base_path, 'analysis')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("📁 Paths in use:")
    print(f"   Input parquet:       {table_parquet_path}")
    print(f"   Output parquet:      {output_parquet}")
    print(f"   Output directory:    {output_dir}")
    print(f"   Figure directory:    {fig_dir}")
    
    DUPLICATE_MAPPING_JSON = os.path.join(output_dir, f"duplicate_mapping{suffix}.json")
    UNIQUE_FILES_TXT = os.path.join(output_dir, f"unique_files{suffix}.txt")
    DUPLICATE_GROUPS_JSON = os.path.join(output_dir, f"duplicate_groups{suffix}.json")
    STATS_PATH = os.path.join(output_dir, f"stats{suffix}.json")

    time_start = time.time()
    # --- Step 1: get the linked set (some files exist in local but not linked to model) ---
    df = pd.read_parquet(table_parquet_path, columns=['modelId', 'hugging_table_list', 'github_table_list', 'html_table_list_mapped', 'llm_table_list_mapped'])
    cols = ["hugging_table_list", "github_table_list", "html_table_list_mapped", "llm_table_list_mapped"]
    
    dirs_to_use = [
        {"path": f"data/processed/deduped_hugging_csvs{v2_suffix}{suffix}", "resource": "hugging", "priority": 1},
        {"path": f"data/processed/deduped_github_csvs{v2_suffix}{suffix}", "resource": "github", "priority": 2},
        {"path": f"data/processed/tables_output{v2_suffix}{suffix}", "resource": "html", "priority": 3},
        {"path": "data/processed/llm_tables", "resource": "llm", "priority": 4},
    ]
    print(f"📁 Using directories:")
    for d in dirs_to_use:
        print(f"   {d['resource']}: {d['path']}")

    linked_set = get_linked_set_from_parquet(df, cols)
    linked_set = set(linked_set)
    # Match by (resource, basename) so CitationLake/ModelTables/relative paths all align
    linked_set_basename_keys = set(
        (infer_resource_from_path(p), os.path.basename(p))
        for p in linked_set
        if isinstance(p, str) and os.path.basename(p) and infer_resource_from_path(p) is not None
    )
    print(f"Linked set size from parquet: {len(linked_set)} (basename keys: {len(linked_set_basename_keys)})")
    # intersection
    """
    file_paths = [item['path'] for item in dirs_to_use]
    existing_set = []
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
    files_info = valid_filelist_with_qc_from_local(dirs_to_use)
    # Filter local files that unlinked to modelcard and generic tables
    resource_totals = {res: 0 for res in RESOURCE_PRIORITY.keys()}
    resource_filtered = {res: 0 for res in RESOURCE_PRIORITY.keys()}
    resource_generic_filtered = {res: 0 for res in RESOURCE_PRIORITY.keys()}
    # Match by (resource, basename): parquet paths (CitationLake/ModelTables/any) align with local scan
    filtered_files_info = []
    for fi in tqdm(files_info, desc="Filtering files"):
        res = fi["resource"]
        resource_totals[res] += 1
        key = (res, os.path.basename(fi["file_path"]))
        if key not in linked_set_basename_keys:
            continue
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
    overall_unique_file = os.path.join(output_dir, f"overall_unique{suffix}.txt")
    with open(overall_unique_file, "w") as f:
        for file_path in overall_unique:
            f.write(file_path + "\n")
    print(f"Overall unique file count: {len(overall_unique)}")
    # save cross_unique_files
    cross_unique_files = stats["cross_unique_files"]
    for res, files in cross_unique_files.items():
        res_unique_file = os.path.join(output_dir, f"{res}_unique{suffix}.txt")
        with open(res_unique_file, "w") as f:
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
    # (resource, basename) -> local path; parquet paths (any prefix) map to local via this
    path_by_key = {(fi["resource"], os.path.basename(fi["file_path"])): fi["file_path"] for fi in filtered_files_info}
    for col in cols:
        total_before = df[col].apply(lambda x: len(to_list_safe(x)) if is_list_like(x) else 0).sum()
        print(f"Filtering {col}... Before: {total_before}")
        # Keep only paths that exist locally; map parquet path -> local path by (resource, basename)
        def map_to_local(path_list):
            if not is_list_like(path_list):
                return []
            out = []
            for p in to_list_safe(path_list):
                local = parquet_path_to_local_path(p, path_by_key)
                if local is not None:
                    out.append(local)
            return out
        df[col] = df[col].apply(map_to_local)
        total_after = df[col].apply(lambda x: len(to_list_safe(x)) if is_list_like(x) else 0).sum()
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
    to_parquet(df, output_parquet)
    print(f"Updated parquet saved as {output_parquet}")
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
    dup_matrix_file = os.path.join(output_dir, f"dup_matrix{suffix}.pkl")
    dup_matrix.to_pickle(dup_matrix_file)
    print(f"Dup matrix saved to {dup_matrix_file}")

    # Save both absolute and percentage heatmaps (with file_suffix for versioning: v2 vs v2_251117)
    save_heatmap(dup_matrix, stats["cross_unique_counts"], table_parquet_path, fig_dir, file_suffix=suffix, v2_suffix=v2_suffix)
    save_heatmap(dup_matrix, stats["cross_unique_counts"], table_parquet_path, fig_dir, is_percentage=True, file_suffix=suffix, v2_suffix=v2_suffix)
    print(f"Time taken: {time.time() - time_start} seconds")
