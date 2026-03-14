#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-04-15
Last Modified: 2025-04-30
Script Description:
    This script scans several fixed source folders (relative to the repository root) for CSV files and 
    creates symbolic links in a target directory (also fixed relative to the repository root).
    It uses an incremental cache to only process new files and prints statistics for each folder.
    Parallel processing is implemented with joblib and a progress bar is shown using tqdm.

    The target directory is derived from the fixed base path:
        <repo_root>/starmie_internal/data/scilake_final{dir_suffix}/datalake
    And based on the provided mode, a suffix is appended to the filename:
        - base     -> dir_suffix="",         file_suffix=""
        - str      -> dir_suffix="_str",     file_suffix="_s"
        - tr       -> dir_suffix="_tr",      file_suffix="_t"
        - tr_str   -> dir_suffix="_tr_str",  file_suffix="_s_t"

Usage:
    python -m src.data_symlink.ln_scilake --repo_root /u501/z6dong/Repo --mode base --tag 251117
""" 

import os
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

# Mapping from mode to (directory suffix, file suffix)
MODE_SUFFIX = {
    "base":   ("",         ""),     ######## define suffix mapping
    "str":    ("_str",     "_s"),    ########
    "tr":     ("_tr",      "_t"),    ########
    #"tr_str": ("_tr_str",  "_s_t"),  ######## renamed from str_tr to tr_str
}

def create_symlink(src, target_dir, cache, file_suffix=""):
    """
    Create a symlink for src in target_dir, appending file_suffix if not already present.
    Target dir is symlinks only; overwrites existing (e.g. broken) links.
    """
    basename = os.path.basename(src)
    name, ext = os.path.splitext(basename)
    if file_suffix and not basename.endswith(f"{file_suffix}{ext}"):
        target_name = f"{name}{file_suffix}{ext}"  ######## only append if missing
    else:
        target_name = basename
    target_path = os.path.join(target_dir, target_name)
    if basename in cache:
        return False
    try:
        if os.path.lexists(target_path):
            os.remove(target_path)
        os.symlink(src, target_path)
    except Exception as e:
        print(f"Error linking {src} -> {target_path}: {e}")
        return False
    return True


def load_mask_file(mask_file_path):
    """
    Load mask file and return a set of allowed base filenames (without file_suffix).
    The mask file contains full paths, we extract basenames and normalize them.
    Returns a set of base names (e.g., 'file.csv' for both 'file.csv' and 'file_s.csv').
    """
    if not mask_file_path or not os.path.exists(mask_file_path):
        return None
    
    allowed_base_names = set()
    with open(mask_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract basename from full path
            basename = os.path.basename(line)
            if basename.endswith('.csv'):
                name, ext = os.path.splitext(basename)
                # Remove file_suffix if present (e.g., 'file_s' -> 'file', 'file_t' -> 'file')
                for suffix in ['_s_t', '_t_s', '_s', '_t']:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break
                # Store the base name (without suffix)
                allowed_base_names.add(f"{name}{ext}")
    
    return allowed_base_names


def process_folder(source_folder, target_dir, cache, file_suffix, mask_set=None):
    csvs = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
            if f.lower().endswith('.csv') and os.path.isfile(os.path.join(source_folder, f))]
    
    # Apply mask filter if provided
    if mask_set is not None:
        # Check if file's base name (without file_suffix) is in mask
        filtered_csvs = []
        for csv_path in csvs:
            basename = os.path.basename(csv_path)
            name, ext = os.path.splitext(basename)
            # Remove file_suffix to get base name
            base_name = name
            if file_suffix and name.endswith(file_suffix):
                base_name = name[:-len(file_suffix)]
            # Check if base name is in mask
            base_filename = f"{base_name}{ext}"
            if base_filename in mask_set:
                filtered_csvs.append(csv_path)
        csvs = filtered_csvs
    
    to_link = [p for p in csvs if os.path.basename(p) not in cache]   ########
    if not to_link:
        print(f"{source_folder}: no new files to link.")
        return

    print(f"{source_folder}: linking {len(to_link)} new CSVs...")
    results = Parallel(n_jobs=4, backend='threading')(        ########
        delayed(create_symlink)(path, target_dir, cache, file_suffix)
        for path in to_link
    )
    processed = sum(results)
    skipped = len(to_link) - processed
    print(f"{source_folder}: total_new={len(to_link)}, linked={processed}, skipped={skipped}")
    for path in to_link:
        cache.add(os.path.basename(path))


def main():
    parser = argparse.ArgumentParser(description="Incremental CSV symlinker with mode-based suffixes.")
    parser.add_argument("--repo_root", type=str, default="/u501/z6dong/Repo", help="Repository root path.")
    parser.add_argument("--mode", type=str, choices=list(MODE_SUFFIX.keys())+["all"], default="base", help="Mode for folder and file suffix. ")
    parser.add_argument("--tag", type=str, default=None,help="Tag suffix for versioning (e.g., 251117). If provided, uses tagged folders like deduped_hugging_csvs_v2_<tag>")
    parser.add_argument("--v2_mode", action="store_true", help="Use v2 mode.")
    args = parser.parse_args()

    modes = [args.mode] if args.mode != "all" else list(MODE_SUFFIX.keys())

    # Determine tag suffix for source folders
    v2_suffix = "_v2" if args.v2_mode else ""
    suffix = f"_{args.tag}" if args.tag else ""

    mask_set = load_mask_file(os.path.join(args.repo_root, "ModelTables", "data", "analysis", f"all_valid_title_valid{v2_suffix}{suffix}.txt"))

    for mode in modes:
        dir_suffix, file_suffix = MODE_SUFFIX[mode]
        src_folders = [
            os.path.join(args.repo_root, "ModelTables", "data", "processed", f"deduped_hugging_csvs{v2_suffix}{suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "ModelTables", "data", "processed", f"deduped_github_csvs{v2_suffix}{suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "ModelTables", "data", "processed", f"tables_output{v2_suffix}{suffix}{dir_suffix}"),
            #os.path.join(args.repo_root, "ModelTables", "data", "processed", f"llm_tables{dir_suffix}")
        ]
        target_dir = os.path.join(args.repo_root, "starmie_internal", "data", f"scilake_final{suffix}{dir_suffix}", "datalake")

        print(f"\nMode={mode}, target_dir={target_dir}, file_suffix={file_suffix}")
        print(f"Source folders to scan:")
        for src in src_folders:
            exists = os.path.isdir(src)
            status = "✓ EXISTS" if exists else "✗ MISSING"
            print(f"  {status}: {src}")
        
        os.makedirs(target_dir, exist_ok=True)
        # Only treat as "existing" if target exists (valid link or real file); broken symlinks will be re-linked.
        # Cache = source basenames already present in target (map target name back using file_suffix).
        cache = set()
        for f in os.listdir(target_dir):
            if not f.lower().endswith('.csv'):
                continue
            if not os.path.exists(os.path.join(target_dir, f)):
                continue
            name, ext = os.path.splitext(f)
            base_name = name[:-len(file_suffix)] if file_suffix and name.endswith(file_suffix) else name
            cache.add(base_name + ext)

        for src in src_folders:
            if not os.path.isdir(src):
                print(f"Skip missing source folder {src}")
                continue
            process_folder(src, target_dir, cache, file_suffix, mask_set)

        total = len([f for f in os.listdir(target_dir) if f.lower().endswith('.csv')])
        print(f"Done mode={mode}: total CSV in {target_dir} = {total}\n")

if __name__ == "__main__":
    main()
