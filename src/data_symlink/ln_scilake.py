#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-04-15
Last Modified: 2025-04-15
Script Description:
    This script scans several fixed source folders (relative to the repository root) for CSV files and 
    creates symbolic links in a target directory (also fixed relative to the repository root). 
    It uses an incremental cache to only process new files and prints statistics for each folder.
    Parallel processing is implemented with joblib and a progress bar is shown using tqdm.
    
    The target directory is derived from the fixed base path:
        target_dir = <repo_root>/starmie_internal/data/scilake_large/datalake
    And based on the provided mode, a suffix is appended:
        - "base": links are created in target_dir.
        - "str": links are created in target_dir + "_str".
        - "tr": links are created in target_dir + "_tr".
        - "str_tr": links are created in target_dir + "_tr_str".
    
Usage:
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode base
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode all
"""

import os
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

def create_symlink(src, target_dir, cache):
    """
    Creates a symbolic link for the given source CSV file in the target directory.
    Uses the cache to skip files that have already been processed.
    """
    basename = os.path.basename(src)
    target_path = os.path.join(target_dir, basename)
    if basename in cache:
        return False
    try:
        # If a symbolic link already exists, remove it before creating a new one
        if os.path.lexists(target_path):
            os.remove(target_path)
        os.symlink(src, target_path)
    except Exception as e:
        print(f"Error linking {src} -> {target_path}: {e}")
        return False
    return True

def process_folder(source_folder, target_dir, cache):
    """
    Processes a single source folder:
      - Lists all CSV files in the folder (non-recursively).
      - Uses joblib with tqdm to create symbolic links for new files.
      - Prints statistics including total files, processed count, and skipped count.
      - Updates the cache with processed file basenames.
    """
    # List all CSV files in the source folder (non-recursive)
    csv_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
                 if os.path.isfile(os.path.join(source_folder, f)) and f.lower().endswith('.csv')]
    total_files = len(csv_files)
    # Parallel processing with joblib and tqdm for progress visualization
    results = Parallel(n_jobs=-1)(
        delayed(create_symlink)(csv, target_dir, cache) for csv in tqdm(csv_files, desc=f"Scanning {source_folder}")
    )
    processed = sum(1 for r in results if r)
    skipped = total_files - processed
    print(f"Folder: {source_folder}, CSV count: {total_files}, Processed: {processed}, Skipped: {skipped}")
    # Update cache with all file basenames to avoid re-processing in the future
    for csv in csv_files:
        cache.add(os.path.basename(csv))

def main():
    parser = argparse.ArgumentParser(
        description="Incrementally create CSV symbolic links using joblib and tqdm with mode support. "
                    "Only repository root and mode need to be provided."
    )
    parser.add_argument("--repo_root", type=str, default="/u4/z6dong/Repo", help="Repository root directory.")
    parser.add_argument("--output", type=str, default="starmie_internal/data/scilake_large/datalake",
                        help="Target directory for symbolic links (relative to repo root).")
    parser.add_argument("--mode", type=str, choices=["base", "str", "tr", "str_tr", "all"], default="base",
                        help="Mode for processing to determine target folder naming. Use 'all' to process all modes.")
    args = parser.parse_args()

    # Define fixed target directory base and source folder list relative to the repository root
    target_dir_base = os.path.join(args.repo_root, "starmie_internal", "data", args.output, "datalake")
    folders = [
        os.path.join(args.repo_root, "CitationLake", "data", "processed", "deduped_hugging_csvs"),
        os.path.join(args.repo_root, "CitationLake", "data", "processed", "deduped_github_csvs"),
        os.path.join(args.repo_root, "CitationLake", "data", "processed", "tables_output"),
        os.path.join(args.repo_root, "CitationLake", "data", "processed", "llm_tables")
    ]
    
    # If mode is "all", process each individual mode sequentially
    if args.mode == "all":
        modes = ["base", "str", "tr", "str_tr"]
        for mode in modes:
            if mode == "base":
                final_target_dir = target_dir_base
            elif mode == "str":
                final_target_dir = target_dir_base + "_str"
            elif mode == "tr":
                final_target_dir = target_dir_base + "_tr"
            elif mode == "str_tr":
                final_target_dir = target_dir_base + "_tr_str"
            else:
                final_target_dir = target_dir_base

            print(f"\nProcessing mode: {mode}")
            # Create target directory if it doesn't exist
            os.makedirs(final_target_dir, exist_ok=True)

            # Build cache: gather all CSV files already present in the target directory by filename
            cache = set()
            for f in os.listdir(final_target_dir):
                if os.path.isfile(os.path.join(final_target_dir, f)) and f.lower().endswith('.csv'):
                    cache.add(f)
            print(f"Initial processed CSV count in target ({final_target_dir}): {len(cache)}")

            # Process each fixed source folder and create symbolic links for CSV files
            for source_folder in folders:
                if not os.path.exists(source_folder):
                    print(f"Source folder does not exist: {source_folder}, skipping.")
                    continue
                process_folder(source_folder, final_target_dir, cache)

            # After processing, count total CSV files (symbolic links) in the final target directory
            total_count = len([f for f in os.listdir(final_target_dir)
                               if os.path.isfile(os.path.join(final_target_dir, f)) and f.lower().endswith('.csv')])
            print(f"Total CSV count in target directory ({final_target_dir}): {total_count}\n")
    else:
        # Determine the final target directory based on the specified mode
        if args.mode == "base":
            final_target_dir = target_dir_base
        elif args.mode == "str":
            final_target_dir = target_dir_base + "_str"
        elif args.mode == "tr":
            final_target_dir = target_dir_base + "_tr"
        elif args.mode == "str_tr":
            final_target_dir = target_dir_base + "_tr_str"
        else:
            final_target_dir = target_dir_base

        # Create target directory if it doesn't exist
        os.makedirs(final_target_dir, exist_ok=True)

        # Build cache: gather all CSV files already present in the target directory by filename
        cache = set()
        for f in os.listdir(final_target_dir):
            if os.path.isfile(os.path.join(final_target_dir, f)) and f.lower().endswith('.csv'):
                cache.add(f)
        print(f"Initial processed CSV count in target: {len(cache)}")

        # Process each fixed source folder and create symbolic links for CSV files
        for source_folder in folders:
            if not os.path.exists(source_folder):
                print(f"Source folder does not exist: {source_folder}, skipping.")
                continue
            process_folder(source_folder, final_target_dir, cache)

        # After processing, count total CSV files (symbolic links) in the final target directory
        total_count = len([f for f in os.listdir(final_target_dir)
                           if os.path.isfile(os.path.join(final_target_dir, f)) and f.lower().endswith('.csv')])
        print(f"Total CSV count in target directory: {total_count}")

if __name__ == "__main__":
    main()
