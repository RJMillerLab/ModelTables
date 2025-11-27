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
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode base
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode str
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode all
    python -m src.data_symlink.ln_scilake --repo_root /u4/z6dong/Repo --mode base --tag 251117
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
    "tr_str": ("_tr_str",  "_s_t"),  ######## renamed from str_tr to tr_str
}


def create_symlink(src, target_dir, cache, file_suffix=""):
    """
    Create a symlink for src in target_dir, appending file_suffix if not already present.
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


def process_folder(source_folder, target_dir, cache, file_suffix):
    csvs = [os.path.join(source_folder, f) for f in os.listdir(source_folder)
            if f.lower().endswith('.csv') and os.path.isfile(os.path.join(source_folder, f))]
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
    parser.add_argument("--repo_root", type=str, default="/u4/z6dong/Repo", help="Repository root path.")
    parser.add_argument("--mode", type=str, choices=list(MODE_SUFFIX.keys())+["all"], default="base",
                        help="Mode for folder and file suffix. Use 'all' to run every mode.")
    parser.add_argument("--dir-name", type=str, default=None,
                        help="Override target directory name (default: scilake_final{suffix}). E.g., scilake_final_v2")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag suffix for versioning (e.g., 251117). If provided, uses tagged folders like deduped_hugging_csvs_v2_<tag>")
    args = parser.parse_args()

    modes = [args.mode] if args.mode != "all" else list(MODE_SUFFIX.keys())

    # Determine tag suffix for source folders
    tag_suffix = f"_{args.tag}" if args.tag else ""

    for mode in modes:
        dir_suffix, file_suffix = MODE_SUFFIX[mode]
        # Use v2 source directories for base data; augmented variants append suffixes to v2 dirs
        base_dirs = {
            "hugging": "deduped_hugging_csvs_v2",
            "github":  "deduped_github_csvs_v2",
            "html":    "tables_output_v2",
            "llm":     "llm_tables"
        }
        # Apply tag_suffix and dir_suffix to create augmented folder names
        # Note: llm_tables only gets tag_suffix if tag is provided, otherwise stays as "llm_tables"
        src_folders = [
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['hugging']}{tag_suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['github']}{tag_suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['html']}{tag_suffix}{dir_suffix}"),
            os.path.join(args.repo_root, "CitationLake", "data", "processed", f"{base_dirs['llm']}{tag_suffix}{dir_suffix}" if args.tag else f"{base_dirs['llm']}{dir_suffix}")
        ]
        dir_name = args.dir_name if args.dir_name else f"scilake_final{dir_suffix}"
        target_dir = os.path.join(
            args.repo_root,
            "starmie_internal", "data",
            dir_name,
            "datalake"
        )  ######## build correct target directory

        print(f"\nMode={mode}, target_dir={target_dir}, file_suffix={file_suffix}")
        if args.tag:
            print(f"Using tag: {args.tag}")
        print(f"Source folders to scan:")
        for src in src_folders:
            exists = os.path.isdir(src)
            status = "✓ EXISTS" if exists else "✗ MISSING"
            print(f"  {status}: {src}")
        
        os.makedirs(target_dir, exist_ok=True)
        cache = {f for f in os.listdir(target_dir) if f.lower().endswith('.csv')}

        for src in src_folders:
            if not os.path.isdir(src):
                print(f"Skip missing source folder {src}")
                continue
            process_folder(src, target_dir, cache, file_suffix)

        total = len([f for f in os.listdir(target_dir) if f.lower().endswith('.csv')])
        print(f"Done mode={mode}: total CSV in {target_dir} = {total}\n")

if __name__ == "__main__":
    main()
