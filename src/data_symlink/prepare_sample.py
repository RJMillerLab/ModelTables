#!/usr/bin/env python3

import os
import csv
import argparse
import random

def collect_files_from_dir(directory, limit, seed=None):
    files = []
    for fname in os.listdir(directory):
        full_path = os.path.join(directory, fname)
        if os.path.isfile(full_path):  # accept both symlinks and regular files
            #files.append(fname)
            try:                     
                with open(full_path, newline='', encoding='utf-8') as f:  
                    reader = csv.reader(f)                                
                    header = next(reader, None)                          
                if not header or len(header) <= 1:                        
                    continue                                              
            except Exception:                                           
                continue                                                  
            files.append(fname)
    if seed is not None:
        random.seed(seed)
    return random.sample(files, min(limit, len(files)))

def main():
    parser = argparse.ArgumentParser(description="Randomly sample files from multiple subdirectories.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory, e.g., /u501/z6dong/Repo')
    parser.add_argument('--output_file', type=str, default='file_list.txt', help='Output file path')
    parser.add_argument('--limit', type=int, default=1000, help='Max files per subdir')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--tag', type=str, default=None, help='Tag suffix for versioning (e.g., 251117). If provided, uses tagged folders like deduped_hugging_csvs_v2_<tag>')
    parser.add_argument('--v2_mode', action='store_true', help='Use v2 mode.')

    args = parser.parse_args()

    v2_suffix = "_v2" if args.v2_mode else ""
    suffix = f"_{args.tag}" if args.tag else ""

    # Subdirectories to scan
    subdirs = [
        #"llm_tables",
        f"deduped_github_csvs{v2_suffix}{suffix}",
        f"deduped_hugging_csvs{v2_suffix}{suffix}",
        f"tables_output{v2_suffix}{suffix}"
    ]

    val_output_file = args.output_file.replace('.txt', '_val.txt') 
    with open(args.output_file, 'w') as train_f, open(val_output_file, 'w') as val_f: 
        for subdir in subdirs:
            abs_path = os.path.join(args.root_dir, "ModelTables", "data", "processed", subdir)
            if not os.path.exists(abs_path):
                print(f"Warning: {abs_path} does not exist. Skipping.")
                continue
            # Sample twice the limit and split into train/val 
            sampled_files = collect_files_from_dir(abs_path, args.limit * 2, args.seed)
            train_samples = sampled_files[:args.limit]
            val_samples = sampled_files[args.limit:]
            for fname in train_samples:
                train_f.write(fname + '\n')
            for fname in val_samples:
                val_f.write(fname + '\n')

    print(f"Generated {args.output_file}")
    print(f"Generated {val_output_file}")

if __name__ == '__main__':
    main()

