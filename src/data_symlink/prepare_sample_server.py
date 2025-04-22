#!/usr/bin/env python3

import os
import re
import argparse
import random

# Regex patterns for each type
PATTERNS = {
    "type1": re.compile(r'^[0-9]+\.[0-9]+v[0-9]+_table[0-9]+\.csv$'),
    "type2": re.compile(r'^[0-9]+_table[0-9]+\.csv$'),
    "type3": re.compile(r'^[a-f0-9]{32}_table_[0-9]+\.csv$'),
    "type4": re.compile(r'^(?![a-f0-9]{32}_table_)[a-f0-9]+_table[0-9]+\.csv$'),
}

def is_symlink_file(path):
    return os.path.islink(path) and os.path.isfile(path)

def collect_files(directory, pattern, limit, seed=None):
    files = []
    for fname in os.listdir(directory):
        full_path = os.path.join(directory, fname)
        if is_symlink_file(full_path) and pattern.match(fname):
            files.append(fname)
    if seed is not None:
        random.seed(seed)
    return random.sample(files, min(limit, len(files)))

def main():
    parser = argparse.ArgumentParser(description="Randomly sample matching symlink files from a datalake.")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory (e.g. /u4/z6dong/Repo)')
    parser.add_argument('--output', type=str, required=True, help='Subdirectory under root (e.g. scilake_large)')
    parser.add_argument('--output_file', type=str, default='file_list.txt', help='Output file path')
    parser.add_argument('--limit', type=int, default=1000, help='Max files per type')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducibility')

    args = parser.parse_args()
    datalake_dir = os.path.join(args.root_dir, 'starmie_internal', 'data', args.output, 'datalake')

    with open(args.output_file, 'w') as out_f:
        for tname, pattern in PATTERNS.items():
            matched = collect_files(datalake_dir, pattern, args.limit, args.seed)
            for fname in matched:
                out_f.write(fname + '\n')

    print(f"Generated {args.output_file}")

if __name__ == '__main__':
    main()

