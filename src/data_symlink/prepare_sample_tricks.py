#!/usr/bin/env python3

import os

def generate_augmented_filenames(input_file):
    with open(input_file, 'r') as f:
        original_files = [line.strip() for line in f if line.strip()]

    def with_suffix(suffix):
        return [f"{os.path.splitext(f)[0]}{suffix}.csv" for f in original_files]

    str_files = with_suffix("_str")
    tr_files = with_suffix("_tr")
    str_tr_files = with_suffix("_str_tr")

    base_name = os.path.splitext(input_file)[0]

    with open(f"{base_name}_str_filelist.txt", 'w') as f:
        f.write('\n'.join(str_files) + '\n')

    with open(f"{base_name}_tr_filelist.txt", 'w') as f:
        f.write('\n'.join(tr_files) + '\n')

    with open(f"{base_name}_str_tr_filelist.txt", 'w') as f:
        f.write('\n'.join(str_tr_files) + '\n')

    print("Generated:")
    print(f"  {base_name}_str_filelist.txt")
    print(f"  {base_name}_tr_filelist.txt")
    print(f"  {base_name}_str_tr_filelist.txt")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate *_str, *_tr, *_str_tr filelists from a base filelist.")
    parser.add_argument('--input_file', type=str, required=True, help='Input file containing list of .csv filenames')
    args = parser.parse_args()
    generate_augmented_filenames(args.input_file)

