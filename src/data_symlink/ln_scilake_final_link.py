#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate variant filelists for _s, _t, and _s_t CSVs
based on an existing filelist.
"""

import os
import argparse

def generate_variant_filelists(filelist_path: str):
    # e.g. filelist_path = "scilake_final_filelist.txt"
    base_name, ext = os.path.splitext(os.path.basename(filelist_path))
    if ext.lower() != ".txt":
        raise ValueError(f"Expect a .txt filelist, got {ext}")
    dir_path = os.path.dirname(filelist_path) or "."

    # 1) Read the original filelist
    with open(filelist_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 2) Define variants
    #variants = {
    #    '_s'  : f"{base_name}_s{ext}",
    #    '_t'  : f"{base_name}_t{ext}",
    #    '_s_t': f"{base_name}_s_t{ext}",
    #}
    # Determine output filenames, preserving _val in list name but not in content suffix
    if base_name.endswith('_val'):
        # root without '_val'
        root = base_name[:-4]
        content_suffixes = ['_s', '_t', '_s_t']
        variants = {}
        for cs in content_suffixes:
            # output name keeps _val after suffix
            variants[cs] = f"{root}{cs}_val{ext}"
    else:
        content_suffixes = ['_s', '_t', '_s_t']
        variants = {cs: f"{base_name}{cs}{ext}" for cs in content_suffixes}
    # 3) Write new list
    for suffix, output_name in variants.items():
        output_path = os.path.join(dir_path, output_name)
        with open(output_path, 'w', encoding='utf-8') as fout:
            for filename in lines:
                name, file_ext = os.path.splitext(filename)
                # only add content suffix (no '_val')
                fout.write(f"{name}{suffix}{file_ext}\n")
        print(f"[OK] Generated {output_path} ({len(lines)} entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scilake_final_filelist_* variants"
    )
    parser.add_argument(
        '-f', '--filelist',                ########
        dest='filelists',                  ########
        nargs='+',                         ########
        required=True,                     ########
        help="Paths to filelists (original and optional _val), e.g. scilake_final_filelist.txt scilake_final_filelist_val.txt"
    )
    args = parser.parse_args()
    for fl in args.filelists:              ########
        generate_variant_filelists(fl)