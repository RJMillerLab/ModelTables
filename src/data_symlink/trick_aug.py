#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Created: 2025-04-15
Last Modified: 2025-04-15
Description: This script processes CSV files in a specified folder using different modes:
- "transpose": Transposes the CSV data.
- "str": Converts each cell to the string format "colname-value".
- "str_transpose": Converts each cell to "colname-value" and then transposes the data.
Usage: python -m src.data_symlink.trick_aug --mode str --repo_root /u4/z6dong/Repo
"""

import os
import csv
from joblib import Parallel, delayed  # for parallel processing
from tqdm import tqdm  # for progress bar display
import argparse  # for command-line argument parsing

def remove_suffixes(base_with_suffix):
    if base_with_suffix.endswith("_s_t") or base_with_suffix.endswith("_t_s"):
        return base_with_suffix[:-4]
    elif base_with_suffix.endswith("_t"):
        return base_with_suffix[:-2]
    elif base_with_suffix.endswith("_s"):
        return base_with_suffix[:-2]
    else:
        return base_with_suffix

def get_processed_bases(target_folder):
    """
    Reads the processed CSV files in the target folder.
    Removes suffixes (_t, _s, _s_t, _t_s) from filenames and returns a set of base names.
    """
    processed_bases = set()
    if os.path.exists(target_folder):
        for file in os.listdir(target_folder):
            if file.endswith(".csv"):
                base_with_suffix = file[:-4]  # remove ".csv"
                base = remove_suffixes(base_with_suffix)
                processed_bases.add(base)
    return processed_bases

def process_file_transpose(csv_in, csv_out):
    """
    Mode: "transpose"
    1. Moves the header to the first row of data,
    2. Normalizes row lengths,
    3. Transposes the CSV data.
    """
    try:
        with open(csv_in, "r", newline="", encoding="utf-8") as f_in:
            reader = list(csv.reader(f_in))
    except Exception as e:
        return f"Error reading file {csv_in}: {e}"
    if not reader:
        open(csv_out, "w", encoding="utf-8").close()
        return f"Empty file, created output: {csv_out}"

    header = reader[0]
    data_rows = reader[1:]
    data_with_header = [header] + data_rows
    max_len = max(len(row) for row in data_with_header)
    normalized = [row + [""] * (max_len - len(row)) for row in data_with_header]
    transposed = list(zip(*normalized))
    try:
        with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            for row in transposed:
                writer.writerow(row)
    except Exception as e:
        return f"Error writing file {csv_out}: {e}"
    return f"Processed (transpose): {csv_in} -> {csv_out}"

def process_file_str(csv_in, csv_out):
    """
    Mode: "str"
    Processes each row by converting each value to the string format "colname-value"
    without transposing the data.
    """
    try:
        with open(csv_in, "r", newline="", encoding="utf-8") as f_in:
            reader = list(csv.reader(f_in))
    except Exception as e:
        return f"Error reading file {csv_in}: {e}"
    if not reader:
        open(csv_out, "w", encoding="utf-8").close()
        return f"Empty file, created output: {csv_out}"

    header = reader[0]
    data_rows = reader[1:]
    processed_data = []
    for row in data_rows:
        new_row = []
        for i, cell in enumerate(row):
            col_name = header[i] if i < len(header) else f"col{i}"
            new_row.append(f"{col_name}-{cell}")
        processed_data.append(new_row)

    try:
        with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(processed_data)
    except Exception as e:
        return f"Error writing file {csv_out}: {e}"
    return f"Processed (str conversion): {csv_in} -> {csv_out}"

def process_file_str_transpose(csv_in, csv_out):
    """
    Mode: "str_transpose"
    Converts each cell to the "colname-value" format and then transposes the processed data.
    """
    try:
        with open(csv_in, "r", newline="", encoding="utf-8") as f_in:
            reader = list(csv.reader(f_in))
    except Exception as e:
        return f"Error reading file {csv_in}: {e}"
    if not reader:
        open(csv_out, "w", encoding="utf-8").close()
        return f"Empty file, created output: {csv_out}"

    header = reader[0]
    data_rows = reader[1:]
    # Normalize each row to the header length
    normalized_rows = [row + [""] * (len(header) - len(row)) for row in data_rows]
    processed_data = []
    for row in normalized_rows:
        new_row = []
        for i, cell in enumerate(row):
            col_name = header[i] if i < len(header) else f"col{i}"
            new_row.append(f"{col_name}-{cell}")
        processed_data.append(new_row)
    # Transpose the processed data: each column becomes a row, with the header as the first element
    out_rows = []
    for i in range(len(header)):
        new_row = [header[i]]
        for row in processed_data:
            new_row.append(row[i] if i < len(row) else "")
        out_rows.append(new_row)
    try:
        with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerows(out_rows)
    except Exception as e:
        return f"Error writing file {csv_out}: {e}"
    return f"Processed (str conversion and transpose): {csv_in} -> {csv_out}"

def process_folder(mode, folder, repo_root):
    """
    Processes CSV files in the specified folder using the given mode.
    - Constructs input and output folder paths based on the mode.
    - Retrieves the list of original CSV files and the set of processed base names.
    - Only processes files that have not been processed yet.
    - Prints out statistical information.
    """
    input_folder = os.path.join(repo_root, folder)
    if mode == "transpose":
        output_folder = input_folder + "_tr"
    elif mode == "str":
        output_folder = input_folder + "_str"
    elif mode == "str_transpose":
        output_folder = input_folder + "_tr_str"
    else:
        return f"Invalid mode: {mode}"

    os.makedirs(output_folder, exist_ok=True)

    # Recursively collect all original CSV files
    original_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                original_files.append(os.path.join(root, file))
    total_files = len(original_files)

    # Get the set of processed file base names in the output folder
    processed_bases = get_processed_bases(output_folder)

    # Filter files to process (those whose base name is not in processed_bases)
    files_to_process = []
    for csv_file in original_files:
        base_with_suffix = os.path.basename(csv_file)[:-4]
        base = remove_suffixes(base_with_suffix)
        if base not in processed_bases:
            files_to_process.append(csv_file)

    num_to_process = len(files_to_process)
    num_skipped = total_files - num_to_process

    print(f"Source folder: {input_folder}")  ########
    print(f"Target folder: {output_folder}")  ########
    print(f"  Total original CSV files: {total_files}")
    print(f"  Already processed (skipped): {num_skipped}")
    print(f"  To process (update): {num_to_process}")

    # Choose processing function and file suffix based on mode
    if mode == "transpose":
        process_func = process_file_transpose
        suffix = "_t.csv"
    elif mode == "str":
        process_func = process_file_str
        suffix = "_s.csv"
    elif mode == "str_transpose":
        process_func = process_file_str_transpose
        suffix = "_s_t.csv"
    else:
        return

    # Process files in parallel with a progress bar
    results = Parallel(n_jobs=-1)(
        delayed(process_func)(
            csv_file,
            os.path.join(output_folder, os.path.splitext(os.path.basename(csv_file))[0] + suffix)
        )
        for csv_file in tqdm(files_to_process, desc=f"Processing {input_folder}", leave=False)
    )
    return results

def main():
    # Parse command-line arguments to choose the processing mode
    parser = argparse.ArgumentParser(description="Process CSV files in different modes")
    parser.add_argument("--mode", type=str, choices=["transpose", "str", "str_transpose"], default="transpose", help="Processing mode")
    parser.add_argument("--repo_root", type=str, default="/u4/z6dong/Repo", help="Repository root directory.")
    args = parser.parse_args()
    
    folders = [
        "CitationLake/data/processed/deduped_hugging_csvs",
        "CitationLake/data/processed/deduped_github_csvs",
        "CitationLake/data/processed/tables_output",
        "CitationLake/data/processed/llm_tables"
    ]
    for folder in folders:
        full_folder = os.path.join(args.repo_root, folder)
        print(f"\nProcessing folder: {full_folder} with mode {args.mode}")
        results = process_folder(args.mode, folder, args.repo_root)
        if isinstance(results, list):
            for res in results:
                print(res)
        else:
            print(results)
    print("All processing done.")

if __name__ == "__main__":
    main()
