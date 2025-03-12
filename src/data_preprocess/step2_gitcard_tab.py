"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2025-03-11
Description: Extract markdown tables from GitHub | Huggingface html, modelcards, readme files, and save them to CSV files.
Usage:
    python -m src.data_preprocess.step2_gitcard_tab
Tips: We deduplicate the card content, and use the symlink to save data storage.
"""

import os, re, time, logging, hashlib, json
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from shutil import copytree
import shutil
from src.data_ingestion.readme_parser import MarkdownHandler
from src.utils import load_data, load_config

tqdm.pandas()

def compute_file_hash(file_path):
    """
    Compute SHA256 hash of a file for deduplication.
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def setup_logging(log_filename):
    """
    Set up logging configuration.
    """
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging started.")

########
def detect_and_extract_markdown_tables(content: str):
    """
    Detect and extract all Markdown tables (supporting multi-line) from the given text.
    Returns a tuple (bool, list), where the boolean indicates whether at least one table was found,
    and the list contains all matched table strings; if no match is found, returns (False, []).
    """
    if not isinstance(content, str):
        return (False, [])
    markdown_table_pattern = r"(?:\|[^\n]*?\|[\s]*\n)+\|[-:| ]*\|[\s]*\n(?:\|[^\n]*?\|(?:\n|$))+"
    matches = re.findall(markdown_table_pattern, content, re.MULTILINE)
    matches = [match.strip() for match in matches] if matches else []
    return (len(matches) > 0, matches)
########

def extract_markdown_tables_in_parallel(df_unique, col_name, n_jobs=-1):
    """
    Perform parallel extraction of Markdown tables from the specified column (a string)
    in the DataFrame using joblib.Parallel.
    Returns a list of tuples (bool, list) for each row.
    """
    #results = Parallel(n_jobs=n_jobs)(
    #    delayed(detect_and_extract_markdown_tables)(content) for content in tqdm(df[col_name], total=len(df))
    #)
    #return results
    contents = df_unique[col_name].tolist()
    indices = df_unique.index.tolist()
    results = Parallel(n_jobs=n_jobs)(
        delayed(detect_and_extract_markdown_tables)(content) for content in tqdm(contents, total=len(contents))
    )
    # Build a dict of row_index -> (found_tables_bool, tables_list)
    row_index_to_result = {}
    for i, r_idx in enumerate(indices):
        row_index_to_result[r_idx] = results[i]
    return row_index_to_result

########
def save_markdown_to_csv_from_content(model_id, content, source, file_idx, output_folder):
    """
    Given file content, extract Markdown tables and save them to CSV files.
    The CSV filenames are generated as "modelId_{source}_git_readme{file_idx}_table{table_idx}.csv".
    Returns the list of saved CSV file paths.
    """
    _, tables = detect_and_extract_markdown_tables(content)
    saved_paths = []
    for table_idx, table in enumerate(tables, start=1):
        identifier = f"git_readme{file_idx}_table{table_idx}"
        csv_path = generate_csv_path(model_id, source, identifier, output_folder)
        MarkdownHandler.markdown_to_csv(table, csv_path)
        saved_paths.append(csv_path)
    return saved_paths
########

def generate_csv_path(model_id, source, identifier, folder):
    """
    Generate a unique CSV file path using modelId, source (e.g., hugging or github),
    and an identifier string (e.g., "git_readme1_table2").
    """
    model_id = model_id.replace('/', '_') # Bug here: notice if not replaced, it will cause saving error without any warning
    filename = f"{model_id}_{source}_{identifier}.csv"
    return os.path.join(folder, filename)

########
def process_github_readmes(row, output_folder, config):
    """
    Process GitHub readme files for one row.
    For each file in row['readme_path'] (a list of paths), read the file,
    extract all Markdown tables, save each table to a CSV file, and return a list
    of the saved CSV file paths.
    """
    model_id = row["modelId"]
    readme_paths = row["readme_path"]
    readme_paths = [os.path.join(config.get('base_path'), csv_path.split("data", 1)[-1].lstrip("/\\")) for csv_path in readme_paths]
    
    #print('readme_paths: ', readme_paths)
    #print('type:', type(readme_paths))
    csv_files = []
    if not isinstance(readme_paths, (list, np.ndarray, tuple)) or len(readme_paths) == 0:
        print(f"Skipping {model_id} due to missing or empty readme paths.")
        return csv_files
    # Process each file (with index starting at 1 for naming)
    for file_idx, path in enumerate(readme_paths, start=1):
        if path and os.path.exists(path):
            #print('exists path: ', path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                saved = save_markdown_to_csv_from_content(model_id, content, "git", file_idx, output_folder)
                #print('saved: ', saved)
                csv_files.extend(saved)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return csv_files
########

def process_markdown_files(github_folder, output_folder):
    """
    Process all markdown files in github_folder and save them as CSV in output_folder.
    Skips files larger than 100MB, but still records them in the mapping with value None.
    If no tables are found, records an empty list.
    """
    os.makedirs(output_folder, exist_ok=True)
    markdown_files = [f for f in os.listdir(github_folder) if f.endswith(".md")]
    md_to_csv_mapping = {}
    for md_file in tqdm(markdown_files, desc="Processing Markdown files"):
        md_path = os.path.join(github_folder, md_file)
        # Check file size (skip if > 5MB, but still record in mapping)
        if os.path.getsize(md_path) > 5 * 1024 * 1024:  # 5MB threshold
            print(f"⚠️ Skipping {md_file} (File too large: {os.path.getsize(md_path) / (1024 * 1024):.2f} MB)")
            md_to_csv_mapping[md_file.replace(".md", "")] = None  # Record as None
            continue
        base_csv_name = md_file.replace(".md", "")
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        _, tables = detect_and_extract_markdown_tables(md_content)
        table_csv_basenames = []
        for i, table in enumerate(tables):
            csv_basename = f"{base_csv_name}_table_{i}.csv"
            csv_path = os.path.join(output_folder, csv_basename)
            MarkdownHandler.markdown_to_csv(table, csv_path)
            table_csv_basenames.append(csv_basename)
        # If no tables found, store an empty list instead of skipping
        md_to_csv_mapping[base_csv_name] = table_csv_basenames if table_csv_basenames else []
    # Save mapping as JSON for reference
    mapping_json_path = os.path.join(output_folder, "md_to_csv_mapping.json")
    with open(mapping_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(md_to_csv_mapping, json_file, indent=4)
    return md_to_csv_mapping

def create_symlinks(md_to_csv_mapping, output_folder_symlinks):
    """
    Create symbolic links for GitHub processed CSVs in a separate directory to avoid data duplication.
    """
    os.makedirs(output_folder_symlinks, exist_ok=True)
    symlink_mapping = {}  # Store symlinked paths for later use
    for md_basename, csv_paths in md_to_csv_mapping.items():
        if not csv_paths:
            symlink_mapping[md_basename] = []
            continue
        symlinked_csv_paths = []
        for csv_basename in csv_paths:
            csv_full_path = os.path.join(output_folder_symlinks.replace("symlinked_github_csvs", "cleaned_markdown_csvs_github"), csv_basename)
            if not os.path.exists(csv_full_path):
                # If we don't find the actual CSV in "cleaned_markdown_csvs_github", skip
                # (or you can adapt logic if you want to locate them differently)
                continue
            symlink_path = os.path.join(output_folder_symlinks, os.path.basename(csv_basename))
            try:
                if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                    os.unlink(symlink_path)  # Remove existing symlink if it exists
                os.symlink(os.path.abspath(csv_full_path), symlink_path)  # Create symlink
                symlinked_csv_paths.append(symlink_path)
            except Exception as e:
                print(f"❌ Error creating symlink for {csv_full_path}: {e}")
        symlink_mapping[md_basename] = symlinked_csv_paths
    return symlink_mapping

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    print("⚠️Step 1: Loading modelcard_step1 data...")
    df_modelcard = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_step1.parquet"), columns=['modelId', 'card_readme', 'github_link'])
    df_giturl = pd.read_parquet(os.path.join(processed_base_path, "github_readmes_info.parquet"))
    df_merged = pd.merge(df_modelcard, df_giturl[['modelId', 'readme_path']], on='modelId', how='left')
    print(f"✅ After merge: {len(df_merged)} rows.")

    # ---------- HuggingFace part (use original code) ----------
    print("⚠️Step 2: Extracting markdown tables from 'card_readme' (HuggingFace)...")
    # 1) Compute a hash for each row's card_readme.
    df_merged['readme_hash'] = df_merged['card_readme'].apply(
        lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest() if isinstance(x, str) else None
    )
    # 2) Identify the unique readme rows to parse exactly once
    df_unique = df_merged.drop_duplicates(subset=['readme_hash'], keep='first').copy()
    print(f"   Found {len(df_unique)} unique readme content out of {len(df_merged)} rows.")
    # 3) Extract markdown tables for each *unique* readme
    row_index_to_result = extract_markdown_tables_in_parallel(df_unique, col_name='card_readme', n_jobs=4)
    # 4) Store the extraction results back in df_unique (two columns: 'found', 'extracted_tables')
    df_unique['found_tables'] = df_unique.index.map(lambda idx: row_index_to_result[idx][0])
    df_unique['extracted_markdown_table_hugging'] = df_unique.index.map(lambda idx: row_index_to_result[idx][1])
    # We'll store these results in a dict: readme_hash -> list_of_tables
    hash_to_tables_map = {}
    for idx, row in df_unique.iterrows():
        hval = row['readme_hash']
        hash_to_tables_map[hval] = row['extracted_markdown_table_hugging']
    # 5) We need to store the CSV for the *first* row (the "master row") that has each readme,
    #    then for subsequent rows that share the same readme, we only create symlinks
    output_folder_hugging = os.path.join(processed_base_path, "cleaned_markdown_csvs_hugging")
    os.makedirs(output_folder_hugging, exist_ok=True)
    # We'll track readme_hash -> list of "master" CSV paths
    hash_to_master_csv_paths = {}
    # Prepare a placeholder to fill final CSV paths for each row
    df_merged['hugging_csv_files'] = [[] for _ in range(len(df_merged))]
    # 6) For each unique readme row (master), create CSV files
    for hval, subdf_indices in df_merged.groupby('readme_hash').groups.items():
        # subdf_indices is a list of all rows that share the same readme_hash
        # the "master index" is the FIRST one in that group, i.e. the first row in that group
        # since we used drop_duplicates(keep='first'), that first row in df_merged
        # should match the row in df_unique as well
        master_idx = sorted(subdf_indices)[0]  # pick the smallest index or in any order
        master_model_id = df_merged.at[master_idx, 'modelId']
        # Get the extracted tables from hash_to_tables_map
        tables = hash_to_tables_map.get(hval, [])
        master_csv_paths = []
        # If we found tables for this readme, save them under the master model's name
        for j, table_content in enumerate(tables, start=1):
            csv_path = generate_csv_path(master_model_id, "hugging", f"table{j}", output_folder_hugging)
            # Remove old file/symlink if it exists
            if os.path.lexists(csv_path):
                os.remove(csv_path)
            MarkdownHandler.markdown_to_csv(table_content, csv_path)
            master_csv_paths.append(csv_path)
        # Save these "master" CSV paths to the dictionary
        hash_to_master_csv_paths[hval] = master_csv_paths
        # Update the master row's hugging_csv_files in df_merged
        df_merged.at[master_idx, 'hugging_csv_files'] = master_csv_paths
    # 7) Create symlinks for subsequent rows that share the same readme_hash
    for hval, master_csv_paths in hash_to_master_csv_paths.items():
        # For all rows that share this hval, after the first, create symlinks
        subdf_indices = df_merged.groupby('readme_hash').groups[hval]
        sorted_indices = sorted(subdf_indices)
        # skip index 0 in sorted_indices if that was the master
        if len(sorted_indices) <= 1:
            continue  # nothing else to symlink

        master_idx = sorted_indices[0]  # the first row that "owns" these CSV files
        master_model_id = df_merged.at[master_idx, 'modelId']

        # For subsequent rows in the group
        for row_idx in sorted_indices[1:]:
            row_model_id = df_merged.at[row_idx, 'modelId']
            symlink_paths = []
            # We'll create a symlink for each table in master_csv_paths
            for j, master_csv_path in enumerate(master_csv_paths, start=1):
                row_csv_path = generate_csv_path(row_model_id, "hugging", f"table{j}", output_folder_hugging)
                if os.path.lexists(row_csv_path):
                    os.remove(row_csv_path)
                try:
                    os.symlink(os.path.abspath(master_csv_path), row_csv_path)
                except Exception as e:
                    print(f"❌ Error creating symlink: {row_csv_path} -> {master_csv_path}: {e}")
                symlink_paths.append(row_csv_path)
            df_merged.at[row_idx, 'hugging_csv_files'] = symlink_paths
    ########
    # ---------- End of HuggingFace part ----------
    
    # ---------- GitHub part ----------
    print("⚠️Step 3: Processing GitHub readme files and saving extracted tables to CSV...")
    output_folder_github = os.path.join(processed_base_path, "cleaned_markdown_csvs_github")
    os.makedirs(output_folder_github, exist_ok=True)
    input_folder_github = os.path.join(config.get('base_path'), 'downloaded_github_readmes')
    md_to_csv_mapping = process_markdown_files(input_folder_github, output_folder_github)

    output_folder_symlinks = os.path.join(processed_base_path, "symlinked_github_csvs")
    symlinked_mapping = create_symlinks(md_to_csv_mapping, output_folder_symlinks)

    github_csv_paths = []
    github_csv_paths_symlink = []
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Processing GitHub readmes"):
        #csv_list = process_github_readmes(row, output_folder_github, config)
        readme_paths = row.get("readme_path", [])
        if isinstance(readme_paths, str):
            readme_paths = [readme_paths]
        mapped_csvs = []
        mapped_symlinks = []
        for path in readme_paths:
            md_basename = os.path.basename(path)
            if md_basename in md_to_csv_mapping and md_to_csv_mapping[md_basename]:
                mapped_csvs.extend(md_to_csv_mapping[md_basename])
            if md_basename in symlinked_mapping and symlinked_mapping[md_basename]:
                mapped_symlinks.extend(symlinked_mapping[md_basename])
        github_csv_paths.append(mapped_csvs)
        github_csv_paths_symlink.append(mapped_symlinks)
    df_merged["github_csv_files"] = github_csv_paths
    # ---------- End of GitHub part ----------

    # ---------- Final Step ----------
    print("⚠️Step 4: Saving integrated DataFrame to Parquet file...")
    output_file = os.path.join(processed_base_path, f"{data_type}_step2.parquet")
    df_merged.to_parquet(output_file, index=False)
    print(f"✅ All done. Results saved to: {output_file}")

if __name__ == "__main__":
    main()


"""
Exampled Output

⚠️Step 1: Loading modelcard_step1 data...
✅ After merge: 1108759 rows.
⚠️Step 2: Extracting markdown tables from 'card_readme' (HuggingFace)...
100%|████████████████| 1108759/1108759 [02:49<00:00, 6554.99it/s]
Saving HuggingFace CSVs: 100%|█| 1108759/1108759 [17:01<00:00, 10
⚠️Step 3: Processing GitHub readme files and saving extracted tables to CSV...
Processing Markdown files:   3%| | 512/18121 [01:37<1:22:4

⚠️Step 4: Saving integrated DataFrame to Parquet file...
"""
