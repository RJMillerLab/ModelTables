import os, re, time, logging
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from shutil import copytree
import shutil
from src.data_ingestion.readme_parser import MarkdownHandler
from src.utils import load_data, load_config

tqdm.pandas()

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

def extract_markdown_tables_in_parallel(df, col_name, n_jobs=-1):
    """
    Perform parallel extraction of Markdown tables from the specified column (a string)
    in the DataFrame using joblib.Parallel.
    Returns a list of tuples (bool, list) for each row.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(detect_and_extract_markdown_tables)(content) for content in tqdm(df[col_name], total=len(df))
    )
    return results

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
        # Check file size (skip if > 100MB, but still record in mapping)
        if os.path.getsize(md_path) > 100 * 1024 * 1024:  # 100MB threshold
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
        symlinked_csv_paths = []
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                print(f"⚠️ Skipping missing CSV: {csv_path}")
                continue
            symlink_path = os.path.join(output_folder_symlinks, os.path.basename(csv_path))
            try:
                if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                    os.unlink(symlink_path)  # Remove existing symlink if it exists
                os.symlink(os.path.abspath(csv_path), symlink_path)  # Create symlink
                symlinked_csv_paths.append(symlink_path)
            except Exception as e:
                print(f"❌ Error creating symlink for {csv_path}: {e}")
        symlink_mapping[md_basename] = symlinked_csv_paths
    return symlink_mapping

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    
    print("⚠️Step 1: Loading modelcard_step1 data...")
    df_modelcard = pd.read_parquet(
        os.path.join(processed_base_path, f"{data_type}_step1.parquet"), 
        columns=['modelId', 'card_readme', 'github_link']
    )
    print(f"✅ Loaded {len(df_modelcard)} rows from modelcard_step1.")
    df_giturl = pd.read_parquet(
        os.path.join(processed_base_path, "giturl_info.parquet")
    )
    print(f"✅ Loaded {len(df_giturl)} rows from giturl_info.")
    df_merged = pd.merge(
        df_modelcard, 
        df_giturl[['modelId', 'readme_path']],
        on='modelId', 
        how='left'
    )
    print(f"✅ After merge: {len(df_merged)} rows.")

    # ---------- HuggingFace part (use original code) ----------
    print("⚠️Step 2: Extracting markdown tables from 'card_readme' (HuggingFace)...")
    output_folder_hugging = os.path.join(processed_base_path, "cleaned_markdown_csvs_hugging")
    os.makedirs(output_folder_hugging, exist_ok=True)
    rerun_hugging = True
    results_hugging = extract_markdown_tables_in_parallel(df_merged, col_name='card_readme', n_jobs=4)
    # Save the extracted tables list into a new column
    df_merged['extracted_markdown_table_hugging'] = [res[1] for res in results_hugging]
    # Save CSV files for HuggingFace part and store file paths in a new column
    hugging_csv_paths = []
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Saving HuggingFace CSVs"):
        cell = row['extracted_markdown_table_hugging']
        row_paths = []
        #if isinstance(cell, (list, np.ndarray, tuple)):
        for j, table in enumerate(cell, start=1):
            #if table:
            identifier = f"table{j}"
            csv_path = generate_csv_path(row["modelId"], "hugging", identifier, output_folder_hugging)
            csv_path = os.path.join(config.get('base_path'), csv_path.split("data", 1)[-1].lstrip("/\\"))
            if rerun_hugging:
                MarkdownHandler.markdown_to_csv(table, csv_path)
            row_paths.append(csv_path)
        hugging_csv_paths.append(row_paths)
    df_merged['hugging_csv_files'] = hugging_csv_paths
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
        for path in readme_paths:
            md_basename = os.path.basename(path)
            if md_basename in md_to_csv_mapping:
                mapped_csvs.extend(md_to_csv_mapping[md_basename])
            if md_basename in symlinked_mapping:
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
✅ Loaded 1108759 rows from modelcard_step1.
✅ Loaded 1108759 rows from giturl_info.
✅ After merge: 1108759 rows.
⚠️Step 2: Extracting markdown tables from 'card_readme' (HuggingFace)...
100%|████████████████| 1108759/1108759 [02:49<00:00, 6554.99it/s]
Saving HuggingFace CSVs: 100%|█| 1108759/1108759 [17:01<00:00, 10
⚠️Step 3: Processing GitHub readme files and saving extracted tables to CSV...
Processing Markdown files:   3%| | 512/18121 [01:37<1:22:4

⚠️Step 4: Saving integrated DataFrame to Parquet file...
"""
