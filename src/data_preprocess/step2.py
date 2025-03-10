import os
import re
import time
import logging
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

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
    filename = f"{model_id}_{source}_{identifier}.csv"
    return os.path.join(folder, filename)

########
def process_github_readmes(row, output_folder):
    """
    Process GitHub readme files for one row.
    For each file in row['readme_path'] (a list of paths), read the file,
    extract all Markdown tables, save each table to a CSV file, and return a list
    of the saved CSV file paths.
    """
    model_id = row["modelId"]
    readme_paths = row["readme_path"]
    print('readme_paths: ', readme_paths)
    print('type:', type(readme_paths))
    csv_files = []
    if not isinstance(readme_paths, (list, np.ndarray, tuple)) or len(readme_paths) == 0:
        return csv_files
    # Process each file (with index starting at 1 for naming)
    for file_idx, path in enumerate(readme_paths, start=1):
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                saved = save_markdown_to_csv_from_content(model_id, content, "git", file_idx, output_folder)
                csv_files.extend(saved)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return csv_files
########

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
    results_hugging = extract_markdown_tables_in_parallel(df_merged, col_name='card_readme', n_jobs=4)
    # Save the extracted tables list into a new column
    df_merged['extracted_markdown_table_hugging'] = [res[1] for res in results_hugging]
    
    # Save CSV files for HuggingFace part and store file paths in a new column
    output_folder_hugging = os.path.join(processed_base_path, "cleaned_markdown_csvs_hugging")
    os.makedirs(output_folder_hugging, exist_ok=True)
    hugging_csv_paths = []
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Saving HuggingFace CSVs"):
        cell = row['extracted_markdown_table_hugging']
        row_paths = []
        if isinstance(cell, (list, np.ndarray, tuple)):
            for j, table in enumerate(cell, start=1):
                if table:
                    identifier = f"table{j}"
                    csv_path = generate_csv_path(row["modelId"], "hugging", identifier, output_folder_hugging)
                    MarkdownHandler.markdown_to_csv(table, csv_path)
                    row_paths.append(csv_path)
        hugging_csv_paths.append(row_paths)
    df_merged['hugging_csv_files'] = hugging_csv_paths
    # ---------- End of HuggingFace part ----------
    
    # ---------- GitHub part ----------
    print("⚠️Step 3: Processing GitHub readme files and saving extracted tables to CSV...")
    output_folder_github = os.path.join(processed_base_path, "cleaned_markdown_csvs_github")
    os.makedirs(output_folder_github, exist_ok=True)
    github_csv_paths = []
    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Processing GitHub readmes"):
        csv_list = process_github_readmes(row, output_folder_github)
        github_csv_paths.append(csv_list)
    df_merged['github_csv_files'] = github_csv_paths
    # ---------- End of GitHub part ----------

    # ---------- Final Step ----------
    print("⚠️Step 4: Saving integrated DataFrame to Parquet file...")
    output_file = os.path.join(processed_base_path, f"{data_type}_step2.parquet")
    df_merged.to_parquet(output_file, index=False)
    print(f"✅ All done. Results saved to: {output_file}")

if __name__ == "__main__":
    main()
