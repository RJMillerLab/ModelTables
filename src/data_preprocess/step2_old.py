"""
Author: Zhengyuan Dong
Created: 2025-02-12
Last Modified: 2025-03-09
Description: Extract BibTeX entries from the 'card_readme' column and save to CSV files.
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from src.data_ingestion.readme_parser import MarkdownHandler
import os, re, time
from src.utils import load_data, load_config

tqdm.pandas()

def setup_logging(log_filename):
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging started.")

def analyze_results(df):
    """Analyze the results and print statistics."""
    print("Analyzing results...")
    df_failed_parsing = df[
        (df["extracted_bibtex_tuple"].notnull()) & 
        (df["parsed_bibtex_tuple_list"].apply(lambda x: x is None or len([i for i in x if i]) == 0))
    ]
    #df_failed_parsing = df[
    #    (df["extracted_bibtex_tuple"].notnull()) & 
    #    (df["parsed_bibtex_tuple_list"].apply(lambda x: x is None or len([i for i in x if i]) == 0), meta=('parsed_bibtex_empty', 'bool'))
    #]
    total_items = len(df[df["extracted_bibtex_tuple"].notnull()])
    total_failed = len(df_failed_parsing)
    print(f"\nProcessed {total_items} BibTeX tuples.")
    print(f"Failed parses: {total_failed} ({(total_failed / total_items) * 100:.2f}% failure rate)\n")
    # Output sample of failed parsing
    return df_failed_parsing

def generate_csv_path(model_id, index, folder):
    """Generate a unique file path using modelId and index."""
    sanitized_model_id = re.sub(r"[^\w\-]", "_", str(model_id) if model_id else "unknown_model")
    return os.path.join(folder, f"{sanitized_model_id}_markdown_{index}.csv")

def save_markdown_to_csv(df, output_folder = "cleaned_markdown_csvs", key="extracted_markdown_table", new_key="csv_path"):
    """Extract markdown and save to local files."""
    os.makedirs(output_folder, exist_ok=True)
    # Apply the MarkdownHandler with a tqdm progress bar
    df[new_key] = df.progress_apply(
        lambda row: MarkdownHandler.markdown_to_csv(
            row[key],
            generate_csv_path(row["modelId"], row.name, output_folder)
        ) if pd.notnull(row[key]) else None,
        axis=1
    )

def detect_and_extract_markdown_table(card_content: str):
    """
    Detect and extract Markdown tables (supporting multi-line) from the given `card_content`.
    Returns a tuple of (whether a Markdown table is present, the extracted table text).
    """
    if not isinstance(card_content, str):
        return (False, None)
    markdown_table_pattern = (
        r"(?:\|[^\n]*?\|[\s]*\n)+\|[-:| ]*\|[\s]*\n(?:\|[^\n]*?\|(?:\n|$))+"
    )
    markdown_match = re.search(markdown_table_pattern, card_content, re.MULTILINE)
    if markdown_match:
        return (True, markdown_match.group(0).strip())
    return (False, None)

def extract_markdown(df, col_name='card_readme', n_jobs=-1):
    """
    Extract Markdown tables from the given DataFrame `df` in parallel.
    """
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(detect_and_extract_markdown_table, df[col_name]), total=len(df)))
    return results

def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'
    
    # Load data
    start_time = time.time()
    print("⚠️Step 1: Loading data...")
    df = load_data(os.path.join(processed_base_path, f"{data_type}_step1.parquet"), columns=['modelId', 'downloads', 'card_readme', 'contains_markdown_table', 'extracted_markdown_table'])
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 2: Extracting markdown tables...")
    start_time = time.time()
    results = extract_markdown(df_split, col_name='card_readme')
    df_split_temp[['contains_markdown_table', 'extracted_markdown_table']] = pd.DataFrame(results, index=df_split_temp.index)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️Step 3: Adding extracted tuples for uniqueness checks...")
    start_time = time.time()
    df["extracted_markdown_table_tuple"] = df["extracted_markdown_table"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))
    print("New attributes added: 'extracted_markdown_table_tuple'.")
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    #print("⚠️Step 7: Analyzing results...")
    #start_time = time.time()
    #df_failed_sample = analyze_results(processed_entries)
    #print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️Step 5: Saving card markdown to CSV files...")
    start_time = time.time()
    save_markdown_to_csv(df, output_folder = os.path.join(processed_base_path, "cleaned_markdown_csvs"))
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️Step 6: Saving results to Parquet file...")
    start_time = time.time()
    output_file = os.path.join(processed_base_path, f"{data_type}_step2.parquet")
    df.to_parquet(output_file, compression="zstd", engine="pyarrow", index=False)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("Results saved to:", output_file)

if __name__ == "__main__":
    main()
