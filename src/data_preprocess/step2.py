import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from data_ingestion.readme_parser import BibTeXExtractor, MarkdownHandler
from data_ingestion.bibtex_parser import BibTeXFactory
import os, re, time
from utils import load_data
tqdm.pandas()

def setup_logging(log_filename):
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging started.")

def ensure_string(entry):
    """Ensure the input is a string."""
    return str(entry) if entry is not None else None

def extract_bibtex(df, readme_key="card_readme", new_key="extracted_bibtex"):
    """Extract BibTeX entries from the 'card_readme' column."""
    print("Extracting BibTeX entries...")
    #df["extracted_bibtex"] = df["card_readme"].apply(lambda x: BibTeXExtractor().extract(x) if isinstance(x, str) else [], meta=('extracted_bibtex', 'object'))
    df[new_key] = df[readme_key].apply(lambda x: BibTeXExtractor().extract(x) if isinstance(x, str) else [])
    print(f"New attributes added: {new_key}")

def add_extracted_tuples(df, key_bibtex = "extracted_bibtex", key_markdown_table = "extracted_markdown_table", add_bibtex_tuple="extracted_bibtex_tuple", add_markdown_tuple="extracted_markdown_table_tuple"):
    """Convert extracted BibTeX and markdown table to tuples for uniqueness checks."""
    print("Adding extracted tuples for uniqueness checks...")
    #df["extracted_bibtex_tuple"] = df[key_bibtex].apply(lambda x: tuple(x) if isinstance(x, list) else (x,), meta=('extracted_bibtex', 'object'))
    #df["extracted_markdown_table_tuple"] = df[key_markdown_table].apply(lambda x: tuple(x) if isinstance(x, list) else (x,), meta=('extracted_bibtex', 'object'))
    df[add_bibtex_tuple] = df[key_bibtex].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))
    df[add_markdown_tuple] = df[key_markdown_table].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))
    print("New attributes added: 'extracted_bibtex_tuple', 'extracted_markdown_table_tuple'.")

def process_bibtex_tuple(entry):
    """Process a BibTeX entry and return parsed results."""
    parsed_results = []
    success_count = 0
    if isinstance(entry, (tuple, np.ndarray)):
        entry = list(entry)
    if isinstance(entry, list):
        for single_entry in entry:
            single_entry = ensure_string(single_entry)
            if single_entry:
                parsed_entry, flag = BibTeXFactory.parse_bibtex(single_entry)
                parsed_results.append(parsed_entry)
                if flag:
                    success_count += 1
    elif isinstance(entry, str):
        single_entry = ensure_string(entry)
        parsed_entry, flag = BibTeXFactory.parse_bibtex(single_entry)
        parsed_results.append(parsed_entry)
        if flag:
            success_count += 1
    return parsed_results, success_count

def process_bibtex_entries(df):
    """Process BibTeX entries in the DataFrame and return results."""
    print("Processing BibTeX entries...")
    non_null_entries = df[df["extracted_bibtex_tuple"].notnull()]
    # Initialize tqdm progress bar
    results = []
    with tqdm(total=len(non_null_entries), desc="Processing BibTeX tuples") as pbar:
        results = Parallel(n_jobs=-1)(
            delayed(process_bibtex_tuple)(entry) for entry in non_null_entries["extracted_bibtex_tuple"]
        )
        pbar.update(len(non_null_entries))  # Update progress bar after processing
    # Create new columns for parsed results
    non_null_entries["parsed_bibtex_tuple_list"] = [result[0] for result in results]
    non_null_entries["successful_parse_count"] = [result[1] for result in results]
    df.loc[non_null_entries.index, 'parsed_bibtex_tuple_list'] = non_null_entries["parsed_bibtex_tuple_list"]
    df.loc[non_null_entries.index, 'successful_parse_count'] = non_null_entries["successful_parse_count"]
    print("New attributes added: 'parsed_bibtex_tuple_list', 'successful_parse_count'.")
    return non_null_entries

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

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    data_type = 'modelcard'
    # Load data
    start_time = time.time()
    t1 = start_time
    print("⚠️Step 1: Loading data...")
    df = load_data(f"{output_dir}/{data_type}_step1.parquet", columns=['modelId', 'downloads', 'card_readme', 'contains_markdown_table', 'extracted_markdown_table'])
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 2: Extracting BibTeX entries...")
    start_time = time.time()
    extract_bibtex(df)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 3: Adding extracted tuples for uniqueness checks...")
    start_time = time.time()
    add_extracted_tuples(df)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 4: Processing BibTeX entries...")
    start_time = time.time()
    processed_entries = process_bibtex_entries(df)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    #print("⚠️Step 7: Analyzing results...")
    #start_time = time.time()
    #df_failed_sample = analyze_results(processed_entries)
    #print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 5: Saving results to CSV files...")
    start_time = time.time()
    save_markdown_to_csv(df, output_folder = "cleaned_markdown_csvs")
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 6: Saving results to Parquet file...")
    start_time = time.time()
    output_file = f"{output_dir}/{data_type}_step2.parquet"
    df.to_parquet(output_file)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("Results saved to:", output_file)
    elapsed_time = time.time() - t1
    print(f"Total processing time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
