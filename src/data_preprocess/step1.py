import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, parallel_backend
import re, os, json, time
from utils import load_data
from concurrent.futures import ThreadPoolExecutor
from ruamel.yaml import YAML

tqdm.pandas()

# Initialize the YAML parser
yaml_parser = YAML(typ="safe")

# Function to clean up line breaks and whitespace
def clean_content(content):
    if content is None:
        return None
    # Normalize line endings (but keep everything else intact)
    return content.replace('\r\n', '\n').replace('\r', '\n')

# Separate tags and README using split and replace
def separate_tags_and_readme(card_content):
    tags, readme = None, None
    try:
        # Clean up content minimally
        card_content = clean_content(card_content)
        if card_content.startswith("---\n"):
            # Split only on the first two "---\n"
            parts = card_content.split("---\n", 2)
            if len(parts) > 2:
                tags = parts[1]  # Keep tags part intact
                readme = parts[2]  # Keep readme part intact
            else:
                readme = parts[1]  # Handle case where only readme exists
        else:
            readme = card_content  # No tags part, entire content is readme
    except Exception as e:
        print(f"Error parsing content: {e}")
    return tags, readme

# Process to extract tags and README using tqdm and apply
def extract_tags_and_readme_parallel(df, n_jobs=-1):
    # Process a single row
    def process_card(card_content):
        return separate_tags_and_readme(card_content)
    # Use joblib to parallelize the map function
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_card)(content) for content in tqdm(df["card"], desc="Extracting Tags and README")
    )
    # Split the results into separate columns
    df["card_tags"] = [x[0] for x in results]
    df["card_readme"] = [x[1] for x in results]
    return df

# Clean and compare restored content using replace and split
def clean_for_comparison(content):
    if content is None:
        return ""
    return content.replace("\n", "").replace("\r", "").replace("---", "").replace(" ", "").strip()

# Validate parsed content using tqdm
def validate_parsing(df):
    inconsistencies = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating Parsed Cards"):
        original_card = row["card"]  # Preserve the original structure
        restored_card = ""
        if row["card_tags"] is not None:
            restored_card = f"---\n{row['card_tags']}\n---\n{row['card_readme']}"
        else:
            restored_card = row["card_readme"]
        if original_card.strip() != restored_card.strip():
            inconsistencies.append({
                "original_card": row["card"],
                "restored_card": restored_card,
                "card_tags": row["card_tags"],
                "card_readme": row["card_readme"]
            })
    return pd.DataFrame(inconsistencies)

def clean_yaml_content(content):
    if content is None:
        return None
    # Replace tabs with spaces and normalize line endings
    content = content.replace('\t', ' ')
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return content

def parse_card_tags_dynamic(card_tag, full_card_content=None, model_id=None):
    """
    Parse the YAML in card_tag. Returns (dict, bool, str):
        dict: Parsed tags or {}
        bool: has_error flag
        str: error_message if any
    """
    if not card_tag:
        return {}, False, None
    card_tag = clean_yaml_content(card_tag)
    try:
        parsed_data = yaml_parser.load(card_tag.strip())
        if isinstance(parsed_data, dict):
            return parsed_data, False, None
        else:
            return {}, False, None
    except Exception as e:
        error_message = f"Error parsing card_tags: {e}"
        # Optionally log or store: the model ID, full card content, etc.
        # print(error_message)
        # print(f"Problematic card_tags content:\n{card_tag}")
        # if model_id: 
        #     print(f"Model ID: {model_id}")
        # if full_card_content:
        #     print(f"Full card content:\n{full_card_content}")
        return {}, True, error_message  

def process_tags_and_combine_dynamic_parallel(df, n_jobs=4):
    """
    Parse each row's card_tags in parallel, 
    then combine results with the original dataframe.
    """
    def process_row(row):
        parsed_tags, has_error, error_message = parse_card_tags_dynamic(
            row.card_tags,
            full_card_content=row.card,
            model_id=getattr(row, 'modelId', None)
        )
        # Prefix keys with 'card_tags_'
        prefixed_tags = {f"card_tags_{k}": v for k, v in parsed_tags.items()}
        return prefixed_tags, set(prefixed_tags.keys()), has_error, error_message

    with tqdm_joblib(tqdm(desc="Processing rows", total=len(df))) as progress_bar:
        with parallel_backend('loky'):
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_row)(row) for row in df.itertuples()
            )
    """with parallel_backend('loky'):  # or 'multiprocessing'
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_row)(row) for row in df.itertuples()
        )"""
    # Unzip the results
    parsed_data, all_keys_list, error_flags, error_messages = zip(*results)
    # Extract all unique keys
    all_keys = set().union(*all_keys_list)
    # Create a dict of columns -> values for the parsed tags
    parsed_results = {
        key: [row_dict.get(key) for row_dict in parsed_data] 
        for key in all_keys
    }
    # Construct a new DataFrame with parsed data + error info
    parsed_df = pd.DataFrame(parsed_results)
    parsed_df['error_flag'] = error_flags
    parsed_df['error_message'] = error_messages
    # Combine
    combined_df = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
    return combined_df

def chunked_parallel_processing(df, chunk_size=1000, n_jobs=4):
    """
    Process the DataFrame in chunks to manage memory usage.
    Returns a list of processed DataFrames.
    """
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    chunks = (df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size))

    processed_list = []
    with tqdm(total=total_chunks, desc="Processing Chunks") as pbar:
        for chunk in chunks:
            processed_chunk = process_tags_and_combine_dynamic_parallel(chunk, n_jobs=n_jobs)
            processed_list.append(processed_chunk)
            pbar.update(1)
    return processed_list

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

def process_row(row):
    return detect_and_extract_markdown_table(row)

def extract_markdown(df, col_name='card_readme', n_jobs=4):
    """
    Extract Markdown tables from the given DataFrame `df` in parallel.
    """
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(process_row, df[col_name]), total=len(df)))
    return results

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    file_type = "modelcard"
    
    print("⚠️ Step 1: Loading data...")
    start_time = time.time()
    from utils import load_combined_data
    df = load_combined_data(data_type, file_path="~/Repo/CitationLake/data/")
    #df = load_data(f"{output_dir}/{file_type}_step1.parquet") # load them all
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 2: Splitting readme and tags...")
    start_time = time.time()
    df_split = extract_tags_and_readme_parallel(df)
    #inconsistencies_df = validate_parsing(df_split)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    
    print("⚠️ Step 3: Processing card_tags...")
    start_time = time.time()
    processed_chunks = chunked_parallel_processing(df, chunk_size=1000, n_jobs=-1)
    df_split = pd.concat(processed_chunks, ignore_index=True)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 4: Extracting markdown tables...")
    start_time = time.time()
    results = extract_markdown(df_split, col_name='card_readme')
    df_split_temp[['contains_markdown_table', 'extracted_markdown_table']] = pd.DataFrame(results, index=df_split_temp.index)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    
    print("⚠️ Step 5: Saving results to Parquet file...")
    start_time = time.time()
    output_file = f"{output_dir}/{file_type}_step1.parquet"
    for col in df_split_temp.columns:
        if df_split_temp[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
            df_split_temp[col] = df_split_temp[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple, np.ndarray)) else x
            )
    df_split_temp.to_parquet(output_file)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("Results saved to:", output_file)

if __name__ == "__main__":
    main()
