import os
import time
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, parallel_backend
from src.utils import load_config, load_combined_data, safe_json_dumps

# Initialize the YAML parser
from ruamel.yaml import YAML
yaml_parser = YAML(typ="safe")


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

def process_tags_and_combine_dynamic_parallel(df, n_jobs=-1):
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

def chunked_parallel_processing(df, chunk_size=1000, n_jobs=-1):
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


def main():
    config = load_config('config.yaml')
    processed_base_path = os.path.join(config.get('base_path'), 'processed')
    data_type = 'modelcard'

    start_time = time.time()
    print("⚠️Step 1: Loading data...")
    df = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_step1.parquet"), columns=['modelId', 'downloads', 'card_readme', 'contains_markdown_table', 'extracted_markdown_table'])
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️ Step 2: Processing card_tags...")
    start_time = time.time()
    processed_chunks = chunked_parallel_processing(df, chunk_size=1000, n_jobs=-1)
    df_split = pd.concat(processed_chunks, ignore_index=True)
    print("✅ Done. Time cost: {:.2f} seconds.".format(time.time() - start_time))

    print("⚠️Step 3: Saving results to Parquet file...")
    start_time = time.time()
    output_file = os.path.join(processed_base_path, f"{data_type}_step1_tags.parquet")
    df.to_parquet(output_file, compression="zstd", engine="pyarrow", index=False)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("Results saved to:", output_file)

if __name__ == "__main__":
    main()
