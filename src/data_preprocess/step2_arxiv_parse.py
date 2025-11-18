"""
Author: Zhengyuan Dong
Created: 25-03-30
Edited: 25-03-30
Description: Determine whethter html is metadata page or full page, extract tables and save table path list into parquet
"""
import sys  
import json
import os
import shutil
import argparse
from bs4 import BeautifulSoup
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from src.utils import to_parquet, load_config


def classify_page(html_path):
    if not os.path.exists(html_path):
        # If file doesn't exist, return None or you can return something else
        return None
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    # Rule 1: Check for <meta> tags with the 'name' attribute starting with "citation_"
    meta_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.lower().startswith('citation_')})
    if meta_tags and len(meta_tags) >= 3:
        return 'metadata'
    # Rule 2: Check if there are multiple <section> or <article> tags present
    sections = soup.find_all(['section', 'article'])
    if len(sections) >= 2:
        return 'fulltext'
    # Rule 3: Check if the body text contains both "introduction" and "conclusion"
    body_text = soup.get_text(separator=' ').lower()
    if "introduction" in body_text and "conclusion" in body_text:
        return 'fulltext'
    # Default to 'metadata' if none of the above rules are met
    return 'metadata'

def extract_tables_and_save(html_path, paper_id, output_dir):
    """
    Extracts all <table> tags from the given HTML file, converts them to CSV,
    and saves them in a new folder: {output_dir}/{paper_id}/{paper_id}_table{idx}.csv

    Returns a list of the CSV file paths created.
    """
    table_paths = []
    if not os.path.exists(html_path):
        return table_paths

    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')

    os.makedirs(output_dir, exist_ok=True)
    # Create an output directory for this paper if it doesn't exist
    #paper_output_dir = os.path.join(output_dir, paper_id)
    #os.makedirs(paper_output_dir, exist_ok=True)

    for idx, table in enumerate(tables):
        # Extract rows
        rows = table.find_all('tr')
        table_data = []
        for row in rows:
            cols = row.find_all(['td', 'th'])
            cols_text = [col.get_text(strip=True) for col in cols]
            table_data.append(cols_text)
        # Convert to DataFrame
        if table_data:
            #df_table = pd.DataFrame(table_data)
            if len(table_data) > 1:
                new_header = [str(item).strip() for item in table_data[0]]
                data_rows = table_data[1:]
                if all(len(row) == len(new_header) for row in data_rows):
                    df_table = pd.DataFrame(data_rows, columns=new_header)
                else:
                    df_table = pd.DataFrame(table_data)
                #df_table = pd.DataFrame(data_rows, columns=new_header)
            else:
                df_table = pd.DataFrame(table_data)
            csv_path = os.path.join(output_dir, f"{paper_id}_table{idx}.csv")
            df_table.to_csv(csv_path, index=False)
            table_paths.append(csv_path)
    return table_paths

def process_item(item, output_dir):
    """
    This function runs classification and table extraction for a single row (paper_id, html_path).
    Returns a dict with updated info: paper_id, html_path, page_type, table_list.
    """
    paper_id, html_path = item
    page_type = classify_page(html_path)
    table_list = extract_tables_and_save(html_path, paper_id, output_dir=output_dir)
    return {
        'paper_id': paper_id,
        'html_path': html_path,
        'page_type': page_type,
        'table_list': table_list
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify HTML pages and extract basic tables")
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--html-cache', dest='html_cache', default=None,
                        help='Path to arxiv_html_cache JSON (default: auto-detect from tag)')
    parser.add_argument('--html-dir', dest='html_dir', default=None,
                        help='Directory containing HTML files (default: auto-detect from tag)')
    parser.add_argument('--output-dir', dest='output_dir', default=None,
                        help='Directory to save extracted tables (default: auto-detect from tag)')
    parser.add_argument('--results-parquet', dest='results_parquet', default=None,
                        help='Path to html_table parquet (default: auto-detect from tag)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel jobs for extraction')
    args = parser.parse_args()

    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""

    html_cache_path = args.html_cache or os.path.join(processed_base_path, f"arxiv_html_cache{suffix}.json")
    html_dir = args.html_dir or (os.path.join(base_path, f"arxiv_fulltext_html_{tag}") if tag else os.path.join(base_path, "arxiv_fulltext_html"))
    tables_output_dir = args.output_dir or os.path.join(processed_base_path, f"tables_output{suffix or ''}")
    parquet_file = args.results_parquet or os.path.join(processed_base_path, f"html_table{suffix}.parquet")

    print(f"📁 HTML cache: {html_cache_path}")
    print(f"📁 HTML directory: {html_dir}")
    print(f"📁 Tables output dir: {tables_output_dir}")
    print(f"📁 Results parquet: {parquet_file}")

    if not os.path.exists(html_cache_path):
        raise FileNotFoundError(f"HTML cache file not found: {html_cache_path}")
    if not os.path.isdir(html_dir):
        raise FileNotFoundError(f"HTML directory not found: {html_dir}")

    os.makedirs(tables_output_dir, exist_ok=True)

    print('loading the json file')
    with open(html_cache_path) as f:
        data = json.load(f)  # {'2109.13855v3': 'arxiv_fulltext_html/2109.13855v3.html'}

    # Normalize html paths if they are relative to default folder
    normalized_items = []
    for paper_id, rel_path in data.items():
        html_path = rel_path
        if not os.path.isabs(html_path):
            # If the stored path already contains the base folder name, keep as is; otherwise join
            if not html_path.startswith(html_dir):
                html_path = os.path.join(html_dir, os.path.basename(html_path))
        normalized_items.append((paper_id, html_path))

    print('processing the html files')
    df = pd.DataFrame(normalized_items, columns=['paper_id', 'html_path'])

    ######## # 1) Check if parquet already exists; if yes, load it
    if os.path.exists(parquet_file):
        df_existing = pd.read_parquet(parquet_file)
        print(f"Loaded existing {parquet_file}, found {len(df_existing)} records.")
    else:
        # If no parquet file exists, create an empty one with the correct columns
        df_existing = pd.DataFrame(columns=['paper_id', 'html_path', 'page_type', 'table_list'])
        print("No existing parquet found. Starting fresh.")
    ######## # 2) Identify new/unseen paper_ids by comparing to df_existing
    existing_ids = set(df_existing['paper_id'])
    df_new = df[~df['paper_id'].isin(existing_ids)]
    print(f"Found {len(df_new)} new items to process.")
    ######## # 3) If df_new is empty, skip processing
    if len(df_new) > 0:
        # Parallel processing for classification + table extraction only on new items
        if args.n_jobs == 1:
            new_results = [process_item(row, tables_output_dir) for row in tqdm(df_new.itertuples(index=False), total=len(df_new))]
        else:
            new_results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_item)(row, tables_output_dir) for row in tqdm(df_new.itertuples(index=False), total=len(df_new))
            )
        # Convert new results to DataFrame
        df_processed_new = pd.DataFrame(new_results, columns=['paper_id', 'html_path', 'page_type', 'table_list'])
        # Concatenate old + new
        df_final = pd.concat([df_existing, df_processed_new], ignore_index=True)
        # Optional: remove duplicates in case of overlap
        df_final.drop_duplicates(subset=['paper_id'], keep='last', inplace=True)
    else:
        # If there is nothing new, just keep the old
        df_final = df_existing
    ######## # 4) Save final results back to the parquet file
    to_parquet(df_final, parquet_file)
    print(f"Done! Updated {parquet_file} with {len(df_final)} total records.")
