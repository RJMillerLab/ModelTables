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
from bs4 import BeautifulSoup
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

print('loading the json file')
with open('arxiv_html_cache.json') as f:
    data = json.load(f)  # {'2109.13855v3': 'arxiv_fulltext_html/2109.13855v3.html'}

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

def extract_tables_and_save(html_path, paper_id, output_dir='tables_output'):
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

    # Create an output directory for this paper if it doesn't exist
    paper_output_dir = os.path.join(output_dir, paper_id)
    os.makedirs(paper_output_dir, exist_ok=True)

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
            df_table = pd.DataFrame(table_data)
            csv_path = os.path.join(paper_output_dir, f"{paper_id}_table{idx}.csv")
            df_table.to_csv(csv_path, index=False)
            table_paths.append(csv_path)

    return table_paths

def process_item(item):
    """
    This function runs classification and table extraction for a single row (paper_id, html_path).
    Returns a dict with updated info: paper_id, html_path, page_type, table_list.
    """
    paper_id, html_path = item
    page_type = classify_page(html_path)
    table_list = extract_tables_and_save(html_path, paper_id, output_dir='tables_output')
    return {
        'paper_id': paper_id,
        'html_path': html_path,
        'page_type': page_type,
        'table_list': table_list
    }

print('processing the html files')
df = pd.DataFrame(data.items(), columns=['paper_id', 'html_path'])

# Parallel processing of classification + table extraction
results = Parallel(n_jobs=-1)(
    delayed(process_item)(row) for row in tqdm(df.itertuples(index=False), total=len(df))
)

# Build a new DataFrame from the results
df_processed = pd.DataFrame(results, columns=['paper_id', 'html_path', 'page_type', 'table_list'])

# If columns did not match the dictionary keys directly, you can do:
# df_processed = pd.DataFrame(results)

# Save as Parquet
df_processed.to_parquet("html_table.parquet", index=False)
print("Done! Saved to html_table.parquet.")
