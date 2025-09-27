"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2025-09-26
Description: Extract markdown tables from GitHub | Huggingface html, modelcards, readme files, and save them to CSV files.
Enhanced version with improved HTML table parsing and using raw card data.
Usage:
    python -m src.data_preprocess.step2_gitcard_tab_v2
Tips: We deduplicate the card content, and use the symlink to save data storage later.
"""

import os, re, time, logging, hashlib, json
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from shutil import copytree
import shutil
from bs4 import BeautifulSoup
from src.data_ingestion.readme_parser import MarkdownHandler
from src.utils import load_config, to_parquet

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
def extract_math_text_gitcard(cell):
    """Extract text from math elements and convert to readable format."""
    math_elements = cell.find_all('math')
    if not math_elements:
        text = cell.get_text(strip=True)
    else:
        text = str(cell)
        for math in math_elements:
            alttext = math.get('alttext', '')
            if alttext:
                text = text.replace(str(math), alttext)
            else:
                annotation = math.find('annotation', encoding='application/x-tex')
                if annotation:
                    text = text.replace(str(math), annotation.get_text(strip=True))
        
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(strip=True)
    
    # Clean up math formatting
    text = re.sub(r'\{_\{\\text\{base\}\}\}', '_base', text)
    text = re.sub(r'\{_\{\\text\{large\}\}\}', '_large', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'_+', '_', text)
    text = text.rstrip('_')
    text = re.sub(r'([A-Za-z]+)_\}', r'\1_base', text)
    
    return text

def parse_html_table_gitcard(table):
    """Parse HTML table with proper rowspan and colspan handling."""
    rows = table.find_all('tr')
    if not rows:
        return []
    
    # First pass: extract all cell data with their positions
    cell_data = []
    for row_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            text = extract_math_text_gitcard(cell)
            
            # Clean HTML content (remove bold tags for cleaner text)
            text = re.sub(r'<[^>]+>', '', text)
            text = text.strip()
            
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            cell_data.append({
                'row': row_idx,
                'text': text,
                'colspan': colspan,
                'rowspan': rowspan
            })
    
    # Find max columns
    max_cols = 0
    for row in rows:
        cells = row.find_all(['td', 'th'])
        col_count = sum(int(cell.get('colspan', 1)) for cell in cells)
        max_cols = max(max_cols, col_count)
    
    # Create grid
    grid = []
    for row_idx in range(len(rows)):
        grid.append([''] * max_cols)
    
    # Fill grid with cell data
    for cell_info in cell_data:
        row_start = cell_info['row']
        text = cell_info['text']
        colspan = cell_info['colspan']
        rowspan = cell_info['rowspan']
        
        # Find the correct column position for this cell
        col_idx = 0
        for col in range(max_cols):
            if grid[row_start][col] == '':
                col_idx = col
                break
        
        # Fill the grid with this cell's data
        for r in range(row_start, row_start + rowspan):
            for c in range(col_idx, col_idx + colspan):
                if r < len(grid) and c < max_cols:
                    grid[r][c] = text
    
    return grid

def detect_and_extract_markdown_tables_v2(content: str):
    """
    Enhanced version: Detect and extract all Markdown tables (both | format and HTML <table>) from the given text.
    Returns a tuple (bool, list), where the boolean indicates whether at least one table was found,
    and the list contains all matched table data (as list of lists).
    """
    if not isinstance(content, str) or not content.strip():
        return (False, [])
    
    all_tables = []
    
    # 1. Extract markdown tables (| format)
    markdown_table_pattern = r"(?:\|[^\n]*?\|[\s]*\n)+\|[-:| ]*\|[\s]*\n(?:\|[^\n]*?\|(?:\n|$))+"
    md_matches = re.findall(markdown_table_pattern, content, re.MULTILINE)
    
    for table in md_matches:
        try:
            # Parse markdown table manually for better control
            lines = table.strip().split('\n')
            if len(lines) < 3:  # Must have header, separator, and at least one data row
                continue
            
            # Clean lines and split by |
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('|') and line.endswith('|'):
                    line = line[1:-1]  # Remove outer pipes
                # Skip separator lines (containing only :, -, |, and spaces)
                if re.match(r'^[\s\-:|]*$', line):
                    continue
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at the end
                while cells and not cells[-1]:
                    cells.pop()
                if cells:  # Only add non-empty rows
                    cleaned_lines.append(cells)
            
            if len(cleaned_lines) >= 2:  # Must have header and at least one data row
                # Ensure all rows have the same number of columns
                max_cols = max(len(row) for row in cleaned_lines)
                for row in cleaned_lines:
                    while len(row) < max_cols:
                        row.append('')
                
                all_tables.append(cleaned_lines)
        except Exception as e:
            print(f"Error extracting markdown table: {e}")
            continue
    
    # 2. Extract HTML tables (<table> format)
    try:
        # Suppress BeautifulSoup warnings
        from bs4 import MarkupResemblesLocatorWarning
        import warnings
        warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
        
        # Clean content before parsing to avoid parser errors
        cleaned_content = content
        
        # Remove problematic CDATA sections and other invalid markup
        cleaned_content = re.sub(r'<!\[CDATA\[.*?\]\]>', '', cleaned_content, flags=re.DOTALL)
        cleaned_content = re.sub(r'<![^>]*>', '', cleaned_content)  # Remove other invalid comments
        
        # Try different parsers for problematic content
        soup = None
        for parser in ['html.parser', 'lxml', 'html5lib']:
            try:
                soup = BeautifulSoup(cleaned_content, parser)
                break
            except Exception as e:
                print(f"Parser {parser} failed: {e}")
                continue
        
        if soup is None:
            # If all parsers fail, try with minimal content
            try:
                # Extract only table tags using regex as fallback
                table_pattern = r'<table[^>]*>.*?</table>'
                table_matches = re.findall(table_pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
                if table_matches:
                    for table_html in table_matches:
                        try:
                            table_soup = BeautifulSoup(table_html, 'html.parser')
                            table = table_soup.find('table')
                            if table:
                                table_data = parse_html_table_gitcard(table)
                                if table_data and len(table_data) >= 2:
                                    all_tables.append(table_data)
                        except Exception as e:
                            print(f"Error parsing table with regex fallback: {e}")
                            continue
            except Exception as e:
                print(f"Regex fallback failed: {e}")
        else:
            html_tables = soup.find_all('table')
            for table in html_tables:
                try:
                    # Use the proven HTML table parser
                    table_data = parse_html_table_gitcard(table)
                    if table_data and len(table_data) >= 2:
                        all_tables.append(table_data)
                except Exception as e:
                    print(f"Error extracting HTML table: {e}")
                    continue
    except Exception as e:
        print(f"Error parsing HTML content: {e}")
        # Continue with markdown tables only
    
    return (len(all_tables) > 0, all_tables)

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

def extract_markdown_tables_in_parallel(df_unique, col_name, n_jobs=-1):
    """
    Perform parallel extraction of Markdown tables from the specified column (a string)
    in the DataFrame using joblib.Parallel.
    Returns a list of tuples (bool, list) for each row.
    """
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

def extract_markdown_tables_in_parallel_v2(df_unique, col_name, n_jobs=-1):
    """
    Enhanced parallel extraction of Markdown tables from the specified column
    using the improved table extraction logic.
    """
    def process_single_content(content):
        try:
            # Limit content size to avoid memory issues
            if isinstance(content, str) and len(content) > 100000:  # 100KB limit
                content = content[:100000]
                print(f"Truncated large content")
            
            return detect_and_extract_markdown_tables_v2(content)
        except Exception as e:
            print(f"Error processing content: {e}")
            return (False, [])
    
    contents = df_unique[col_name].tolist()
    indices = df_unique.index.tolist()
    
    # Use fewer jobs to avoid memory issues
    if n_jobs == -1:
        n_jobs = min(4, os.cpu_count() or 1)
    
    results = Parallel(n_jobs=n_jobs, batch_size=50)(
        delayed(process_single_content)(content) for content in tqdm(contents, total=len(contents))
    )
    # Build a dict of row_index -> (found_tables_bool, tables_list)
    row_index_to_result = {}
    for i, r_idx in enumerate(indices):
        row_index_to_result[r_idx] = results[i]
    return row_index_to_result

########
def save_markdown_to_csv_from_content_v2(model_id, content, source, file_idx, output_folder):
    """
    Enhanced version: Extract markdown tables using improved parser and save to CSV files.
    Simple naming: model_id_table_x.csv
    """
    # Use the improved table extraction
    _, tables = detect_and_extract_markdown_tables_v2(content)
    saved_paths = []
    for table_idx, table_data in enumerate(tables):
        # Simple naming: model_id_table_x.csv (same as original format)
        csv_path = os.path.join(output_folder, f"{model_id}_table_{table_idx}.csv")
        
        # Convert table_data to DataFrame and save
        try:
            if len(table_data) >= 2:  # Has header and data
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                df.to_csv(csv_path, index=False)
                saved_paths.append(csv_path)
        except Exception as e:
            print(f"Error saving table {table_idx}: {e}")
            continue
    return saved_paths

########
def generate_csv_path_for_dedup(hash_str, table_idx, folder):
    """
    Generate CSV path for deduplicated content using hash.
    """
    short_hash = hash_str[:10]  # Use first 10 characters of hash
    filename = f"{short_hash}_table{table_idx}.csv"
    return os.path.join(folder, filename)

########
def process_github_readmes_v2(row, output_folder, config):
    """
    Enhanced process GitHub readme files for one row using improved parsing.
    """
    model_id = row["modelId"]
    readme_paths = row["readme_path"]
    readme_paths = [os.path.join(config.get('base_path'), csv_path.split("data", 1)[-1].lstrip("/\\")) for csv_path in readme_paths]
    csv_files = []
    if not isinstance(readme_paths, (list, np.ndarray, tuple)) or len(readme_paths) == 0:
        print(f"Skipping {model_id} due to missing or empty readme paths.")
        return csv_files
    # Process each file (with index starting at 1 for naming)
    for file_idx, path in enumerate(readme_paths, start=1):
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                saved = save_markdown_to_csv_from_content_v2(model_id, content, "git", file_idx, output_folder)
                csv_files.extend(saved)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return csv_files

########
def process_markdown_files_v2(github_folder, output_folder):
    """
    Enhanced process all markdown files using improved parsing.
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
        
        # Use the improved extraction
        _, tables = detect_and_extract_markdown_tables_v2(md_content)
        table_csv_basenames = []
        for i, table_data in enumerate(tables):
            csv_basename = f"{base_csv_name}_table_{i}.csv"
            csv_path = os.path.join(output_folder, csv_basename)
            try:
                # Convert table_data to DataFrame and save
                if len(table_data) >= 2:  # Has header and data
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    df.to_csv(csv_path, index=False)
                    table_csv_basenames.append(csv_basename)
            except Exception as e:
                print(f"Error saving {csv_basename}: {e}")
                continue
        # If no tables found, store an empty list instead of skipping
        md_to_csv_mapping[base_csv_name] = table_csv_basenames if table_csv_basenames else []
    # Save mapping as JSON for reference
    mapping_json_path = os.path.join(output_folder, "md_to_csv_mapping.json")
    with open(mapping_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(md_to_csv_mapping, json_file, indent=4)

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
    print('...adding readme_hash')
    # 2) Identify the unique readme rows to parse exactly once
    df_unique = df_merged.drop_duplicates(subset=['readme_hash'], keep='first').copy()
    df_unique = df_unique[df_unique['readme_hash'].notnull()]
    print(f"...Found {len(df_unique)} unique readme content out of {len(df_merged)} rows.")
    output_file = os.path.join(processed_base_path, f"{data_type}_step2_v2.parquet")
    df_merged.drop(columns=['card_readme', 'github_link'], inplace=True, errors='ignore')
    to_parquet(df_merged, output_file)
    print(f"✅ Results saved to: {output_file}")

    # 3) Extract markdown tables for each *unique* readme
    row_index_to_result = extract_markdown_tables_in_parallel(df_unique, col_name='card_readme', n_jobs=-1)
    # 4) Store the extraction results back in df_unique (two columns: 'found', 'extracted_tables')
    df_unique['found_tables_modelcard'] = df_unique.index.map(lambda idx: row_index_to_result[idx][0])
    df_unique['extracted_tables_modelcard'] = df_unique.index.map(lambda idx: row_index_to_result[idx][1])

    #df_unique['extracted_markdown_table_hugging'] = df_unique.index.map(lambda idx: row_index_to_result[idx][1])
    # We'll store these results in a dict: readme_hash -> list_of_tables
    print('Start creating dictionary for {readme_path: list_of_tables}...')
    dedup_folder_hugging = os.path.join(processed_base_path, "deduped_hugging_csvs_v2")  ########
    os.makedirs(dedup_folder_hugging, exist_ok=True)  ########
    hash_to_csv_map = {}  # readme_hash -> [list_of_csv_paths (absolute paths)]  ########
    for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Storing deduped CSVs"):
        hval = row['readme_hash']
        tables = row['extracted_tables_modelcard']
        csv_list = []
        if not tables:
            hash_to_csv_map[hval] = []
            continue
        for j, table_content in enumerate(tables, start=1):
            out_csv_path = generate_csv_path_for_dedup(hval, j, dedup_folder_hugging)  ########
            if os.path.lexists(out_csv_path):
                os.remove(out_csv_path)
            tmp_csv_path = MarkdownHandler.markdown_to_csv(table_content, out_csv_path)
            if tmp_csv_path:
                csv_list.append(os.path.abspath(out_csv_path))  ########
        hash_to_csv_map[hval] = csv_list
    hugging_map_json_path = os.path.join(processed_base_path, "hugging_deduped_mapping_v2.json")  ########
    with open(hugging_map_json_path, 'w', encoding='utf-8') as jf:
        json.dump(hash_to_csv_map, jf, indent=2)
    print(f"✅ Deduped CSV mapping saved to: {hugging_map_json_path}")

    # ---------- End of HuggingFace part ----------
    
    # ---------- GitHub part ----------
    print("⚠️Step 3: Processing GitHub readme files and saving extracted tables to CSV...")
    output_folder_github = os.path.join(processed_base_path, "deduped_github_csvs_v2")
    os.makedirs(output_folder_github, exist_ok=True)
    input_folder_github = os.path.join(config.get('base_path'), 'downloaded_github_readmes')
    process_markdown_files_v2(input_folder_github, output_folder_github) # save csv and md_to_csv_mapping

if __name__ == "__main__":
    main()
