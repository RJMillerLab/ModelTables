"""
Author: Zhengyuan Dong
Created: 2025-03-11
Last Modified: 2025-04-02
Description: Extract markdown tables from GitHub | Huggingface html, modelcards, readme files, and save them to CSV files.
Usage:
    python -m src.data_preprocess.step2_gitcard_tab
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

# Compile regex patterns once for performance
SEPARATOR_PATTERN = re.compile(r'^\|?\s*[:\-\| ]+\|?\s*$')
PIPE_SEP_PATTERN = re.compile(r'^[\|\-\s:]+$')
BADGE_INDICATOR_PATTERN = re.compile(r'img\.shields\.io|!\[|badge', re.IGNORECASE)
PIPE_NORMALIZE_PATTERN = re.compile(r'\s*\|\s*')
SPACE_NORMALIZE_PATTERN = re.compile(r'\s+')

def normalize_table_for_dedup(table_str):
    """Normalize table content for duplicate detection (optimized version)."""
    lines = [ln.strip() for ln in table_str.split('\n') if ln.strip()]
    # Remove separators (use precompiled pattern)
    data = [ln for ln in lines if not SEPARATOR_PATTERN.fullmatch(ln.strip())]
    # Normalize pipes and spaces
    normalized = []
    for ln in data:
        ln = PIPE_NORMALIZE_PATTERN.sub('|', ln)
        ln = SPACE_NORMALIZE_PATTERN.sub(' ', ln)
        normalized.append(ln.strip())
    return '||'.join(normalized)

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
    Also extracts HTML tables and converts them to markdown format.
    Returns a tuple (bool, list), where the boolean indicates whether at least one table was found,
    and the list contains all matched table strings; if no match is found, returns (False, []).
    """
    if not isinstance(content, str):
        return (False, [])
    
    # Step 1: Extract and convert HTML tables first, then remove them from content
    # This prevents markdown extraction from picking up partial HTML content
    html_tables_extracted = []
    content_without_html_tables = content
    
    # Quick pre-check: only parse HTML if '<table' exists (significant speedup)
    if '<table' not in content:
        # No HTML tables, skip BeautifulSoup parsing
        pass
    else:
        try:
            # Use lxml parser for faster parsing (fallback to html.parser if lxml not available)
            try:
                soup = BeautifulSoup(content, 'lxml')
            except:
                soup = BeautifulSoup(content, 'html.parser')
            html_table_elements = soup.find_all('table')
            
            for html_table in html_table_elements:
                # Parse HTML table to markdown format with proper rowspan/colspan handling
                rows = html_table.find_all('tr')
                if not rows or len(rows) < 2:  # Need at least header + 1 data row
                    continue
                
                # Calculate column count
                max_cols = 0
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    col_count = 0
                    for cell in cells:
                        colspan = int(cell.get('colspan', 1))
                        col_count += colspan
                    max_cols = max(max_cols, col_count)
                
                # Convert HTML table to a 2D grid to handle rowspan/colspan
                # Initialize grid with None (unfilled) cells
                grid = [[None] * max_cols for _ in range(len(rows))]
                
                # Fill the grid with cell contents
                for row_idx, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    col_idx = 0
                    
                    for cell in cells:
                        # Skip cells that are already filled by rowspan
                        while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                            col_idx += 1
                        
                        if col_idx >= max_cols:
                            break
                        
                        text = cell.get_text(strip=True)
                        colspan = int(cell.get('colspan', 1))
                        rowspan = int(cell.get('rowspan', 1))
                        
                        # Fill ALL cells in rowspan/colspan area with the text (like ArXiv processing)
                        # This replicates the cell content across all spanned rows and columns
                        for r in range(row_idx, row_idx + rowspan):
                            for c in range(colspan):
                                if r < len(rows) and col_idx + c < max_cols:
                                    grid[r][col_idx + c] = text
                        
                        col_idx += colspan
                
                # Replace None with empty string (keep empty cells as empty)
                for row_idx in range(len(grid)):
                    for col_idx in range(len(grid[row_idx])):
                        if grid[row_idx][col_idx] is None:
                            grid[row_idx][col_idx] = ''
                
                # Convert grid to markdown rows
                markdown_rows = []
                for row in grid:
                    # Only add row if it has non-empty cells
                    if any(cell.strip() for cell in row):
                        markdown_row = '| ' + ' | '.join(row) + ' |'
                        markdown_rows.append(markdown_row)
                
                # Add separator after header (first row)
                if len(markdown_rows) >= 2:
                    separator = '|' + '|'.join(['---'] * len(markdown_rows[0].split('|')[1:-1])) + '|'
                    markdown_table = markdown_rows[0] + '\n' + separator + '\n' + '\n'.join(markdown_rows[1:])
                    html_tables_extracted.append(markdown_table)
                
                # Remove this HTML table from content to avoid confusion in markdown extraction
                html_str = str(html_table)
                content_without_html_tables = content_without_html_tables.replace(html_str, '')
        except Exception:
            # If HTML parsing fails, just use original content
            pass
    
    # Step 2: Extract markdown tables from the cleaned content (without HTML tables)
    
    # Enhanced pattern that handles indentation and various table formats (GitHub-Flavored Markdown)
    lines = content_without_html_tables.split('\n')
    markdown_tables = []
    current_table = []
    in_table = False
    seen_non_sep_row = False  # ensure we have at least one non-separator row to avoid header-only captures
    
    for line in lines:
        raw_line = line
        line = line.strip()
        if not line:
            if in_table and current_table:
                # End of table
                markdown_tables.append('\n'.join(current_table))
                current_table = []
                in_table = False
                seen_non_sep_row = False
            continue
            
        # Consider lines that either:
        # - start and end with '|'
        # - contain '|' and we can normalize by adding missing leading/trailing pipes
        is_pipe_row = ('|' in line)
        starts_with_pipe = line.startswith('|')
        ends_with_pipe = line.endswith('|')

        if starts_with_pipe and ends_with_pipe:
            if not in_table:
                in_table = True
                seen_non_sep_row = False
            
            # Handle lines with double pipes (||) - these create empty cells
            # Example: |||A|B| means 2 empty cells, then A, then B
            # We should NOT split by || as that breaks empty cell preservation
            # Instead, just append the line as-is to preserve all empty cells
            current_table.append(line)
            
            # Track if this is a non-separator data row
            inner = line[1:-1]
            cells = [c.strip() for c in inner.split('|')]
            if not re.fullmatch(r'[\s:\-\|]*', ''.join(cells)):
                seen_non_sep_row = True
        elif is_pipe_row:
            # Normalize rows that have pipes but missing borders
            normalized = line
            if not starts_with_pipe:
                normalized = '|' + normalized
            if not normalized.endswith('|'):
                normalized = normalized + '|'

            if not in_table:
                in_table = True
                seen_non_sep_row = False

            # Accept the normalized row
            current_table.append(normalized)

            # Mark if non-separator
            inner = normalized[1:-1]
            cells = [c.strip() for c in inner.split('|')]
            if not re.fullmatch(r'[\s:\-\|]*', ''.join(cells)):
                seen_non_sep_row = True

        elif in_table:
            # Check if this is a separator line (contains |, -, :, spaces only)
            if PIPE_SEP_PATTERN.match(line):
                current_table.append(line)
            else:
                # End of table
                if current_table:
                    # Only keep if at least one non-separator row exists
                    if seen_non_sep_row:
                        markdown_tables.append('\n'.join(current_table))
                current_table = []
                in_table = False
                seen_non_sep_row = False
    
    # Don't forget the last table if content ends with a table
    if in_table and current_table:
        if seen_non_sep_row:
            markdown_tables.append('\n'.join(current_table))
    
    # Step 3: Combine HTML tables and markdown tables, then filter
    # HTML tables go first (they were extracted first)
    all_tables = html_tables_extracted + markdown_tables
    
    # Filter out tables that are too short (less than header + one data row)
    valid_tables = []
    for table in all_tables:
        non_empty_lines = [ln for ln in table.split('\n') if ln.strip()]
        if len(non_empty_lines) < 2:
            continue
        # Require presence of a separator row OR at least one data row with multiple cells
        has_sep = any(SEPARATOR_PATTERN.fullmatch(ln.strip()) for ln in non_empty_lines)
        if not has_sep:
            # Fallback: treat as table if every line contains at least one pipe (>=2 cells expected)
            multi_cell = sum(1 for ln in non_empty_lines if ln.count('|') >= 2)
            if multi_cell < 2:
                continue
        
        # Filter out badge/decoration tables (common in READMEs)
        # Characteristics: empty or minimal headers, mostly badges/links, very few data rows
        is_badge_table = False
        
        # Identify separator lines more precisely (must contain at least one dash)
        separator_lines = [ln for ln in non_empty_lines if '-' in ln and SEPARATOR_PATTERN.fullmatch(ln.strip())]
        data_lines = [ln for ln in non_empty_lines if ln not in separator_lines]
        
        if len(data_lines) <= 2:  # Only 1-2 data rows (including header)
            # Check if first row (header) is mostly empty
            if data_lines:
                header = data_lines[0]
                # Parse cells properly (split by | and remove leading/trailing empty strings)
                header_parts = header.split('|')
                # Remove leading/trailing empty parts from pipe-delimited format
                if len(header_parts) > 0 and not header_parts[0].strip():
                    header_parts = header_parts[1:]
                if len(header_parts) > 0 and not header_parts[-1].strip():
                    header_parts = header_parts[:-1]
                header_cells = [c.strip() for c in header_parts]
                
                # Calculate empty ratio
                if len(header_cells) > 0:
                    empty_count = sum(1 for c in header_cells if not c or c == '')
                    empty_header_ratio = empty_count / len(header_cells)
                    
                    # Check if content is mostly badges/images (use precompiled regex)
                    all_text = ' '.join(data_lines)
                    badge_indicators = len(BADGE_INDICATOR_PATTERN.findall(all_text))
                    has_many_badges = badge_indicators >= 3  # At least 3 badge indicators
                    
                    # Mark as badge table if header is mostly empty AND has many badges
                    if empty_header_ratio > 0.5 and has_many_badges:
                        is_badge_table = True
        
        if not is_badge_table:
            valid_tables.append(table)
    
    return (len(valid_tables) > 0, valid_tables)
########

def sanitize_markdown_table_separators(table: str) -> str:
    from src.utils import sanitize_table_separators
    return sanitize_table_separators(table)
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
        table = sanitize_markdown_table_separators(table)
        tmp_path = MarkdownHandler.markdown_to_csv(table, csv_path, verbose=True) # print failed saving, avoid it work silently
        if tmp_path:
            #saved_paths.append(tmp_path)
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

def generate_csv_path_for_dedup(hash_str, table_idx, folder):
    """
    Generate a unique CSV file path based on the readme_hash and table index.
    Example: /path/to/deduped_hugging_csvs/<hash>_table1.csv
    """
    short_hash = hash_str[:10]
    filename = f"{short_hash}_table{table_idx}.csv"
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
            table = sanitize_markdown_table_separators(table)
            csv_basename = f"{base_csv_name}_table_{i}.csv"
            csv_path = os.path.join(output_folder, csv_basename)
            tmp_path = MarkdownHandler.markdown_to_csv(table, csv_path)
            if tmp_path:
                table_csv_basenames.append(csv_basename)
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
    
    # Define processing function for parallel execution at hash level
    def process_and_save_tables_for_hash(row_tuple, output_folder):
        """Process tables for one readme_hash and save CSVs. Returns (hash, csv_list)."""
        idx, row_series = row_tuple
        hval = row_series['readme_hash']
        tables = row_series['extracted_tables_modelcard']
        csv_list = []
        
        if not tables:
            return (hval, [])
        
        # Additional deduplication before saving: remove duplicate tables within same readme
        # Use the optimized global function
        seen_signatures = set()
        unique_tables = []
        for table_content in tables:
            sig = normalize_table_for_dedup(table_content)
            if sig not in seen_signatures:
                unique_tables.append(table_content)
                seen_signatures.add(sig)
        
        # Step 1: Sanitize all tables first
        sanitized_tables = [sanitize_markdown_table_separators(t) for t in unique_tables]
        
        # Step 2: Try converting all tables and track success/failure
        conversion_results = []
        for j, table_content in enumerate(sanitized_tables, start=1):
            temp_out_path = generate_csv_path_for_dedup(hval, j, dedup_folder_hugging)
            tmp_csv_path = MarkdownHandler.markdown_to_csv(table_content, temp_out_path, verbose=False)
            conversion_results.append({
                'original_index': j,
                'success': tmp_csv_path is not None,
                'temp_path': temp_out_path if tmp_csv_path else None
            })
        
        # Step 3: Assign sequential numbers to successful conversions
        table_counter = 0
        for result in conversion_results:
            if result['success']:
                table_counter += 1
                result['final_index'] = table_counter
        
        # Step 4: Rename files to sequential numbers and collect paths
        for result in conversion_results:
            if result['success']:
                temp_path = result['temp_path']
                final_index = result['final_index']
                final_out_path = generate_csv_path_for_dedup(hval, final_index, output_folder)
                
                # Rename if needed
                if temp_path != final_out_path:
                    if os.path.lexists(final_out_path):
                        os.remove(final_out_path)
                    os.rename(temp_path, final_out_path)
                
                csv_list.append(os.path.abspath(final_out_path))
            else:
                # Clean up failed conversion temp file
                temp_path = generate_csv_path_for_dedup(hval, result['original_index'], output_folder)
                if os.path.lexists(temp_path):
                    os.remove(temp_path)
        
        return (hval, csv_list)
    
    # Parallel processing at hash level (each hash's tables processed independently)
    print(f"Processing {len(df_unique)} unique readme hashes in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(process_and_save_tables_for_hash)(row, dedup_folder_hugging)
        for row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Storing deduped CSVs")
    )
    
    # Build the hash_to_csv_map from results
    hash_to_csv_map = {hval: csv_list for hval, csv_list in results}
    
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
    process_markdown_files(input_folder_github, output_folder_github) # save csv and md_to_csv_mapping

if __name__ == "__main__":
    main()


"""
Exampled Output

⚠️⚠️Step 1: Loading modelcard_step1 data...
✅ After merge: 1108759 rows.
⚠️Step 2: Extracting markdown tables from 'card_readme' (HuggingFace)...
...adding readme_hash
...Found 387883 unique readme content out of 1108759 rows.
100%|███████████████████████████████████████████████████| 387883/387883 [01:20<00:00, 4793.16it/s]
Start creating dictionary for {readme_path: list_of_tables}...
Storing deduped CSVs: 100%|██████████████████████████████| 387883/387883 [13:59<00:00, 461.85it/s]
✅ Deduped CSV mapping saved to: /Users/doradong/Repo/CitationLake/data/processed/hugging_deduped_mapping.json
⚠️Step 2.2: Creating symlinks to 'sym_hugging_csvs' ...
Linking CSVs: 100%|███████████████████████████████████| 1108759/1108759 [04:25<00:00, 4178.61it/s]
End of HuggingFace part.

⚠️Step 3: Processing GitHub readme files and saving extracted tables to CSV...
Processing Markdown files:   7%|█▉                           | 1304/19916 [00:19<04:15, 72.82it/s]⚠️ Skipping 8d77428a41be7b07877aed1208e7106a.md (File too large: 14.38 MB)
Processing Markdown files:  11%|███▏                         | 2156/19916 [00:30<03:20, 88.65it/s]⚠️ Skipping 3769fd2d562ed5bfb1412b4be4ea91f4.md (File too large: 49.72 MB)
Processing Markdown files:  11%|███▏                         | 2199/19916 [00:33<09:34, 30.81it/s]⚠️ Skipping 57e6cbf06af2077ad39f6dc022f448a0.md (File too large: 8.64 MB)
Processing Markdown files:  12%|███▌                         | 2457/19916 [00:36<03:11, 91.12it/s]⚠️ Skipping 88f0e7f02c674aa25e628a3d79e9bbdb.md (File too large: 12.53 MB)
Processing Markdown files:  38%|███████████                  | 7616/19916 [01:49<06:46, 30.29it/s]⚠️ Skipping 6880c06f052a3e6d52e5f6da9258bb1a.md (File too large: 7.17 MB)
Processing Markdown files:  45%|████████████▉                | 8903/19916 [02:06<02:11, 83.58it/s]⚠️ Skipping fc2193033fac642fbc50a27e31291ca7.md (File too large: 12.78 MB)
Processing Markdown files:  50%|██████████████▍              | 9900/19916 [02:36<02:39, 62.71it/s]⚠️ Skipping 665b970b8e4c4aeead66dd542331c37e.md (File too large: 5.95 MB)
Processing Markdown files:  62%|█████████████████▍          | 12394/19916 [11:46<01:23, 90.36it/s]⚠️ Skipping 44a0008dcc96aad0da0c0ac00fdc4d09.md (File too large: 8.12 MB)
Processing Markdown files:  66%|██████████████████▍         | 13093/19916 [11:55<01:15, 90.02it/s]⚠️ Skipping b7c00a7974825548053cb4a8501f1440.md (File too large: 22.23 MB)
Processing Markdown files:  67%|██████████████████▊         | 13350/19916 [11:59<01:56, 56.30it/s]⚠️ Skipping de3a438afd20c4649d2b2fbe800e71e4.md (File too large: 10.60 MB)
Processing Markdown files:  67%|██████████████████▊         | 13365/19916 [11:59<02:01, 53.93it/s]⚠️ Skipping b6c76f03d949332d18f77b4030d3493d.md (File too large: 6.07 MB)
Processing Markdown files:  69%|███████████████████▎        | 13772/19916 [12:05<01:27, 70.34it/s]⚠️ Skipping fc856ed1392888cab683eed4deb8e61b.md (File too large: 14.79 MB)
Processing Markdown files:  71%|███████████████████▊        | 14106/19916 [12:10<01:12, 80.65it/s]⚠️ Skipping bbb830ff97cb048d9dbd1fd0523f28b4.md (File too large: 29.72 MB)
Processing Markdown files:  71%|████████████████████        | 14231/19916 [12:11<01:08, 83.58it/s]⚠️ Skipping ca9977191569120dcb613632526bede5.md (File too large: 11.89 MB)
Processing Markdown files:  78%|█████████████████████▉      | 15633/19916 [12:31<01:03, 67.55it/s]⚠️ Skipping d288d0c1b17b6c35cb56533d4b75c3b8.md (File too large: 34.09 MB)
Processing Markdown files:  85%|███████████████████████▉    | 16989/19916 [12:51<00:38, 75.88it/s]⚠️ Skipping 874f2976b310e16e87d24db0eee0e440.md (File too large: 24.67 MB)
Processing Markdown files:  88%|████████████████████████▌   | 17452/19916 [12:59<00:56, 43.41it/s]⚠️ Skipping 3803c3116946fc9c859c576a724b0a47.md (File too large: 12.10 MB)
Processing Markdown files:  94%|██████████████████████████▎ | 18744/19916 [13:18<00:13, 84.76it/s]⚠️ Skipping 50acb0cbbf98479df6ed5323953efc13.md (File too large: 14.31 MB)
Processing Markdown files: 100%|████████████████████████████| 19916/19916 [13:54<00:00, 23.87it/s]
Processing GitHub readmes: 100%|█████████████████████| 1108759/1108759 [01:42<00:00, 10784.97it/s]
⚠️Step 4: Saving integrated DataFrame to Parquet file...
✅ All done. Results saved to: /Users/doradong/Repo/CitationLake/data/processed/modelcard_step2.parquet
"""