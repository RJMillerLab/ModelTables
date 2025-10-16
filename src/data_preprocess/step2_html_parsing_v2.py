"""
Author: Zhengyuan Dong
Created: 25-09-25
Edited: 25-09-25
Description: Improved HTML table parsing with DuckDB storage support
"""
import sys  
import json
import os
import shutil
from bs4 import BeautifulSoup
import pandas as pd
import duckdb
import sqlite3
from joblib import Parallel, delayed
from tqdm import tqdm
from src.utils import to_parquet
SOUP_PARSER = 'lxml'  # faster than the default html.parser
VERBOSE = False
PROFILE = False
from src.utils import sanitize_table_separators
from tqdm_joblib import tqdm_joblib


def classify_page_from_soup(soup):
    """Classify page as metadata or fulltext using an existing soup (no reparse)."""
    
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

def load_soup_and_tables(html_path):
    """Single-parse loader: returns (soup, figures) where figures are table figures."""
    if not os.path.exists(html_path):
        return None, []
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, SOUP_PARSER)
    figures = soup.select('figure.ltx_table')
    return soup, figures


def extract_math_text(cell):
    """Extract text from math elements and convert to readable format."""
    # Fast path: avoid reparsing HTML strings; modify the Tag in-place for math alttext
    math_elements = cell.find_all('math')
    if math_elements:
        for m in math_elements:
            alttext = m.get('alttext')
            repl = None
            if alttext:
                repl = alttext
            else:
                annotation = m.find('annotation', encoding='application/x-tex')
                if annotation:
                    repl = annotation.get_text(strip=True)
            if repl is not None:
                m.replace_with(repl)
    # Single get_text with separator preserves spaces between inline parts
    text = cell.get_text(separator=' ', strip=True)
    
    # Clean up math formatting like BERTbasebase{}_{\text{base}} to BERT_base
    import re
    # Remove {}_{\text{base}} and similar patterns
    text = re.sub(r'\{_\{\\text\{base\}\}\}', '_base', text)
    text = re.sub(r'\{_\{\\text\{large\}\}\}', '_large', text)
    # Remove any remaining LaTeX formatting
    text = re.sub(r'\{[^}]*\}', '', text)
    # Clean up multiple underscores
    text = re.sub(r'_+', '_', text)
    # Remove trailing underscores
    text = text.rstrip('_')
    # Fix cases like "BERT_}" to "BERT_base"
    text = re.sub(r'([A-Za-z]+)_\}', r'\1_base', text)
    text = re.sub(r'([A-Za-z]+)_\}', r'\1_large', text)
    
    return text

def parse_table_with_nested_structure(table, preserve_bold=True):
    """Parse HTML table while handling nested tables properly."""
    # Check if this is a nested table structure
    nested_tables = table.find_all('table')
    
    if len(nested_tables) > 1:
        # This is a nested table structure, use the outermost table to preserve colspan info
        if VERBOSE:
            print(f"  Found nested table structure with {len(nested_tables)} levels")
        # Use the first (outermost) table to preserve colspan information
        table = nested_tables[0]
    
    rows = table.find_all('tr')
    if not rows:
        return []
    
    # First pass: extract all cell data with their positions
    cell_data = []
    for row_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            text = extract_math_text(cell)
            
            # Check for bold formatting
            if preserve_bold and cell.find('span', class_='ltx_text ltx_font_bold'):
                text = f"**{text}**"
            
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


def create_structured_dataframe(table_data):
    """Create DataFrame with proper structure handling."""
    if not table_data:
        return None
    
    # Find the maximum number of columns
    max_cols = max(len(row) for row in table_data) if table_data else 0
    
    # Pad all rows to have the same number of columns
    padded_data = []
    for row in table_data:
        padded_row = row + [""] * (max_cols - len(row))
        padded_data.append(padded_row)
    
    # Create DataFrame
    df = pd.DataFrame(padded_data)
    
    # Remove first row if it's just column numbers (0,1,2,3,4...)
    if not df.empty and len(df) > 0:
        first_row = df.iloc[0]
        if all(str(val).strip().isdigit() for val in first_row if str(val).strip() != ''):
            df = df.iloc[1:].reset_index(drop=True)
    
    return df


def clean_final_dataframe(df):
    """Final DataFrame cleaning with all requirements."""
    if df is None or df.empty:
        return df
    
    # 1. Optionally remove purely index-like rows (0..N or 1..N). Keep numeric data rows like "11 17".
    def is_index_like_row(row):
        vals = [str(val).strip() for val in row if str(val).strip() != '']
        if len(vals) == 0:
            return True  # empty -> drop later
        # All ints?
        try:
            ints = [int(v) for v in vals]
        except ValueError:
            return False
        if len(ints) < 2:
            return False
        # contiguous sequence starting at 0 or 1
        start = ints[0]
        if start not in (0, 1):
            return False
        expected = list(range(start, start + len(ints)))
        return ints == expected

    df = df[~df.apply(is_index_like_row, axis=1)]
    
    # 1.5. Remove first row if it's just column numbers (0,1,2,3,4...)
    if not df.empty and len(df) > 0:
        first_row = df.iloc[0]
        if all(str(val).strip().isdigit() for val in first_row if str(val).strip() != ''):
            df = df.iloc[1:].reset_index(drop=True)
    
    # 2/3. Preserve column alignment: do NOT drop all-empty columns.
    # Keeping structural empty columns is important for arXiv tables with spacing placeholders.
    
    # 4. Remove columns with unnamed headers (like "Unnamed: 0")
    df = df.loc[:, ~df.columns.astype(str).str.contains('Unnamed', na=False)]
    
    # 5. Remove completely empty rows
    df = df.dropna(axis=0, how='all')
    
    # 6. Reset index
    df = df.reset_index(drop=True)
    
    return df


def extract_tables_and_save_to_duckdb(html_path, paper_id, duckdb_path='data/processed/tables_output.db', preserve_bold=False):
    """
    Extracts all tables from HTML file and saves them to DuckDB database.
    
    Args:
        html_path: Path to HTML file
        paper_id: Paper identifier
        duckdb_path: Path to DuckDB database
        preserve_bold: Whether to preserve bold formatting
    
    Returns:
        List of table names saved to database
    """
    table_names = []
    
    if not os.path.exists(html_path):
        return table_names
    
    soup, table_figures = load_soup_and_tables(html_path)
    if soup is None:
        return table_names
    if not table_figures:
        return table_names
    
    # Connect to DuckDB
    conn = duckdb.connect(duckdb_path)
    
    try:
        for fig_idx, figure in enumerate(table_figures):
            # Extract caption - only process tables with captions
            caption_elem = figure.select_one('figcaption')
            if not caption_elem:
                if VERBOSE:
                    print(f"Skipping table {fig_idx}: No caption found")
                continue
                
            caption = caption_elem.get_text(strip=True)
            # print(f"Processing table {fig_idx}: {caption[:50]}...")  # Commented for speed
            
            # Find the actual table within the figure
            table = figure.select_one('table')
            if not table:
                if VERBOSE:
                    print(f"Skipping table {fig_idx}: No table element found")
                continue
            
            # Parse the table structure with proper colspan/rowspan handling
            table_data = parse_table_with_nested_structure(table, preserve_bold)
            
            if not table_data or len(table_data) < 2:
                if VERBOSE:
                    print(f"Skipping table {fig_idx}: Insufficient data")
                continue
            
            # Create DataFrame with proper structure
            df = create_structured_dataframe(table_data)
            
            if df is None or df.empty:
                if VERBOSE:
                    print(f"Skipping table {fig_idx}: Empty DataFrame")
                continue
            
            # Clean the DataFrame according to requirements
            df = clean_final_dataframe(df)
            # Remove repeated all-separator rows like ':--:' while preserving the first
            df = sanitize_table_separators(df)
            
            if df.empty:
                if VERBOSE:
                    print(f"Skipping table {fig_idx}: Empty DataFrame after cleaning")
                continue
            
            # Create table name using paper_id and table index (replace dots with underscores and add prefix for DuckDB compatibility)
            table_name = f"table_{paper_id.replace('.', '_')}_{fig_idx}"
            
            # Save to DuckDB
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            
            table_names.append(table_name)
            if VERBOSE:
                print(f"Saved table to DuckDB: {table_name}")
    
    finally:
        conn.close()
    
    return table_names


def extract_tables_and_save_to_sqlite(html_path, paper_id, sqlite_path='data/processed/tables_output.db', preserve_bold=False):
    """
    Extracts all tables from HTML file and saves them to SQLite database.
    
    Args:
        html_path: Path to HTML file
        paper_id: Paper identifier
        sqlite_path: Path to SQLite database
        preserve_bold: Whether to preserve bold formatting
    
    Returns:
        List of table names saved to database
    """
    table_names = []
    
    if not os.path.exists(html_path):
        return table_names
    
    soup, table_figures = load_soup_and_tables(html_path)
    if soup is None:
        return table_names
    if not table_figures:
        return table_names
    
    # Connect to SQLite
    conn = sqlite3.connect(sqlite_path)
    
    try:
        for fig_idx, figure in enumerate(table_figures):
            # Extract caption - only process tables with captions
            caption_elem = figure.select_one('figcaption')
            if not caption_elem:
                print(f"Skipping table {fig_idx}: No caption found")
                continue
                
            caption = caption_elem.get_text(strip=True)
            # print(f"Processing table {fig_idx}: {caption[:50]}...")  # Commented for speed
            
            # Find the actual table within the figure
            table = figure.select_one('table')
            if not table:
                print(f"Skipping table {fig_idx}: No table element found")
                continue
            
            # Parse the table structure with proper colspan/rowspan handling
            table_data = parse_table_with_nested_structure(table, preserve_bold)
            
            if not table_data or len(table_data) < 2:
                print(f"Skipping table {fig_idx}: Insufficient data")
                continue
            
            # Create DataFrame with proper structure
            df = create_structured_dataframe(table_data)
            
            if df is None or df.empty:
                print(f"Skipping table {fig_idx}: Empty DataFrame")
                continue
            
            # Clean the DataFrame according to requirements
            df = clean_final_dataframe(df)
            # Remove repeated all-separator rows
            df = sanitize_table_separators(df)
            
            if df.empty:
                print(f"Skipping table {fig_idx}: Empty DataFrame after cleaning")
                continue
            
            # Create table name using paper_id and table index (replace dots with underscores for SQLite compatibility)
            table_name = f"table_{paper_id.replace('.', '_')}_{fig_idx}"
            
            # Save to SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            table_names.append(table_name)
            print(f"Saved table to SQLite: {table_name}")
    
    finally:
        conn.close()
    
    return table_names


def extract_tables_and_save(html_path, paper_id, output_dir='data/processed/tables_output_v2', preserve_bold=False):
    """
    Extracts all <table> tags from the given HTML file, converts them to CSV,
    and saves them directly in the output directory: {output_dir}/{paper_id}_table{idx}.csv

    Returns a list of the CSV file paths created.
    """
    table_paths = []
    if not os.path.exists(html_path):
        return table_paths

    soup, table_figures = load_soup_and_tables(html_path)
    if soup is None or not table_figures:
        return table_paths
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for fig_idx, figure in enumerate(table_figures):
        # Extract caption - only process tables with captions
        caption_elem = figure.select_one('figcaption')
        if not caption_elem:
            if VERBOSE:
                print(f"Skipping table {fig_idx}: No caption found")
            continue
            
        caption = caption_elem.get_text(strip=True)
        # print(f"Processing table {fig_idx}: {caption[:50]}...")  # Commented for speed
        
        # Find the actual table within the figure
        table = figure.select_one('table')
        if not table:
            if VERBOSE:
                print(f"Skipping table {fig_idx}: No table element found")
            continue
        
        # Parse using the same robust parser used for DB outputs
        table_data = parse_table_with_nested_structure(table, preserve_bold)
        df = create_structured_dataframe(table_data)
        df = clean_final_dataframe(df)
        df = sanitize_table_separators(df)
        if df is None or df.empty:
            continue
        # Save as CSV directly in output directory
        csv_path = os.path.join(output_dir, f"{paper_id}_table{fig_idx}.csv")
        df.to_csv(csv_path, index=False, header=False)
        table_paths.append(csv_path)
    
    return table_paths


def process_single_html(html_path, paper_id, output_dir='data/processed/tables_output_v2', duckdb_path='data/processed/tables_output.db', preserve_bold=False, save_mode='csv'):
    """
    Process a single HTML file and save results based on save_mode.
    
    Args:
        html_path: Path to HTML file
        paper_id: Paper identifier
        output_dir: Directory for CSV output
        duckdb_path: Path to database file
        preserve_bold: Whether to preserve bold formatting
        save_mode: 'csv', 'duckdb', 'sqlite', or 'both'
    
    Returns:
        Dictionary with processing results
    """
    result = {
        'paper_id': paper_id,
        'html_path': html_path,
        'page_type': None,
        'csv_paths': [],
        'db_tables': [],
        'error': None
    }
    
    import time
    stage_times = {}
    t0 = time.perf_counter()
    try:
        # Single-parse classification to avoid reparse
        soup, _ = load_soup_and_tables(html_path)
        t_load = time.perf_counter()
        result['page_type'] = classify_page_from_soup(soup) if soup else None
        t_class = time.perf_counter()
        
        # Extract tables based on save_mode (csv and duckdb must be run in separate passes)
        if save_mode == 'csv':
            result['csv_paths'] = extract_tables_and_save(html_path, paper_id, output_dir, preserve_bold)
        elif save_mode == 'duckdb':
            result['db_tables'] = extract_tables_and_save_to_duckdb(html_path, paper_id, duckdb_path, preserve_bold)
        elif save_mode == 'sqlite':
            # TODO: Implement SQLite support
            result['db_tables'] = extract_tables_and_save_to_sqlite(html_path, paper_id, duckdb_path, preserve_bold)
        t_save = time.perf_counter()
        stage_times = {
            'load_soup': t_load - t0,
            'classify': t_class - t_load,
            'extract_save': t_save - t_class,
        }
        
    except Exception as e:
        result['error'] = str(e)
        print(f"Error processing {html_path}: {e}")
    
    if PROFILE:
        result['profile'] = stage_times
    return result


def main():
    """Main function to process HTML files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process HTML files and extract tables')
    parser.add_argument('--input_dir', default='arxiv_fulltext_html', help='Input directory containing HTML files')
    parser.add_argument('--output_dir', default='data/processed/tables_output_v2', help='Output directory for CSV files')
    parser.add_argument('--db_path', default='data/processed/tables_output.db', help='Path to database file')
    parser.add_argument('--preserve_bold', action='store_true', default=False, help='Preserve bold formatting')
    parser.add_argument('--save_mode', default='csv', choices=['csv', 'duckdb', 'sqlite'], help='Save mode: choose one (csv or duckdb or sqlite). Run separately if you need both.')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs (set 1 for sequential)')
    parser.add_argument('--sequential', action='store_true', help='Force sequential processing (disables joblib Parallel)')
    parser.add_argument('--verbose', action='store_true', help='Print per-table logs (skips, saves)')
    parser.add_argument('--profile', action='store_true', help='Collect coarse timing per stage and report mean/P95')
    
    args = parser.parse_args()
    
    global VERBOSE, PROFILE
    VERBOSE = bool(args.verbose)
    PROFILE = bool(args.profile)

    # Get list of HTML files
    html_files = []
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.html'):
            paper_id = filename.replace('.html', '')
            html_path = os.path.join(args.input_dir, filename)
            html_files.append((html_path, paper_id))
    
    print(f"Found {len(html_files)} HTML files to process")
    if VERBOSE:
        print(f"Save mode: {args.save_mode}")
        print(f"Preserve bold: {args.preserve_bold}")
    
    # Process files (sequential or parallel)
    results = []
    if args.sequential or args.n_jobs <= 1:
        for html_path, paper_id in tqdm(html_files, desc="Processing HTML files (seq)"):
            results.append(process_single_html(html_path, paper_id, args.output_dir, args.db_path, args.preserve_bold, args.save_mode))
    else:
        with tqdm_joblib(tqdm(total=len(html_files), desc="Processing HTML files")):
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_single_html)(
                    html_path, paper_id, args.output_dir, args.db_path, args.preserve_bold, args.save_mode
                ) for html_path, paper_id in html_files
            )
    
    # Save results to parquet
    results_df = pd.DataFrame(results)
    to_parquet(results_df, 'data/processed/html_parsing_results_v2.parquet')
    
    # Print summary
    successful = results_df[results_df['error'].isna()]
    failed = results_df[results_df['error'].notna()]
    
    print(f"\nProcessing Summary:")
    print(f"Total files: {len(results_df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if len(failed) > 0 and VERBOSE:
        print(f"\nFailed files:")
        for _, row in failed.iterrows():
            print(f"  {row['paper_id']}: {row['error']}")

    # Profile report
    if PROFILE and 'profile' in results_df.columns:
        prof = results_df['profile'].dropna().tolist()
        if prof:
            import numpy as np
            keys = ['load_soup', 'classify', 'extract_save']
            print("\nTiming (seconds): mean | p95")
            for k in keys:
                vals = np.array([p.get(k, 0.0) for p in prof], dtype=float)
                if vals.size:
                    mean = float(vals.mean())
                    p95 = float(np.percentile(vals, 95))
                    print(f"  {k:12s}: {mean:.4f} | {p95:.4f}")
    
    # Print database summary
    if args.save_mode == 'duckdb':
        total_tables = sum(len(r['db_tables']) for r in results if r['db_tables'])
        print(f"\nTotal tables saved to DuckDB: {total_tables}")
    
    if args.save_mode == 'sqlite':
        total_tables = sum(len(r['db_tables']) for r in results if r['db_tables'])
        print(f"\nTotal tables saved to SQLite: {total_tables}")
    
    if args.save_mode == 'csv':
        total_csvs = sum(len(r['csv_paths']) for r in results if r['csv_paths'])
        print(f"\nTotal CSV files created: {total_csvs}")


if __name__ == "__main__":
    main()
