"""
Author: Zhengyuan Dong
Created: 25-09-25
Edited: 25-09-25
Description: Improved markdown parsing for GitHub and HuggingFace READMEs with table extraction
"""
import os
import argparse
import traceback
import html2text
import re
import hashlib
import json
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import pandas as pd
import duckdb
import sqlite3
from bs4 import BeautifulSoup
from src.utils import to_parquet
from src.data_ingestion.readme_parser import MarkdownHandler


def extract_math_text_md(cell):
    """Extract text from math elements and convert to readable format."""
    # Find all math elements
    math_elements = cell.find_all('math')
    if not math_elements:
        text = cell.get_text(strip=True)
    else:
        # Replace math elements with their alttext or annotation
        text = str(cell)
        for math in math_elements:
            alttext = math.get('alttext', '')
            if alttext:
                text = text.replace(str(math), alttext)
            else:
                # Try to get annotation
                annotation = math.find('annotation', encoding='application/x-tex')
                if annotation:
                    text = text.replace(str(math), annotation.get_text(strip=True))
        
        # Parse the modified HTML
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


def parse_html_table_with_rowspan_colspan(table):
    """Parse HTML table with proper rowspan and colspan handling."""
    rows = table.find_all('tr')
    if not rows:
        return []
    
    # First pass: extract all cell data with their positions
    cell_data = []
    for row_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            text = extract_math_text_md(cell)
            
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


def detect_and_extract_markdown_tables(content: str):
    """
    Detect and extract all Markdown tables from the given text.
    Returns a tuple (bool, list), where the boolean indicates whether at least one table was found,
    and the list contains all matched table strings.
    """
    if not isinstance(content, str):
        return (False, [])
    
    all_tables = []
    
    # 1. Extract markdown tables (| format)
    markdown_table_pattern = r"(?:\|[^\n]*?\|[\s]*\n)+\|[-:| ]*\|[\s]*\n(?:\|[^\n]*?\|(?:\n|$))+"
    md_matches = re.findall(markdown_table_pattern, content, re.MULTILINE)
    if md_matches:
        all_tables.extend([match.strip() for match in md_matches])
    
    # 2. Extract HTML tables (<table> format)
    soup = BeautifulSoup(content, 'html.parser')
    html_tables = soup.find_all('table')
    for table in html_tables:
        # Convert HTML table to markdown-like format for processing
        table_text = str(table)
        all_tables.append(table_text)
    
    return (len(all_tables) > 0, all_tables)


def clean_markdown_content(content: str, preserve_formatting: bool = True):
    """
    Clean markdown content while optionally preserving formatting.
    
    Args:
        content: Raw markdown content
        preserve_formatting: Whether to preserve bold/italic formatting
    
    Returns:
        Cleaned markdown content
    """
    if not content:
        return content
    
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Clean up table formatting
    lines = content.split('\n')
    cleaned_lines = []
    in_table = False
    
    for line in lines:
        # Detect table start
        if '|' in line and not in_table:
            in_table = True
        # Detect table end
        elif in_table and '|' not in line and line.strip():
            in_table = False
        
        if in_table:
            # Don't skip separator lines - they're needed for markdown tables!
            # Ensure proper table formatting
            if '|' in line and not (line.startswith('|') and line.endswith('|')):
                line = '|' + line.strip() + '|'
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_tables_from_markdown(content: str, preserve_formatting: bool = True):
    """
    Extract tables from markdown content and return structured data.
    
    Args:
        content: Markdown content
        preserve_formatting: Whether to preserve formatting in table cells
    
    Returns:
        List of table data (list of lists)
    """
    found, tables = detect_and_extract_markdown_tables(content)
    if not found:
        return []
    
    extracted_tables = []
    for table in tables:
        try:
            # Check if this is an HTML table
            if table.strip().startswith('<table'):
                # Parse HTML table using the same advanced logic as HTML parser
                try:
                    soup = BeautifulSoup(table, 'html.parser')
                    html_table = soup.find('table')
                    if html_table:
                        # Use the same logic from step2_arxiv_parse_v2.py
                        table_data = parse_html_table_with_rowspan_colspan(html_table)
                        if table_data and len(table_data) >= 2:
                            extracted_tables.append(table_data)
                except Exception as e:
                    print(f"Error parsing HTML table: {e}")
                    continue
            else:
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
                    
                    extracted_tables.append(cleaned_lines)
        except Exception as e:
            print(f"Error extracting table: {e}")
            continue
    
    return extracted_tables


def process_md_file(filepath, input_root, output_dir, preserve_formatting=True, 
                   extract_tables=True, save_mode='csv', db_path=None):
    """
    Process a single markdown file with enhanced parsing.
    
    Args:
        filepath: Path to markdown file
        input_root: Root directory for relative path calculation
        output_dir: Output directory for processed files
        preserve_formatting: Whether to preserve markdown formatting
        extract_tables: Whether to extract tables
        save_mode: 'csv', 'duckdb', 'sqlite', or 'both'
        db_path: Database path for DB modes
    
    Returns:
        Dictionary with processing results
    """
    rel_path = os.path.relpath(filepath, input_root)
    result = {
        'input_path': filepath,
        'relative_path': rel_path,
        'output_path': None,
        'converted_from_html': False,
        'tables_found': False,
        'tables_extracted': 0,
        'table_paths': [],
        'db_tables': [],
        'error': None,
    }
    
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Detect HTML-like content
        is_html_like = ("<!DOCTYPE" in content.upper() or 
                       content.strip().lower().startswith("<html") or
                       "<body>" in content.lower())
        
        if is_html_like:
            # Convert HTML to markdown
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.body_width = 0  # Don't wrap lines
            new_content = h.handle(content)
            result['converted_from_html'] = True
        else:
            new_content = content
        
        # Clean the markdown content
        cleaned_content = clean_markdown_content(new_content, preserve_formatting)
        
        # Extract tables if requested
        if extract_tables:
            tables = extract_tables_from_markdown(cleaned_content, preserve_formatting)
            result['tables_found'] = len(tables) > 0
            result['tables_extracted'] = len(tables)
            
            if tables and save_mode in ['csv', 'both']:
                # Save tables as CSV files
                base_name = os.path.splitext(rel_path)[0]
                for i, table in enumerate(tables):
                    csv_filename = f"{base_name}_table_{i}.csv"
                    csv_path = os.path.join(output_dir, csv_filename)
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df.to_csv(csv_path, index=False)
                    result['table_paths'].append(csv_path)
            
            if tables and save_mode in ['duckdb', 'both']:
                # Save tables to DuckDB
                conn = duckdb.connect(db_path or 'data/processed/md_tables.db')
                try:
                    base_name = os.path.splitext(rel_path)[0].replace('/', '_').replace('.', '_')
                    for i, table in enumerate(tables):
                        table_name = f"md_table_{base_name}_{i}"
                        df = pd.DataFrame(table[1:], columns=table[0])
                        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                        result['db_tables'].append(table_name)
                finally:
                    conn.close()
            
            if tables and save_mode == 'sqlite':
                # Save tables to SQLite
                conn = sqlite3.connect(db_path or 'data/processed/md_tables.db')
                try:
                    base_name = os.path.splitext(rel_path)[0].replace('/', '_').replace('.', '_')
                    for i, table in enumerate(tables):
                        table_name = f"md_table_{base_name}_{i}"
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df.to_sql(table_name, conn, if_exists='replace', index=False)
                        result['db_tables'].append(table_name)
                finally:
                    conn.close()
        
        # Save processed markdown content
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
        result['output_path'] = out_path
        
    except Exception as e:
        result['error'] = f"{e}\n{traceback.format_exc()}"
    
    return result


def process_md_files_parallel(input_dir, output_dir, n_jobs=4, preserve_formatting=True,
                             extract_tables=True, save_mode='csv', db_path=None,
                             results_parquet="data/processed/md_parsing_results_v2.parquet"):
    """
    Process markdown files in parallel with enhanced features.
    
    Args:
        input_dir: Input directory containing markdown files
        output_dir: Output directory for processed files
        n_jobs: Number of parallel jobs
        preserve_formatting: Whether to preserve markdown formatting
        extract_tables: Whether to extract tables
        save_mode: 'csv', 'duckdb', 'sqlite', or 'both'
        db_path: Database path for DB modes
        results_parquet: Path to save results summary
    
    Returns:
        List of processing results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all markdown files
    md_files = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(('.md', '.markdown')):
                md_files.append(os.path.join(root, name))
    
    print(f"Found {len(md_files)} markdown files under {input_dir}")
    print(f"Save mode: {save_mode}")
    print(f"Extract tables: {extract_tables}")
    print(f"Preserve formatting: {preserve_formatting}")
    
    # Process files in parallel
    with tqdm_joblib(tqdm(total=len(md_files), desc="Processing MD files")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_md_file)(
                filepath, input_dir, output_dir, preserve_formatting, 
                extract_tables, save_mode, db_path
            ) for filepath in md_files
        )
    
    # Save results summary
    df = pd.DataFrame(results)
    try:
        to_parquet(df, results_parquet)
        print(f"Saved results parquet to {results_parquet}")
    except Exception as e:
        print(f"Warning: failed to save results parquet: {e}")
    
    # Print summary
    successful = df[df['error'].isna()]
    failed = df[df['error'].notna()]
    
    print(f"\nProcessing Summary:")
    print(f"Total files: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if extract_tables:
        total_tables = df['tables_extracted'].sum()
        files_with_tables = len(df[df['tables_found'] == True])
        print(f"Files with tables: {files_with_tables}")
        print(f"Total tables extracted: {total_tables}")
        
        if save_mode in ['csv', 'both']:
            total_csvs = df['table_paths'].apply(len).sum()
            print(f"Total CSV files created: {total_csvs}")
        
        if save_mode in ['duckdb', 'both']:
            total_db_tables = df['db_tables'].apply(len).sum()
            print(f"Total tables saved to DuckDB: {total_db_tables}")
        
        if save_mode == 'sqlite':
            total_db_tables = df['db_tables'].apply(len).sum()
            print(f"Total tables saved to SQLite: {total_db_tables}")
    
    if len(failed) > 0:
        print(f"\nFailed files:")
        for _, row in failed.iterrows():
            print(f"  {row['relative_path']}: {row['error'][:100]}...")
    
    return results


def main():
    """Main function to process markdown files."""
    parser = argparse.ArgumentParser(description='Enhanced markdown parsing for GitHub/HuggingFace READMEs')
    parser.add_argument('--input_dir', default='data/downloaded_github_readmes', 
                       help='Input directory containing markdown files')
    parser.add_argument('--output_dir', default='data/processed/md_processed_v2', 
                       help='Output directory for processed files')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')
    parser.add_argument('--preserve_formatting', action='store_true', default=True,
                       help='Preserve markdown formatting')
    parser.add_argument('--extract_tables', action='store_true', default=True,
                       help='Extract tables from markdown')
    parser.add_argument('--save_mode', default='csv', 
                       choices=['csv', 'duckdb', 'sqlite', 'both'],
                       help='Save mode: csv, duckdb, sqlite, or both')
    parser.add_argument('--db_path', default='data/processed/md_tables_v2.db',
                       help='Database path for DB modes')
    parser.add_argument('--results_parquet', default='data/processed/md_parsing_results_v2.parquet',
                       help='Path to save results summary')
    
    args = parser.parse_args()
    
    process_md_files_parallel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        preserve_formatting=args.preserve_formatting,
        extract_tables=args.extract_tables,
        save_mode=args.save_mode,
        db_path=args.db_path,
        results_parquet=args.results_parquet,
    )


if __name__ == "__main__":
    main()
