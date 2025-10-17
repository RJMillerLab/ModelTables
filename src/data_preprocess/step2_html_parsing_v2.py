"""
Author: Zhengyuan Dong
Created: 25-09-25
Edited: 25-10-16
Description: HTML table parsing with rowspan/colspan fix and DuckDB/SQLite/CSV support.
"""

import os
import pandas as pd
import duckdb
import sqlite3
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from src.utils import to_parquet, sanitize_table_separators

SOUP_PARSER = 'lxml'
VERBOSE = False
PROFILE = False


def classify_page_from_soup(soup):
    meta_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.lower().startswith('citation_')})
    if meta_tags and len(meta_tags) >= 3:
        return 'metadata'
    sections = soup.find_all(['section', 'article'])
    if len(sections) >= 2:
        return 'fulltext'
    body_text = soup.get_text(separator=' ').lower()
    if "introduction" in body_text and "conclusion" in body_text:
        return 'fulltext'
    return 'metadata'


def load_soup_and_tables(html_path):
    if not os.path.exists(html_path):
        return None, []
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, SOUP_PARSER)
    figures = soup.select('figure.ltx_table')
    return soup, figures


def extract_math_text(cell):
    math_elements = cell.find_all('math')
    if math_elements:
        for m in math_elements:
            alttext = m.get('alttext')
            repl = alttext or m.get_text(" ", strip=True)
            if repl is not None:
                m.replace_with(repl)
    import re
    text = cell.get_text(" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text


def parse_table_with_nested_structure(table, preserve_bold=True):
    """Improved table parser preserving rowspan, colspan, and empty cells."""
    nested_tables = table.find_all('table')
    if len(nested_tables) > 1:
        table = nested_tables[0]

    rows = table.find_all('tr')
    if not rows:
        return []

    cell_data = []
    for row_idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            text = extract_math_text(cell)
            if preserve_bold and cell.find('span', class_='ltx_text ltx_font_bold'):
                text = f"**{text}**"
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            cell_data.append({'row': row_idx, 'text': text, 'colspan': colspan, 'rowspan': rowspan})

    max_cols = max(sum(int(c.get('colspan', 1)) for c in r.find_all(['td', 'th'])) for r in rows)
    grid = [['' for _ in range(max_cols)] for _ in range(len(rows))]
    occupied = [[False] * max_cols for _ in range(len(rows))]

    for cell_info in cell_data:
        row_start = cell_info['row']
        text = cell_info['text']
        colspan = cell_info['colspan']
        rowspan = cell_info['rowspan']

        # find next available col that isn't occupied
        col_idx = 0
        while col_idx < max_cols and occupied[row_start][col_idx]:
            col_idx += 1

        for r in range(row_start, row_start + rowspan):
            for c in range(col_idx, col_idx + colspan):
                if r < len(grid) and c < max_cols:
                    if grid[r][c] == '':
                        grid[r][c] = text if text != '' else ''
                    occupied[r][c] = True

    return grid


def create_structured_dataframe(table_data):
    if not table_data:
        return None
    max_cols = max(len(row) for row in table_data)
    padded_data = [row + [''] * (max_cols - len(row)) for row in table_data]
    df = pd.DataFrame(padded_data)
    if not df.empty and len(df) > 0:
        first_row = df.iloc[0]
        if all(str(val).strip().isdigit() for val in first_row if str(val).strip() != ''):
            df = df.iloc[1:].reset_index(drop=True)
    return df


def clean_final_dataframe(df):
    if df is None or df.empty:
        return df

    def is_index_like_row(row):
        vals = [str(val).strip() for val in row if str(val).strip() != '']
        if len(vals) == 0:
            return True
        try:
            ints = [int(v) for v in vals]
        except ValueError:
            return False
        if len(ints) < 2:
            return False
        start = ints[0]
        if start not in (0, 1):
            return False
        expected = list(range(start, start + len(ints)))
        return ints == expected

    df = df[~df.apply(is_index_like_row, axis=1)]
    if not df.empty and len(df) > 0:
        first_row = df.iloc[0]
        if all(str(val).strip().isdigit() for val in first_row if str(val).strip() != ''):
            df = df.iloc[1:].reset_index(drop=True)
    df = df.loc[:, ~df.columns.astype(str).str.contains('Unnamed', na=False)]
    df = df.dropna(axis=0, how='all')
    df = df.reset_index(drop=True)
    return df


def extract_tables_and_save_to_duckdb(html_path, paper_id, duckdb_path='data/processed/tables_output.db', preserve_bold=False):
    table_names = []
    if not os.path.exists(html_path):
        return table_names
    soup, table_figures = load_soup_and_tables(html_path)
    if soup is None or not table_figures:
        return table_names
    conn = duckdb.connect(duckdb_path)
    try:
        for fig_idx, figure in enumerate(table_figures):
            caption_elem = figure.select_one('figcaption')
            if not caption_elem:
                continue
            table = figure.select_one('table')
            if not table:
                continue
            table_data = parse_table_with_nested_structure(table, preserve_bold)
            df = create_structured_dataframe(table_data)
            df = clean_final_dataframe(df)
            df = sanitize_table_separators(df)
            if df is None or df.empty:
                continue
            table_name = f"table_{paper_id.replace('.', '_')}_{fig_idx}"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            table_names.append(table_name)
    finally:
        conn.close()
    return table_names


def extract_tables_and_save(html_path, paper_id, output_dir='data/processed/tables_output_v2', preserve_bold=False):
    table_paths = []
    if not os.path.exists(html_path):
        return table_paths
    soup, table_figures = load_soup_and_tables(html_path)
    if soup is None or not table_figures:
        return table_paths
    os.makedirs(output_dir, exist_ok=True)
    for fig_idx, figure in enumerate(table_figures):
        caption_elem = figure.select_one('figcaption')
        if not caption_elem:
            continue
        table = figure.select_one('table')
        if not table:
            continue
        table_data = parse_table_with_nested_structure(table, preserve_bold)
        df = create_structured_dataframe(table_data)
        df = clean_final_dataframe(df)
        df = sanitize_table_separators(df)
        # Normalize whitespace-only cells to NA and drop fully-empty rows
        try:
            df = df.replace(r'^\s*$', pd.NA, regex=True)
            df = df.dropna(axis=0, how='all')
        except Exception:
            pass
        if df is None or df.empty:
            continue
        csv_path = os.path.join(output_dir, f"{paper_id}_table{fig_idx}.csv")
        df.to_csv(csv_path, index=False, header=False, na_rep='')  # keep blanks
        table_paths.append(csv_path)
    return table_paths


def process_single_html(html_path, paper_id, output_dir='data/processed/tables_output_v2', duckdb_path='data/processed/tables_output.db', preserve_bold=False, save_mode='csv'):
    result = {'paper_id': paper_id, 'html_path': html_path, 'page_type': None, 'csv_paths': [], 'db_tables': [], 'error': None}
    import time
    t0 = time.perf_counter()
    try:
        soup, _ = load_soup_and_tables(html_path)
        result['page_type'] = classify_page_from_soup(soup) if soup else None
        if save_mode == 'csv':
            result['csv_paths'] = extract_tables_and_save(html_path, paper_id, output_dir, preserve_bold)
        elif save_mode == 'duckdb':
            result['db_tables'] = extract_tables_and_save_to_duckdb(html_path, paper_id, duckdb_path, preserve_bold)
    except Exception as e:
        result['error'] = str(e)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process HTML files and extract tables')
    parser.add_argument('--input_dir', default='arxiv_fulltext_html')
    parser.add_argument('--output_dir', default='data/processed/tables_output_v2')
    parser.add_argument('--db_path', default='data/processed/tables_output.db')
    parser.add_argument('--preserve_bold', action='store_true')
    parser.add_argument('--save_mode', default='csv', choices=['csv', 'duckdb'])
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()
    global VERBOSE, PROFILE
    html_files = [(os.path.join(args.input_dir, f), f.replace('.html', '')) for f in os.listdir(args.input_dir) if f.endswith('.html')]
    print(f"Found {len(html_files)} HTML files to process")
    results = []
    if args.sequential or args.n_jobs <= 1:
        for html_path, paper_id in tqdm(html_files):
            results.append(process_single_html(html_path, paper_id, args.output_dir, args.db_path, args.preserve_bold, args.save_mode))
    else:
        with tqdm_joblib(tqdm(total=len(html_files), desc="Processing HTML files")):
            results = Parallel(n_jobs=args.n_jobs)(delayed(process_single_html)(
                html_path, paper_id, args.output_dir, args.db_path, args.preserve_bold, args.save_mode
            ) for html_path, paper_id in html_files)
    df = pd.DataFrame(results)
    to_parquet(df, 'data/processed/html_parsing_results_v2.parquet')
    print(f"✅ Done. Saved {len(df)} results.")


if __name__ == "__main__":
    main()
