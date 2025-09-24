#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-02-24
Description: Extract tables from .tex files and save to CSV and Parquet files.
"""

import os, re, tarfile
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from src.utils import load_config, to_parquet

def process_tex_content(tex_content):
    paragraphs = re.split(r'\n\s*\n', tex_content)
    table_data = []
    for match in re.finditer(r'\\begin{table}(.*?)\\end{table}', tex_content, re.DOTALL):
        table = match.group(1)
        caption = (re.search(r'\\caption{(.*?)}', table, re.DOTALL) or [None, "No caption"])[1]
        tabular = (re.search(r'\\begin{tabular}(.*?)\\end{tabular}', table, re.DOTALL) or [None, "No table"])[1]
        table_label = (re.search(r'\\label{([^}]+)}', table) or [None, None])[1]
        prev_context = next_context = None
        for i, paragraph in enumerate(paragraphs):
            par_start = tex_content.find(paragraph)
            par_end = par_start + len(paragraph)
            if par_end < match.start(): 
                prev_context = paragraph.strip()
            if par_start > match.end() and next_context is None:
                next_context = paragraph.strip()
                break
        context = []
        if prev_context: context.append(prev_context)
        if next_context: context.append(next_context)
        table_data.append({
            "caption": caption.strip(),
            "table": tabular.strip(),
            "label": table_label,
            "context": context
        })
    return table_data

def get_tex_content(local_path):
    if local_path.endswith('.tar.gz'):
        if not tarfile.is_tarfile(local_path):
            print(f"File {local_path} is not a valid tar.gz archive.")
            return None
        try:
            with tarfile.open(local_path, 'r:gz') as tar:
                members = [m for m in tar.getmembers() if m.name.endswith('.tex')]
                if members:
                    with tar.extractfile(members[0]) as f:
                        return f.read().decode("utf-8")
        except Exception as e:
            print(f"Error processing tar file {local_path}: {e}")
        return None
    elif os.path.isfile(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {local_path}: {e}")
    return None

def process_tex_row(row, output_folder="tex_extracted_tables"):
    os.makedirs(output_folder, exist_ok=True)
    local_path = row.get('local_path')
    tex_content = get_tex_content(local_path) if local_path else None
    if tex_content:
        tables = process_tex_content(tex_content)
        if tables:
            csv_filename = os.path.splitext(os.path.basename(local_path))[0] + ".csv"
            output_csv_path = os.path.join(output_folder, csv_filename)
            try:
                pd.DataFrame(tables).to_csv(output_csv_path, index=False, encoding="utf-8")
            except Exception as e:
                print(f"Error saving CSV for {local_path}: {e}")
        return tables
    return None

def parallel_process_tex_entries(df):
    results = Parallel(n_jobs=-1)(
        delayed(process_tex_row)(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing .tex files")
    )
    df = df.copy()
    df['extracted_tex_table'] = results
    return df

def main():
    config = load_config('config.yaml')
    base_path = config.get('base_path')
    tex_csv_path = os.path.join(base_path, "downloaded_tex_info.parquet")
    print("Loading CSV files...")
    df_tex = pd.read_parquet(tex_csv_path)
    df_tex = df_tex[df_tex['local_path'].notnull()].copy()
    print("Extracting tables from .tex files in parallel...")
    df_tex = parallel_process_tex_entries(df_tex)
    to_parquet(df_tex, os.path.join(base_path, "step_tex_table.parquet"))
    print("Final data saved as Parquet:", output_parquet)

if __name__ == "__main__":
    main()