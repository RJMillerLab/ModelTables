#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import fitz
from tqdm import tqdm
from joblib import Parallel, delayed

from data_ingestion.readme_parser import BibTeXExtractor, MarkdownHandler
from data_ingestion.bibtex_parser import BibTeXFactory
from data_preprocess.step2 import save_markdown_to_csv, extract_bibtex, add_extracted_tuples
from data_preprocess.step1 import extract_markdown

def extract_table_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return None

    table_data = []
    for page in doc:
        text = page.get_text("text")
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            row = re.split(r'\s{2,}', line)
            if len(row) > 1:
                table_data.append(row)
    doc.close()
    return table_data if table_data else None

def process_pdf_row(row):
    local_path = row.get('local_path')
    if pd.notnull(local_path) and os.path.isfile(local_path):
        table = extract_table_from_pdf(local_path)
        if table:
            output_folder = "pdf_extracted_csv"
            os.makedirs(output_folder, exist_ok=True)
            csv_filename = os.path.splitext(os.path.basename(local_path))[0] + ".csv"
            output_csv_path = os.path.join(output_folder, csv_filename)
            try:
                pd.DataFrame(table).to_csv(output_csv_path, index=False, header=False, encoding="utf8")
            except Exception as e:
                print(f"Error saving CSV for {local_path}: {e}")
        return table
    else:
        return None

def load_github_readme(row):
    readme_path = row.get('readme_path')
    if pd.notnull(readme_path) and os.path.isfile(readme_path):
        try:
            with open(readme_path, "r", encoding="utf8") as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Error reading {readme_path}: {e}")
            return ""
    else:
        return ""

def parallel_process_pdf_entries(df):
    results = Parallel(n_jobs=-1)(
        delayed(process_pdf_row)(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing PDFs")
    )
    df = df.copy()
    df['extracted_pdf_table'] = results
    return df

def parallel_load_github_readme(df):
    results = Parallel(n_jobs=-1)(
        delayed(load_github_readme)(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing READMEs")
    )
    df = df.copy()
    df['github_readme'] = results
    return df

def main():
    readme_csv_path = "../github_readmes_info.csv"
    print("Loading CSV files...")
    df_readme = pd.read_csv(readme_csv_path)
    df_readme = df_readme[df_readme['readme_path'].notnull()].copy()
    print("Extracting content from README files in parallel...")
    df_readme = parallel_load_github_readme(df_readme)
    df_readme["github_readme"] = df_readme["github_readme"].apply(lambda x: str(x) if pd.notnull(x) else "")
    print("Extracting markdown tables from README files...")
    #extract_bibtex(df_readme, readme_key="github_readme", new_key="github_extracted_markdown_table")
    #df_readme["github_extracted_markdown_table_tuple"] = df_readme["github_extracted_markdown_table"].apply(lambda x: tuple(x) if isinstance(x, list) else (x,))
    results = extract_markdown(df_readme, col_name='github_readme')
    df_readme[['github_contains_markdown_table', 'github_extracted_markdown_table']] = pd.DataFrame(results, index=df_readme.index)
    print("Saving markdown extraction results to CSV files...")
    save_markdown_to_csv(df_readme, key="github_extracted_markdown_table", output_folder="github_markdown_csvs", new_key="github_csv_path")
    output_parquet = "data/step_add_github.parquet"
    df_readme.to_parquet(output_parquet, index=False)
    print("Final data saved as Parquet file:", output_parquet)

if __name__ == "__main__":
    main()
