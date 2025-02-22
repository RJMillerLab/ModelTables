
import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from data_ingestion.readme_parser import BibTeXExtractor, MarkdownHandler
from data_ingestion.bibtex_parser import BibTeXFactory
import os, re, time, json
from utils import load_data, get_statistics_table, clean_title
tqdm.pandas()

def remove_duplicates(df):
    """Remove duplicates based on markdown table and downloads."""
    print("Removing duplicates...")
    filtered_df = df.sort_values(by="downloads", ascending=False)
    unique_by_markdown = filtered_df.drop_duplicates(subset=["extracted_markdown_table_tuple"], keep="first")
    unique_by_markdown['tupel_str'] = unique_by_markdown.apply(lambda row: str(row['extracted_bibtex_tuple']) + str(row['extracted_markdown_table_tuple']), axis=1)
    final_unique_df = unique_by_markdown.drop_duplicates(subset=["tupel_str"])
    print("Number of unique rows after full filtering:", len(unique_by_markdown), len(final_unique_df))
    return unique_by_markdown, final_unique_df

def filter_entries(df):
    """Filter the DataFrame based on specified conditions."""
    print("Filtering entries based on conditions...")
    filtered_df = df[
        df["extracted_bibtex"].notnull() & 
        (df["extracted_bibtex"].apply(lambda x: len(x) > 0)) & 
        df["extracted_markdown_table_tuple"].notnull() & 
        (df["extracted_markdown_table_tuple"].apply(lambda x: len(x) > 0))
    ]
    print("Number of rows after first filter:", len(filtered_df))
    return filtered_df

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    data_type = 'modelcard'
    # Load data
    start_time = time.time()
    t1 = start_time
    print("⚠️Step 1: Loading data...")
    df_new = load_data(f"{output_dir}/{data_type}_step4.parquet", columns=['modelId', 'extracted_markdown_table_tuple', 'extracted_bibtex_tuple', 'extracted_bibtex', 'csv_path'])
    df_makeup = load_data(f"{output_dir}/{data_type}_step3.parquet", columns=['modelId', 'downloads'])
    df = df_new.merge(df_makeup, on='modelId')
    del df_new, df_makeup
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 2: Filtering entries...")
    start_time = time.time()
    filtered_df = filter_entries(df)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 3: Removing duplicates...")
    start_time = time.time()
    unique_by_markdown, final_unique_df = remove_duplicates(filtered_df)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("⚠️Step 4: Getting statistics table...")
    start_time = time.time()
    benchmark_df = get_statistics_table(unique_by_markdown)
    os.makedirs("statistics", exist_ok=True)
    benchmark_df.to_csv("statistics/benchmark_results.csv", index=False)
    print("✅ done. Time cost: {:.2f} seconds.".format(time.time() - start_time))
    print("Final time cost: {:.2f} seconds.".format(time.time() - t1))

if __name__ == "__main__":
    main()