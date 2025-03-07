import pandas as pd
import numpy as np
import os, json
from src.utils import load_data, load_config, load_combined_data

data_type = "modelcard"
config = load_config('config.yaml')
processed_base_path = os.path.join(config.get('base_path'), 'processed')

df = load_data(f"{processed_base_path}/{data_type}_step1.parquet")

# Check if each link type column has a non-empty value (avoids empty strings, lists, or NaNs)
df_split_temp['all_links_non_empty'] = df_split_temp['all_links'].apply(lambda x: bool(x) and x != '[]' and x != '')
df_split_temp['pdf_link_non_empty'] = df_split_temp['pdf_link'].apply(lambda x: bool(x) and x != '[]' and x != '')
df_split_temp['github_link_non_empty'] = df_split_temp['github_link'].apply(lambda x: bool(x) and x != '[]' and x != '')

# Count the occurrences and calculate the ratios
pdf_link_count = df_split_temp['pdf_link_non_empty'].sum()
total_count = len(df_split_temp)
pdf_link_ratio = (pdf_link_count / total_count) * 100

# Count rows where 'all_links' is non-empty
all_link_count = df_split_temp['all_links_non_empty'].sum()
all_link_ratio = (all_link_count / total_count) * 100

github_link_count = df_split_temp['github_link_non_empty'].sum()
github_link_ratio = (github_link_count / total_count) * 100

# Count entries with no PDF but with GitHub links
no_pdf_has_github_count = df_split_temp[~df_split_temp['pdf_link_non_empty'] & df_split_temp['github_link_non_empty']].shape[0]
no_pdf_has_github_ratio = (no_pdf_has_github_count / total_count) * 100

# Output the results
print(f"Model cards with all links: {all_link_count}/{total_count} = {all_link_ratio:.2f}%")
print(f"Model cards with GitHub links: {github_link_count}/{total_count} = {github_link_ratio:.2f}%")
print(f"Model cards with PDF links: {pdf_link_count}/{total_count} = {pdf_link_ratio:.2f}%")
print(f"Model cards with NO PDF but HAS GitHub links: "
      f"{no_pdf_has_github_count}/{total_count} = {no_pdf_has_github_ratio:.2f}%")
