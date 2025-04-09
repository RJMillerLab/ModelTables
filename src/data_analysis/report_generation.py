"""
Author: Zhengyuan Dong
Created: 2025-04-08
Last Modified: 2025-04-08
Description: Generate a markdown report for table retrieval results, automatically from JSON files.
Usage: 
 python -m src.data_analysis.report_generation
"""

import json
import os
import re
import pandas as pd
from datetime import datetime

BASE_PATH = "/Users/doradong/Repo/CitationLake"

def get_file_path(filename):
    hugging_pattern = r'^[0-9a-f]{10}_table\d+\.csv$'
    tables_output_pattern = r'^\d{4}\.\d{5}.*\.csv$'
    llm_tables_pattern = r'^\d+_table\d+\.csv$'
    github_pattern = r'^[0-9a-f]{32}_table_\d+\.csv$'
    
    if re.match(hugging_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/deduped_hugging_csvs', filename)
    elif re.match(tables_output_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/tables_output', filename)
    elif re.match(llm_tables_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/llm_tables', filename)
    elif re.match(github_pattern, filename):
        return os.path.join(BASE_PATH, 'data/processed/deduped_github_csvs', filename)
    else:
        raise ValueError(f"Unknown: {filename}")

def df_to_markdown(df, max_rows=5):
    if isinstance(df, pd.DataFrame):
        return df.head(max_rows).to_markdown(index=False) + "\n..."
    return df

def generate_md_report(json_path, output_file=None):
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = f"table_report_{timestamp}.md"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    report = []
    for query_file, retrieved_files in data.items():
        report.append(f"# Query Table: `{query_file}`\n")
        
        try:
            query_path = get_file_path(query_file)
            query_df = pd.read_csv(query_path)
            report.append("## Query Table Content\n")
            report.append(df_to_markdown(query_df))
        except Exception as e:
            report.append(f"**Loading failure: {str(e)}**")
        
        report.append("\n## Retrieved Tables\n")
        for idx, file in enumerate(retrieved_files[1:], 1):
            report.append(f"### Top {idx}: `{file}`\n")
            try:
                file_path = get_file_path(file)
                df = pd.read_csv(file_path)
                report.append(df_to_markdown(df))
            except Exception as e:
                report.append(f"*Loading failure: {str(e)}*")
            report.append("\n")
        
        report.append("\n---\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"Report generated: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a markdown report from JSON files.")
    parser.add_argument("--json_path", type=str, default="test_hnsw_search_scilake_large_first10.json", help="Path to the JSON file.")
    args = parser.parse_args()
    generate_md_report(args.json_path)
