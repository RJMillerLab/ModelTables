"""
Author: Zhengyuan Dong
Created: 2025-03-31
Last edited: 2025-04-05
Description: This script save the polished markdown tables to CSV files.
"""
import os, re
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from src.data_ingestion.readme_parser import MarkdownHandler
from src.utils import to_parquet, load_config

# Flag: Virtual mode - generate paths but don't actually create CSV files
VIRTUAL_CSV_GENERATION = True  ######## Set to True to generate llm_table_list paths without creating actual CSV files


def clean_markdown_block(md_block: str):
    """
    Remove ```markdown and ``` wrappers from the markdown code block.
    """
    if md_block.startswith("```markdown"):
        md_block = md_block[len("```markdown"):].strip()
    if md_block.endswith("```"):
        md_block = md_block[:-3].strip()
    return md_block

def process_markdown_and_save_paths(df: pd.DataFrame, output_dir: str, key_column: str = "corpusid", skip_if_html_fulltext: bool = True, virtual_mode: bool = False):
    """
    For each row in df, extract markdown tables from 'llm_response_raw',
    save them as individual CSV files, and collect their paths.
    Returns updated DataFrame with a new column: 'llm_table_list'.
    """
    os.makedirs(output_dir, exist_ok=True)
    df["llm_table_list"] = [[] for _ in range(len(df))]  ######## initialize empty list

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing LLM tables"):
        # --- skip if HTML fulltext is available ---
        if skip_if_html_fulltext:
            html_path = row["html_html_path"]
            html_type = row["html_page_type"]
            html_tables = row["html_table_list"]

            has_valid_html_path = pd.notna(html_path) and str(html_path).strip()
            has_valid_table_list = (
                isinstance(html_tables, (list, tuple, np.ndarray)) and len(html_tables) > 0
            ) or (
                isinstance(html_tables, str) and html_tables.strip() not in ["[]", ""]
            )
            if html_type == "fulltext" and has_valid_html_path and has_valid_table_list:
                continue  ######## skip if HTML fulltext + valid tables exist
            #if pd.notna(html_path) and str(html_path).strip() and html_type == "fulltext":
            #    continue  ######## skip processing if HTML fulltext exists

        raw_response = row['llm_response_raw']
        if pd.isna(raw_response) or not raw_response.strip():
            continue
        # updated processing logic
        table_blocks = []
        try:
            if raw_response.strip().startswith("["):
                parsed_json = json.loads(raw_response)
                if isinstance(parsed_json, list):
                    for item in parsed_json:
                        if isinstance(item, str):
                            table_blocks.extend(re.findall(r"```markdown\s*(.*?)\s*```", item, re.DOTALL))
            else:
                table_blocks = re.findall(r"```markdown\s*(.*?)\s*```", raw_response, re.DOTALL)
        except Exception as e:
            print(f"❌ JSON parse failed at row {idx}: {e}")
            table_blocks = re.findall(r"```markdown\s*(.*?)\s*```", raw_response, re.DOTALL)
        if not table_blocks and raw_response.strip():
            table_blocks = [raw_response.strip()]
        table_blocks = list(dict.fromkeys(table_blocks))  ######## remove duplicates
        ######## End updated markdown block extraction ########

        key_value = row[key_column]
        if pd.isna(key_value) or not str(key_value).strip():
            key_value = f"row_{idx}"
        safe_key = str(key_value).strip().replace(" ", "_").replace("/", "_")

        csv_paths = []
        for i, block in enumerate(table_blocks):
            if not isinstance(block, str) or not block.strip():
                continue
            markdown_table = clean_markdown_block(block)
            filename = f"{safe_key}_table{i}.csv"
            out_csv_path = os.path.join(output_dir, filename)
            
            if virtual_mode:
                # Virtual mode: just generate the path without creating the file
                csv_paths.append(out_csv_path)
            else:
                # Normal mode: actually create the CSV file
                if markdown_table:
                    try:
                        tmp_csv_path = MarkdownHandler.markdown_to_csv(markdown_table, out_csv_path)
                        if tmp_csv_path:
                            csv_paths.append(out_csv_path)
                    except Exception as e:
                        print('----------------------')
                        print(f"⚠️ Failed to convert markdown for {safe_key}, table {i}: {e}")
                        print(markdown_table)
                        continue
                else:
                    continue
        df.at[idx, "llm_table_list"] = csv_paths
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save LLM-processed markdown tables to CSV files")
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--input', dest='input', default=None,
                        help='Path to llm_markdown_table_results parquet (default: auto-detect from tag)')
    parser.add_argument('--output-dir', dest='output_dir', default=None,
                        help='Directory to save CSV files (default: auto-detect from tag)')
    parser.add_argument('--output-parquet', dest='output_parquet', default=None,
                        help='Path to final_integration_with_paths parquet (default: auto-detect from tag)')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""
    
    # Determine input/output paths based on tag
    input_csv = args.input or os.path.join(processed_base_path, f"llm_markdown_table_results{suffix}.parquet")
    output_dir = args.output_dir or os.path.join(processed_base_path, f"llm_tables{suffix}")
    updated_parquet_path = args.output_parquet or os.path.join(processed_base_path, f"final_integration_with_paths_v2{suffix}.parquet")
    
    print("📁 Paths in use:")
    print(f"   Input parquet:      {input_csv}")
    print(f"   Output CSV dir:     {output_dir}")
    print(f"   Output parquet:     {updated_parquet_path}")
    
    df_parquet = pd.read_parquet(input_csv)
    print(df_parquet.head(5))
    print(df_parquet.columns)

    os.makedirs(output_dir, exist_ok=True)
    
    # Process tables and write csvs (or just generate paths in virtual mode)
    df_parquet = process_markdown_and_save_paths(df_parquet, output_dir, key_column="corpusid", skip_if_html_fulltext=False, virtual_mode=VIRTUAL_CSV_GENERATION)
    
    if VIRTUAL_CSV_GENERATION:
        print(f"\n🎉 Virtual mode: Generated llm_table_list paths (no CSV files created).")
    else:
        print(f"\n🎉 All markdown tables saved. Paths recorded in 'llm_table_list'.")
    
    # Save updated parquet (always save, regardless of skip flag)
    to_parquet(df_parquet, updated_parquet_path)
    print(f"\n🎉 All markdown tables saved. Paths recorded in 'llm_table_list'.")
    print(f"📝 Updated parquet saved to: {updated_parquet_path}")
