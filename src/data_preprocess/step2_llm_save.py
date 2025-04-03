"""
Author: Zhengyuan Dong
Created: 2025-03-31
Description: This script save the polished markdown tables to CSV files.
"""
import os, re
import pandas as pd
import numpy as np
import json
from src.data_ingestion.readme_parser import MarkdownHandler


def clean_markdown_block(md_block: str) -> str:
    """
    Remove ```markdown and ``` wrappers from the markdown code block.
    """
    if md_block.startswith("```markdown"):
        md_block = md_block[len("```markdown"):].strip()
    if md_block.endswith("```"):
        md_block = md_block[:-3].strip()
    return md_block

def process_markdown_and_save_paths(df: pd.DataFrame, output_dir: str, key_column: str = "corpusid", skip_if_html_fulltext: bool = True) -> pd.DataFrame:
    """
    For each row in df, extract markdown tables from 'llm_response_raw',
    save them as individual CSV files, and collect their paths.
    Returns updated DataFrame with a new column: 'llm_table_list'.
    """
    os.makedirs(output_dir, exist_ok=True)
    df["llm_table_list"] = [[] for _ in range(len(df))]  ######## initialize empty list

    for idx, row in df.iterrows():
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

        """try:
            table_blocks = json.loads(raw_response)
            if not isinstance(table_blocks, (list, tuple, np.ndarray)):
                continue
        except Exception as e:
            print(f"❌ Failed to parse JSON for row {idx}: {e}")
            continue"""
        
        pattern = re.compile(r"```markdown\s*(.*?)\s*```", re.DOTALL)  ########
        table_blocks = pattern.findall(raw_response)  ########
        if not table_blocks:  ########
            table_blocks = [raw_response.strip()]  ########

        key_value = row[key_column]
        if pd.isna(key_value) or not str(key_value).strip():
            key_value = f"row_{idx}"  ######## fallback ID
        safe_key = str(key_value).strip().replace(" ", "_").replace("/", "_")

        csv_paths = []
        for i, block in enumerate(table_blocks):
            if not isinstance(block, str) or not block.strip():
                continue
            markdown_table = clean_markdown_block(block)
            filename = f"{safe_key}_table{i}.csv"  ########
            out_csv_path = os.path.join(output_dir, filename)
            try:
                MarkdownHandler.markdown_to_csv(markdown_table, out_csv_path)
                csv_paths.append(out_csv_path)  ######## collect path
                print(f"✅ Saved: {out_csv_path}")
            except Exception as e:
                print(f"⚠️ Failed to convert markdown for {safe_key}, table {i}: {e}")
                continue

        df.at[idx, "llm_table_list"] = csv_paths  ######## update the row

    return df

if __name__ == "__main__":
    input_csv = os.path.join("data/processed/llm_markdown_table_results.parquet")
    df_parquet = pd.read_parquet(input_csv)
    print(df_parquet.head(5))
    print(df_parquet.columns)

    output_dir =  "data/processed/llm_tables"
    os.makedirs(output_dir, exist_ok=True)
    # Process tables and write csvs
    df_parquet = process_markdown_and_save_paths(df_parquet, output_dir, key_column="corpusid", skip_if_html_fulltext=True) ########
    # Save updated parquet
    updated_parquet_path = "data/processed/final_integration_with_paths.parquet" ########
    df_parquet.to_parquet(updated_parquet_path, index=False)
    print(f"\n🎉 All markdown tables saved. Paths recorded in 'llm_table_list'.")
    print(f"📝 Updated parquet saved to: {updated_parquet_path}")
