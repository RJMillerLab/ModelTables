"""
Author: Zhengyuan Dong
Created: 2025-03-31
Description: This script save the polished markdown tables to CSV files.
"""
import os
import pandas as pd
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

def process_markdown_and_save_paths(df: pd.DataFrame, output_dir: str, key_column: str = "arxiv_id") -> pd.DataFrame:
    """
    For each row in df, extract markdown tables from 'llm_response_raw',
    save them as individual CSV files, and collect their paths.
    Returns updated DataFrame with a new column: 'saved_csv_paths'.
    """
    os.makedirs(output_dir, exist_ok=True)
    df["saved_csv_paths"] = [[] for _ in range(len(df))]  ######## initialize empty list

    for idx, row in df.iterrows():
        raw_response = row.get("llm_response_raw", "")
        if pd.isna(raw_response) or not raw_response.strip():
            continue

        try:
            table_blocks = json.loads(raw_response)
            if not isinstance(table_blocks, list):
                continue
        except Exception as e:
            print(f"❌ Failed to parse JSON for row {idx}: {e}")
            continue

        key_value = row.get(key_column)
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

        df.at[idx, "saved_csv_paths"] = csv_paths  ######## update the row

    return df

if __name__ == "__main__":
    LLM_OUTPUT_FOLDER = "llm_outputs"
    input_csv = os.path.join(LLM_OUTPUT_FOLDER, "llm_markdown_table_results.csv")
    input_parquet = os.path.join(LLM_OUTPUT_FOLDER, "final_integration.parquet") ######## original parquet
    output_dir =  "llm_tables"
    os.makedirs(output_dir, exist_ok=True)
    # Load original dataframe (parquet) and updated csv result
    df_parquet = pd.read_parquet(input_parquet)
    df_llm = pd.read_csv(input_csv)
    # Merge CSV back into original df by index
    df_parquet.update(df_llm) ######## keep updated LLM responses
    # Process tables and write csvs
    df_parquet = process_markdown_and_save_paths(df_parquet, output_dir, key_column="arxiv_id") ########
    # Save updated parquet
    updated_parquet_path = os.path.join(LLM_OUTPUT_FOLDER, "final_integration_with_paths.parquet") ########
    df_parquet.to_parquet(updated_parquet_path, index=False)
    print(f"\n🎉 All markdown tables saved. Paths recorded in 'saved_csv_paths'.")
    print(f"📝 Updated parquet saved to: {updated_parquet_path}")
