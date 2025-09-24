# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Date: 2025-04-01
Description: Update the CSV file's llm_response_raw field using the LLM outputs from the log file,
             aligning responses to their corresponding title. Then, save the updated CSV and Parquet files.
"""

import re
import pandas as pd
from src.utils import to_parquet

# -------------- Helper Functions -------------- #
def parse_llm_log(log_file_path: str) -> dict:
    """
    Parse the log file and return a dictionary mapping {title: llm_response_raw}.
    The response is extracted from after the "📝 Response:" marker up to the next block of "=" lines.
    """
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Updated regex pattern: Capture title and response until the next line with 10 or more "=" characters ########
    pattern = re.compile(
        r"==================== LLM RESULT ====================\s+📄 Title\s+:\s+(.*?)\s+🧠 Prompt\s+:.*?📝 Response:\s+(.*?)(?=\n={10,})",
        re.DOTALL
    )  ########
    
    matches = pattern.findall(log_content)
    llm_responses = {}
    for title, response in matches:
        title = title.strip()  ######## Remove whitespace from title ########
        response = response.strip()  ######## Remove whitespace from response ########
        llm_responses[title] = response
    return llm_responses

def update_llm_responses_in_df(df: pd.DataFrame, llm_responses: dict) -> pd.DataFrame:
    """
    Update the DataFrame by adding a new 'llm_response_raw' field using the llm_responses dictionary based on the 'retrieved_title' field.
    Note: Old 'llm_response_raw' column is deleted if it exists.
    """
    if "llm_response_raw" in df.columns:
        df.drop("llm_response_raw", axis=1, inplace=True)  ########

    df["llm_response_raw"] = df["retrieved_title"].apply(lambda title: llm_responses.get(title.strip(), ""))  ########
    return df

# -------------- Main Process -------------- #
def main():
    csv_file_path = "data/processed/llm_markdown_table_results.parquet"  ######## Original CSV file path ########
    log_file_path = "step2_integration_order_3.log"               ######## Log file path ########
    updated_csv_path = "data/processed/llm_markdown_table_results_aligned.parquet"  ######## Updated CSV file path ########

    llm_responses = parse_llm_log(log_file_path)
    print(f"Parsed {len(llm_responses)} LLM responses from log file.")  ########

    df = pd.read_parquet(csv_file_path)
    print(f"Loaded CSV with {len(df)} rows.")  ########

    df_updated = update_llm_responses_in_df(df, llm_responses)
    print("Updated the 'llm_response_raw' column based on log file.")  ########

    to_parquet(df_updated, updated_csv_path)
    print(f"Saved updated CSV to {updated_csv_path}.")  ########

if __name__ == "__main__":
    main()
