# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Date: 2025-03-30
Description: Integration code for combining HTML, PDF, and extracted annotations,
             labeling the source for each item (HTML, PDF, or extracted),
             and saving final results.
             (Note: LLM call is not performed in this version.)
"""

import os, re
import json
import hashlib
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import asyncio                                           ########
from typing import Tuple, List                            ########
import numpy as np
from src.llm.model import LLM_response             ########

# --------------- Fixed Path Constants --------------- ########
TITLE2ARXIV_JSON = "title2arxiv_new_cache.json"       ######## # Mapping: title -> arxiv_id
HTML_TABLE_PARQUET = "html_table.parquet"             ######## # Contains: paper_id, html_path, page_type, table_list
ANNOTATIONS_PARQUET = "extracted_annotations.parquet" ######## # base
PDF_CACHE_PATH = "pdf_download_cache.json"            ######## #
LLM_OUTPUT_FOLDER = "llm_outputs"                     ######## # Folder for output CSV files (even if LLM is not called)
FINAL_PARQUET = "final_integration.parquet"           ########

# ---------------------- Helper Functions ---------------------- #

def normalize_title(title):
    """
    Normalize the title by converting to lower-case and reducing whitespace.
    """
    return " ".join(title.lower().split())

def preprocess_title(title):
    title = re.sub(r"[-:_*@&'\"]+", " ", title)
    return " ".join(title.split())

def load_json_cache(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json_cache(cache, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def is_valid_pdf(pdf_path):
    """Check PDF validity by reading header."""
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except:
        return False

def safe_list(value):
    """Convert value to a Python list if possible."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return value.tolist() if hasattr(value, "tolist") else list(value)
    except Exception:
        return []

def safe_scalar(value):
    """Convert value to a scalar (if it's a list or Series, take the first element)."""
    if isinstance(value, (list, pd.Series)):
        if isinstance(value, list):
            return value[0] if value else None
        else:
            return value.iloc[0] if not value.empty else None
    return value

def query_llm(tables_data, raw_text=""):
    """
    Simulate calling an LLM to convert table data to Markdown format.
    (Not used in this version.)
    """
    prompt = f"Please convert the following table data to Markdown:\n{raw_text}\n"
    markdown = "| Column1 | Column2 |\n| --- | --- |\n"
    return markdown

def non_empty(x):
    if isinstance(x, (list, pd.Series, np.ndarray)):
        return len(x) > 0
    if pd.isna(x):
        return False
    if isinstance(x, str):
        return len(x.strip()) > 0
    return False

async def async_LLM_response(prompt: str, GPT_model: str, history: list = [], kwargs: dict = {}) -> Tuple[str, list]:
    """
    Asynchronously sends a prompt to a language model and awaits the response.

    Parameters
    ----------
    prompt : str
        The prompt to send to the language model.
    GPT_model : str
        The specific GPT model version to use.
    history : list, optional
        Previous interactions with the language model, if applicable.
    kwargs : dict, optional
        Additional keyword arguments for the language model interaction.

    Returns
    -------
    Tuple[str, list]
        A tuple containing the model's response and the interaction history.
    """
    model_version = "gpt-4o-mini"# if GPT_model == 'gpt4'# else "gpt-3.5-turbo-0125"
    loop = asyncio.get_event_loop()
    print('-----------------'*10)
    print(prompt)
    response, history = await loop.run_in_executor(None, LLM_response, prompt, model_version, history, kwargs)
    print(response)
    #print('-----------------'*10)
    return response, history

def parse_json_response(response: str):
    """
    Very simple parser for demonstration.
    Adjust to your real parsing logic.
    """
    try:
        return json.loads(response)
    except:
        return []

########
# We define the concurrency semaphore for controlling parallel requests:
SEMAPHORE = asyncio.Semaphore(5)  ########

########
prompt_template = (
    "Your task is to convert the extracted table text from s2orc "
    "into a well-organized Markdown table that is easy to recognize. "  ######## Updated wording for clarity
    "Return only a list containing the Markdown table, with no extra content. "  ######## Refined output instructions
    "Here is the input table: {}\nNow, please provide your answer:"  ######## Streamlined prompt for input
)

async def process_table_with_llm(row) -> str:
    # preprocessing
    table_entries = row.get("extracted_tables", [])
    table_texts = []
    if isinstance(table_entries, (list, tuple, np.ndarray)):
        for entry in table_entries:
            if isinstance(entry, dict):
                text = entry.get("extracted_text", "")
                if isinstance(text, str) and text.strip():
                    table_texts.append(text.strip())
            elif isinstance(entry, str):
                if entry.strip():
                    table_texts.append(entry.strip())
    else:
        text = str(table_entries)
        if text.strip():
            table_texts.append(text.strip())
    # Extract figure texts from extracted_figures with id starting with 'tab'
    figure_entries = row.get("extracted_figures", [])
    figure_texts = []
    if isinstance(figure_entries, (list, tuple, np.ndarray)):
        for entry in figure_entries:
            if isinstance(entry, dict):
                fig_id = entry.get("id", "")
                if isinstance(fig_id, str) and fig_id.startswith("tab"):
                    text = entry.get("extracted_text", "")
                    if isinstance(text, str) and text.strip():
                        figure_texts.append(text.strip())
    # Combine texts from tables and figures
    combined_texts = table_texts + figure_texts
    if not combined_texts:
        return ""
    combined_text = "\n".join(combined_texts)

    prompt = prompt_template.format(combined_text)

    async with SEMAPHORE:
        response, _ = await async_LLM_response(prompt, GPT_model="gpt4")
    parsed = parse_json_response(response)
    # Suppose the LLM returns a JSON array containing the markdown table as a string
    if isinstance(parsed, (list, tuple, np.ndarray)) and len(parsed) > 0:
        return parsed[0]  # e.g. the first item is our table
    else:
        return ""

########
async def run_parallel_llm(df: pd.DataFrame) -> pd.DataFrame:    ########
    """
    Execute the LLM calls in parallel for each row in df.
    Returns a copy of df with an additional column 'llm_markdown_table'.
    """
    rows = df.to_dict("records")
    tasks = [process_table_with_llm(row) for row in rows]
    results = await asyncio.gather(*tasks)
    # Attach to DataFrame
    new_df = df.copy()
    new_df["llm_markdown_table"] = results
    return new_df

# ---------------------- Main Process ---------------------- #

def main():
    # --- Step 4: Load extracted annotations --- 
    df_anno = pd.read_parquet(ANNOTATIONS_PARQUET)
    df_anno["norm_title"] = df_anno["retrieved_title"].apply(normalize_title) ########
    df_anno["preproc_title"] = df_anno["retrieved_title"].apply(preprocess_title) ########
    # Expected columns include: retrieved_title, extracted_openaccessurl, extracted_tables, extracted_figures, etc.
    print("📝 df_anno shape:", df_anno.shape)

    # --- Step 1: Load title2arxiv mapping (title -> arxiv_id) --- 
    title2arxiv_map = load_json_cache(TITLE2ARXIV_JSON)
    # Example: { "Some paper title": "2301.12345v2", ... }
    df_title2arxiv = pd.DataFrame(
        [(t, a) for t, a in title2arxiv_map.items()],
        columns=["retrieved_title", "arxiv_id"]
    )
    df_title2arxiv["norm_title"] = df_title2arxiv["retrieved_title"].apply(normalize_title) ########
    df_title2arxiv["preproc_title"] = df_title2arxiv["retrieved_title"].apply(preprocess_title) ########
    print("📝 df_title2arxiv shape:", df_title2arxiv.shape)

    # --- Step 2: Load html_table.parquet which contains HTML info including table_list --- 
    df_html = pd.read_parquet(HTML_TABLE_PARQUET)
    # Columns: [paper_id, html_path, page_type, table_list]
    print("📝 df_html shape:", df_html.shape)

    # --- Step 3: Merge title mapping with HTML table info on arxiv_id/paper_id --- 
    df_html_merged = pd.merge(
        df_title2arxiv, df_html,
        left_on="arxiv_id", right_on="paper_id",
        how="left"
    )
    df_html_merged.rename(columns={
        "html_path": "html_html_path", 
        "page_type": "html_page_type", 
        "table_list": "html_table_list", 
        "paper_id": "html_paper_id" 
    }, inplace=True)
    print("📝 df_html_merged shape:", df_html_merged.shape)
    
    # --- Step 5: Merge annotations with the title-HTML mapping on retrieved_title --- 
    # 先用原始 retrieved_title 对齐
    df_merged = pd.merge(
        df_anno,
        df_html_merged,
        on="retrieved_title",
        how="left",
        suffixes=("", "_temp")
    )

    # 对于 arxiv_id 为空的记录，尝试用 norm_title 对齐
    mask_missing = df_merged["arxiv_id"].isna() ########
    if mask_missing.any():
        df_missing = df_merged[mask_missing].copy() ########
        df_missing2 = pd.merge(
            df_missing.drop(columns=["arxiv_id"]),
            df_title2arxiv[["norm_title", "arxiv_id"]],
            left_on="norm_title",
            right_on="norm_title",
            how="left"
        ) ########
        df_merged.loc[mask_missing, "arxiv_id"] = df_missing2["arxiv_id"].values ########

    # 对于仍然 arxiv_id 为空的记录，尝试用 preproc_title 对齐
    mask_missing = df_merged["arxiv_id"].isna() ########
    if mask_missing.any():
        df_missing = df_merged[mask_missing].copy() ########
        df_missing2 = pd.merge(
            df_missing.drop(columns=["arxiv_id"]),
            df_title2arxiv[["preproc_title", "arxiv_id"]],
            left_on="preproc_title",
            right_on="preproc_title",
            how="left"
        ) ########
        df_merged.loc[mask_missing, "arxiv_id"] = df_missing2["arxiv_id"].values ########

    # 删除辅助列，保持数据原有字段不变
    df_merged.drop(columns=["norm_title", "preproc_title"], inplace=True) ########
    print("📝 After merging title mapping, shape:", df_merged.shape)
    #print("📝 After merging HTML info, shape:", df_merged.shape) 
    
    # --- Step 6: Load PDF cache (already downloaded PDFs) --- ########
    pdf_cache = load_json_cache(PDF_CACHE_PATH)
    print("🔎 PDF cache loaded with", len(pdf_cache), "entries.")
    # Filter out invalid PDFs from cache
    for url, path in pdf_cache.items():
        if path and os.path.isfile(path):
            if not is_valid_pdf(path):
                pdf_cache[url] = None
    df_pdf = pd.DataFrame([(url, path) for url, path in pdf_cache.items()], columns=["openaccessurl", "pdf_pdf_path"])
    #df_pdf = pd.read_parquet(PDF_CACHE_PARQUET)
    print("📝 df_pdf shape:", df_pdf.shape)

    # --- Step 7: Merge PDF info into extraction based on extracted_openaccessurl --- 
    df_final = pd.merge(
        df_merged,
        df_pdf,
        left_on="extracted_openaccessurl",
        right_on="openaccessurl",
        how="left"
    )
    # Optionally, drop the redundant 'openaccessurl' column from df_pdf 
    df_final.drop(columns=["openaccessurl"], inplace=True)
    print("📝 After merging PDF info, final shape:", df_final.shape)

    # --- Step 8: Save final integration results ---
    df_final.to_parquet(FINAL_PARQUET, index=False)
    print(f"\n🎉 Integration done. Output saved to {FINAL_PARQUET}, total {len(df_final)} rows.\n")

    # --- Step 9: 打印统计信息 ---
    # 统计具有 HTML 信息且 html_page_type 为 fulltext 的记录
    df_html_items = df_final[(df_final["html_html_path"].notna()) & (df_final["html_html_path"] != "") & (df_final["html_page_type"] == "fulltext")]
    print(f"Items with HTML (fulltext): {len(df_html_items)}")

    # 剩下的记录
    df_remaining = df_final[~((df_final["html_html_path"].notna()) & (df_final["html_html_path"] != "") & (df_final["html_page_type"] == "fulltext"))]

    # 统计剩余记录中有本地 PDF 路径的记录
    df_pdf_items = df_remaining[(df_remaining["pdf_pdf_path"].notna()) & (df_remaining["pdf_pdf_path"] != "")]
    print(f"Remaining items with local PDF path: {len(df_pdf_items)}")

    # 统计剩余记录中 extracted_tables 或 extracted_figures 不为空的记录
    df_extracted = df_remaining[(df_remaining["extracted_tables"].apply(non_empty)) | (df_remaining["extracted_figures"].apply(non_empty))]
    print(f"Remaining items with non-empty extracted tables or figures: {len(df_extracted)}")

    def count_tokens(text):
        return len(text.split())

    # count token for the extracted figures
    total_tokens_pdf = 0
    for idx, row in df_pdf_items.iterrows():
        content = ""
        if non_empty(row.get("extracted_tables")):
            if isinstance(row.get("extracted_tables"), list):
                content += " ".join(row.get("extracted_tables"))
            else: 
                content += str(row.get("extracted_tables"))
        if non_empty(row.get("extracted_figures")):
            if isinstance(row.get("extracted_figures"), list):
                content += " ".join(row.get("extracted_figures"))
            else: 
                content += str(row.get("extracted_figures"))
        if content:
            llm_result = content
            tokens = count_tokens(llm_result)
            total_tokens_pdf += tokens
    print(f"Total tokens from LLM queries for items with local PDF path: {total_tokens_pdf}")

    # -------------- Parallel querying LLM (example) --------------

    if not df_extracted.empty:                                                    
        print("⚙️ Running parallel LLM queries for extracted tables...")         
        # Convert df_extracted to a separate DataFrame so we don't re-query everything
        df_extracted = df_extracted.copy()                                        

        # For demonstration, let's just do it for those that have 'extracted_tables'.
        # Then store the results in a new DataFrame with an added column "llm_markdown_table".
        loop = asyncio.get_event_loop()                                          
        df_extracted_llm = loop.run_until_complete(run_parallel_llm(df_extracted))

        # You can now merge or join df_extracted_llm back into df_final on index, etc.
        df_final.loc[df_extracted_llm.index, "llm_markdown_table"] = df_extracted_llm["llm_markdown_table"]
        
        # Optionally, save or examine the updated df_final.
        # For example:
        output_path = os.path.join(LLM_OUTPUT_FOLDER, "llm_markdown_table_results.csv") 
        os.makedirs(LLM_OUTPUT_FOLDER, exist_ok=True)                                    
        df_final.to_csv(output_path, index=False)                                        
        print(f"✅ LLM results saved to {output_path}")                                   


if __name__ == "__main__":
    main()
