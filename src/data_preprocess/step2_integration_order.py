# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Date: 2025-03-30
Last edited: 2025-04-01
Description: Integration code for combining HTML, PDF, and extracted annotations,
             labeling the source for each item (HTML, PDF, or extracted),
             and saving final results.
"""

import os, re, json, tiktoken
import hashlib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Tuple, List
from src.llm.model import LLM_response


# --------------- Fixed Path Constants --------------- #
TITLE2ARXIV_JSON = "title2arxiv_new_cache.json"       ######## # Mapping: title -> arxiv_id
HTML_TABLE_PARQUET = "html_table.parquet"             ######## # Contains: paper_id, html_path, page_type, table_list
ANNOTATIONS_PARQUET = "extracted_annotations.parquet" ######## # base
PDF_CACHE_PATH = "pdf_download_cache.json"            ######## #
#LLM_OUTPUT_FOLDER = "llm_outputs"                     ######## # Folder for output CSV files (even if LLM is not called)
#FINAL_PARQUET = "final_integration.parquet"           ########
FINAL_OUTPUT_CSV = "llm_markdown_table_results.csv"
BATCH_OUTPUT_PATH = "data/processed/batch_output.jsonl"


from src.llm.batch import main_batch_query
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

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text, disallowed_special=()))

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

def non_empty(x):
    if isinstance(x, (list, pd.Series, np.ndarray)):
        return len(x) > 0
    if pd.isna(x):
        return False
    if isinstance(x, str):
        return len(x.strip()) > 0
    return False

def combine_table_and_figure_text(row) -> str:
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

    combined_texts = table_texts + figure_texts
    if not combined_texts:
        return ""
    return "\n".join(combined_texts)

prompt_template = (
    "The following text may contain multiple tables, including descriptions, metadata captions, and body content. "
    "Some tables may be poorly formatted (e.g., missing delimiters between columns). "
    "Please identify and extract each table, and convert it into a separate Markdown code block. "
    "For each, return only a JSON array of Markdown code blocks, formatted like: "
    "For each, return only a single string including Markdown code blocks, separated by triple backticks (```markdown). For example:"
    "\"```markdown\\n| Header1 | Header2 |\\n| --- | --- |\\n| value1 | value2 |\\n```\"... "
    "Ensure the output reflects the same information as the original, but with clearer structure and improved readability where possible. "
    "Do not include any explanations or extra text.\n\n"
    "Here is the input text:\n{}\nNow, please provide your answer:"
)

# ---------------------- Main Process ---------------------- #

def main():
    # --- Step 1: Load extracted annotations ---
    df_anno = pd.read_parquet(ANNOTATIONS_PARQUET)
    df_anno["norm_title"] = df_anno["retrieved_title"].apply(normalize_title) ########
    df_anno["preproc_title"] = df_anno["retrieved_title"].apply(preprocess_title) ########
    # Expected columns include: retrieved_title, extracted_openaccessurl, extracted_tables, extracted_figures, etc.
    print("📝 df_anno shape:", df_anno.shape)

    # --- Step 2: Load title2arxiv mapping (title -> arxiv_id) ---
    title2arxiv_map = load_json_cache(TITLE2ARXIV_JSON) # Example: { "Some paper title": "2301.12345v2", ... }
    df_title2arxiv = pd.DataFrame(
        [(t, a) for t, a in title2arxiv_map.items()],
        columns=["retrieved_title", "arxiv_id"]
    )
    df_title2arxiv["norm_title"] = df_title2arxiv["retrieved_title"].apply(normalize_title) ########
    df_title2arxiv["preproc_title"] = df_title2arxiv["retrieved_title"].apply(preprocess_title) ########
    print("📝 df_title2arxiv shape:", df_title2arxiv.shape)

    # --- Step 3: Load html_table.parquet which contains HTML info including table_list ---
    df_html = pd.read_parquet(HTML_TABLE_PARQUET) # Columns: [paper_id, html_path, page_type, table_list]
    print("📝 df_html shape:", df_html.shape)

    # 2502.12345v1 => (2502.12345, 1)
    def parse_arxiv_id(paper_id):
        match = re.match(r"(\d{4}\.\d{5})(v(\d+))?", paper_id)
        if match:
            arxiv_id_pure = match.group(1)
            arxiv_id_version = int(match.group(3)) if match.group(3) else 1
            return pd.Series([arxiv_id_pure, arxiv_id_version])
        else:
            return pd.Series([paper_id, 1])  # fallback
    df_html[['arxiv_id_pure', 'arxiv_id_version']] = df_html['paper_id'].apply(parse_arxiv_id)
    # keep the latest version
    df_html = df_html.sort_values('arxiv_id_version', ascending=False).drop_duplicates('arxiv_id_pure', keep='first')
    # --- Step 4: Merge title mapping with HTML table info on arxiv_id/paper_id ---
    def parse_arxiv_id_simple(arxiv_id):
        match = re.match(r"(\d{4}\.\d{5})(v(\d+))?", str(arxiv_id))
        if match:
            return match.group(1)
        else:
            return arxiv_id
    df_title2arxiv['arxiv_id_pure'] = df_title2arxiv['arxiv_id'].apply(parse_arxiv_id_simple)
    df_html_merged = pd.merge(df_title2arxiv, df_html,left_on="arxiv_id_pure", right_on="arxiv_id_pure",how="left")
    #df_html_merged = pd.merge(df_title2arxiv, df_html,left_on="arxiv_id", right_on="paper_id",how="left")
    
    df_html_merged.rename(columns={
        "html_path": "html_html_path", 
        "page_type": "html_page_type", 
        "table_list": "html_table_list", 
        "paper_id": "html_paper_id"
    }, inplace=True)
    print(df_html_merged[df_html_merged['html_paper_id'] == '1508.00305'].iloc[0])
    print("📝 df_html_merged shape:", df_html_merged.shape)
    # --- Step 5: Merge annotations with the title-HTML mapping on retrieved_title ---
    df_merged = pd.merge(
        df_anno,
        df_html_merged,
        on="retrieved_title",
        how="left",
        suffixes=("", "_temp")
    )

    mask_missing = df_merged["arxiv_id"].isna()
    if mask_missing.any():
        df_missing = df_merged[mask_missing].copy()
        df_missing2 = pd.merge(
            df_missing.drop(columns=["arxiv_id"]),
            df_title2arxiv[["norm_title", "arxiv_id"]],
            left_on="norm_title",
            right_on="norm_title",
            how="left"
        )
        df_merged.loc[mask_missing, "arxiv_id"] = df_missing2["arxiv_id"].values

    mask_missing = df_merged["arxiv_id"].isna()
    if mask_missing.any():
        df_missing = df_merged[mask_missing].copy()
        df_missing2 = pd.merge(
            df_missing.drop(columns=["arxiv_id"]),
            df_title2arxiv[["preproc_title", "arxiv_id"]],
            left_on="preproc_title",
            right_on="preproc_title",
            how="left"
        )
        df_merged.loc[mask_missing, "arxiv_id"] = df_missing2["arxiv_id"].values

    df_merged.drop(columns=["norm_title", "preproc_title"], inplace=True)
    print("📝 After merging title mapping, shape:", df_merged.shape)
    #print("📝 After merging HTML info, shape:", df_merged.shape) 
    
    # --- Step 6: Load PDF cache (already downloaded PDFs) ---
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
    #df_final.to_parquet(FINAL_PARQUET, index=False)
    #print(f"\n🎉 Integration done. Output saved to {FINAL_PARQUET}, total {len(df_final)} rows.\n")

    # --- Step 9: Stats ---
    df_html_items = df_final[(df_final["html_html_path"].notna()) & (df_final["html_html_path"] != "") & (df_final["html_page_type"] == "fulltext")]
    print(f"Items with HTML (fulltext): {len(df_html_items)}")

    df_remaining = df_final[~((df_final["html_html_path"].notna()) & (df_final["html_html_path"] != "") & (df_final["html_page_type"] == "fulltext"))]

    df_pdf_items = df_remaining[(df_remaining["pdf_pdf_path"].notna()) & (df_remaining["pdf_pdf_path"] != "")]
    print(f"Remaining items with local PDF path: {len(df_pdf_items)}")

    df_final.loc[:, "combined_text"] = df_final.apply(combine_table_and_figure_text, axis=1) 
    df_final["llm_prompt"] = df_final["combined_text"].apply(
        lambda x: prompt_template.format(x) if isinstance(x, str) and x.strip() else ""
    )
    df_extracted = df_final[df_final["combined_text"].str.strip().astype(bool)]

    # Keep only items with non-empty extracted tables or figures
    has_html = (df_final["html_html_path"].notna()) & (df_final["html_html_path"] != "") & (df_final["html_page_type"] == "fulltext")
    has_pdf = (df_final["pdf_pdf_path"].notna()) & (df_final["pdf_pdf_path"] != "")
    df_remaining = df_final[~has_html & ~has_pdf]
    # Then filter the remaining items with non-empty extracted tables or figures
    df_remaining_tmp = df_remaining[df_remaining["combined_text"].str.strip().astype(bool)]
    print(f"Remaining items (no HTML or PDF) with non-empty extracted tables or figures: {len(df_remaining)}")  
    print(f"Remaining items with non-empty extracted tables or figures: {len(df_remaining_tmp)}")

    # count token for the extracted figures
    df_final['token_count_combined_text'] = df_final['combined_text'].apply(count_tokens)
    print(f"Total tokens from LLM queries for items with local PDF path: {df_final['token_count_combined_text'].sum()}")
    print(f"Average tokens per item: {df_final['token_count_combined_text'].mean()}")
    print(f"Max tokens in a single item: {df_final['token_count_combined_text'].max()}")
    #print(f"Min tokens in a single item: {df_final['token_count_combined_text'].min()}")
    print(f"Token count for prompt template: {count_tokens(prompt_template)}")
    print(f"⚠️ Items with token count > 16000: {(df_final['token_count_combined_text'] > 16000).sum()}")  ########

    # -------------- Parallel querying LLM (example) --------------
    
    df_extracted.to_parquet("before_llm_output.parquet", index=False)
    print("combined text finished, next step is to run LLM")

    if not df_extracted.empty:
        print("⚙️ Preparing batch_input.jsonl ...")
        input_path = "data/processed/batch_input.jsonl"
        max_context = 16384
        token_buffer = 300
        model_name = "gpt-4o-mini"
        ######## Recompute prompt_token_count
        df_final["adaptive_max_tokens"] = df_final["token_count_combined_text"].apply(lambda x: min(x + token_buffer, max_context))  ########
        ######## Build list of (index, prompt, max_tokens)
        batch_entries = df_final[["llm_prompt", "adaptive_max_tokens"]].to_dict(orient="index")
        def build_jsonl_line(row_index, row_data):
            prompt = row_data["llm_prompt"]
            max_tok = row_data["adaptive_max_tokens"]
            ########## ! or max_tok = max_context directly
            if not isinstance(prompt, str) or not prompt.strip():
                return None
            return json.dumps({
                "custom_id": str(row_index),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tok
                }
            }, ensure_ascii=False)
        
        print("⚙️ Building JSONL lines in parallel...")
        jsonl_lines = Parallel(n_jobs=-1)(
        delayed(build_jsonl_line)(idx, data) for idx, data in tqdm(batch_entries.items())
        )
        jsonl_lines = [line for line in jsonl_lines if line]
        with open(input_path, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines) + "\n")
        print(f"✅ Created {input_path} with {len(jsonl_lines)} entries (parallelized)")  ########

        run_llm = True #####
        if run_llm:
            main_batch_query(input_path, BATCH_OUTPUT_PATH) # batch query and save
        # 5) Parse the output_file to attach responses back to df_extracted
        print(f"⚙️ Parsing {BATCH_OUTPUT_PATH} ...")
        with open(BATCH_OUTPUT_PATH, "r", encoding="utf-8") as in_f:
            for line in in_f:
                obj = json.loads(line.strip())
                c_id = obj.get("custom_id", "")
                # e.g. "req-123" => index=123
                #if not c_id.startswith("req-"):
                #    continue
                # I have removed the req-, directly use row index as batch index
                row_index = int(c_id.replace("req-", ""))
                # Attempt to get the raw content
                resp = obj.get("response", {})
                body = resp.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content = choices[0]["message"].get("content", "")
                else:
                    content = "No content or error."
                df_extracted.loc[row_index, "llm_response_raw"] = content
        
        # 6) Merge results back into df_final
        df_final["llm_response_raw"] = df_extracted["llm_response_raw"]
        
        # 7) Save final CSV with the new LLM responses
        #os.makedirs(LLM_OUTPUT_FOLDER, exist_ok=True)
        df_final.to_csv(FINAL_OUTPUT_CSV, index=False)
        print(f"✅ LLM results saved to {FINAL_OUTPUT_CSV}")
    else:
        print("No items require LLM batch processing. Skipping batch step.")


if __name__ == "__main__":
    main()
