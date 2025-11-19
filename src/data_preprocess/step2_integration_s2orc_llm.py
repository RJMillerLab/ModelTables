# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Date: 2025-03-30
Last edited: 2025-04-04
Description: Integration code for combining HTML, PDF, and extracted annotations,
             labeling the source for each item (HTML, PDF, or extracted),
             and saving final results.
"""

import os, re, json, tiktoken, argparse
import hashlib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Tuple, List
from src.llm.model import LLM_response
from src.utils import to_parquet, load_config, is_list_like, to_list_safe


# --------------- Fixed Path Constants --------------- #
# These will be updated dynamically in main()
TITLE2ARXIV_JSON = "data/processed/title2arxiv_new_cache.json"
HTML_TABLE_PARQUET = "data/processed/html_table.parquet"
HTML_TABLE_PARQUET_V2 = "data/processed/html_parsing_results_v2.parquet"
ANNOTATIONS_PARQUET = "data/processed/extracted_annotations.parquet"
PDF_CACHE_PATH = "data/processed/pdf_download_cache.json"
FINAL_OUTPUT_CSV = "data/processed/llm_markdown_table_results.parquet"
BATCH_OUTPUT_PATH = "data/processed/batch_output.jsonl"
BATCH_INPUT_PATH = "data/processed/batch_input.jsonl"

MAX_CONTEXT = 16384
TOKEN_BUFFER = 300 # for symbol like ```markdown ```
MODEL_NAME = "gpt-4o-mini"

# Debug: if the below is set is False, we just update FINAL_OUTPUT_CSV file
RUN_LLM = False ##### whether we re-run llm, set to False if we want to skip the LLM call and use previous results
RUN_REBUILD_BATCHINPUT = False ### whether we want to rebuild the batch input file
# ---------------------- Imports ---------------------- #


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
    if isinstance(x, pd.Series):
        return len(x) > 0
    elif is_list_like(x):
        return len(to_list_safe(x)) > 0
    if pd.isna(x):
        return False
    if isinstance(x, str):
        return len(x.strip()) > 0
    return False

def get_extracted_blocks(row):
    """
    Return a list of formatted blocks from extracted tables and figures, each wrapped as a text block.
    """
    blocks = []
    # Process table entries
    table_entries = safe_list(row.get("extracted_tables", []))
    for entry in table_entries:
        text = ""
        if isinstance(entry, dict):
            text = entry.get("extracted_text", "").strip()
            if "id" in entry:
                text = f"Table {entry['id']}:\n{text}"
        elif isinstance(entry, str):
            text = entry.strip()
        if text:
            blocks.append(f"```\n{text}\n```")
    # Process figure entries
    figure_entries = safe_list(row.get("extracted_figures", []))
    for entry in figure_entries:
        text = ""
        if isinstance(entry, dict):
            if "id" in entry and str(entry["id"]).startswith("tab"):
                text = entry.get("extracted_text", "").strip()
                text = f"Figure {entry['id']}:\n{text}"
        elif isinstance(entry, str):
            text = entry.strip()
        if text:
            blocks.append(f"```\n{text}\n```")
    return blocks

def combine_table_and_figure_text(row) -> str:
    """Return the extracted content as a single string by joining formatted blocks."""
    blocks = get_extracted_blocks(row)
    return "\n".join(blocks)

def split_row_text(row, max_tokens=16000, token_buffer=300, model="gpt-4o-mini"):
    """
    Split the row's formatted extracted content (obtained per cell) into chunks without breaking individual blocks.
    """
    blocks = get_extracted_blocks(row)
    if not blocks:
        return []
    block_tokens = [count_tokens(block, model=model) for block in blocks]
    total_tokens = sum(block_tokens)
    n_chunks = max(1, -(-total_tokens // (max_tokens - token_buffer)))
    target_tokens = total_tokens / n_chunks
    chunks = []
    current_chunk = ""
    current_chunk_token_count = 0
    for block, blk_tokens in zip(blocks, block_tokens):
        if blk_tokens > (max_tokens - token_buffer):
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_chunk_token_count = 0
            chunks.append(block)
            continue
        if current_chunk and (current_chunk_token_count + blk_tokens > target_tokens):
            chunks.append(current_chunk)
            current_chunk = block
            current_chunk_token_count = blk_tokens
        else:
            if current_chunk:
                current_chunk += "\n" + block
            else:
                current_chunk = block
            current_chunk_token_count += blk_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def get_truncated_prompts(row, max_tokens=16000, token_buffer=300, model="gpt-4o-mini"):
    """
    Split the row's formatted extracted content into chunks and return a list of truncated prompts.
    Each prompt is created using the prompt_template.
    """
    chunks = split_row_text(row, max_tokens=max_tokens, token_buffer=token_buffer, model=model)
    prompts = [prompt_template.format(chunk) for chunk in chunks]
    return json.dumps(prompts, ensure_ascii=False)

def build_jsonl_lines(row_index, row_data, model_name="gpt-4o-mini", token_buffer=300, max_context=16384):  ########
    """
    Build one or more JSONL entries for a row using the precomputed truncated prompts.
    """
    try:
        prompts_list = json.loads(row_data["llm_prompt_truncated"])  ########
    except Exception:
        prompts_list = []
    if not prompts_list:  ########
        return []  ########
    if len(prompts_list) == 1:  ########
        return [(f"{row_index}", prompts_list[0], row_data["adaptive_max_tokens"])]  ########
    else:  ########
        entries = []  ########
        for idx, prompt_line in enumerate(prompts_list):  ########
            entries.append((f"{row_index}-{idx+1}", prompt_line, row_data["adaptive_max_tokens"]))  ########
        return entries  ########

prompt_template = (
    "The following text may contain multiple tables, including descriptions, metadata captions, and body content. "
    "Some tables may be poorly formatted (e.g., missing delimiters between columns). "
    "Please identify and extract each table, and convert it into a separate Markdown code block. "
    "For each, return only a single string including Markdown code blocks, separated by triple backticks (```markdown). For example:"
    "\"```markdown\\n| Header1 | Header2 |\\n| value1 | value2 |\\n```\\n```markdown\\n...\\n```"
    "Ensure the output reflects the same tabular information as the original, but with clearer structure and improved readability where possible. "
    "Do not include any explanations or extra text.\n\n"
    "Here is the input text:\n{}\nNow, please provide your answer:"
)

# ---------------------- Main Process ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Integrate HTML/PDF/annotation tables and prepare LLM inputs")
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--annotations', dest='annotations', default=None,
                        help='Path to extracted annotations parquet (default: auto-detect from tag)')
    parser.add_argument('--title2arxiv', dest='title2arxiv', default=None,
                        help='Path to title→arxiv cache JSON (default: auto-detect from tag)')
    parser.add_argument('--html-table', dest='html_table', default=None,
                        help='Path to html_table parquet (default: auto-detect from tag)')
    parser.add_argument('--html-table-v2', dest='html_table_v2', default=None,
                        help='Path to html_parsing_results_v2 parquet (default: auto-detect from tag)')
    parser.add_argument('--pdf-cache', dest='pdf_cache', default=None,
                        help='Path to pdf_download_cache JSON (default: auto-detect from tag)')
    parser.add_argument('--output', dest='output', default=None,
                        help='Path to final llm_markdown_table_results parquet (default: auto-detect from tag)')
    parser.add_argument('--batch-input', dest='batch_input', default=None,
                        help='Path to batch input JSONL (default: auto-detect from tag)')
    parser.add_argument('--batch-output', dest='batch_output', default=None,
                        help='Path to batch output JSONL (default: auto-detect from tag)')
    args = parser.parse_args()

    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""

    global TITLE2ARXIV_JSON, HTML_TABLE_PARQUET, HTML_TABLE_PARQUET_V2, ANNOTATIONS_PARQUET
    global PDF_CACHE_PATH, FINAL_OUTPUT_CSV, BATCH_INPUT_PATH, BATCH_OUTPUT_PATH

    TITLE2ARXIV_JSON = args.title2arxiv or os.path.join(processed_base_path, f"title2arxiv_new_cache{suffix}.json")
    HTML_TABLE_PARQUET = args.html_table or os.path.join(processed_base_path, f"html_table{suffix}.parquet")
    HTML_TABLE_PARQUET_V2 = args.html_table_v2 or os.path.join(processed_base_path, f"html_parsing_results_v2{suffix}.parquet")
    ANNOTATIONS_PARQUET = args.annotations or os.path.join(processed_base_path, f"extracted_annotations{suffix}.parquet")
    PDF_CACHE_PATH = args.pdf_cache or os.path.join(processed_base_path, f"pdf_download_cache{suffix}.json")
    FINAL_OUTPUT_CSV = args.output or os.path.join(processed_base_path, f"llm_markdown_table_results{suffix}.parquet")
    BATCH_INPUT_PATH = args.batch_input or os.path.join(processed_base_path, f"batch_input{suffix}.jsonl")
    BATCH_OUTPUT_PATH = args.batch_output or os.path.join(processed_base_path, f"batch_output{suffix}.jsonl")

    print("📁 Paths in use:")
    print(f"   Annotations:        {ANNOTATIONS_PARQUET}")
    print(f"   Title→arxiv cache:  {TITLE2ARXIV_JSON}")
    print(f"   HTML table v2:      {HTML_TABLE_PARQUET_V2}")
    print(f"   HTML table v1:      {HTML_TABLE_PARQUET}")
    print(f"   PDF cache:          {PDF_CACHE_PATH}")
    print(f"   Output parquet:     {FINAL_OUTPUT_CSV}")
    print(f"   Batch input JSONL:  {BATCH_INPUT_PATH}")
    print(f"   Batch output JSONL: {BATCH_OUTPUT_PATH}")

    # --- Step 1: Load extracted annotations ---
    df_anno = pd.read_parquet(ANNOTATIONS_PARQUET, columns=['query', 'retrieved_title', 'paperId', 'corpusid', 'paper_identifier', 'rank', 'score', 'filename', 'line_index', 'title', 'raw_json', 'extracted_openaccessurl', 'extracted_tables', 'extracted_tablerefs', 'extracted_figures', 'extracted_figure_captions', 'extracted_figurerefs'])
    # 'raw_json'

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
    # Try v2 first, fallback to v1
    if os.path.exists(HTML_TABLE_PARQUET_V2):
        print(f"📦 Loading HTML tables from v2: {HTML_TABLE_PARQUET_V2}")
        df_html = pd.read_parquet(HTML_TABLE_PARQUET_V2) # Columns: [paper_id, html_path, page_type, csv_paths]
        # Rename csv_paths to table_list for compatibility
        if 'csv_paths' in df_html.columns and 'table_list' not in df_html.columns:
            # Handle both list and numpy.ndarray types
            def convert_to_list(x):
                if is_list_like(x):
                    return to_list_safe(x)
                elif x is None or pd.isna(x):
                    return []
                else:
                    return []
            df_html['table_list'] = df_html['csv_paths'].apply(convert_to_list)
    else:
        print(f"📦 Loading HTML tables from v1: {HTML_TABLE_PARQUET}")
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
    #print(df_html_merged[df_html_merged['html_paper_id'] == '1508.00305'].iloc[0])
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
    df_final['orig_index'] = df_final.index

    # --- Step 9: Stats ---
    df_html_items = df_final[(df_final["html_html_path"].notna()) & (df_final["html_html_path"] != "") & (df_final["html_page_type"] == "fulltext")]
    print(f"Items with HTML (fulltext): {len(df_html_items)}")

    df_remaining = df_final[~((df_final["html_html_path"].notna()) & (df_final["html_html_path"] != "") & (df_final["html_page_type"] == "fulltext"))]

    df_pdf_items = df_remaining[(df_remaining["pdf_pdf_path"].notna()) & (df_remaining["pdf_pdf_path"] != "")]
    print(f"Remaining items with local PDF path: {len(df_pdf_items)}")

    df_final.loc[:, "combined_text"] = df_final.apply(combine_table_and_figure_text, axis=1) 
    #df_final["llm_prompt"] = df_final["combined_text"].apply(
    #    lambda x: prompt_template.format(x) if isinstance(x, str) and x.strip() else ""
    #)
    df_final["llm_prompt_truncated"] = df_final.apply(lambda row: get_truncated_prompts(row, max_tokens=16384, token_buffer=TOKEN_BUFFER, model=MODEL_NAME) if isinstance(row["combined_text"], str) and row["combined_text"].strip() else "[]", axis=1)
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
    print(f"⚠️ Items with token count > 16000: {(df_final['token_count_combined_text'] > 16000).sum()}")

    # -------------- Parallel querying LLM (example) --------------
    
    if not df_extracted.empty:
        ######## Recompute prompt_token_count
        df_final["adaptive_max_tokens"] = df_final["token_count_combined_text"].apply(lambda x: min(x + TOKEN_BUFFER, MAX_CONTEXT))

        if RUN_REBUILD_BATCHINPUT:
            print("⚙️ Preparing data/processed/batch_input.jsonl ...")
            ######## Build list of (index, prompt, max_tokens)
            #batch_entries = df_final[["llm_prompt_truncated", "adaptive_max_tokens"]].to_dict(orient="index")
            batch_entries = df_final.set_index('orig_index')[["llm_prompt_truncated", "adaptive_max_tokens"]].to_dict(orient="index")
            
            print("⚙️ Building JSONL lines in parallel with splitting...")
            all_entries = Parallel(n_jobs=-1)(
                delayed(build_jsonl_lines)(idx, data, model_name=MODEL_NAME, token_buffer=TOKEN_BUFFER, max_context=MAX_CONTEXT)
                for idx, data in tqdm(batch_entries.items())  ########
            )
            jsonl_lines = []
            for entry_list in all_entries:
                if entry_list:
                    for custom_id, prompt_line, max_tok in entry_list:
                        jsonl_lines.append(json.dumps({
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": MODEL_NAME,
                                "messages": [{"role": "user", "content": prompt_line}],
                                "max_tokens": max_tok
                            }
                        }, ensure_ascii=False))
            with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(jsonl_lines) + "\n")
            print(f"✅ Created {BATCH_INPUT_PATH} with {len(jsonl_lines)} entries (parallelized)")
        else:
            print(f"⚙️ Skipping batch input file generation, using previous results...")

        if RUN_LLM:
            print("⚙️ Running LLM batch query...")
            main_batch_query(BATCH_INPUT_PATH, BATCH_OUTPUT_PATH) # batch query and save
        else:
            print("⚙️ Skipping LLM batch query, using previous results...")
        # 5) Parse the output_file to attach responses back to df_extracted
        print(f"⚙️ Parsing {BATCH_OUTPUT_PATH} and aggregating responses...")
        responses_dict = {}
        with open(BATCH_OUTPUT_PATH, "r", encoding="utf-8") as in_f:
            for line in in_f:
                obj = json.loads(line.strip())
                c_id = obj.get("custom_id", "")
                # Use the part before the '-' as the original row index
                original_index = c_id.split("-")[0]
                resp = obj.get("response", {})
                body = resp.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content = choices[0]["message"].get("content", "")
                else:
                    content = "No content or error."
                responses_dict.setdefault(original_index, []).append((c_id, content))

        # Aggregate responses for each original row (sorting by chunk order if available)
        for original_index, responses in responses_dict.items():
            sorted_responses = sorted(responses, key=lambda x: int(x[0].split("-")[1]) if "-" in x[0] else 0)
            aggregated_response = "\n".join([resp for _, resp in sorted_responses])
            df_extracted.loc[int(original_index), "llm_response_raw"] = aggregated_response
        
        # 6) Merge results back into df_final
        df_final["llm_response_raw"] = df_extracted["llm_response_raw"]
        df_final = df_final.sort_values('orig_index')
        df_final.reset_index(drop=True, inplace=True) 
        df_final.drop(columns=['orig_index'], inplace=True)
        to_parquet(df_final, FINAL_OUTPUT_CSV)
        print(f"✅ LLM results saved to {FINAL_OUTPUT_CSV}")
    else:
        print("No items require LLM batch processing. Skipping batch step.")


if __name__ == "__main__":
    main()
