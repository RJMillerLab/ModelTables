"""
Author: Zhengyuan Dong
Created: 2025-04-12
Last Modified: 2025-04-12
Description: This script merges multiple DataFrames from S2ORC data, processes JSON fields, and saves the final DataFrame to a Parquet file.
Usage:
    python -m src.data_preprocess.s2orc_merge --tag 251117
    python -m src.data_preprocess.s2orc_merge --tag 251117 --add-missing  # after s2orc_retry_missing
Updates:
    There are some missing citations/references in the titles2ids parquet file. This script will identify them and print the missing items.
    There are some 429 errors in the API query. This script will identify them and print the missing items.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from collections import Counter, defaultdict
from glob import glob
from src.utils import to_parquet

DATA_FOLDER = "data/processed"
MERGE_KEY = "corpusId"

def print_key_stats(df, key, df_name):
    """
    Print basic statistics of a DataFrame column.

    Parameters:
        df (pd.DataFrame): The DataFrame.
        key (str): The column name.
        df_name (str): A descriptive name for the DataFrame.
    """
    total = len(df)
    unique = df[key].nunique()
    duplicates = total - unique
    print(f"DataFrame '{df_name}': Total rows = {total}, Unique '{key}' = {unique}, Duplicates = {duplicates}")
    if duplicates > 0:
        print("Duplicate key counts:")
        print(df[key].value_counts()[df[key].value_counts() > 1])
    print("-" * 40)

def parse_cit_papers(json_str, id_key = "citingcorpusid"):
    """
    Parse the input JSON string and extract cited paper details by intent.
    
    This function extracts lists of paper IDs and contexts for three intent types:
      - "methodology"
      - "background"
      - "result"
      
    It also produces 'overall' lists that aggregate all cited papers with any intent.
    
    Returns:
      tuple: (
            method_ids, method_contexts,
            background_ids, background_contexts,
            result_ids, result_contexts,
            overall_ids, overall_contexts
      )
    """
    cit_key = "data" # "cited_papers" or "citing_papers"
    method_ids = []
    method_contexts = []
    background_ids = []
    background_contexts = []
    result_ids = []
    result_contexts = []
    overall_ids = []
    none_ids = []
    
    method_infl_ids = []
    method_infl_ctxs = []
    background_infl_ids = []
    background_infl_ctxs = []
    result_infl_ids = []
    result_infl_ctxs = []
    overall_infl_ids = []

    if pd.isna(json_str) or not isinstance(json_str, str):
        return (method_ids, method_contexts,
                background_ids, background_contexts,
                result_ids, result_contexts,
                overall_ids,
                method_infl_ids, method_infl_ctxs,                   
                background_infl_ids, background_infl_ctxs,           
                result_infl_ids, result_infl_ctxs,                   
                overall_infl_ids)
    #try:
    if True:
        data = json.loads(json_str)
        cit_papers = data[cit_key]
        for item in cit_papers:
            paper_id = item[id_key]
            intents_nested = item["intents"]
            contexts = item["contexts"]
            if paper_id is None or not intents_nested:
                none_ids.append(paper_id)
                overall_ids.append(paper_id)
                #print(f"Missing paper_id or intents_nested or contexts: {item}")
                continue
            # Flatten: intents_nested = [['methodology'], ['result']] -> ['methodology', 'result']
            intents_flat = [i for sub in intents_nested for i in (sub if isinstance(sub, list) else [sub])]

            if len(intents_flat) == len(contexts):
                pairs = zip(intents_flat, contexts)
            else:
                # fallback: align all intents with a combined context string
                joined_context = " ".join(contexts)
                pairs = zip(intents_flat, [joined_context] * len(intents_flat))
            influential = item["isinfluential"]
            for intent, ctx in pairs:
                if intent == "methodology":
                    method_ids.append(paper_id)
                    method_contexts.append(ctx)
                    if influential:
                        method_infl_ids.append(paper_id)
                        method_infl_ctxs.append(ctx)
                elif intent == "background":
                    background_ids.append(paper_id)
                    background_contexts.append(ctx)
                    if influential:
                        background_infl_ids.append(paper_id)
                        background_infl_ctxs.append(ctx)
                elif intent == "result":
                    result_ids.append(paper_id)
                    result_contexts.append(ctx)
                    if influential:
                        result_infl_ids.append(paper_id)
                        result_infl_ctxs.append(ctx)
                elif intent in ["None", "none", None]:
                    none_ids.append(paper_id)
                else:
                    raise ValueError(f"Unknown intent: {intent}")
                # All intents contribute to overall
                overall_ids.append(paper_id)
                if influential:
                    overall_infl_ids.append(paper_id)
    # make them unique
    method_ids        = list(dict.fromkeys(method_ids))
    background_ids    = list(dict.fromkeys(background_ids))
    result_ids        = list(dict.fromkeys(result_ids))          
    overall_ids       = list(dict.fromkeys(overall_ids))         
    method_infl_ids   = list(dict.fromkeys(method_infl_ids))     
    background_infl_ids = list(dict.fromkeys(background_infl_ids)) 
    result_infl_ids   = list(dict.fromkeys(result_infl_ids))     
    overall_infl_ids  = list(dict.fromkeys(overall_infl_ids))    
    return (method_ids,
            background_ids, 
            result_ids, 
            overall_ids,
            method_infl_ids, 
            background_infl_ids, 
            result_infl_ids, 
            overall_infl_ids)

def count_intents(final_df, col_name="original_response_references", cit_key="data"):
    """
    Count the occurrences of each intent in the specified JSON column of the DataFrame.
    
    Parameters:
        final_df (pd.DataFrame): DataFrame containing the JSON strings.
        col_name (str): The column name with JSON strings (default is "original_response_references").
    
    Returns:
        Counter: A Counter object with intent counts.
    """
    counter = Counter()
    for json_str in final_df[col_name].dropna():
        try:
            data = json.loads(json_str)
            cit_papers = data.get(cit_key, [])
            for item in cit_papers:
                intents = item["intents"]
                if intents:
                    flat_intents = [i for sub in intents for i in (sub if isinstance(sub, list) else [sub])]
                    counter.update(flat_intents)
                else:
                    counter.update(["None"])
        except Exception as e:
            print(f"Error parsing JSON in count_intents: {e}")
    return counter

def analyze_intent_influential_correlation(json_series, cit_key="data"):
    """
    Analyze the co-occurrence of each intent with the 'isInfluential' flag from a Series of JSON strings.
    
    For each JSON string, this function parses the cited papers and, for each paper with an explicit 'isInfluential'
    value, counts how many times each intent appears when 'isInfluential' is True or False.
    
    Returns:
        dict: A mapping of intent to a dictionary {'True': count, 'False': count}.
    """
    result = defaultdict(lambda: {"True": 0, "False": 0})
    for json_str in json_series.dropna():
        try:
            data = json.loads(json_str)
            cit_papers = data.get(cit_key, [])
            for item in cit_papers:
                influential = item.get("isInfluential", None)
                if influential is None:
                    continue
                influential_key = "True" if influential else "False"
                intents = item.get("intents", [])
                for intent in intents:
                    result[intent][influential_key] += 1
        except Exception as e:
            print(f"Error parsing JSON in analyze_intent_influential_correlation: {e}")
    return dict(result)

def load_and_concat(pattern, data_path):
    ######## Helper function to concatenate files by pattern
    files = list(data_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

def load_parquet_if_exists(path):
    if path.exists():
        return pd.read_parquet(path)
    return None

def load_main_and_429(stem, tag, data_path):
    """
    Load the base file and its 429 variant for a given stem and tag.

    Rules:
      - tag is None: use stem.parquet and stem_429.parquet (if it exists)
      - tag is set:  use stem_{tag}.parquet and stem_{tag}_429.parquet (if it exists)

    Only these two variants for the current tag are considered; files from other
    tags are ignored.
    """
    suffix = f"_{tag}" if tag else ""
    candidates = [
        data_path / f"{stem}{suffix}.parquet",
        data_path / f"{stem}{suffix}_429.parquet",
    ]
    dfs = []
    for p in candidates:
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError(
            f"No files found for stem={stem!r} with tag={tag!r}; "
            f"checked: {', '.join(str(p) for p in candidates)}"
        )
    return pd.concat(dfs, ignore_index=True)

def remerge_multiple_files(data_path, tag, add_missing=False):
    """
    Re-merge caches for a given tag by combining main + 429 (+ optional missing)
    files.

      - tag is None:      use untagged *.parquet and *_429.parquet
      - tag like 251117:  use *_{tag}.parquet and *_{tag}_429.parquet

    This keeps different tags completely separated.
    """
    # 1) citations / references 主 cache（主 + 429）
    citations_cache_main = load_main_and_429("s2orc_citations_cache", tag, data_path)
    references_cache_main = load_main_and_429("s2orc_references_cache", tag, data_path)

    # 2) missing 结果（如果有的话，同样主 + 429）
    if add_missing:
        try:
            citations_missing = load_main_and_429("s2orc_citations_missing", tag, data_path)
        except FileNotFoundError:
            citations_missing = None
        try:
            references_missing = load_main_and_429("s2orc_references_missing", tag, data_path)
        except FileNotFoundError:
            references_missing = None

        if citations_missing is not None:
            citations_cache = pd.concat([citations_cache_main, citations_missing], ignore_index=True)
        else:
            citations_cache = citations_cache_main
        if references_missing is not None:
            references_cache = pd.concat([references_cache_main, references_missing], ignore_index=True)
        else:
            references_cache = references_cache_main
    else:
        citations_cache = citations_cache_main
        references_cache = references_cache_main

    # 3) titles / query_results 也按同样规则：只看当前 tag 的 base + 429
    query_results = load_main_and_429("s2orc_query_results", tag, data_path)
    titles2ids = load_main_and_429("s2orc_titles2ids", tag, data_path)

    from src.data_preprocess.s2orc_API_query import merge_cit_ref
    # Rename cache columns to match merge_all_results output (original_response -> original_response_citations/_references)
    df_cit = citations_cache.rename(columns={"original_response": "original_response_citations"})
    df_ref = references_cache.rename(columns={"original_response": "original_response_references"})
    suffix = f"_{tag}" if tag else ""
    output_path = data_path / f"s2orc_query_results{suffix}_merged.parquet"
    final_merged_df = merge_cit_ref(titles2ids, df_cit, df_ref, str(output_path), MERGE_KEY="paperId")
    return final_merged_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge S2ORC parquet files and parse citations/references")
    parser.add_argument(
        "--tag",
        required=False,
        default=None,
        help="Tag suffix (e.g. 251117). Must match s2orc_API_query --tag; "
             "if omitted, uses untagged base + base_429 files.",
    )
    parser.add_argument("--add-missing", action="store_true", help="Include s2orc_*_missing_{tag}.parquet from retry")
    args = parser.parse_args()
    tag = args.tag
    add_missing = args.add_missing

    data_path = Path(DATA_FOLDER)
    suffix = f"_{tag}" if tag else ""
    output_file = data_path / f"s2orc_rerun{suffix}.parquet"

    # Always use remerge_multiple_files to combine main + 429 (+ missing)
    final_merged_df = remerge_multiple_files(data_path, tag, add_missing=add_missing)
    cit_new_cols = final_merged_df["original_response_citations"].apply(
        lambda x: pd.Series(
            parse_cit_papers(x, id_key="citingcorpusid"),
            index=[
                "cit_papers_methodology_ids", 
                "cit_papers_background_ids",
                "cit_papers_result_ids", 
                "cit_papers_overall_ids",
                "cit_papers_methodology_infl_ids", 
                "cit_papers_background_infl_ids", 
                "cit_papers_result_infl_ids",
                "cit_papers_overall_infl_ids"
            ]
        )
    )
    ref_new_cols = final_merged_df["original_response_references"].apply(
        lambda x: pd.Series(
            parse_cit_papers(x, id_key="citedcorpusid"),
            index=[
                "ref_papers_methodology_ids",
                "ref_papers_background_ids", 
                "ref_papers_result_ids", 
                "ref_papers_overall_ids",
                "ref_papers_methodology_infl_ids",
                "ref_papers_background_infl_ids",
                "ref_papers_result_infl_ids", 
                "ref_papers_overall_infl_ids"
            ]
        )
    )
    final_merged_df = pd.concat([final_merged_df, cit_new_cols, ref_new_cols], axis=1)
    to_parquet(final_merged_df, output_file)
    
    # Compute and print the intents counter statistics
    intents_counter = count_intents(final_merged_df, col_name="original_response_references")
    print("Intent Counter Stats:")
    for intent, count in intents_counter.items():
        print(f"{intent}: {count}")
    
    # Compute and print the co-occurrence statistics of intents and the isInfluential flag
    intent_influential_stats = analyze_intent_influential_correlation(final_merged_df["original_response_references"])
    print("\nIntent and isInfluential Co-occurrence Stats:")
    for intent, stats in intent_influential_stats.items():
        print(f"{intent}: {stats}")
    print('Save merged dataframe to', output_file)
    