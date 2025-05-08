"""
Author: Zhengyuan Dong
Created: 2025-04-12
Last Modified: 2025-04-12
Description: This script merges multiple DataFrames from S2ORC data, processes JSON fields, and saves the final DataFrame to a Parquet file.
Usage:
    python -m src.data_preprocess.s2orc_merge
Updates:
    There are some missing citations/references in the titles2ids parquet file. This script will identify them and print the missing items.
    There are some 429 errors in the API query. This script will identify them and print the missing items.
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter, defaultdict
from glob import glob

DATA_FOLDER = "data/processed"
MERGE_KEY = "corpusId"
MERGED_RESULTS_FILE = f"{DATA_FOLDER}/s2orc_query_results.parquet"
OUTPUT_FILE = f"{DATA_FOLDER}/s2orc_rerun.parquet"

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

def merge_dataframes(query_results, titles2ids, citations_cache, references_cache):
    """
    Merge four DataFrames with query_results as the primary table (preserving its row count).
    For query_results and titles2ids (which may have duplicate),
    an occurrence counter ('occ') is added to perform a one-to-one merge using (occ).
    The other two DataFrames have unique keys, so suffixes '_citation' and '_reference'
    are added to differentiate their fields.

    Finally, duplicate fields from query_results and the title mapping are dropped.
    
    Parameters:
        query_results (pd.DataFrame): Primary query results table.
        titles2ids (pd.DataFrame): DataFrame mapping titles to IDs.
        citations_cache (pd.DataFrame): Response DataFrame with one set of details.
        references_cache (pd.DataFrame): Response DataFrame with another set of details.
    
    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    print_key_stats(query_results, MERGE_KEY, "query_results")
    print_key_stats(titles2ids, MERGE_KEY, "titles2ids")
    print_key_stats(citations_cache, MERGE_KEY, "citations_cache")
    print_key_stats(references_cache, MERGE_KEY, "references_cache")

    # Merge query_results and titles2ids using an occurrence counter to ensure one-to-one mapping
    query_results = query_results.copy()
    query_results["occ"] = query_results.groupby(MERGE_KEY).cumcount()
    query_results[MERGE_KEY] = query_results[MERGE_KEY].astype(str)
    titles2ids[MERGE_KEY] = titles2ids[MERGE_KEY].astype(str)
    titles2ids["occ"] = titles2ids.groupby(MERGE_KEY).cumcount()
    merged_main = pd.merge(query_results, titles2ids, on=[MERGE_KEY, "occ"], suffixes=("", "_titles"), how="left")
    merged_main = merged_main.drop(columns=["occ"])

    # Merge the two unique-key DataFrames; add distinct suffixes to differentiate their fields
    citations_cache[MERGE_KEY] = citations_cache[MERGE_KEY].astype(str)
    references_cache[MERGE_KEY] = references_cache[MERGE_KEY].astype(str)
    merged_aux = pd.merge(citations_cache, references_cache, on=MERGE_KEY, suffixes=("_citation", "_reference"), how="outer")

    # Merge the auxiliary data into the main table
    final_df = pd.merge(merged_main, merged_aux, on=MERGE_KEY, how="left")

    # Remove redundant fields inherited from the primary query_results and title mapping
    """redundant_cols = [
        "original_response_citations",
        "parsed_response_citations",
        "original_response_references",
        "parsed_response_references"
    ]
    final_df = final_df.drop(columns=redundant_cols, errors="ignore")"""
    final_df = final_df.drop(columns=[col for col in final_df.columns if col.endswith('_titles')])
    
    print("Final merged DataFrame shape:", final_df.shape)
    return final_df

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
    
    if pd.isna(json_str) or not isinstance(json_str, str):
        return (method_ids, method_contexts,
                background_ids, background_contexts,
                result_ids, result_contexts,
                overall_ids)
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

            for intent, ctx in pairs:
                if intent == "methodology":
                    method_ids.append(paper_id)
                    method_contexts.append(ctx)
                elif intent == "background":
                    background_ids.append(paper_id)
                    background_contexts.append(ctx)
                elif intent == "result":
                    result_ids.append(paper_id)
                    result_contexts.append(ctx)
                elif intent in ["None", "none", None]:
                    none_ids.append(paper_id)
                else:
                    raise ValueError(f"Unknown intent: {intent}")
                # All intents contribute to overall
                overall_ids.append(paper_id)
    #except Exception as e:
    #    print(f"Error parsing JSON: {e}")
    return (method_ids, method_contexts,
            background_ids, background_contexts,
            result_ids, result_contexts,
            overall_ids)

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

def remerge_multiple_files(data_path, add_missing=False):
    """
    This function loads multiple files and concatenates them.
    It also loads missing files and concatenates them if add_missing is True.
    It then merges the citations and references caches with the query results and titles mapping.
    """
    # Load original citations and references caches (including _429 versions)
    citations_cache_main = load_and_concat("s2orc_citations_cache*.parquet", data_path)  ######## Concat multiple citations cache files
    references_cache_main = load_and_concat("s2orc_references_cache*.parquet", data_path)  ######## Concat multiple references cache files
    if add_missing:
        # Load missing citations and references caches (including _429 versions)
        citations_missing = load_and_concat("s2orc_citations_missing*.parquet", data_path)  ######## Concat missing citations files
        references_missing = load_and_concat("s2orc_references_missing*.parquet", data_path)  ######## Concat missing references files
        # Combine original and missing caches for final merge
        citations_cache = pd.concat([citations_cache_main, citations_missing], ignore_index=True)  ######## Combined citations
        references_cache = pd.concat([references_cache_main, references_missing], ignore_index=True)  ######## Combined references
    else:
        citations_cache = citations_cache_main
        references_cache = references_cache_main
    query_results = load_and_concat("s2orc_query_results*.parquet", data_path)  ######## Concat multiple query results files
    titles2ids = load_and_concat("s2orc_titles2ids*.parquet", data_path)  ######## Concat multiple titles mapping files

    from src.data_preprocess.s2orc_API_query import merge_cit_ref
    final_merged_df = merge_cit_ref(query_results, titles2ids, citations_cache, references_cache)
    return final_merged_df

if __name__ == "__main__":
    data_path = Path(DATA_FOLDER)  ######## Define data_path using Path
    final_merged_df = pd.read_parquet(MERGED_RESULTS_FILE)
    cit_new_cols = final_merged_df["original_response_citations"].apply(
        lambda x: pd.Series(
            parse_cit_papers(x, id_key="citingcorpusid"),
            index=[
                "cit_papers_methodology_ids", "cit_papers_methodology_contexts",
                "cit_papers_background_ids", "cit_papers_background_contexts",
                "cit_papers_result_ids", "cit_papers_result_contexts",
                "cit_papers_overall_ids"
            ]
        )
    )
    ref_new_cols = final_merged_df["original_response_references"].apply(
        lambda x: pd.Series(
            parse_cit_papers(x, id_key="citedcorpusid"),
            index=[
                "ref_papers_methodology_ids", "ref_papers_methodology_contexts",
                "ref_papers_background_ids", "ref_papers_background_contexts",
                "ref_papers_result_ids", "ref_papers_result_contexts",
                "ref_papers_overall_ids"
            ]
        )
    )
    final_merged_df = pd.concat([final_merged_df, cit_new_cols, ref_new_cols], axis=1)
    final_merged_df.to_parquet(OUTPUT_FILE)
    
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
    print('Save merged dataframe to', OUTPUT_FILE)
    