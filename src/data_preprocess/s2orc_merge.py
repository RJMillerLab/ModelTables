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
    For query_results and titles2ids (which may have duplicate 'paperId' values),
    an occurrence counter ('occ') is added to perform a one-to-one merge using (paperId, occ).
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
    print_key_stats(query_results, "paperId", "query_results")
    print_key_stats(titles2ids, "paperId", "titles2ids")
    print_key_stats(citations_cache, "paperId", "citations_cache")
    print_key_stats(references_cache, "paperId", "references_cache")

    # Merge query_results and titles2ids using an occurrence counter to ensure one-to-one mapping
    query_results = query_results.copy()
    query_results["occ"] = query_results.groupby("paperId").cumcount()
    titles2ids = titles2ids.copy()
    titles2ids["occ"] = titles2ids.groupby("paperId").cumcount()
    merged_main = pd.merge(query_results, titles2ids, on=["paperId", "occ"],
                           suffixes=("", "_titles"), how="left")
    merged_main = merged_main.drop(columns=["occ"])

    # Merge the two unique-key DataFrames; add distinct suffixes to differentiate their fields
    merged_aux = pd.merge(citations_cache, references_cache, on="paperId",
                          suffixes=("_citation", "_reference"), how="outer")

    # Merge the auxiliary data into the main table (using paperId only)
    final_df = pd.merge(merged_main, merged_aux, on="paperId", how="left")

    # Remove redundant fields inherited from the primary query_results and title mapping
    redundant_cols = [
        "original_response_citations",
        "parsed_response_citations",
        "original_response_references",
        "parsed_response_references"
    ]
    final_df = final_df.drop(columns=redundant_cols, errors="ignore")
    final_df = final_df.drop(columns=[col for col in final_df.columns if col.endswith('_titles')])
    
    print("Final merged DataFrame shape:", final_df.shape)
    return final_df

def parse_cited_papers(json_str):
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
    method_ids = []
    method_contexts = []
    background_ids = []
    background_contexts = []
    result_ids = []
    result_contexts = []
    overall_ids = []
    overall_contexts = []
    
    if pd.isna(json_str) or not isinstance(json_str, str):
        return (method_ids, method_contexts, 
                background_ids, background_contexts, 
                result_ids, result_contexts, 
                overall_ids, overall_contexts)
    try:
        data = json.loads(json_str)
        cited_papers = data.get("cited_papers", [])
        for item in cited_papers:
            intents = item.get("intents", [])
            contexts = item.get("contexts", [])
            citedPaper = item.get("citedPaper", {})
            paperId = citedPaper.get("paperId", None)
            if paperId is None:
                continue
            if "methodology" in intents:
                method_ids.append(paperId)
                method_contexts.append(contexts)
            if "background" in intents:
                background_ids.append(paperId)
                background_contexts.append(contexts)
            if "result" in intents:
                result_ids.append(paperId)
                result_contexts.append(contexts)
            if intents:
                overall_ids.append(paperId)
                overall_contexts.append(contexts)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    return (method_ids, method_contexts, 
            background_ids, background_contexts, 
            result_ids, result_contexts, 
            overall_ids, overall_contexts)

def count_intents(final_df, col_name="parsed_response_reference"):
    """
    Count the occurrences of each intent in the specified JSON column of the DataFrame.
    
    Parameters:
        final_df (pd.DataFrame): DataFrame containing the JSON strings.
        col_name (str): The column name with JSON strings (default is "parsed_response_reference").
    
    Returns:
        Counter: A Counter object with intent counts.
    """
    counter = Counter()
    for json_str in final_df[col_name].dropna():
        try:
            data = json.loads(json_str)
            cited_papers = data.get("cited_papers", [])
            for item in cited_papers:
                intents = item.get("intents", [])
                counter.update(intents)
        except Exception as e:
            print(f"Error parsing JSON in count_intents: {e}")
    return counter

def analyze_intent_influential_correlation(json_series):
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
            cited_papers = data.get("cited_papers", [])
            for item in cited_papers:
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

# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    data_path = Path(DATA_FOLDER)  ######## Define data_path using Path
    prefix = ""  ######## Define prefix as needed (here empty; modify if needed)
    
    def load_and_concat(pattern):
        ######## Helper function to concatenate files by pattern
        files = list(data_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    
    # Load original citations and references caches (including _429 versions)
    citations_cache_main = load_and_concat("s2orc_citations_cache*.parquet")  ######## Concat multiple citations cache files
    references_cache_main = load_and_concat("s2orc_references_cache*.parquet")  ######## Concat multiple references cache files
    # Load missing citations and references caches (including _429 versions)
    citations_missing = load_and_concat("s2orc_citations_missing*.parquet")  ######## Concat missing citations files
    references_missing = load_and_concat("s2orc_references_missing*.parquet")  ######## Concat missing references files
    # Combine original and missing caches for final merge
    citations_cache = pd.concat([citations_cache_main, citations_missing], ignore_index=True)  ######## Combined citations
    references_cache = pd.concat([references_cache_main, references_missing], ignore_index=True)  ######## Combined references
    query_results = load_and_concat("s2orc_query_results*.parquet")  ######## Concat multiple query results files
    titles2ids = load_and_concat("s2orc_titles2ids*.parquet")  ######## Concat multiple titles mapping files

    final_merged_df = merge_dataframes(query_results, titles2ids, citations_cache, references_cache)
    new_cols = final_merged_df["parsed_response_reference"].apply(
        lambda x: pd.Series(
            parse_cited_papers(x),
            index=[
                "cited_papers_methodology_ids", "cited_papers_methodology_contexts",
                "cited_papers_background_ids", "cited_papers_background_contexts",
                "cited_papers_result_ids", "cited_papers_result_contexts",
                "cited_papers_overall_ids", "cited_papers_overall_contexts"
            ]
        )
    )
    final_merged_df = pd.concat([final_merged_df, new_cols], axis=1)
    final_merged_df.to_parquet(OUTPUT_FILE)
    
    # Compute and print the intents counter statistics
    intents_counter = count_intents(final_merged_df)
    print("Intent Counter Stats:")
    for intent, count in intents_counter.items():
        print(f"{intent}: {count}")
    
    # Compute and print the co-occurrence statistics of intents and the isInfluential flag
    intent_influential_stats = analyze_intent_influential_correlation(final_merged_df["parsed_response_reference"])
    print("\nIntent and isInfluential Co-occurrence Stats:")
    for intent, stats in intent_influential_stats.items():
        print(f"{intent}: {stats}")
    print('Save merged dataframe to', OUTPUT_FILE)
    
    ######## New Block: Identify missing citations/references based on titles2ids parquet ########
    # Retrieve paper IDs from citations cache (English: get paper IDs from citations_cache)
    citations_ids = set(citations_cache["paperId"].unique())  ######## English: Retrieve paper IDs from citations cache
    # Retrieve paper IDs from references cache (English: get paper IDs from references_cache)
    references_ids = set(references_cache["paperId"].unique())  ######## English: Retrieve paper IDs from references cache
    
    # Filter titles2ids DataFrame for rows whose paperId is missing in either citations or references caches
    missing_df = titles2ids[~(titles2ids["paperId"].isin(citations_ids)) | ~(titles2ids["paperId"].isin(references_ids))]  ######## English: Find missing paper IDs
    
    if not missing_df.empty:
        print("\nMissing Items that require re-query (based on titles2ids parquet):")
        for _, row in missing_df.iterrows():
            missing_citation = row["paperId"] not in citations_ids  ######## English: Flag missing citation
            missing_reference = row["paperId"] not in references_ids  ######## English: Flag missing reference
            print("PaperId:", row.get("paperId"))
            print("Query:", row.get("query", "N/A"))  ######## English: Print query if available
            print("Retrieved Title:", row.get("retrieved_title", "N/A"))  ######## English: Print retrieved title if available
            print("Missing Citation:", missing_citation)
            print("Missing Reference:", missing_reference)
            print("-" * 40)
    else:
        print("\nNo missing items found in citation or reference caches based on titles2ids parquet.")
