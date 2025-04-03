# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-04-02
Last Modified: 2025-04-02
Description:
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ============ Placeholders / Configurations ============
DF_PATH                     = "placeholder_for_citation.parquet"
MERGE_PATH                  = "data/processed/modelcard_step4.parquet" ########
OUTPUT_GROUNDTRUTH          = "data/gt/scilakeUnionBenchmark_by_ids.pickle"

THRESHOLD_VALUE             = 0.6       # threshold for citation overlap count
DISCOUNT_RATE               = 0.5       # when title is not original, apply discount rate
IS_STRICT_MODE              = True # mode placeholder
USE_SYMLINK        = False     # True => use *_table_list_sym; False => use the "original" columns

def discount_logic(is_strict: bool, raw_score: int, discount_rate: float) -> float:
    if is_strict:
        return raw_score
    else:
        return raw_score * discount_rate

def gather_tables_with_discount(row, use_symlink: bool, is_strict: bool, discount_rate: float):
    """
    Gather four lists (hugging/github/html/llm) from a row and assign weights.
    For "hugging" list, weight is always 1.0. For the other three lists,
    if is_strict is False, they are discounted by discount_rate; otherwise, their weight is also 1.0.
    
    If use_symlink is True, it uses the *_table_list_sym columns;
    otherwise, it uses the original columns (hugging_table_list, github_table_list,
    html_table_list_mapped, llm_table_list_mapped).

    Returns:
        dict: { table_path -> accumulated float weight }
    """
    # 1. Determine which columns to use
    if use_symlink:
        hugging_col = "hugging_table_list_sym"
        github_col  = "github_table_list_sym"
        html_col    = "html_table_list_sym"
        llm_col     = "llm_table_list_sym"
    else:
        hugging_col = "hugging_table_list"
        github_col  = "github_table_list"
        html_col    = "html_table_list_mapped"
        llm_col     = "llm_table_list_mapped"

    # 2. Extract the four lists from the row
    hugging_list = row.get(hugging_col, [])
    github_list  = row.get(github_col, [])
    html_list    = row.get(html_col, [])
    llm_list     = row.get(llm_col, [])

    # 3. Set discount value for non-hugging lists
    #    If strict mode is enabled, all lists have a weight of 1.0.
    discount_val = 1.0 if is_strict else discount_rate

    # 4. Combine lists into a single dictionary with accumulated weights
    result = {}
    # For hugging, add weight 1.0 for each path
    for path in hugging_list:
        result[path] = result.get(path, 0.0) + 1.0
    # For github, add discounted weight
    for path in github_list:
        result[path] = result.get(path, 0.0) + discount_val
    # For html, add discounted weight
    for path in html_list:
        result[path] = result.get(path, 0.0) + discount_val
    # For llm, add discounted weight
    for path in llm_list:
        result[path] = result.get(path, 0.0) + discount_val

    return result

def compute_overlap_score(row1, row2, is_strict: bool = True, mode: str = 'both') -> float:
    # Safely convert to sets
    set1_citing   = set(row1.get('citing_id_list', []))
    set1_citation = set(row1.get('citation_id_list', []))
    set2_citing   = set(row2.get('citing_id_list', []))
    set2_citation = set(row2.get('citation_id_list', []))
    if mode == 'citing':
        overlap = len(set1_citing.intersection(set2_citing))  ######## Calculate citing overlap only
        total_length = len(set1_citing) + len(set2_citing)  ######## Total length for citing lists
    else:
        overlap = (len(set1_citing.intersection(set2_citing)) + 
                   len(set1_citation.intersection(set2_citation)))  ######## Calculate combined overlap
        total_length = ((len(set1_citing) + len(set1_citation)) + 
                        (len(set2_citing) + len(set2_citation)))  ######## Total length for both lists
    raw_score = (overlap * 2 / total_length) if total_length > 0 else 0  ######## Compute score in the range 0-1
    final_score = discount_logic(is_strict, raw_score, DISCOUNT_RATE)
    return final_score

def main():
    print(f"Loading modelcard Step4 from: {MERGE_PATH}")
    # merge table list to original modelcard dataset
    df2 = pd.read_parquet(MERGE_PATH)
    print(f"  df2 shape: {df2.shape}")

    print("\nStep 1: Compute pairwise row relationships based on overlap of citing/citation IDs...")
    # We'll build a dictionary that tracks each row's related rows
    row_relationships = defaultdict(list)
    # Naive O(n^2) approach for demonstration
    # For large data, you'd want a more efficient method.
    df = pd.read_parquet(DF_PATH, columns=['query', 'title', 'citing_id_list', 'citation_id_list'])
    n = df.shape[0]
    print("  Building adjacency among df rows (this might be slow for large n)...")
    for i in tqdm(range(n)):
        row_i = df.iloc[i]
        for j in range(i+1, n):
            row_j = df.iloc[j]
            score = compute_overlap_score(row_i, row_j, IS_STRICT_MODE)
            if score >= THRESHOLD_VALUE:
                # Mark them as related
                row_relationships[i].append(j)
                row_relationships[j].append(i)
    print(f"  Found relationships for {len(row_relationships)} out of {n} rows.")
    
    print("\nStep 2: Map each df row to modelids (via df2) using 'title' <-> 'all_title_list'...")
    # (A) explode df2 to quickly build a lookup from title -> set of modelids
    title_to_modelids = defaultdict(set)
    df2_exploded = df2.explode('all_title_list')  # each row is (modelid, single_title)
    df2_exploded = df2_exploded.dropna(subset=['all_title_list'])
    for idx2, row2 in df2_exploded.iterrows():
        t = row2['all_title_list']
        title_to_modelids[t].add(row2['modelid'])
    # (B) for each df row, gather the modelids that correspond to df['title']
    #     (some rows might have a single title, some might have multiple - adjust as needed)
    row_index_to_modelids = defaultdict(set)
    for i in range(n):
        row_i = df.iloc[i]
        t = row_i.get('title', None)
        if t in title_to_modelids:
            row_index_to_modelids[i] = title_to_modelids[t]
        else:
            row_index_to_modelids[i] = set()

    # Now we have row_relationships (which row is related to which row)
    # and row_index_to_modelids (which modelids each row belongs to).

    print("\nStep 3: Construct modelid-level relationships from row-level adjacency...")
    modelid_relationships = defaultdict(set)

    # For each row i, all related rows j => unify their modelids
    # i.e. if row i maps to M1, M2 and row j maps to M3, M4 => all of them are pairwise related
    # (depending on your definition, you might pair M1->M3, M1->M4, M2->M3, M2->M4, etc.)
    for i, related_rows in row_relationships.items():
        i_modelids = row_index_to_modelids[i]
        # for each related row j
        for j in related_rows:
            j_modelids = row_index_to_modelids[j]
            # build pairwise relationships
            for mid1 in i_modelids:
                for mid2 in j_modelids:
                    if mid1 == mid2:
                        continue
                    modelid_relationships[mid1].add(mid2)
                    modelid_relationships[mid2].add(mid1)

    print(f"  We have relationships for {len(modelid_relationships)} modelids in total.")

    print("\nStep 4: Gather union of all 4 table-lists for each related modelid.")
    df2_indexed = df2.set_index('modelId', drop=False)

    # groundtruth structure: { mid: {table_path: weight, ...}, ... }
    groundtruth_tables = {}

    for mid, related_mids in modelid_relationships.items():
        if mid not in df2_indexed.index:
            continue
        
        # collect mid table
        row_mid = df2_indexed.loc[mid]
        combined_dict = gather_tables_with_discount(
            row_mid,
            use_symlink=USE_SYMLINK,
            is_strict=IS_STRICT_MODE,
            discount_rate=DISCOUNT_RATE
        )
        
        # add related mid table
        for other_mid in related_mids:
            if other_mid not in df2_indexed.index:
                continue
            row_other = df2_indexed.loc[other_mid]
            other_dict = gather_tables_with_discount(
                row_other,
                use_symlink=USE_SYMLINK,
                is_strict=IS_STRICT_MODE,
                discount_rate=DISCOUNT_RATE
            )
            # save to combined_dict
            for path, wt in other_dict.items():
                combined_dict[path] = combined_dict.get(path, 0.0) + wt

        # save groundtruth
        groundtruth_tables[mid] = combined_dict

    print(f"  Built groundtruth_tables for {len(groundtruth_tables)} modelIds in total.")

    print("\nStep 5: Save final groundtruth to a pickle file...")
    os.makedirs(os.path.dirname(OUTPUT_GROUNDTRUTH), exist_ok=True)
    with open(OUTPUT_GROUNDTRUTH, "wb") as f:
        pickle.dump(groundtruth_tables, f)

    print(f"Done! Groundtruth saved to {OUTPUT_GROUNDTRUTH}.")


if __name__ == "__main__":
    main()