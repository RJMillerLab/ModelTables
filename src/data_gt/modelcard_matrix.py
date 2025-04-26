"""
Author: Zhengyuan Dong
Created: 2025-04-22
Description: Build a full-model relation adjacency matrix from the model card data,
             with full model inclusion, cleaning of extracted_base_model tags, case-insensitive matching,
             and support for 'directed' and 'related' linking modes. Enhanced to clean symbols
             from extracted names, rematch, extract Hugging Face links, and preserve
             original modelId casing in final mappings. Uses modelcard_step1.parquet.
"""

import re                                                       
import pandas as pd
import numpy as np                                              
import pickle
from scipy.sparse import dok_matrix, csr_matrix, save_npz
import argparse
import os                                                     

MODEL_REL_PATH      = "data/tmp/model_relation_adjacency.npz"
MODEL_REL_INDEXPATH = "data/tmp/model_relation_index.pickle"
DATA_PATH           = "data/processed/modelcard_step1.parquet"  ########
CARD_TAGS_KEY       = "card_tags"
CARD_README_KEY     = 'card_readme'


def build_model_relation_matrix(
    df_rel: pd.DataFrame,
    all_models: set,
    src_col: str = "modelId",
    tgt_col: str = "final_base_model",
    mode: str = "directed"
) -> (csr_matrix, list):
    """
    Build adjacency matrix including all_models.
    mode='directed': only direct src->tgt edges.
    mode='related': fully connect models sharing the same base.
    """
    model_index = sorted(all_models)  ########
    pos = {m: i for i, m in enumerate(model_index)}

    M = len(model_index)
    mat = dok_matrix((M, M), dtype=bool)

    if mode == 'directed':
        for _, row in df_rel.iterrows():
            src = row.get(src_col)
            tgt = row.get(tgt_col)
            if src in pos and tgt in pos:
                mat[pos[src], pos[tgt]] = True
    else:  # related mode
        groups = df_rel.groupby(tgt_col)[src_col].unique()
        for members in groups:
            for m1 in members:
                for m2 in members:
                    if m1 in pos and m2 in pos:
                        mat[pos[m1], pos[m2]] = True

    # self-loops for all nodes
    for i in range(M): mat[i, i] = True

    csr_mat = mat.tocsr()
    total_nonzeros = csr_mat.nnz
    num_edges = total_nonzeros - M
    print(f"Adjacency matrix shape: {csr_mat.shape}")
    print(f"Total edges (excluding self-loops): {num_edges}")

    return csr_mat, model_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['directed','related'], default='directed')
    parser.add_argument('--stats-only', action='store_true', help='Only compute and print stats, then exit')  ########
    args = parser.parse_args()

    # load full dataframe
    df_full = pd.read_parquet(DATA_PATH)
    # record full model set for matrix dimension
    full_models = set(df_full['modelId'])  ########

    df = df_full.copy()

    # initial extraction of base_model tag, with cleaning of quotes/backticks
    df['extracted_base_model'] = df[CARD_TAGS_KEY].str.extract(
        r'base_model:\s*([^\s]+)', flags=re.IGNORECASE
    ).replace({np.nan: None})
    df['extracted_base_model'] = df['extracted_base_model'].str.replace(r"[\"'`]", "", regex=True)  ########
    print(f"Unique extracted_base_model before filtering: {df['extracted_base_model'].nunique()}")

    # build mapping counts and identify invalid
    mapping_counts = df.groupby(['modelId','extracted_base_model']).size().reset_index(name='count')
    valid_models = full_models
    valid_map = {m.lower(): m for m in valid_models}  ########

    mapping_invalid = mapping_counts[~mapping_counts['extracted_base_model']\
        .fillna('')\
        .str.lower()\
        .isin(valid_map)]
    num_invalid = mapping_invalid['extracted_base_model'].nunique()
    print(f"Unique invalid extracted_base_model: {num_invalid}")

    # correction via repository merge
    models_df = df[['modelId','downloads']].drop_duplicates()
    models_df['repo'] = models_df['modelId'].str.split('/', n=1).str[1]

    invalid_df = pd.DataFrame(mapping_invalid['extracted_base_model'].unique(),
                              columns=['extracted_base_model'])
    merged = invalid_df.merge(models_df,
                              left_on='extracted_base_model',
                              right_on='repo', how='left')
    merged_sorted = merged.sort_values(['extracted_base_model','downloads'],
                                      ascending=[True, False])
    merged_valid = merged_sorted[merged_sorted['modelId'].notna()]
    best = merged_valid.drop_duplicates(subset='extracted_base_model', keep='first')
    correction_map = dict(zip(best['extracted_base_model'], best['modelId']))
    num_corrected = len(correction_map)
    print(f"Corrected invalid extracted_base_model: {num_corrected}")

    # clean & case-insensitive rematch
    unmatched = set(invalid_df['extracted_base_model']) - set(correction_map)
    cleaned = {u: re.sub(r"[\"'`]", "", u) for u in unmatched}
    for raw, clean in cleaned.items():
        lc = clean.lower()
        if lc in valid_map:
            correction_map[raw] = valid_map[lc]  ########
    num_rematched = len(correction_map) - num_corrected
    print(f"Rematched after cleaning symbols: {num_rematched}")

    final_unmatched = list(unmatched - set(correction_map))
    print("Final unmatched extracted_base_model:")
    print(final_unmatched)

    # extract HF links if README exists and print stats
    if CARD_README_KEY in df.columns:
        hf_match = df[CARD_README_KEY].str.extract(
            r'https?://huggingface\.co/([^/\s]+)/([^/\s]+)'
        )
        df[['hf_user','hf_repo']] = hf_match
        df['hf_modelid'] = df['hf_user'].str.lower() + '/' + df['hf_repo'].str.lower()  ########
        df['hf_modelid'] = df['hf_modelid'].map(valid_map).fillna(df['hf_modelid'])  ########
        num_hf_links = df['hf_modelid'].notna().sum()
        matched_hf = df['hf_modelid'].isin(valid_models).sum()
        unmatched_hf = num_hf_links - matched_hf
        print(f"Found HF model links: {num_hf_links}, matched: {matched_hf}, unmatched: {unmatched_hf}")
        ds_match = df[CARD_README_KEY].str.extract(
            r'https?://huggingface\.co/datasets/([^/\s]+)/([^/\s]+)'
        )
        df[['hf_ds_org','hf_ds_name']] = ds_match
        df['hf_datasetid'] = df['hf_ds_org'] + '/' + df['hf_ds_name']
        num_ds_links = df['hf_datasetid'].notna().sum()
        num_unique_ds = df['hf_datasetid'].nunique(dropna=True)  ########
        print(f"Found HF dataset links in {num_ds_links} cards")
        print(f"Unique HF dataset IDs: {num_unique_ds}")
    else:
        print("Column CARD_README_KEY not found: skipping HF link extraction.")
    
    if args.stats_only:
        print("Stats-only mode: exiting before building adjacency matrix.")
        exit(0)

    # apply corrections and map back to original casing
    def map_final(x):  ########
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return x
        val = correction_map.get(x, x)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return val
        return valid_map.get(val.lower(), val)  ########

    df['final_base_model'] = df['extracted_base_model'].map(map_final)  ########
    # prepare df_rel for adjacency building
    df_rel = df[df['final_base_model'].isin(valid_models)].copy()  ########

    # build and save adjacency including all models
    adj, index = build_model_relation_matrix(df_rel, full_models, mode=args.mode)  ########

    degrees = np.array(adj.sum(axis=1)).flatten()
    print(f"Unlinked modelIds (only self-loop): {int(np.sum(degrees==1))}")

    os.makedirs(os.path.dirname(MODEL_REL_PATH), exist_ok=True)
    save_npz(MODEL_REL_PATH, adj)
    with open(MODEL_REL_INDEXPATH,'wb') as f:
        pickle.dump(index, f)

    print(f"✔️  Saved adjacency ({args.mode}) to {MODEL_REL_PATH}")
    print(f"✔️  Saved index to {MODEL_REL_INDEXPATH}")