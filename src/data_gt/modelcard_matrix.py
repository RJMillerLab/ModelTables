"""
Author: Zhengyuan Dong
Created: 2025-04-22
Last Edited: 2025-04-28
Description: Build a full-model relation adjacency matrix from the model card data,
             with full model inclusion, cleaning of extracted_base_model tags, case-insensitive matching,
             and support for 'directed' and 'related' linking modes. Enhanced to clean symbols
             from extracted names, rematch, extract Hugging Face links, and preserve
             original modelId casing in final mappings. Also saves HF model/dataset link lists
             (all and unmatched) to TXT and Parquet for inspection. Uses modelcard_step1.parquet.
Usage: 
    python -m src.data_gt.modelcard_matrix
"""

import re, os
import requests                                    
import pandas as pd
import numpy as np                                              
import pickle
from scipy.sparse import dok_matrix, csr_matrix, save_npz
import argparse
from tqdm import tqdm                            
from src.utils import load_combined_data                      

MODEL_REL_PATH      = "data/tmp/model_relation_adjacency.npz"
MODEL_REL_INDEXPATH = "data/tmp/model_relation_index.pickle"
DATA_PATH           = "data/processed/modelcard_step1.parquet"  
CARD_TAGS_KEY       = "card_tags"
CARD_README_KEY     = 'card_readme'

# ---------- fast regex helpers (no PyYAML) ---------- 
_DS_INLINE_RE = re.compile(r'^datasets?\s*:\s*\[?([^\[\]\n]+)', re.I)
_TAG_INLINE_RE = re.compile(r'^tags?\s*:\s*\[?([^\[\]\n]+)', re.I)

def _split_csv(txt: str):                                       
    return [t.strip().lower() for t in re.split(r'[,\s]+', txt) if t]

def extract_datasets_tags(card_text: str) -> (list, list):
    """Very fast single-pass scan of the YAML header using regex only."""
    if not isinstance(card_text, str):
        return [], []
    ds, tg = [], []
    for ln in card_text.splitlines():
        ln = ln.strip()
        m_ds = _DS_INLINE_RE.match(ln)
        if m_ds:
            ds = _split_csv(m_ds.group(1))
            continue
        m_tg = _TAG_INLINE_RE.match(ln)
        if m_tg:
            tg = _split_csv(m_tg.group(1))
            continue
    # prune special keys you already handle elsewhere
    tg = [t for t in tg if not t.startswith(('arxiv:', 'base_model'))]
    return ds, tg
# ----------------------------------------------------


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
    model_index = sorted(all_models)  
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
    valid_ds_df       = load_combined_data(                          
                        data_type="datasetcard",
                        file_path=os.path.expanduser("~/Repo/CitationLake/data/raw/"))
    valid_dataset_ids = set(valid_ds_df["datasetId"].str.lower())
    print(f"Loaded {len(valid_dataset_ids):,} valid dataset IDs")

    df_full = pd.read_parquet(DATA_PATH)
    full_models = set(df_full['modelId'])
    df = df_full.copy()
    # initial extraction of base_model tag, with cleaning of quotes/backticks
    df['extracted_base_model'] = df[CARD_TAGS_KEY].str.extract(
        r'base_model:\s*([^\s]+)', flags=re.IGNORECASE
    ).replace({np.nan: None})
    df['extracted_base_model'] = df['extracted_base_model'].str.replace(r"[\"'`]", "", regex=True)
    df['extracted_base_model'] = df['extracted_base_model'].str.replace(r"[\[\]\(\)\{\}]", "", regex=True) # remove some  illegal chars
    df['extracted_base_model'] = df['extracted_base_model'].str.replace('https://huggingface.co/', '')
    
    print(f"Unique extracted_base_model before filtering: {df['extracted_base_model'].nunique()}")
    # ---------- re-direct the link  ----------
    def normalize_extracted(link: str):                       
        if not isinstance(link, str) or link.lower() in ['none', 'nan']:   
            return None                                                    
        # 1) remove https://huggingface.co/
        link = re.sub(r'^https?://huggingface\.co/', '', link, flags=re.I) 
        # use user/repo format
        if '/' in link:
            return link.lower()                                            
        # 2) try HTML canonical
        try:                                                               
            html = requests.get(f"https://huggingface.co/{link}",
                                headers={'User-Agent': 'Mozilla/5.0'},
                                timeout=4).text                            
            m = re.search(r'<link[^>]+rel="canonical"[^>]+href="https://huggingface\.co/([^/]+)/([^"/]+)"',
                        html, flags=re.I)                                
            if m:
                return f"{m.group(1)}/{m.group(2)}".lower()                
        except requests.RequestException:
            pass                                                           
        return link.lower()                                                

    """
    # deprecated, because too slow
    df['extracted_base_model'] = (                                         
        df[CARD_TAGS_KEY]
        .str.extract(r'base_model:\s*([^\s]+)', flags=re.IGNORECASE)
        .iloc[:, 0]                                                      
        .apply(normalize_extracted)                                      
    )
    print(f"Unique extracted_base_model before filtering: {df['extracted_base_model'].nunique()}")  
    """ 
    # build mapping counts and identify invalid
    mapping_counts = df.groupby(['modelId','extracted_base_model']).size().reset_index(name='count')
    valid_models = full_models
    valid_map = {m.lower(): m for m in valid_models}

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
            correction_map[raw] = valid_map[lc]
    num_rematched = len(correction_map) - num_corrected
    print(f"Rematched after cleaning symbols: {num_rematched}")

    final_unmatched = list(unmatched - set(correction_map))
    print("Final unmatched extracted_base_model:", len(final_unmatched))
    print(final_unmatched)

    salvage_count = 0     
    for raw in tqdm(final_unmatched):
        if not isinstance(raw, str):
            continue                                                    
        norm = re.sub(r'^https?://huggingface\.co/', '', raw, flags=re.I).lower() 
        if norm in models_df['repo'].values:
            best_id = models_df.loc[models_df['repo'] == norm]          
            best_id = best_id.sort_values('downloads', ascending=False)['modelId'].iloc[0] 
            correction_map[raw] = best_id
            salvage_count += 1
        elif norm in valid_map:
            correction_map[raw] = valid_map[norm]
            salvage_count += 1
    if salvage_count:
        print(f"Salvaged {salvage_count} unmatched base_model entries") 
    # ---------- add the final_base_model column ----------
    df['final_base_model'] = df['extracted_base_model'].map(correction_map).fillna(df['extracted_base_model'])


    # extract HF links if README exists and print stats
    if CARD_README_KEY in df.columns:
        """df.loc[mask_only, 'hf_modelid'] = df.loc[mask_only, 'hf_repo'].map(
            lambda repo: models_df[models_df['repo']==repo]
                            .sort_values('downloads', ascending=False)
                            ['modelId'].iloc[0]
        )"""
        hf_match = df[CARD_README_KEY].str.extract(
            r'https?://huggingface\.co/([^/\s]+)/([^/\s]+)'    
        )
        df[['hf_user','hf_repo']] = hf_match
        df['hf_modelid'] = (df['hf_user'].str.lower() + '/' + df['hf_repo'].str.lower())
        df['hf_modelid'] = df['hf_modelid'].map(valid_map).fillna(df['hf_modelid'])

        """
        hf_match = df[CARD_README_KEY].str.extract(
            r'https?://huggingface\.co/(?:datasets/)?' +
            r'(?:(?P<hf_user>[^/\s]+)/(?P<hf_repo>[^/\s]+)' +
            r'|(?P<hf_repo_only>[^/\s]+))'
        )
        print('extracted!')
        df['hf_user'] = hf_match['hf_user']
        df['hf_repo'] = hf_match['hf_repo'].fillna(hf_match['hf_repo_only'])
        mask_only = hf_match['hf_repo_only'].notna()
        def resolve_repo_only(repo):
            subset = models_df[models_df['repo'] == repo]
            if not subset.empty:
                return subset.sort_values('downloads', ascending=False)['modelId'].iloc[0]  
            else:
                return None

        df.loc[mask_only, 'hf_modelid'] = df.loc[mask_only, 'hf_repo'].map(resolve_repo_only)

        mask_full = ~mask_only
        df.loc[mask_full, 'hf_modelid'] = df.loc[mask_full].apply(
            lambda r: f"{r['hf_user'].lower()}/{r['hf_repo'].lower()}", axis=1
        )
        df['hf_modelid'] = df['hf_modelid'].map(valid_map).fillna(df['hf_modelid'])"""
        num_hf_links = df['hf_modelid'].notna().sum()
        matched_hf = df['hf_modelid'].isin(valid_models).sum()
        unmatched_hf = num_hf_links - matched_hf
        print(f"Found HF model links: {num_hf_links}, matched: {matched_hf}, unmatched: {unmatched_hf}")
        # save HF model link lists
        all_model_links = df['hf_modelid'].dropna().unique()
        pd.Series(all_model_links).to_csv('data/tmp/all_hf_modelids.txt', index=False, header=False)
        pd.DataFrame({'hf_modelid': all_model_links}).to_parquet('data/tmp/all_hf_modelids.parquet')
        unmatched_model_links = [m for m in all_model_links if m not in valid_models]
        pd.Series(unmatched_model_links).to_csv('data/tmp/unmatched_hf_modelids.txt', index=False, header=False)
        pd.DataFrame({'hf_modelid': unmatched_model_links}).to_parquet('data/tmp/unmatched_hf_modelids.parquet')
        # ---------- NEW: grab datasets+tags from YAML header ----------
        df[['datasets_tag_list','card_tag_list']] = (df[CARD_TAGS_KEY]
            .apply(lambda txt: pd.Series(extract_datasets_tags(txt))) )

        # README hyperlinks → list
        df['hf_datasetid_list'] = df[CARD_README_KEY].str.findall(r'https?://huggingface\.co/datasets/([^/\s]+)/([^/\s]+)',  flags=re.I).apply(lambda lst: [f"{org.lower()}/{name.lower()}" for org, name in lst])

        # merge YAML + README
        df['all_dataset_list'] = df.apply(
            lambda r: list(set( (r['datasets_tag_list'] or []) +
                                (r['hf_datasetid_list'] or []) )), axis=1)

        # flatten to one row per (modelId, dataset)
        df_ds = (df[['modelId','all_dataset_list']]
                   .explode('all_dataset_list')
                   .dropna(subset=['all_dataset_list'])
                   .rename(columns={'all_dataset_list':'dataset'}))
        # only keep the whitelist dataset
        before_cnt = len(df_ds)                                           ########
        df_ds      = df_ds[df_ds['dataset'].isin(valid_dataset_ids)]      ########
        after_cnt  = len(df_ds)                                           ########
        print(f"Filtered datasets by whitelist: {before_cnt:,} → {after_cnt:,}") ########

        num_ds_links = len(df_ds)
        all_dataset_links = df_ds['dataset'].unique()
        num_unique_ds    = len(all_dataset_links)
        print(f"Found HF dataset links total: {num_ds_links}, unique datasets: {num_unique_ds}") 
        # save HF dataset link lists
        pd.Series(all_dataset_links).to_csv('data/tmp/all_hf_datasetids.txt', index=False, header=False) 
        pd.DataFrame({'hf_datasetid': all_dataset_links}).to_parquet('data/tmp/all_hf_datasetids.parquet') 

    else:
        print("Column CARD_README_KEY not found: skipping HF link extraction.")
    
    # apply corrections and map back to original casing
    def map_final(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return x
        val = correction_map.get(x, x)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return val
        return valid_map.get(val.lower(), val)

    df_tag_rel = df[df['final_base_model'].isin(valid_models)].copy()
    tag_adj, model_index = build_model_relation_matrix(df_tag_rel, full_models, mode='undirected')

    # 2) HF-model link matrix (README links) 
    df_hfmodel_rel = df[df['hf_modelid'].isin(valid_models)].copy()
    df_hfmodel_rel.rename(columns={'hf_modelid':'final_base_model'}, inplace=True) 
    hfmodel_adj, _ = build_model_relation_matrix(df_hfmodel_rel, full_models, mode='undirected') 

    # 3) Dataset co-link matrix (models sharing datasets)
    dataset_groups = df_ds.groupby('dataset')['modelId'].unique()
    M = len(model_index) 
    pos = {m:i for i,m in enumerate(model_index)}
    mat_ds = dok_matrix((M, M), dtype=bool)
    for members in dataset_groups:
        for m1 in members:
            for m2 in members:
                if m1 in pos and m2 in pos:
                    mat_ds[pos[m1], pos[m2]] = True
    for i in range(M): mat_ds[i, i] = True 
    dataset_adj = mat_ds.tocsr()

    # 4) Tag adjacency (shared YAML tags) ----------
    df_tags = (df[['modelId','card_tag_list']]
                 .explode('card_tag_list')
                 .dropna(subset=['card_tag_list'])
                 .rename(columns={'card_tag_list':'tag'}))
    tag_groups = df_tags.groupby('tag')['modelId'].unique()
    mat_tags = dok_matrix((M, M), dtype=bool)
    for members in tag_groups:
        for m1 in members:
            for m2 in members:
                if m1 in pos and m2 in pos:
                    mat_tags[pos[m1], pos[m2]] = True
    for i in range(M): mat_tags[i, i] = True
    tags_adj = mat_tags.tocsr()
    # ------------------------------------------------

    # Save all three sparse matrices and the shared index
    # Print relation statistics
    # Tag-based matrix stats
    tag_nnz = tag_adj.nnz - len(model_index)
    tag_deg = tag_adj.sum(axis=1).A1
    tag_linked = int((tag_deg > 1).sum())
    tag_unlinked = int((tag_deg == 1).sum())
    print(f"Tag-based: {tag_nnz} edges (excl. self-loops), {tag_linked} linked models, {tag_unlinked} unlinked models") 
    # HF-model matrix stats 
    hf_nnz = hfmodel_adj.nnz - len(model_index)
    hf_deg = hfmodel_adj.sum(axis=1).A1
    hf_linked = int((hf_deg > 1).sum())
    hf_unlinked = int((hf_deg == 1).sum())
    print(f"HF-model: {hf_nnz} edges (excl. self-loops), {hf_linked} linked models, {hf_unlinked} unlinked models") 
    # Dataset matrix stats 
    ds_nnz = dataset_adj.nnz - len(model_index)
    ds_deg = dataset_adj.sum(axis=1).A1
    ds_linked = int((ds_deg > 1).sum())
    ds_unlinked = int((ds_deg == 1).sum())
    print(f"Dataset: {ds_nnz} edges (excl. self-loops), {ds_linked} linked models, {ds_unlinked} unlinked models") 

    # Save all three sparse matrices and the shared index
    save_path = 'data/tmp/relations_all.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model_index': model_index, 
            'tag_adj': tag_adj, 
            'hfmodel_adj': hfmodel_adj, 
            'dataset_adj': dataset_adj,
            'tags_adj'   : tags_adj
        }, f)
    print(f"✔️  Saved all relations (4 levels) to {save_path}") 

    print(f"✔️  Saved adjacency to {MODEL_REL_PATH}")
    print(f"✔️  Saved index to {MODEL_REL_INDEXPATH}")

    df['extracted_base_model'].value_counts().to_frame('count').to_csv('data/tmp/extracted_base_model_counts.csv')
    df['final_base_model'].value_counts().to_frame('count').to_csv('data/tmp/final_base_model_counts.csv')
    df['hf_modelid'].value_counts().to_frame('count').to_csv('data/tmp/hf_modelid_counts.csv')

    df['hf_datasetid'] = (df['hf_ds_org'].str.lower() + '/' + df['hf_ds_name'].str.lower())
    df['hf_datasetid'].value_counts().to_frame('count').to_csv('data/tmp/hf_datasetid_counts.csv')
