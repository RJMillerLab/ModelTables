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

import re, os, json
import requests                                    
import pandas as pd
import numpy as np                                              
import pickle
from scipy.sparse import dok_matrix, csr_matrix, save_npz
import sqlite3
from tqdm import tqdm                            
from src.utils import load_combined_data    
from itertools import combinations
from scipy.sparse import coo_matrix

MODEL_REL_DB   = "data/tmp/model_rel.db"
DATASET_REL_DB = "data/tmp/dataset_rel.db"
DATA_PATH           = "data/processed/modelcard_step1.parquet" 
DATA_2_PATH         = "data/processed/modelcard_step3_dedup.parquet"
CARD_TAGS_KEY       = "card_tags"
CARD_README_KEY     = 'card_readme'

os.makedirs('data/tmp', exist_ok=True)

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
def init_edge_db(db_path):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous  = OFF;
        CREATE TABLE IF NOT EXISTS edges (
            src TEXT NOT NULL,
            tgt TEXT NOT NULL,
            rel TEXT NOT NULL,
            UNIQUE(src, tgt, rel)
        );
        CREATE INDEX IF NOT EXISTS idx_src ON edges(src);
        CREATE INDEX IF NOT EXISTS idx_tgt ON edges(tgt);
    """)
    conn.commit()
    return conn, cur

def insert_edges(cur, edge_iter, rel_type, batch=5000):
    buf = []
    for s, t in edge_iter:
        buf.append((s, t, rel_type))
        if len(buf) >= batch:
            cur.executemany("INSERT INTO edges VALUES(?,?,?)", buf)
            buf.clear()
    if buf:
        cur.executemany("INSERT INTO edges VALUES(?,?,?)", buf)

def build_edges_sql(
    df_rel: pd.DataFrame,
    valid_model_ids: set,
    rel_type: str,
    src_col="modelId",
    tgt_col="final_base_model",
    db_path=MODEL_REL_DB):
    print(f"building {rel_type} edges to {db_path}")
    conn, cur = init_edge_db(db_path)
    """groups = df_rel.groupby(tgt_col)[src_col]
    total = len(groups)  # number of cliques
    for base, members in tqdm(groups,
                              desc=f"[{rel_type}] clique", 
                              total=total):
        try:
            members = [m for m in members if m in valid_model_ids]
        except Exception as e:
            print(f"ERROR filtering members for base={base}: {e}")
            raise
        for i, j in tqdm(combinations(members, 2),
                          desc=f"[{rel_type}] edges in {base}", 
                          leave=False):
            try:
                insert_edges(cur, ((i, j), (j, i)), rel_type)
            except Exception as e:
                print(f"ERROR inserting edge ({i},{j}) rel={rel_type}: {e}")
                raise"""
    """groups = df_rel.groupby(tgt_col)[src_col]                    ########
    for base, members in tqdm(groups,                            ########
                              desc=f"[{rel_type}] clique"):    ########
        members = [m for m in members if m in valid_model_ids]    ########
        if len(members) < 2:                                     ########
            continue                                             ########
        buf = []                                                 ########
        for i, j in combinations(members, 2):                    ########
            buf.append((i, j, rel_type))                        ########
            buf.append((j, i, rel_type))                        ########
        cur.executemany("INSERT OR IGNORE INTO edges VALUES(?,?,?)", buf) ########"""
    groups = df_rel.groupby(tgt_col)[src_col]
    commit_every = 500               ######## 每 500 个 clique 提交一次
    for idx, (base, members) in enumerate(tqdm(groups, desc=f"[{rel_type}] clique")):
        members = [m for m in members if m in valid_model_ids]
        if len(members) < 2:
            continue
        buf = []
        for i, j in combinations(members, 2):
            buf.append((i, j, rel_type))
            buf.append((j, i, rel_type))
        cur.executemany("INSERT OR IGNORE INTO edges VALUES(?,?,?)", buf)
        del buf                        ######## 释放 Python list 内存
        # 每 commit_every 个 clique，就提交一次
        if idx % commit_every == 0:
            conn.commit()             ######## 分批 commit，清空 WAL 缓冲
    conn.commit()                     ######## 最后再确认一次
    conn.close()
    print(f"✔️  {rel_type} edges stored → {db_path}")


def build_model_relation_matrix(
    df_rel: pd.DataFrame,
    all_models: set,
    src_col: str = "modelId",
    tgt_col: str = "final_base_model",
    mode: str = "undirected"
) -> (csr_matrix, list):
    """
    Vectorised + COO
    ------------------------------------------------
    directed : single-directional src→tgt
    related  : fully connected within the same base_model (undirected, symmetric)
    """
    print('shape', df_rel.shape)
    print()
    pos = {m: i for i, m in enumerate(model_index)}
    M = len(model_index)
    rows, cols = [], []

    if mode == "directed":
        src_idx = df_rel[src_col].map(pos).to_numpy()
        tgt_idx = df_rel[tgt_col].map(pos).to_numpy()
        mask    = (src_idx >= 0) & (tgt_idx >= 0)
        rows.extend(src_idx[mask])
        cols.extend(tgt_idx[mask])
    else:  # 'related' → clique
        for _, members in df_rel.groupby(tgt_col)[src_col]:
            members = [pos[m] for m in members if m in pos]
            if len(members) < 2:
                continue
            # generate (i,j) & (j,i) to maintain undirected
            for i, j in combinations(members, 2):
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)
    # add self-loops
    rows.extend(range(M)); cols.extend(range(M))
    data = np.ones(len(rows), dtype=bool)
    csr_mat = coo_matrix((data, (rows, cols)), shape=(M, M)).tocsr()

    print(f"Adjacency shape: {csr_mat.shape}  |  "
          f"edges excl.self-loop: {csr_mat.nnz - M:,}")
    return csr_mat, model_index

def build_dataset_edges_sql(ds_groups, db_path=DATASET_REL_DB):
    conn, cur = init_edge_db(db_path)
    for members in ds_groups:
        if len(members) < 2:
            continue
        for i, j in combinations(members, 2):
            insert_edges(cur, ((i, j), (j, i)), rel_type='dataset')
    conn.commit(); conn.close()
    print(f"✔️  dataset edges stored → {db_path}")

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

if __name__ == "__main__":
    # get all valid dataset IDs
    valid_ds_df       = load_combined_data(data_type="datasetcard", file_path=os.path.expanduser("~/Repo/CitationLake/data/raw/"), columns=["datasetId"])
    valid_dataset_ids = set(valid_ds_df["datasetId"].str.lower())
    del valid_ds_df
    print(f"Loaded {len(valid_dataset_ids)} valid dataset IDs")
    # get df which contains modelId, card_tags, downloads and all_table_list_dedup
    df_full = pd.read_parquet(DATA_PATH, columns=['modelId', CARD_TAGS_KEY, CARD_README_KEY, 'downloads'])
    df_full_2 = pd.read_parquet(DATA_2_PATH, columns=['modelId', 'hugging_table_list_dedup', 'github_table_list_dedup', 'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup', 'all_title_list'])
    df_full_2['all_table_list_dedup'] = df_full_2['hugging_table_list_dedup'].apply(list) + df_full_2['github_table_list_dedup'].apply(list) + df_full_2['html_table_list_mapped_dedup'].apply(list) + df_full_2['llm_table_list_mapped_dedup'].apply(list)
    df = pd.merge(df_full, df_full_2[['modelId', 'all_table_list_dedup', 'all_title_list']], on='modelId', how='left')
    del df_full, df_full_2
    # get all valid model IDs
    valid_model_ids= set(df['modelId'])
    # get extracted_base_model from card_tags
    df['extracted_base_model'] = df[CARD_TAGS_KEY].str.extract(r'base_model:\s*([^\s]+)', flags=re.IGNORECASE).replace({np.nan: None})
    df['extracted_base_model'] = df['extracted_base_model'].str.replace(r"[\"'`]", "", regex=True)
    df['extracted_base_model'] = df['extracted_base_model'].str.replace(r"[\[\]\(\)\{\}]", "", regex=True)
    df['extracted_base_model'] = df['extracted_base_model'].str.replace('https://huggingface.co/', '')
    print(f"Unique extracted_base_model before filtering: {df['extracted_base_model'].nunique()}")
    # build mapping counts and identify invalid
    mapping_counts = df.groupby(['modelId','extracted_base_model']).size().reset_index(name='count')
    valid_map = {m.lower(): m for m in valid_model_ids}

    mapping_invalid = mapping_counts[~mapping_counts['extracted_base_model']\
        .fillna('')\
        .str.lower()\
        .isin(valid_map)]
    num_invalid = mapping_invalid['extracted_base_model'].nunique()
    print(f"Unique invalid extracted_base_model: {num_invalid}")

    # correction via repository merge
    models_df = df[['modelId','downloads']].drop_duplicates()
    models_df['repo'] = models_df['modelId'].str.split('/', n=1).str[1]

    invalid_df = pd.DataFrame(mapping_invalid['extracted_base_model'].unique(), columns=['extracted_base_model'])
    merged = invalid_df.merge(models_df, left_on='extracted_base_model', right_on='repo', how='left')
    merged_sorted = merged.sort_values(['extracted_base_model','downloads'], ascending=[True, False])
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
    #print(final_unmatched)

    # corrected the unmatched base_model entries
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
    print(f"Salvaged {salvage_count} unmatched base_model entries")
    print('-'*100)
    # the model which points to same base_model should link to each other
    df['final_base_model'] = df['extracted_base_model'].map(correction_map).fillna(df['extracted_base_model'])
    # extract HF links if README exists and print stats
    assert CARD_README_KEY in df.columns
    """df.loc[mask_only, 'hf_modelid'] = df.loc[mask_only, 'hf_repo'].map(
        lambda repo: models_df[models_df['repo']==repo]
                        .sort_values('downloads', ascending=False)
                        ['modelId'].iloc[0]
    )"""
    hf_match = df[CARD_README_KEY].str.extract(r'https?://huggingface\.co/([^/\s]+)/([^/\s]+)')
    df[['hf_user','hf_repo']] = hf_match
    df['hf_modelid'] = (df['hf_user'].str.lower() + '/' + df['hf_repo'].str.lower())
    df['hf_modelid'] = df['hf_modelid'].map(valid_map).fillna(df['hf_modelid'])
    # because some base_model is only partial, we need to makeup the full modelId

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
    matched_hf = df['hf_modelid'].isin(valid_model_ids).sum()
    unmatched_hf = num_hf_links - matched_hf
    print(f"Found HF model links: {num_hf_links}, matched: {matched_hf}, unmatched: {unmatched_hf}")
    # save HF model link lists
    all_model_links = df['hf_modelid'].dropna().unique()
    pd.Series(all_model_links).to_csv('data/tmp/all_hf_modelids.txt', index=False, header=False)
    #pd.DataFrame({'hf_modelid': all_model_links}).to_parquet('data/tmp/all_hf_modelids.parquet')
    unmatched_model_links = [m for m in all_model_links if m not in valid_model_ids]
    pd.Series(unmatched_model_links).to_csv('data/tmp/unmatched_hf_modelids.txt', index=False, header=False)
    #pd.DataFrame({'hf_modelid': unmatched_model_links}).to_parquet('data/tmp/unmatched_hf_modelids.parquet')

    # ---------- NEW: grab datasets+tags from YAML header ----------
    df[['datasets_tag_list','card_tag_list']] = (df[CARD_TAGS_KEY].apply(lambda txt: pd.Series(extract_datasets_tags(txt))) )
    # README hyperlinks → list
    df['hf_datasetid_list'] = df[CARD_README_KEY].str.findall(r'https?://huggingface\.co/datasets/([^/\s]+)/([^/\s]+)',  flags=re.I).apply(lambda lst: [f"{org.lower()}/{name.lower()}" for org, name in lst])
    # merge YAML + README
    df['all_dataset_list'] = df.apply(lambda r: list(set( (r['datasets_tag_list'] or []) + (r['hf_datasetid_list'] or []) )), axis=1)
    # flatten to one row per (modelId, dataset)
    df_ds = (df[['modelId','all_dataset_list']].explode('all_dataset_list').dropna(subset=['all_dataset_list']).rename(columns={'all_dataset_list':'dataset'}))
    # only keep the whitelist dataset
    before_cnt = len(df_ds)
    df_ds      = df_ds[df_ds['dataset'].isin(valid_dataset_ids)]
    print(f"Filtered datasets by whitelist: {before_cnt:,} → {len(df_ds):,}")
    num_ds_links = len(df_ds)
    all_dataset_links = df_ds['dataset'].unique()
    print(f"Found HF dataset links total: {num_ds_links}, unique datasets: {len(all_dataset_links)}") 
    # save HF dataset link lists
    pd.Series(all_dataset_links).to_csv('data/tmp/all_hf_datasetids.txt', index=False, header=False) 
    #pd.DataFrame({'hf_datasetid': all_dataset_links}).to_parquet('data/tmp/all_hf_datasetids.parquet') 
    print('finished!')
    
    # build edges
    model_index = sorted(valid_model_ids)
    df_tag_rel = df[df['final_base_model'].isin(valid_model_ids)].copy()
    build_edges_sql(df_tag_rel, valid_model_ids, rel_type='base_model', db_path=MODEL_REL_DB)

    df_hf_rel = df[df['hf_modelid'].isin(valid_model_ids)].copy()
    df_hf_rel.rename(columns={'hf_modelid':'final_base_model'}, inplace=True)
    build_edges_sql(df_hf_rel, valid_model_ids, rel_type='readme', db_path=MODEL_REL_DB)

    print('building dataset_adj')
    # 3) Dataset co-link matrix (models sharing datasets)
    """dataset_groups = df_ds.groupby('dataset')['modelId'].unique()
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
    tags_adj = mat_tags.tocsr()"""
    dataset_groups = df_ds.groupby('dataset')['modelId'].unique()
    build_dataset_edges_sql(dataset_groups, db_path=DATASET_REL_DB)
    # ------------------------------------------------

    print('building csv_groundtruth_dataset')
    def _build_csv_gt(adj: csr_matrix, model_index: list, model_to_csvs: dict):
        coo = adj.tocoo()
        csv_gt = {}
        for i, j in zip(coo.row, coo.col):
            if i == j:
                continue
            m1, m2 = model_index[i], model_index[j]
            c1 = model_to_csvs.get(m1, []) or []
            c2 = model_to_csvs.get(m2, []) or []
            for a in c1:
                a_base = os.path.basename(a)
                tgt = csv_gt.setdefault(a_base, set())
                tgt.update(os.path.basename(b) for b in c2)
        return {k: sorted(v - {k}) for k, v in csv_gt.items()}

    model_to_csvs = (df.set_index('modelId')['all_table_list_dedup']
                    .to_dict())
    csv_gt_dataset = _build_csv_gt(dataset_adj, model_index, model_to_csvs)
    with open('data/tmp/csv_groundtruth_dataset.json', 'w') as f:
        json.dump(csv_gt_dataset, f, indent=2)
    print(f"✔️  Saved DATASET-level GT: {len(csv_gt_dataset):,} keys")

    modelcard_adj = (tag_adj.copy()
                    .maximum(hfmodel_adj)
                    .maximum(tags_adj))
    csv_gt_modelcard = _build_csv_gt(modelcard_adj, model_index, model_to_csvs)
    with open('data/tmp/csv_groundtruth_modelcard.json', 'w') as f:
        json.dump(csv_gt_modelcard, f, indent=2)
    print(f"✔️  Saved MODELCARD-level GT: {len(csv_gt_modelcard):,} keys")


    print('saving / stats (dataset + YAML-tag only)')
    # Save all three sparse matrices and the shared index
    # Print relation statistics
    # Tag-based matrix stats
    # Save all three sparse matrices and the shared index
    save_path = 'data/tmp/relations_all.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model_index': model_index, 
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

