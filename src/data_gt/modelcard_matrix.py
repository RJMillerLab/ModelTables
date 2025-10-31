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
from tqdm import tqdm                            
from src.utils import load_combined_data, to_parquet    
from itertools import combinations
from collections import defaultdict
from itertools import product
from scipy.sparse import save_npz, coo_matrix

DATA_PATH           = "data/processed/modelcard_step1.parquet" 
DATA_2_PATH         = "data/processed/modelcard_step3_dedup_v2.parquet"
DATA_3_PATH         = "data/processed/modelcard_step3_merged_v2.parquet"
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

def load_model_with_valid_table():
    df_full = pd.read_parquet(DATA_PATH, columns=['modelId', CARD_TAGS_KEY, CARD_README_KEY, 'downloads'])
    df_full_2 = pd.read_parquet(DATA_2_PATH, columns=['modelId', 'hugging_table_list_dedup', 'github_table_list_dedup', 'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup']) #, 'all_title_list'
    # this data 2 path don't have all title list, please load all title list from DATA_3_PATH, and get modelId and all_title_list, then merge this to df_full_2 please!
    df_full_3 = pd.read_parquet(DATA_3_PATH, columns=['modelId', 'all_title_list'])
    df_full_2 = pd.merge(df_full_2, df_full_3, on='modelId', how='left')
    def _to_list_safe(x):
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return []
    df_full_2['all_table_list_dedup'] = (
        df_full_2['hugging_table_list_dedup'].apply(_to_list_safe)
        + df_full_2['github_table_list_dedup'].apply(_to_list_safe)
        + df_full_2['html_table_list_mapped_dedup'].apply(_to_list_safe)
        + df_full_2['llm_table_list_mapped_dedup'].apply(_to_list_safe)
    )
    df = pd.merge(df_full, df_full_2[['modelId', 'all_table_list_dedup', 'all_title_list']], on='modelId', how='left')
    mask = (df['all_table_list_dedup'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0) & df['all_title_list'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0))
    df = df.loc[mask, ['modelId', 'all_table_list_dedup', 'all_title_list', CARD_TAGS_KEY, CARD_README_KEY, 'downloads']]
    return df

HF_ID_RE    = re.compile(r"([A-Za-z0-9\-_]+)/([A-Za-z0-9\-_\.]+)")
#HF_Q_MODEL  = re.compile(r"[?&]model=([\w\-.\/]+)", re.I)
#HF_Q_SEARCH = re.compile(r"[?&]search=([\w\-.\/]+)", re.I)
HF_Q_MODEL  = re.compile(r"[?&]model=([^\)\]\}\s\"'>]+)", re.I)
HF_Q_SEARCH = re.compile(r"[?&]search=([^\)\]\}\s\"'>]+)", re.I)
VALID_PAT   = re.compile(r"^[A-Za-z0-9_\-]+/[A-Za-z0-9_\-.]+$")

def _clean_token(tok: str) -> str:
    """strip trailing punctuation like ')', '\"', '>' …"""
    return re.sub(r"[)\]\}\"'›>\.,]+$", "", tok)

def extract_modelids_from_readme(text: str, valid_models: set, repo_map: dict) -> list:
    """Return *valid* modelIds found in README (robust, de-dup)."""
    if not isinstance(text, str):
        return []
    ids = set()
    # 1) https://huggingface.co/user/repo --------------------------------
    for u, r in re.findall(r"https?://huggingface\.co/([^\s/]+)/([^\s/?#]+)", text, re.I):
        ids.add(f"{u.lower()}/{_clean_token(r.lower())}")
    # 2) query (?model= , ?search=) --------------------------------------
    #for m in HF_Q_MODEL.findall(text):                                    
    #    if "/" in m: ids.add(_clean_token(m.lower())) 
    for m in HF_Q_MODEL.findall(text):                                    
        #tok = _clean_token(m.lower())       
        tok = _clean_token(m)                              
        if "/" in tok:                                                   
            ids.add(tok)                                                  
        else:
            ids.add(tok)
            # bare repo: resolve to highest-download modelId             
            if tok in repo_map:                                          
                ids.add(repo_map[tok][0])                               
    #for s in HF_Q_SEARCH.findall(text):                                   
    #    ids.add(_clean_token(s.lower()))   
    for s in HF_Q_SEARCH.findall(text):                                   
        #tok = _clean_token(s.lower())
        tok = _clean_token(s)
        if "/" in tok:
            ids.add(tok)
        else:
            ids.add(tok)
            if tok in repo_map:
                ids.add(repo_map[tok][0])
    # 3) bare org/repo tokens -------------------------------------------
    for org, repo in HF_ID_RE.findall(text):                              
        ids.add(f"{org.lower()}/{_clean_token(repo.lower())}")            
    # validate pattern & membership -------------------------------------
    clean_ids = [mid for mid in ids if VALID_PAT.match(mid) and mid in valid_models]
    return sorted(clean_ids)

def extract_datasetids_from_readme(text: str, valid_datasets: set) -> list:
    if not isinstance(text, str):
        return []
    ds = set()
    for org, name in re.findall(r"https?://huggingface\.co/datasets/([^\s/]+)/([^\s/]+)", text, re.I):
        tok = f"{org.lower()}/{_clean_token(name.lower())}"
        ds.add(tok)
    return [d for d in ds if d in valid_datasets]

def extract_datasets_from_tags(tags_text: str, valid_datasets: set):
    ds, _ = extract_datasets_tags(tags_text)
    return [d for d in ds if d in valid_datasets]

def extract_basemodels_from_tags(df: pd.DataFrame):
    #df['extracted_base_model'] = df[CARD_TAGS_KEY].str.extract(r'base_model:\s*([^\s]+)', flags=re.IGNORECASE, expand=False)
    df['extracted_base_model'] = df[CARD_TAGS_KEY].str.extract(r'base_model:\s*([^\s,>\]]+)', flags=re.IGNORECASE, expand=False)
    #cleanup_pattern = r'https?://huggingface\.co/|["\'`\[\]\(\)\{\}]'
    #cleanup_pattern = r'https?://huggingface\.co/|["\'`\[\]\{\}]'
    cleanup_pattern = r'https?://huggingface\.co/|["\'`\[\]\{\}\)\>]+'
    #df['extracted_base_model'] = (df['extracted_base_model'].str.replace(cleanup_pattern, '', regex=True))
    df['extracted_base_model'] = df['extracted_base_model'].str.replace(cleanup_pattern, '', regex=True)
    #df['extracted_base_model'] = df['extracted_base_model'].apply(lambda x: _clean_token(x) if isinstance(x, str) else x)
    df['extracted_base_model'] = df['extracted_base_model'].apply(lambda x: _clean_token(x) if isinstance(x, str) else x)

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
    df['tag_base_model_list'] = df['extracted_base_model'].map(correction_map).fillna(df['extracted_base_model'])
    return df

if __name__ == "__main__":
    ########################################################################
    # 0 )  LOAD DATA  ───────────────────────────────────────────────────────
    ########################################################################
    # get all valid dataset IDs
    valid_ds_df = load_combined_data(data_type="datasetcard", file_path=os.path.expanduser("~/Repo/CitationLake/data/raw/"), columns=["datasetId"])
    valid_dataset_ids = set(valid_ds_df["datasetId"].str.lower())
    del valid_ds_df
    print(f"Loaded {len(valid_dataset_ids)} valid dataset IDs")

    # get df which contains modelId, card_tags, downloads and all_table_list_dedup
    df = load_model_with_valid_table()
    print(f"Loaded {len(df)} rows with valid table list")
    # get all valid model IDs
    valid_model_ids= set(df['modelId'])
    # build repo->modelId map, sorted by downloads
    repo_map = defaultdict(list)
    for mid, dl in zip(df['modelId'], df['downloads']):
        repo = mid.split('/', 1)[1]
        repo_map[repo].append((dl, mid))
    # keep only highest-download first
    for repo, lst in repo_map.items():
        repo_map[repo] = [mid for _, mid in sorted(lst, reverse=True)]
    ########################################################################
    # 1 )  EXTRACT HF LINKS, BASE-MODEL, DATASET
    ########################################################################
    #df["readme_modelid_list"]   = df[CARD_README_KEY].apply(lambda txt: extract_modelids_from_readme(txt, valid_model_ids))
    df["readme_modelid_list"] = df[CARD_README_KEY].apply(
        lambda txt: extract_modelids_from_readme(txt, valid_model_ids, repo_map)
    )
    df["readme_datasetid_list"] = df[CARD_README_KEY].apply(lambda txt: extract_datasetids_from_readme(txt, valid_dataset_ids))
    print(f"Updated readme_modelid_list and readme_datasetid_list")
    df = extract_basemodels_from_tags(df)
    #df["tag_base_model_list"]
    print(f"Updated tag_base_model_list")
    df["tag_dataset_list"]      = df[CARD_TAGS_KEY].apply(lambda txt: extract_datasets_from_tags(txt, valid_dataset_ids))
    print(f"Updated tag_dataset_list")
    def to_list(x):
        if isinstance(x, (list,)):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        # skip Pandas 的 NaN、None、pd.NaT
        if pd.isna(x):
            return []
        return []
    # combine: readme + tag + self modelid
    df['tag_base_model_list']   = df['tag_base_model_list'].apply(to_list)
    df['readme_modelid_list']   = df['readme_modelid_list'].apply(to_list)
    # concat + sort + dedup
    df['hf_modelid_list'] = df.apply(
        lambda r: sorted(set([r['modelId']]
                            + r['tag_base_model_list']
                            + r['readme_modelid_list'])),
        axis=1
    )
    df["tag_dataset_list"] = df['tag_dataset_list'].apply(to_list)
    df["readme_datasetid_list"] = df['readme_datasetid_list'].apply(to_list)
    df["hf_datasetid_list"] = df.apply(
        lambda r: sorted(set(r['tag_dataset_list']
                            + r['readme_datasetid_list'])),
        axis=1
    )
    num_hf_links = df["hf_modelid_list"].apply(len).sum()
    matched_hf   = sum(any(m in valid_model_ids for m in lst) for lst in df["hf_modelid_list"])
    print(f"Found HF model links: {num_hf_links}, matched rows: {matched_hf}")
    #pd.Series(all_model_links).to_csv("data/tmp/all_hf_modelids.txt", index=False, header=False)
    #pd.Series([m for m in all_model_links if m not in valid_model_ids]).to_csv("data/tmp/unmatched_hf_modelids.txt", index=False, header=False)
    pd.Series(sorted({m for lst in df["tag_base_model_list"]   for m in lst})).to_csv("data/tmp/tag_base_modelids.txt", index=False, header=False)
    pd.Series(sorted({m for lst in df["readme_modelid_list"]  for m in lst})).to_csv("data/tmp/readme_modelids.txt", index=False, header=False)
    pd.Series(sorted({d for lst in df["tag_dataset_list"]     for d in lst})).to_csv("data/tmp/tag_datasetids.txt", index=False, header=False)
    pd.Series(sorted({d for lst in df["readme_datasetid_list"]for d in lst})).to_csv("data/tmp/readme_datasetids.txt", index=False, header=False)
    print(f'saved all hf modelids and datasetids to data/tmp/tag_base_modelids.txt, readme_modelids.txt, tag_datasetids.txt, readme_datasetids.txt')
    ########################################################################
    # 2 )  BUILD  *RELATED-MODEL LIST*  INSTEAD OF LARGE MATRICES ##########
    ########################################################################
    # drop rows with empty four lists (speed up later calculation)
    ########################################################################
    # 2a ) BUILD *MODEL-BASED* RELATED-MODEL LIST                              ########
    ########################################################################
    # keep rows with at least one base/model link
    # e.g. B -> A, C -> A
    df_model = df[df.apply(lambda r: bool(r['tag_base_model_list'] or r['readme_modelid_list']), axis=1)]
    related_model = defaultdict(set)                                                             
    for col in ["tag_base_model_list", "readme_modelid_list"]:
        exploded = df_model[["modelId", col]].explode(col).dropna()
        for target, grp in exploded.groupby(col)["modelId"]:
            mem = grp.tolist()
            for a, b in combinations(mem, 2): # (B,C)
                related_model[a].add(b)
                related_model[b].add(a)
            if target in valid_model_ids: # (A,B), (A,C)
                for m in mem:
                    related_model[m].add(target)
                    related_model[target].add(m)
    df["related_model_list"] = df["modelId"].map(lambda m: sorted(related_model.get(m, [])))
    df.drop(columns=['card_tags', 'card_readme', 'downloads'], inplace=True, errors='ignore')
    to_parquet(df, "data/processed/modelcard_gt_related_model.parquet")

    ########################################################################
    # 5 )  BUILD CSV-LEVEL GT via related_model_list （no self-pair） ########
    ########################################################################

    ########################################################################
    # 5a ) BUILD CSV-LEVEL GT FROM related_model_list (Model-based)
    ########################################################################
    model_to_csvs = df.set_index("modelId")["all_table_list_dedup"].to_dict()
    # 1) full model and csv list
    model_ids = list(model_to_csvs.keys())
    all_csvs_m = sorted({c for cs in model_to_csvs.values() for c in cs})
    # 2) construct incidence matrix A_model (models × csvs)
    model2idx = {m:i for i,m in enumerate(model_ids)}
    csv2idx   = {c:i for i,c in enumerate(all_csvs_m)}
    rows, cols = [], []
    for m, cs in model_to_csvs.items():
        i = model2idx[m]
        for c in cs:
            rows.append(i); cols.append(csv2idx[c])
    data = np.ones(len(rows), dtype=bool)
    A_model = coo_matrix((data, (rows, cols)),
                        shape=(len(model_ids), len(all_csvs_m)),
                        dtype=bool).tocsr()

    # 3) construct model-level adjacency matrix P_model (first inter-model, then self-loop to include intra-model)
    row_p, col_p = [], []
    for m, neighs in related_model.items():
        i = model2idx[m]
        for n in neighs:
            j = model2idx[n]
            row_p += [i, j]; col_p += [j, i]
    P_model = coo_matrix((np.ones(len(row_p), bool),
                         (row_p, col_p)),
                        shape=(len(model_ids), len(model_ids)),
                        dtype=bool).tocsr()
    P_model.setdiag(True)   # diag=True auto-count intra-model pairs

    # 4) calculate csv-level adjacency: M_model = A^T · P_model · A
    M_model = (A_model.T.dot(P_model).dot(A_model)).astype(bool).tocsr()
    M_model.setdiag(False) # remove self-loop

    # 5) save npz + csv_list for MODEL-BASED adjacency
    # (a) Print size before trimming
    print(f"[INFO] MODEL csv adjacency before trim: {M_model.shape[0]} items")
    # (b) Trim zero rows/cols
    row_sums = np.array(M_model.sum(axis=1)).ravel()
    keep_idx = np.where(row_sums > 0)[0]
    print(f"[INFO] Dropping {M_model.shape[0] - keep_idx.size} zero rows/cols for MODEL")
    M_model = M_model[keep_idx][:, keep_idx]
    all_csvs_model = [all_csvs_m[i] for i in keep_idx]
    # (c) Print size after trimming
    print(f"[INFO] MODEL csv adjacency after trim: {M_model.shape[0]} items")
    # (d) Save trimmed matrix and updated CSV list
    save_npz('data/gt/scilake_gt_modellink_model_adj.npz', M_model, compressed=True)
    all_csvs_model = [os.path.basename(c) for c in all_csvs_model]
    with open('data/gt/scilake_gt_modellink_model_adj_csv_list.pkl','wb') as f:
        pickle.dump(all_csvs_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✔️ Saved MODEL-BASED CSV adjacency matrix ({M_model.nnz} edges) after trimming")
    '''
    # original for-loop version
    csv_counts_model = defaultdict(int)    
    # inside model               
    for m, csvs in model_to_csvs.items():
        for a, b in combinations(sorted(set(csvs)), 2):
            if a != b:
                key = tuple(sorted((a, b)))
                csv_counts_model[key] += 1          ######## NEW
    # between models                            
    for m, neighs in related_model.items():
        cs_m = model_to_csvs.get(m, [])
        for n in neighs:
            if m >= n: # only count once for each pair
                continue
            cs_n = model_to_csvs.get(n, [])
            for a, b in product(cs_m, cs_n):
                if a == b: # skip self-pair
                    continue
                key = tuple(sorted((a, b)))
                csv_counts_model[key] += 1
    # keep original tuple→count
    #with open('data/gt/scilake_gt_modellink_model_counts.pkl', 'wb') as f:
    #    pickle.dump(dict(csv_counts_model), f)
    # convert to adjacency mapping {csv: [related_csvs]}
    adj_model = defaultdict(set)
    for (a, b), cnt in csv_counts_model.items():
        if cnt > 0:
            adj_model[a].add(b); adj_model[b].add(a)
    adj_model = {k: sorted(v) for k, v in adj_model.items()}
    processed_model_adj = {os.path.basename(k): [os.path.basename(x) for x in v] for k, v in adj_model.items()}
    # save to npz
    from src.data_gt.convert_adj_to_npz import dict_to_boolean_csr
    M, M_csv_list = dict_to_boolean_csr(processed_model_adj)
    save_npz('data/gt/scilake_gt_modellink_model_adj.npz', M, compressed=True)
    with open('data/gt/scilake_gt_modellink_model_csv_list.pkl','wb') as f:
        pickle.dump(list(M_csv_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('data/gt/scilake_gt_modellink_model_adj_processed.pkl', 'wb') as f:
    #    pickle.dump(processed_model_adj, f)
    print(f"✔️  Saved MODEL-BASED CSV adjacency ({len(adj_model):,} keys)")'''

    ########################################################################
    # 5b ) BUILD CSV-LEVEL GT FROM related_dataset_list (Dataset-based)
    ########################################################################
    # Create dataset related map
    df_ds = df[df.apply(lambda r: bool(r['tag_dataset_list'] or r['readme_datasetid_list']), axis=1)]
    related_ds = defaultdict(set)
    for col in ["tag_dataset_list", "readme_datasetid_list"]:
        expl = df_ds[["modelId", col]].explode(col).dropna()
        for _, grp in expl.groupby(col)["modelId"]:
            mem = grp.tolist()
            for a, b in combinations(mem, 2):
                related_ds[a].add(b); related_ds[b].add(a)                                  
    '''
    # original for-loop version
    # cross model→csv to dataset-GT
    csv_counts_ds = defaultdict(int)
    # intra-model
    for m, csvs in model_to_csvs.items():
        for a, b in combinations(sorted(set(csvs)), 2):
            if a != b:
                key = tuple(sorted((a, b)))
                csv_counts_ds[key] += 1
    # between models
    for m, neighs in related_ds.items():
        cs_m = model_to_csvs.get(m, [])
        for n in neighs:
            if m >= n: continue
            cs_n = model_to_csvs.get(n, [])
            for a, b in product(cs_m, cs_n):
                if a == b: continue                                                         
                key = tuple(sorted((a, b)))
                csv_counts_ds[key] += 1
    #with open('data/gt/scilake_gt_modellink_dataset_counts.pkl', 'wb') as f:
    #    pickle.dump(dict(csv_counts_ds), f)
    adj_ds = defaultdict(set)
    for (a, b), cnt in csv_counts_ds.items():
        if cnt > 0:                                                                       
            adj_ds[a].add(b); adj_ds[b].add(a)                                            
    adj_ds = {k: sorted(v) for k, v in adj_ds.items()}
    processed_ds_adj = {os.path.basename(k): [os.path.basename(x) for x in v] for k, v in adj_ds.items()}
    # save npz
    from src.data_gt.convert_adj_to_npz import dict_to_boolean_csr
    D, D_csv_list = dict_to_boolean_csr(processed_ds_adj)
    save_npz('data/gt/scilake_gt_modellink_dataset_adj.npz', D, compressed=True)
    with open('data/gt/scilake_gt_modellink_dataset_csv_list.pkl','wb') as f:
        pickle.dump(list(D_csv_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    #with open('data/gt/scilake_gt_modellink_dataset_adj_processed.pkl', 'wb') as f:
    #    pickle.dump(processed_ds_adj, f)
    print(f"✔️  Saved DATASET-BASED CSV adjacency ({len(adj_ds):,} keys)")'''
    # 1) Full model and csv list (same as above model_ids & all_csvs_m)
    # 2) incidence matrix A_model already constructed (same as above)
    # 3) construct dataset-level adjacency P_ds
    row_d, col_d = [], []
    for m, neighs in related_ds.items():
        i = model2idx[m]
        for n in neighs:
            j = model2idx[n]
            row_d += [i, j]; col_d += [j, i]
    P_ds = coo_matrix((np.ones(len(row_d), bool),
                      (row_d, col_d)),
                      shape=(len(model_ids), len(model_ids)),
                      dtype=bool).tocsr()
    P_ds.setdiag(True)  # include model-specific csv pairs
    # 4) calculate dataset-level csv adjacency: M_ds = A^T · P_ds · A
    M_ds = (A_model.T.dot(P_ds).dot(A_model)).astype(bool).tocsr()
    M_ds.setdiag(False)  # remove self-loop
    # 5) save npz + csv_list for DATASET-BASED adjacency
    # (a) Print size before trimming
    print(f"[INFO] DATASET csv adjacency before trim: {M_ds.shape[0]} items")
    # (b) Trim zero rows/cols
    row_sums_ds = np.array(M_ds.sum(axis=1)).ravel()
    keep_idx_ds = np.where(row_sums_ds > 0)[0]
    print(f"[INFO] Dropping {M_ds.shape[0] - keep_idx_ds.size} zero rows/cols for DATASET")
    M_ds = M_ds[keep_idx_ds][:, keep_idx_ds]
    all_csvs_dataset = [all_csvs_m[i] for i in keep_idx_ds]
    # (c) Print size after trimming
    print(f"[INFO] DATASET csv adjacency after trim: {M_ds.shape[0]} items")
    # (d) Save trimmed matrix and updated CSV list
    save_npz('data/gt/scilake_gt_modellink_dataset_adj.npz', M_ds, compressed=True)
    all_csvs_dataset = [os.path.basename(c) for c in all_csvs_dataset]
    with open('data/gt/scilake_gt_modellink_dataset_adj_csv_list.pkl','wb') as f:
        pickle.dump(all_csvs_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✔️ Saved DATASET-BASED CSV adjacency matrix ({M_ds.nnz} edges) after trimming")