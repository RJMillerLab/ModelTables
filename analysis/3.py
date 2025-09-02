#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

df = pd.read_parquet("data/processed/modelcard_step2.parquet")
print(df.columns)
df.head()


# In[33]:


import json, os
with open(os.path.join("data/processed", "deduped_github_csvs", "md_to_csv_mapping.json"), 'r', encoding='utf-8') as jf:
    md_to_csv_mapping = json.load(jf)


# In[ ]:


md_to_csv_mapping


# In[ ]:


# load final_integration_with_paths.parquet
df = pd.read_parquet("final_integration_with_paths.parquet")
df


# In[ ]:


df['html_table_list'].iloc[1]


# In[ ]:


# load data/processed/modelcard_step3_merged.parquet.
df2 = pd.read_parquet("data/processed/modelcard_step3_merged.parquet")
df2


# In[ ]:


df2['html_table_list_mapped']


# In[ ]:


import numpy as np

# 1. 如果你已经加载了 df
# df = pd.read_parquet("data/processed/modelcard_step2.parquet")

for col in ['html_table_list_mapped', 'llm_table_list_mapped']:
    if col not in df2.columns:
        raise ValueError(f"Column `{col}` not found in DataFrame!")

def count_empty_and_nonempty_array(series):
    empty_count = 0
    nonempty_count = 0
    for val in series:
        if isinstance(val, np.ndarray):
            if len(val) == 0:
                empty_count += 1
            else:
                nonempty_count += 1
        elif isinstance(val, list):
            if len(val) == 0:
                empty_count += 1
            else:
                nonempty_count += 1
    return empty_count, nonempty_count

html_empty, html_nonempty = count_empty_and_nonempty_array(df2['html_table_list_mapped'])
llm_empty, llm_nonempty = count_empty_and_nonempty_array(df2['llm_table_list_mapped'])

print(f"🧾 html_table_list_mapped - empty: {html_empty}, non-empty: {html_nonempty}")
print(f"🤖 llm_table_list_mapped  - empty: {llm_empty}, non-empty: {llm_nonempty}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# -*- coding: utf-8 -*-
"""
Author: Zhengyuan Dong
Created: 2025-04-02
Last Modified: 2025-04-02
Description: Merge tables list from final_integration_with_paths.parquet to modelcard_all_title_list.parquet
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# ============ Placeholders / Configurations ============
FINAL_INTEGRATION_PARQUET   = "final_integration_with_paths.parquet"
ALL_TITLE_PATH              = "data/processed/modelcard_all_title_list.parquet"
MERGE_PATH                  = "data/processed/modelcard_step3_merged.parquet" ########

def _combine_lists(series):
    """
    Helper to combine lists while dropping NaN/None.
    """
    all_items = []
    for x in series.dropna():
        if isinstance(x, (list, tuple, np.ndarray)):
            all_items.extend(x)
        else:
            pass
    return list(set(all_items))


df = pd.read_parquet(FINAL_INTEGRATION_PARQUET, columns=['query', 'html_table_list', 'saved_csv_paths'])
print(f"  df loaded with shape: {df.shape}")
if 'saved_csv_paths' in df.columns:
    df.rename(columns={'saved_csv_paths': 'llm_table_list'}, inplace=True)
df2 = pd.read_parquet(ALL_TITLE_PATH)
print(f"  df2 loaded with shape: {df2.shape}")
print("\nStep 1: Expanding df2 to match df (on df2.all_title_list vs df.query)...")


# In[ ]:


df2


# In[ ]:


df2.columns


# In[ ]:


all_title_list_key = "all_title_list"
df2[all_title_list_key] = df2[all_title_list_key].apply(lambda x: list(dict.fromkeys(x)) if isinstance(x, (list, tuple, np.ndarray)) else x)
df2_exploded = df2.explode(all_title_list_key).rename(columns={all_title_list_key: 'explode_title'})
df2_exploded


# In[ ]:


df


# In[ ]:


# 1. explode df2 so that each title in df2['all_title_list'] becomes its own row

# Now each row has (modelid, title)
# 2. merge with df on (title == query)
merged = pd.merge(
    df2_exploded,
    df,
    how='left',
    left_on='explode_title',
    right_on='query'
)
merged


# In[ ]:


df['llm_table_list'].iloc[1]


# In[ ]:


# merged columns: modelid, title, query, html_table_list, llm_table_list
print("Step 2: Grouping & assembling lists back by modelId...")
grouped = merged.groupby('modelId').agg({
    'html_table_list': lambda x: _combine_lists(x),
    'llm_table_list': lambda x: _combine_lists(x),
}).reset_index()
# Rename to e.g. html_table_list_mapped, llm_table_list_mapped
grouped.rename(columns={'html_table_list': 'html_table_list_mapped', 'llm_table_list': 'llm_table_list_mapped'}, inplace=True)
print("Step 3: Merging the grouped columns back into df2...")
df2_merged = pd.merge(df2, grouped, on='modelId', how='left')
df2_merged['html_table_list_mapped']  = df2_merged['html_table_list_mapped'].apply(lambda v: v if isinstance(v, (list, tuple, np.ndarray)) else [])
df2_merged['llm_table_list_mapped']  = df2_merged['llm_table_list_mapped'].apply(lambda v: v if isinstance(v, (list, tuple, np.ndarray)) else [])

# save df2_merged
df2_merged.to_parquet(MERGE_PATH, index=False)


# In[ ]:


# test html_table.parquet
import pandas as pd
df2_merged = pd.read_parquet("data/processed/final_integration_with_paths.parquet")
df2_merged


# In[ ]:


# load final_integration_with_paths.parquet
df5 = pd.read_parquet("data/processed/final_integration_with_paths.parquet")
df5


# In[ ]:


df5.columns


# In[ ]:


#                            columns=['modelId', 'readme_path', 'readme_hash'])
#                            columns=['modelId', 'readme_path', 'readme_hash'])


# In[ ]:


# load all_title_list.parquet
df2 = pd.read_parquet("data/processed/modelcard_all_title_list.parquet")
df2


# In[ ]:


df2.columns


# In[ ]:


# load modelcard_step2.parquet
df3 = pd.read_parquet("data/processed/modelcard_step2.parquet")
df3
df3.columns


# In[3]:


"""df2_merged["html_path"] = "data/processed/" + df2_merged["html_path"]  ########

# Update each element in table_list column
df2_merged["table_list"] = df2_merged["table_list"].apply(
    lambda lst: ["data/processed/" + path for path in lst]
)
df2_merged
"""


# In[ ]:


# need to update 
llm_tables ok


tables_output ok -> data/processed/html_table.parquet ok
table_list, arxiv, llm_tables


# In[ ]:


# laod merged_df.parquet
df3_merged = pd.read_parquet("data/processed/modelcard_citation_enriched.parquet")
df3_merged


# In[ ]:


import json
print(json.dumps(json.loads(df3_merged['original_response'].iloc[0]),indent=4))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pickle
SCORE_PICKLE = "data/processed/modelcard_citation_overlap_by_paperId_score.pickle"
RELATED_PICKLE = "data/processed/modelcard_citation_overlap_by_paperId_related.pickle"
DIRECT_PICKLE = "data/processed/modelcard_citation_direct_relation.pickle"

# load these and check data format
with open(SCORE_PICKLE, 'rb') as f:
    score = pickle.load(f)
    print(f"Score loaded with {len(score)} items.")
    print(score)
    print(score[0].keys())

with open(RELATED_PICKLE, 'rb') as f:
    related = pickle.load(f)
    print(f"Related loaded with {len(related)} items.")
    print(related[0])
    print(related[0].keys())

with open(DIRECT_PICKLE, 'rb') as f:
    direct = pickle.load(f)
    print(f"Direct loaded with {len(direct)} items.")
    print(direct[0])
    print(direct[0].keys())
# Check if the keys are the same
keys = set(score[0].keys()).union(set(related[0].keys())).union(set(direct[0].keys()))
print(f"Keys: {keys}")


# In[ ]:


# load data/processed/modelcard_citation_enriched.parquet
import pandas as pd
df = pd.read_parquet("data/processed/modelcard_citation_enriched.parquet")
df


# In[ ]:


df[df['corpusId']==253523547]['title'].iloc[0]


# In[ ]:


# load data/processed/llm_markdown_table_results.parquet
#df1 = pd.read_parquet("data/processed/llm_markdown_table_results.parquet")
print(df1[df1['corpusid']==252439001]['llm_response_raw'].iloc[0])


# In[ ]:


import json
print(json.dumps(json.loads(df['parsed_response'].iloc[4007]),indent=4))


# In[ ]:


# df ['paperId']=='98ab627dd147db88b5e5cfa9a74f1bd8da110021.'
print(json.dumps(json.loads(df[df['paperId']=='98ab627dd147db88b5e5cfa9a74f1bd8da110021']['original_response'].iloc[0]), indent=4))


# In[ ]:


df.columns


# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np

#df = pd.read_parquet("data/processed/modelcard_step4_dedup.parquet")
print(df.columns)
def has_valid_key(row, key):
    html_tables = row[key]
    if html_tables is not None:
        return len(html_tables) > 0
    else:
        return False

df['has_html'] = df.apply(lambda row: has_valid_key(row, 'html_table_list_mapped'), axis=1)
df['has_llm'] = df.apply(lambda row: has_valid_key(row, 'llm_table_list_mapped'), axis=1)

df_both = df[df["has_html"] & df["has_llm"]]

print(f"同时存在 HTML 和 LLM 路径的行数: {df_both.shape[0]}")


# In[ ]:


df2 = pd.read_parquet("data/processed/final_integration_with_paths.parquet")
df2["has_html"] = df2.apply(lambda row: has_valid_key(row, "html_table_list"), axis=1)
df2["has_llm"] = df2.apply(lambda row: has_valid_key(row, "llm_table_list"), axis=1)
df_both2 = df2[df2["has_html"] & df2["has_llm"]]
print(f"同时存在 HTML 和 LLM 路径的行数: {df_both.shape[0]}")
df_both2[["html_html_path", "html_table_list", "llm_table_list"]].head()


# In[ ]:


# for modelcard_step3_merged
#df3 = pd.read_parquet("data/processed/modelcard_step3_merged.parquet")
df3.columns
df3['has_html'] = df3.apply(lambda row: has_valid_key(row, 'html_table_list_mapped'), axis=1)
df3['has_llm'] = df3.apply(lambda row: has_valid_key(row, 'llm_table_list_mapped'), axis=1)
df_both3 = df3[df3["has_html"] & df3["has_llm"]]
print(f"同时存在 HTML 和 LLM 路径的行数: {df_both3.shape[0]}")
df_both3[["html_table_list_mapped", "llm_table_list_mapped"]].head()


# In[ ]:


df3.columns


# In[ ]:


len(df), len(df2), len(df3)


# In[ ]:


df['github_table_list_dedup']
# check whether 


# In[ ]:





# In[ ]:


# load data/statistics/benchmark_results.csv
import pandas as pd
df3 = pd.read_parquet("data/analysis/benchmark_results.parquet")
df3


# In[ ]:





# In[ ]:


import os
# load data/processed/modelcard_step3_dedup.parquet
data_type = "modelcard_step3"
processed_base_path = "data/processed"
df_merged = pd.read_parquet(os.path.join(processed_base_path, f"{data_type}_dedup.parquet"),
                                columns=['modelId', 'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup', 'hugging_table_list_dedup', 'github_table_list_dedup'])
df_merged.info()


# In[ ]:


import numpy as np
import pandas as pd
df2 = pd.read_parquet("data/analysis/all_title_list_valid.parquet")
df2[['all_title_list', 'all_title_list_valid']].iloc[0]
#print(df2['all_title_list'].iloc[0])
#df2['all_title_list_valid'].iloc[0]

# can you compute the count of the list of these two keys, for each row
df2['all_title_list_count'] = df2['all_title_list'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
df2['all_title_list_valid_count'] = df2['all_title_list_valid'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
df2[['all_title_list', 'all_title_list_count', 'all_title_list_valid', 'all_title_list_valid_count']].head()


# In[ ]:


df2['all_title_list'].iloc[0]


# In[ ]:


# turn df2['all_title_list'] (np.ndarray) to a list
tmp_all_list = df2['all_title_list'].apply(list).explode().tolist()
tmp_all_list


# In[ ]:


key = 'all_title_list'
tmp_all_list = df2[key].apply(list).explode().tolist()
tmp_all_list_unique = pd.unique(df2[key].explode()).tolist()
print(len(tmp_all_list))
print(len(tmp_all_list_unique))


# In[ ]:


key = 'all_title_list_valid'
tmp_all_list = df2[key].apply(list).explode().tolist()
tmp_all_list_unique = pd.unique(df2[key].explode()).tolist()
print(len(tmp_all_list))
print(len(tmp_all_list_unique))


# In[ ]:


# load step4.parquet
df4 = pd.read_parquet("data/processed/modelcard_step4.parquet")
df4


# In[ ]:


df4.columns


# In[ ]:





# In[ ]:


# df3['query']制作一个list
tmp_list = df3['query'].tolist()
print(len(tmp_list))
# 'gooaq: open question answering with diverse answer types'是否在df3的list中
if 'gooaq: open question answering with diverse answer types' in tmp_list:
    print("Found!")
else:
    print("Not Found!")


# In[ ]:


# test which row has all_title_list_count>0 but all_title_list_valid_count as 0
df2[df2['all_title_list_count'] > 0][df2['all_title_list_valid_count'] == 0]


# In[ ]:


type(df2['all_title_list_valid'].iloc[0])


# In[ ]:


# load data/processed/final_integration_with_paths.parquet
df3 = pd.read_parquet("data/processed/final_integration_with_paths.parquet")
df3


# In[36]:


import pickle
def load_relationships(path):
    """Factory loader for paperId‑level relationship graphs."""
    with open(path, "rb") as f:
        return pickle.load(f)

tmp = load_relationships("data/processed/modelcard_citation_overlap_rate.pickle")


# In[ ]:


tmp


# In[34]:


def query_related_paper_ids(score_matrix, paper_index_list, paper_id, threshold=1.0):
    """
    Given a paper ID, return related paper IDs with score >= threshold.
    
    Parameters:
    - score_matrix: csr_matrix of shape (n_papers, n_papers)
    - paper_index_list: list of paper IDs (length = n_papers)
    - paper_id: str, the paper ID to query
    - threshold: float, minimum score to consider related

    Returns:
    - List of related paper IDs
    """
    try:
        paper_idx = paper_index_list.index(paper_id)
    except ValueError:
        raise ValueError(f"Paper ID {paper_id} not found in paper_index_list.")
    
    row = score_matrix.getrow(paper_idx)  # shape (1, n_papers)
    related_indices = row.indices[row.data >= threshold]  ######## vectorized filtering
    related_ids = [paper_index_list[i] for i in related_indices]
    return related_ids


# In[ ]:


# 你的数据
paper_index = tmp['paper_index']
score_matrix = tmp['score_matrix']

# 选一个 paper ID 查询，比如第一个
target_paper_id = paper_index[0]

related_ids = query_related_paper_ids(score_matrix, paper_index, target_paper_id)
print(f"{target_paper_id} is related to:")
for rid in related_ids:
    print(rid)


# In[ ]:


# load single_citations_204e3073870fae3d05bcbc2f6a8e263d9b72e776.parquet thanks
import pandas as pd
dfm = pd.read_parquet("titles_to_ids.parquet")
dfm


# In[ ]:


dfm['citingPaper'].iloc[0]


# In[ ]:


dfm5 = pd.read_parquet("single_references_cache.parquet")
dfm6 = pd.read_parquet("single_citations_cache.parquet")


# In[ ]:


print(json.dumps(json.loads(dfm6['parsed_response'].iloc[1]), indent=4))


# In[ ]:


# load single_references_204e3073870fae3d05bcbc2f6a8e263d9b72e776.parquet
dfm2 = pd.read_parquet("batch_results.parquet")
dfm2


# In[ ]:


print(json.dumps(json.loads(dfm2['parsed_response'].iloc[0]), indent=4))


# In[ ]:


# titles_to_ids.parquet
dfm3 = pd.read_parquet("single_references_row.parquet")
dfm3


# In[ ]:


print(dfm3['parsed_response'].iloc[0])


# In[ ]:


# laod modelcard_citation_enriched.parquet
dfm4 = pd.read_parquet("data/processed/modelcard_citation_enriched.parquet")
dfm4


# In[ ]:


# load data/porcessed/s2orc_titles2ids.parquet
dfm5 = pd.read_parquet("data/processed/s2orc_titles2ids.parquet")
dfm5


# In[ ]:


final_merged_df


# In[ ]:


print(json.dumps(json.loads(final_merged_df.parsed_response_reference.iloc[0]),indent=4))


# In[ ]:


import pandas as pd
import json  ######## 新增：用于解析 JSON
from pathlib import Path
from collections import Counter  ######## 新增：用于统计 intents

def print_key_stats(df, key, df_name):  ######## 新的辅助函数，用于打印键的统计信息
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
    合并四个 DataFrame，其中 query_results 作为主表（保持原有行数）。
    对于 query_results 与 titles2ids（两个表中都存在重复的 'paperId'），
    通过添加计数器 'occ' 保证使用 (paperId, occ) 键进行一对一合并。
    citations_cache 和 references_cache 含有唯一键，但字段含义不同，
    合并时分别添加后缀 '_citation' 和 '_reference' 来区分。
    
    最终合并后，由于 query_results 本身已经包含了一组引用与参考字段，
    为避免重复，我们删除 query_results 中的对应字段，
    保留来自 citations_cache/references_cache 的数据；
    同时删除合并后产生的标题重复字段（后缀 _titles）。
    
    参数:
        query_results (pd.DataFrame): 查询结果主表（允许重复）。
        titles2ids (pd.DataFrame): 标题到 ID 的映射数据（重复与 query_results 对齐）。
        citations_cache (pd.DataFrame): 引用响应（唯一键）。
        references_cache (pd.DataFrame): 参考文献响应（唯一键）。
    
    返回:
        pd.DataFrame: 最终合并后的 DataFrame，行数与 query_results 相同。
    """
    # 打印各个表的键统计信息
    print_key_stats(query_results, "paperId", "query_results")
    print_key_stats(titles2ids, "paperId", "titles2ids")
    print_key_stats(citations_cache, "paperId", "citations_cache")
    print_key_stats(references_cache, "paperId", "references_cache")

    # -------------------------
    # 利用计数器 'occ' 合并 query_results 与 titles2ids，使得重复记录能一对一对应
    # -------------------------
    query_results = query_results.copy()
    query_results["occ"] = query_results.groupby("paperId").cumcount()  ######## 为 query_results 添加计数器
    
    titles2ids = titles2ids.copy()
    titles2ids["occ"] = titles2ids.groupby("paperId").cumcount()  ######## 为 titles2ids 添加计数器
    
    # 使用 (paperId, occ) 键进行合并，重复出现字段自动加后缀 _titles
    merged_main = pd.merge(query_results, titles2ids, on=["paperId", "occ"],
                           suffixes=("", "_titles"), how="left")  ######## 合并 query_results 和 titles2ids
    
    # 删除计数器字段
    merged_main = merged_main.drop(columns=["occ"])  ######## 删除计数器字段

    # -------------------------
    # 合并 citations_cache 和 references_cache，两个表的键唯一，但字段重叠，为区分添加不同后缀
    # -------------------------
    merged_aux = pd.merge(citations_cache, references_cache, on="paperId",
                          suffixes=("_citation", "_reference"), how="outer")  ######## 合并 citations 和 references 数据，添加后缀

    # -------------------------
    # 最后将辅助数据（citations + references）合并到主表中，合并键仅用 paperId
    # -------------------------
    final_df = pd.merge(merged_main, merged_aux, on="paperId", how="left")  ######## 最终合并得到完整表

    # 删除通过 query_results 得到的重复引用与参考字段，
    # 如果这些字段与最终辅助数据提供的字段内容一致，则可以删除它们
    redundant_cols = [
        "original_response_citations",
        "parsed_response_citations",
        "original_response_references",
        "parsed_response_references"
    ]
    final_df = final_df.drop(columns=redundant_cols, errors="ignore")  ######## 删除重复的引用与参考字段

    # 删除标题合并后重复的字段，如 query_title_titles, retrieved_title_titles, corpusId_titles, paper_identifier_titles
    final_df = final_df.drop(columns=[col for col in final_df.columns if col.endswith('_titles')])  ######## 删除重复的标题字段

    print("Final merged DataFrame shape:", final_df.shape)  ######## 打印最终表的形状
    return final_df

def parse_cited_papers(json_str):
    """
    解析传入的 JSON 字符串，返回包含两组信息：
      - methodology：包含 "methodology" 的 paperId list 和 contexts list
      - background：包含 "background" 的 paperId list 和 contexts list
    如果解析失败或数据为空，则返回空列表。
    
    返回:
        tuple: (method_ids, method_contexts, background_ids, background_contexts)
    """
    method_ids = []
    method_contexts = []
    background_ids = []
    background_contexts = []
    
    if pd.isna(json_str) or not isinstance(json_str, str):
        return method_ids, method_contexts, background_ids, background_contexts
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
    return method_ids, method_contexts, background_ids, background_contexts

def count_intents(final_df, col_name="parsed_response_reference"):
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

if __name__ == "__main__":
    folder = "data/processed"
    
    """citations_cache = pd.read_parquet(Path(folder) / "s2orc_citations_cache.parquet")  ########
    references_cache = pd.read_parquet(Path(folder) / "s2orc_references_cache.parquet")  ########
    query_results = pd.read_parquet(Path(folder) / "s2orc_query_results.parquet")  ########
    titles2ids = pd.read_parquet(Path(folder) / "s2orc_titles2ids.parquet")  ########

    final_merged_df = merge_dataframes(query_results, titles2ids, citations_cache, references_cache)  ########"""

    new_cols = final_merged_df["parsed_response_reference"].apply(lambda x: pd.Series(parse_cited_papers(x), 
                                                     index=["cited_papers_methodology_ids", 
                                                            "cited_papers_methodology_contexts", 
                                                            "cited_papers_background_ids", 
                                                            "cited_papers_background_contexts"]))
    final_merged_df = pd.concat([final_merged_df, new_cols], axis=1)

    print("Example parsed cited_papers (reference):")
    print(json.dumps(json.loads(final_merged_df["parsed_response_reference"].dropna().iloc[0]), indent=4))
    print("Methodology paper IDs (first row):", final_merged_df["cited_papers_methodology_ids"].iloc[0])
    print("Methodology contexts (first row):", final_merged_df["cited_papers_methodology_contexts"].iloc[0])
    print("Background paper IDs (first row):", final_merged_df["cited_papers_background_ids"].iloc[0])
    print("Background contexts (first row):", final_merged_df["cited_papers_background_contexts"].iloc[0])

    intents_counter = count_intents(final_merged_df)
    print("Intents Count results:")
    for intent, count in intents_counter.items():
        print(f"{intent}: {count}")


# In[ ]:


final_merged_df


# In[ ]:


def analyze_intent_influential_correlation(json_series):
    """
    统计传入的 JSON 字符串序列（例如 parsed_response_reference 列）中，
    每个 cited_papers 的 intents 与 isInfluential 同时出现的情况。
    
    对于每一个 JSON 字符串，解析出其中 cited_papers 列表，然后对每一个
    cited paper，依次获取其 'intents' 列表和 'isInfluential' 布尔值。
    最终返回一个字典，格式如下：
      {
         "methodology": {"True": <数值>, "False": <数值>},
         "background": {"True": <数值>, "False": <数值>},
         其他 intent: { ... }
      }
      
    参数:
        json_series (pd.Series): 包含 JSON 字符串的 pandas Series，每个字符串应为
            {
                "cited_papers": [
                    {
                        "intents": [ ... ],
                        "isInfluential": <bool>,
                        ... 
                    },
                    ...
                ]
            }
            
    返回:
        dict: 统计结果的字典，如上所示。
    """
    from collections import defaultdict
    import json

    result = defaultdict(lambda: {"True": 0, "False": 0})
    
    # 遍历每个非空的 JSON 字符串
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
            print("解析 JSON 时出错:", e)
    return dict(result)


final_merged_df_2 = analyze_intent_influential_correlation(final_merged_df["parsed_response_reference"])
print("Intent 与 isInfluential 统计结果:")
for intent, counts in final_merged_df_2.items():
    print(f"{intent}: {counts}")
# 统计每个 intent 的出现次数
intent_counts = Counter()
for json_str in final_merged_df["parsed_response_reference"].dropna():
    try:
        data = json.loads(json_str)
        cited_papers = data.get("cited_papers", [])
        for item in cited_papers:
            intents = item.get("intents", [])
            intent_counts.update(intents)
    except Exception as e:
        print(f"Error parsing JSON in count_intents: {e}")
print("Intent 统计结果:")
for intent, count in intent_counts.items():
    print(f"{intent}: {count}")


# In[ ]:


final_merged_df_3 = analyze_intent_influential_correlation(final_merged_df["parsed_response_citation"])
print("Intent 与 isInfluential 统计结果:")
for intent, counts in final_merged_df_3.items():
    print(f"{intent}: {counts}")
# 统计每个 intent 的出现次数
intent_counts = Counter()
for json_str in final_merged_df["parsed_response_citation"].dropna():
    try:
        data = json.loads(json_str)
        cited_papers = data.get("cited_papers", [])
        for item in cited_papers:
            intents = item.get("intents", [])
            intent_counts.update(intents)
    except Exception as e:
        print(f"Error parsing JSON in count_intents: {e}")
print("Intent 统计结果:")
for intent, count in intent_counts.items():
    print(f"{intent}: {count}")


# In[ ]:


# use citation paperId to query for files thanks


# In[ ]:


#!/usr/bin/env python
"""
Script to load and check the schema of the following Parquet files:
  - extracted_annotations.parquet
  - tmp_extracted_lines.parquet
  - merged_df.parquet

Usage:
    python check_schema.py
"""

import pandas as pd

def print_schema(file_path):
    """
    Load a Parquet file and print its shape, schema (dtypes), column names, and a sample of rows.
    
    Parameters:
        file_path (str): Path to the Parquet file.
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"File: {file_path}")
        print("Shape:", df.shape)
        print("Schema (dtypes):")
        print(df.dtypes)
        print("\nColumns:", list(df.columns))
        print("\nSample Data:")
        print(df.head(), "\n")
        print("=" * 80, "\n")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

def main():
    parquet_files = [
        "query_cache.parquet",
        "s2orc_rerun.parquet",
        "extracted_annotations.parquet"
    ]
    
    for file_path in parquet_files:
        print_schema("data/processed/" + file_path)

if __name__ == "__main__":
    main()


# In[ ]:


# load missing_queries_rows.parquet
df_missing = pd.read_parquet("missing_queries_rows.parquet")
df_missing


# In[ ]:


df_missing_tmp = df_missing[(df_missing['score'] > 100) & (df_missing['retrieved_title'] == df_missing['query'])]
df_missing_tmp


# In[ ]:


df_missing_tmp.iloc[0].query


# In[ ]:


df_missing_tmp.iloc[0].retrieved_title


# In[ ]:


df_missing_tmp.iloc[0].query_processed


# In[ ]:


len(df_missing_tmp)


# In[ ]:


ls data/processed


# In[9]:


# plaes load this extracted_annotations to check the schema
import pandas as pd
df = pd.read_parquet("data/processed/extracted_annotations.parquet")
df.head(5)


# In[21]:


def extract_influential_references(ref_json_str):
    """
    Parse JSON from a single column cell (e.g., original_response_reference),
    returning a list of paperIds with isInfluential == True.
    """
    try:
        parsed = json.loads(ref_json_str)
        if isinstance(parsed, dict) and "data" in parsed:
            # Keep only references that have "isInfluential" == True
            influential_ids = []
            for item in parsed["data"]:
                if item.get("isInfluential") is True and "citedPaper" in item:
                    cp = item["citedPaper"]
                    if "paperId" in cp:
                        influential_ids.append(cp["paperId"])
            return influential_ids
        else:
            # Old format or unexpected JSON structure
            return []
    except Exception:
        return []

df["influential_reference_ids"] = df["original_response_reference"].apply(
        extract_influential_references
    )
df["influential_reference_ids"]


# In[34]:


def parse_overall_with_empty_intents(json_str):
    """
    Extract overall cited paper IDs and contexts from parsed_response_reference.
    Unlike the original parse_cited_papers, this version **includes all papers**
    even if the 'intents' field is empty.

    Returns:
        tuple: (overall_ids, overall_contexts)
    """
    overall_ids = []
    overall_contexts = []
    
    if pd.isna(json_str) or not isinstance(json_str, str):
        return overall_ids, overall_contexts
    
    try:
        data = json.loads(json_str)
        cited_papers = data.get("cited_papers", [])
        for item in cited_papers:
            cited_paper = item.get("citedPaper", {})
            paper_id = cited_paper.get("paperId")
            contexts = item.get("contexts", [])
            if paper_id is not None:
                overall_ids.append(paper_id)
                overall_contexts.append(contexts)
    except Exception as e:
        print(f"[parse_overall_with_empty_intents] JSON parse error: {e}")
    
    return overall_ids, overall_contexts

overall_cols = df["parsed_response_reference"].apply(
    lambda x: pd.Series(parse_overall_with_empty_intents(x),
        index=["cited_papers_overall_ids", "cited_papers_overall_contexts"]
    )
)
df.update(overall_cols)


# In[35]:


df.columns


# In[36]:


df[["cited_papers_overall_ids", "influential_reference_ids"]]


# In[37]:


for idx, (overall, influential) in df[["cited_papers_overall_ids", "influential_reference_ids"]].iterrows():
    # Convert to set and remove None safely
    influential_set = set(influential) - {None} if influential is not None else set()
    overall_set = set(overall) - {None} if overall is not None else set()

    # Skip if influential is empty after cleaning
    if len(influential_set) == 0:
        continue
    # Raise if overall is still empty
    if len(overall_set) == 0:
        raise AssertionError(
            f"Row {idx} has influential refs but no valid overall refs.\n"
            f"> influential_reference_ids: {influential}\n"
            f"> cited_papers_overall_ids: {overall}"
        )

    # Check subset condition
    if not influential_set.issubset(overall_set):
        raise AssertionError(
            f"Row {idx} has influential_reference_ids not contained in cited_papers_overall_ids.\n"
            f"> Missing items: {influential_set - overall_set}\n"
            f"> influential_reference_ids: {influential}\n"
            f"> cited_papers_overall_ids: {overall}"
        )


# In[25]:


df.columns


# In[20]:


import json
print(json.dumps(json.loads(df['original_response_reference'].iloc[0]),indent=4))


# In[19]:


import json
print(json.dumps(json.loads(df['original_response_citation'].iloc[0]),indent=4))


# In[15]:


# load final_integration_with_paths
df2 = pd.read_parquet("data/processed/final_integration_with_paths.parquet")
df2


# In[17]:


df2.columns


# In[18]:


df3.columns


# In[24]:


import json
print(json.dumps(json.loads(df3['original_response'].iloc[0]),indent=4))


# In[25]:


import json
print(json.dumps(json.loads(df['original_response_citation'].iloc[0]),indent=4))


# In[26]:


import json
print(json.dumps(json.loads(df['original_response_reference'].iloc[0]),indent=4))


# In[41]:


result1 = load_paperId_lists(df, mode="reference")
result1


# In[40]:


result2 = load_paperId_lists(df, mode="citation")
result2


# In[13]:


# compare the result1, result2, 
# load final_integration_with_paths
import pandas as pd
df2 = pd.read_parquet("data/processed/final_integration_with_paths.parquet")
df2


# In[16]:


# get the query with bert
df2[df2['query'] == 'bert: pre-training of deep bidirectional transformers for language understanding']['llm_response_raw'].iloc[0]


# In[17]:


df2[df2['query'] == 'bert: pre-training of deep bidirectional transformers for language understanding'].columns


# In[19]:


df2[df2['query'] == 'bert: pre-training of deep bidirectional transformers for language understanding'].extracted_tables.iloc[0]


# In[ ]:


'```markdown\n| Image Sources       | Original Distribution           |         | Sampled Distribution           |         |\n|---------------------|-------------------------------|---------|-------------------------------|---------|\n|                     | MOS ∈ [0, 100)               | Size    | µMOS  | σMOS  | Size    | µMOS  | σMOS  |\n| KonIQ-10k [14]     | 10,073                       | 58.73   | 15.43 | 5,182   | 49.53  | 15.72  |\n| SPAQ [9]           | 11,125                       | 50.32   | 20.90 | 10,797  | 49.46  | 20.63  |\n| LIVE-FB [60]       | 39,810                       | 72.13   | 6.16  | 80      | 60.68  | 17.38  |\n| LIVE-itw [12]      | 1,169                        | 55.38   | 20.27 | 200     | 55.70  | 19.83  |\n| AGIQA-3K [26]      | 2,982                        | 50.00   | 19.80 | 400     | 40.80  | 21.80  |\n| ImageRewardDB [58] | 50,000                       | -w/o MOS| -584  | -w/o MOS| -15    | -distortion|\n| COCO [5]           | 330,000                      | -w/o MOS| -1,012| -w/o MOS| -      | -      |\n| Overall             | 445,159                      | 65.02   | 16.51 | 18,973  | 49.87  | 19.08  |\n```\n```markdown\n| Model (variant)        | Q-Instruct Strategy            | Yes-or-No   | What      | How       | Distortion | Other      | I-C Distortion | I-C Other  | Overall    |\n|------------------------|--------------------------------|--------------|-----------|-----------|------------|------------|----------------|------------|------------|\n| random guess           | -50.00%                       | 27.86%      | 33.31%    | 37.89%    | 38.48%     | 38.28%     | 35.82%        | 37.80%     |\n| no (Baseline)          |                                | 66.36%      | 58.19%    | 50.51%    | 49.42%     | 65.74%     | 54.61%        | 70.61%     | 58.66%     |\n| LLaVA-v1.5 (7B) (a)   | mix with high-level           | 76.18%      | +9.82%    | 66.37%    | +8.18%     | 57.61%     | +7.10%        | 65.18%     | +15.76%    |\n|                        |                                | 67.59%      | +1.85%    | 64.80%    | +10.19%    | 73.06%     | +2.55%        | 67.09%     | +8.43%     |\n|                        | (b) after high-level          | 76.91%      | +10.45%   | 65.04%    | +6.85%     | 55.78%     | +5.27%        | 64.01%     | +14.59%    |\n|                        |                                | 67.13%      | +1.39%    | 64.80%    | +10.19%    | 71.84%     | +1.23%        | 66.35%     | +7.69%     |\n| no (Baseline)          |                                | 65.27%      | 64.38%    | 56.59%    | 56.03%     | 67.13%     | 61.18%        | 67.35%     | 62.14%     |\n| LLaVA-v1.5 (13B) (a)  | mix with high-level           | 76.18%      | +10.91%   | 65.71%    | +1.33%     | 59.23%     | +2.64%        | 64.39%     | +8.36%     |\n|                        | (b) after high-level          | 76.36%      | +11.09%   | 65.04%    | +0.66%     | 58.42%     | +1.83%        | 65.56%     | +9.53%     |\n|                        |                                | 66.44%      | -0.69%    | 64.47%    | +3.29%     | 74.29%     | +6.94%        | 67.02%     | +4.88%     |\n| no (Baseline)          |                                | 72.18%      | 57.96%    | 56.19%    | 56.68%     | 69.21%     | 53.29%        | 72.65%     | 61.61%     |\n| mPLUG-Owl-2 (a)       | mix with high-level           | 75.64%      | +3.46%    | 67.04%    | +9.08%     | 59.03%     | +2.84%        | 71.01%     | +14.33%    |\n|                        | (b) after high-level          | 76.00%      | +3.82%    | 65.04%    | +7.08%     | 61.66%     | +5.47%        | 65.95%     | +9.27%     |\n|                        |                                | 68.75%      | -0.46%    | 65.46%     | +12.17%    | 73.88%     | +1.23%        | 67.96%     | +6.35%     |\n| InternLM-XComposer-VL  | no (Baseline) (a) mix with high-level | 76.73% | +7.28% | 69.45%    |            |            |                |            |            |\n|                        | (b) after high-level          | 78.36%      | +8.91%    | 65.27%    | 69.91%     | +4.64%     | 63.89%        | +3.04%     | 60.85%     |\n|                        |                                | 68.58%      | +3.31%    | 63.08%    | +2.23%     | 61.67%     | 70.23%        | +8.56%     | 65.37%     |\n|                        |                                | 70.14%      | 71.53%    | +1.39%    | 73.15%     | +3.01%     | 56.91%        | +10.52%    | 68.42%     |\n|                        |                                | +11.51%    | 75.10%    | 72.65%    | -2.45%     | 70.43%      | +5.08%        | 65.35%     | 78.37%     |\n|                        |                                | +3.27%     | 70.37%    | +5.02%    |            |            |                |            |            |\n```\n```markdown\n| Model (variant)        | Q-Instruct Strategy            | completeness | precision  | relevance | sum      |\n|------------------------|--------------------------------|--------------|------------|-----------|----------|\n| no (Baseline)          |                                | 0.90         | 1.13       | 1.18      | 3.21     |\n| LLaVA-v1.5 (7B) (a)   | mix w/ high-level              | 1.12         | 1.17       | 1.57      | 3.86     |\n|                        | (b) after high-level           | 1.11         | 1.16       | 1.54      | 3.82     |\n| no (Baseline)          |                                | 0.91         | 1.28       | 1.29      | 3.47     |\n| LLaVA-v1.5 (13B) (a)  | mix w/ high-level              | 1.14         | 1.29       | 1.58      | 4.01     |\n|                        | (b) after high-level           | 1.13         | 1.26       | 1.61      | 4.00     |\n| no (Baseline)          |                                | 1.06         | 1.24       | 1.36      | 3.67     |\n| mPLUG-Owl-2 (a)       | mix w/ high-level              | 1.18         | 1.29       | 1.57      | 4.04     |\n|                        | (b) after high-level           | 1.16         | 1.27       | 1.57      | 3.99     |\n| InternLM-XComposer-VL  | no (Baseline) (a) mix w/ high-level | 1.03    | 1.16       | 1.18      | 3.56     |\n|                        | (b) after high-level           | 1.26         | 1.35       | 1.34      | 4.14     |\n| Average Improvement     |                                | +0.17        | +0.04      | +0.31     | +0.52    |\n```\n```markdown\n| Q-Instruct Strategy    | low-level dataset              | completeness  | precision   | relevance  | sum      |\n|------------------------|--------------------------------|---------------|-------------|------------|----------|\n| no (Baseline)          | None                           | 0.90          | 1.13        | 1.18       | 3.21     |\n| (a) mix w/ high-level  | only Q-Pathway full Q-Instruct | 1.07          | 1.12        | 1.13       | 3.74     |\n|                        |                                | 1.17          | 1.54        | 1.57       | 3.86     |\n| (b) after high-level   | only Q-Pathway full Q-Instruct | 1.02          | 1.11        | 1.12       | 3.69     |\n|                        |                                | 1.16          | 1.55        | 1.54       | 3.82     |\n```\n```markdown\n| Q-Instruct Strategy    | low-level dataset              | Yes-or-No     | What       | How        | Overall  |\n|------------------------|--------------------------------|----------------|------------|------------|----------|\n| no (Baseline)          | None                           | 64.6%         | 59.2%      | 55.8%      | 60.1%    |\n| (a) mix w/ high-level  | only VQA subset full Q-Instruct | 78.1%        | 78.7%      | 61.5%      | 64.0%    |\n|                        |                                | 63.8%         | 69.3%      | 61.5%      | 67.6%    |\n| (b) after high-level   | only VQA subset full Q-Instruct | 77.9%        | 78.5%      | 61.8%      | 63.3%    |\n|                        |                                | 58.9%         | 67.4%      | 56.8%      | 66.1%    |\n```\n```markdown\n| Q-Instruct Strategy    | Yes-or-No                     | What       | How        | Overall    |\n|------------------------|-------------------------------|------------|------------|------------|\n| no (Baseline)          | 64.6%                         | 59.2%      | 55.8%      | 60.1%      |\n| replace high-level      | (not adopted)                 | 75.0%      | 59.4%      | 56.4%      | 64.1%    |\n| mix with high-level    | (ours, strategy (a))         | 78.7%      | 64.0%      | 63.8%      |\n| after high-level       | (ours, strategy (b))         | 78.5%      | 63.3%      | 58.9%      | 67.4%    |\n```\n```markdown\n| Model (variant)        | Q-Instruct Strategy            | completeness | precision  | relevance  | sum      |\n|------------------------|--------------------------------|--------------|------------|------------|----------|\n| no (Baseline)          |                                | 0.90         | 1.13       | 1.18       | 3.21     |\n| LLaVA-v1.5 (7B) (a)   | mix w/ high-level              | 1.12         | 1.17       | 1.57       | 3.86     |\n|                        | (b) after high-level           | 1.11         | 1.16       | 1.54       | 3.82     |\n| no (Baseline)          |                                | 0.91         | 1.28       | 1.29       | 3.47     |\n| LLaVA-v1.5 (13B) (a)  | mix w/ high-level              | 1.14         | 1.29       | 1.58       | 4.01     |\n|                        | (b) after high-level           | 1.13         | 1.26       | 1.61       | 4.00     |\n| no (Baseline)          |                                | 1.06         | 1.24       | 1.36       | 3.67     |\n| mPLUG-Owl-2 (a)       | mix w/ high-level              | 1.18         | 1.29       | 1.57       | 4.04     |\n|                        | (b) after high-level           | 1.16         | 1.27       | 1.57       | 3.99     |\n| InternLM-XComposer-VL  | no (Baseline) (a) mix w/ high-level | 1.03    | 1.16       | 1.18       | 3.56     |\n|                        | (b) after high-level           | 1.26         | 1.35       | 1.34       | 4.14     |\n| Average Improvement     |                                | +0.17        | +0.04      | +0.31      | +0.52    |\n```'


# In[140]:


df2[df2['query'].str.contains('roberta')]


# In[ ]:


#print(json.dumps(json.loads(df2[df2['query'].str.startswith('roberta: ')].original_response_citation.iloc[0]),indent=4))
print(json.dumps(json.loads(df2[df2['query'].str.startswith('roberta: ')].original_response_citation.iloc[0]),indent=4))


# In[135]:


print(json.dumps(json.loads(df2[df2['query'].str.startswith('bert: pre')].original_response_reference.iloc[0]),indent=4))


# In[127]:


print(json.dumps(json.loads(df2[df2['query'].str.startswith('bert: pre')].raw_json.iloc[0]),indent=4))


# In[20]:


{"id": "batch_req_67feee4d9ae8819082903a28aa76cf39", "custom_id": "1067", "response": {"status_code": 200, "request_id": "09390940647cf4f2487ef43f21f35ae2", "body": {"id": "chatcmpl-BMecboWasVFIWoyadZP5gVVAB45ct", "object": "chat.completion", "created": 1744739065, "model": "gpt-4o-mini-2024-07-18", "choices": [{"index": 0, "message": {"role": "assistant", "content":
 "```markdown
 | System                       | MNLI-(m/mm) | QQP  | NLI   | SST-2 | CoLA   | STS-B | MRPC | TE Average |
 |------------------------------|--------------|------|-------|-------|--------|-------|------|------------|
 |                              | 392k        | 363k | 108k  | 67k   | 8.5k   | 5.7k  | 3.5k | 2.5k       |
 | Pre-OpenAI SOTA              | 80.6/80.1   | 66.1 | 82.3  | 93.2  | 35.0   | 81.0  | 86.0 | 61.7       |
 | BiLSTM+ELMo+Attn             | 76.4/76.1   | 64.8 | 79.8  | 90.4  | 36.0   | 73.3  | 84.9 | 56.8       |
 | OpenAI GPT                   | 82.1/81.4   | 70.3 | 87.4  | 91.3  | 45.4   | 80.0  | 82.3 | 56.0       |
 | BERTBASE                     | 84.6/83.4   | 71.2 | 90.5  | 93.5  | 52.1   | 85.8  | 88.9 | 66.4       |
 | BERTLARGE                    | 86.7/85.9   | 72.1 | 92.7  | 94.9  | 60.5   | 86.5  | 89.3 | 70.1       |
 ```\n
 
 ```markdown\n| System                                     | Dev Test EM F1 | EM F1 |\n|--------------------------------------------|----------------|-------|\n| Human                                      | --             | 82.3  | 91.2 |\n| #1 Ensemble -nlnet                         | --             | 86.0  | 91.7 |\n| #2 Ensemble -QANet                         | --             | 84.5  | 90.5 |\n| Published BiDAF+ELMo (Single)             | 85.6           | -     |\n| R.M. Reader (Ensemble)                     | 81.2           | 87.9  | 82.3 | 88.5 |\n| Ours BERTBASE (Single)                    | 80.8           | 88.5  | --    |\n| BERTLARGE (Single)                        | 84.1           | 90.9  | --    |\n| BERTLARGE (Ensemble)                       | 85.8           | 91.8  | --    |\n| BERTLARGE (Sgl.+TriviaQA)                  | 84.2           | 91.1  | 85.1  | 91.8 |\n| BERTLARGE (Ens.+TriviaQA)                  | 86.2           | 92.2  | 87.4  | 93.2 |\n| Human                                      | 86.3           | 89.0  | 86.9  | 89.5 |\n| #1 Single -MIR-MRC (F-Net)                | --             | 74.8  | 78.0 |\n| #2 Single -nlnet                           | --             | 74.2  | 77.1 |\n| Published unet (Ensemble)                  | --             | 71.4  | 74.9 |\n| SLQA+ (Single)                             | -              | 71.4  | 74.4 |\n| Ours BERTLARGE (Single)                   | 78.7           | 81.9  | 80.0  | 83.1 |\n```\n
 
 ```markdown
 | Dev Test          | ESIM+GloVe | ESIM+ELMo | OpenAI GPT | BERTBASE | BERTLARGE | Human (expert) | Human (5 annotations) |
 |------------------|------------|-----------|-------------|----------|------------|-----------------|-----------------------|
 |                  | 51.9      | 52.7      | 78.0        | 81.6     | 86.6      | -               | -                     |
 ```\n
 
 ```markdown
 | Hyperparams | Dev Set Accuracy | #L | #H | #A | LM (ppl) | MNLI-m | MRPC | SST-2 |
 |-------------|------------------|----|----|----|----------|--------|-------|-------|
 |             |                  | 768| 125|  47|  7.9     |  79.8  |  88.4 |  66.4 |
 |             |                  | 768|  35| 248|  6.6     |  82.2  |  90.7 |  70.1 |
 |             |                  | 768| 124| 688|  6.3     |  81.9  |  84.9 |  91.3 |
 |             |                  | 768| 123| 998|  4.8     |  84.4  |  86.7 |  92.9 |
 |             |                  | 1024| 163| 548|  5.7     |  86.9  |  99.3 |  93.7 |
 |             |                  | 1024| 163| 238|  5.0     |  86.6  |  87.3 |  89.1 |
 ```\n
 
 ```markdown
 | Masking Rates | Dev Set Results | MASK | SAME | RND | MNLI NER | Fine-tune | Fine-tune | Feature-based |
 |---------------|-----------------|------|------|-----|----------|-----------|-----------|---------------|
 |               |                 | 80%  |  10% | 10% | 84.2     | 95.4      | 94.9      |
 |               |                 | 100% |  0%  |  0% | 84.3     | 94.9      | 94.0      |
 |               |                 | 80%  |  0%  | 20% | 84.1     | 95.2      | 94.6      |
 |               |                 | 80%  | 20%  |  0% | 84.4     | 95.2      | 94.7      |
 |               |                 | 0%   | 100% |  0% | 83.7     | 94.8      | 94.6      |
 |               |                 | 80%  | 10%  | 10% | 84.1     | 95.4      | 94.9      |
 ```\n
 
 ```markdown\n| System                          | MNLI Dev Accuracy |
 |---------------------------------|------------------|
 | Effect of Number of Training Steps|                  |
 | BERT BASE                        | 1.0              |
 | MLM pre-training                 | 1.0              |
 | LTR pre-training                 | 1.0              |
 ```\n
 
 ```markdown\n| Masking Rates | Dev Set Results | MASK | SAME | RND | MNLI NER | Fine-tune | Fine-tune | Feature-based |
 |---------------|-----------------|------|------|-----|----------|-----------|-----------|---------------|
 |               |                 | 80%  |  10% | 10% | 84.2     | 95.4      | 94.9      |
 |               |                 | 100% |  0%  |  0% | 84.3     | 94.9      | 94.0      |
 |               |                 | 80%  |  0%  | 20% | 84.1     | 95.2      | 94.6      |
 |               |                 | 80%  | 20%  |  0% | 84.4     | 95.2      | 94.7      |
 |               |                 | 0%   | 100% |  0% | 83.7     | 94.8      | 94.6      |
 |               |                 | 80%  | 10%  | 10% | 84.1     | 95.4      | 94.9      |
 ```\n", "refusal": null, "annotations": []}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2356, "completion_tokens": 1899, "total_tokens": 4255, "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}, "service_tier": "default", "system_fingerprint": "fp_64e0ac9789"}}, "error": null}


# ```markdown
# | Masking Rates | Dev Set Results | MASK | SAME | RND | MNLI NER | Fine-tune | Fine-tune | Feature-based |
# |---------------|-----------------|------|------|-----|----------|-----------|-----------|---------------|
# |               |                 | 80%  |  10% | 10% | 84.2     | 95.4      | 94.9      |
# |               |                 | 100% |  0%  |  0% | 84.3     | 94.9      | 94.0      |
# |               |                 | 80%  |  0%  | 20% | 84.1     | 95.2      | 94.6      |
# |               |                 | 80%  | 20%  |  0% | 84.4     | 95.2      | 94.7      |
# |               |                 | 0%   | 100% |  0% | 83.7     | 94.8      | 94.6      |
# |               |                 | 80%  | 10%  | 10% | 84.1     | 95.4      | 94.9      |
# ```

# In[121]:


print(df2[df2['query'].str.startswith('bert: pre')].llm_response_raw.iloc[0])


# In[6]:


# load modelcard_citation_enriched
df3 = pd.read_parquet("data/processed/modelcard_citation_enriched.parquet")
df3


# In[3]:


df2.columns


# In[114]:





# In[113]:


df3[df3['modelId']=='google-bert/bert-base-uncased'].corpusid


# In[8]:


df3.columns


# In[5]:


df2['llm_table_list']


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


df.columns


# In[16]:


# load modelcard_citation_enriched
df3 = pd.read_parquet("data/processed/modelcard_citation_enriched.parquet")
df3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#print(json.dumps(json.loads(df['raw_json'].iloc[0]),indent=4))


# In[ ]:


# plaes load this extracted_annotations_b0413.parquet to check the schema
df1 = pd.read_parquet("data/processed/extracted_annotations_b0413.parquet")
df1


# In[7]:


# 指定需要检查的列
cols = [
    'extracted_tables', 
    'extracted_tablerefs', 
    'extracted_figures', 
    'extracted_figure_captions', 
    'extracted_figurerefs'
]

# 使用 apply 方法对每行统计非空列表的数量（即列表长度大于 0 的个数）
df['non_empty_count'] = df[cols].apply(lambda row: sum(1 for cell in row if len(cell) > 0), axis=1)  ########

# 打印结果以观察每行统计出的非空列表个数
print(df['non_empty_count'])
print(df[df['non_empty_count']>0])


# In[12]:


# load html_table.parquet
df_html = pd.read_parquet("data/processed/html_table.parquet")
df_html


# In[11]:


df_html.columns


# In[2]:


# load data/processed/modelcard_step4.parquet
import pandas as pd
data = pd.read_parquet("data/processed/modelcard_step4.parquet")
data


# In[9]:


tmp = data[data['modelId']=='google-bert/bert-base-uncased']
#tmp
print(tmp.html_table_list_mapped_dedup.iloc[0])
print(tmp.llm_table_list_mapped_dedup.iloc[0])
print(tmp.hugging_table_list_dedup.iloc[0])
print(tmp.github_table_list_dedup.iloc[0])


# In[13]:


tmp = data[data['modelId']=='FacebookAI/roberta-base']
#tmp
print(tmp.html_table_list_mapped_dedup.iloc[0])
print(tmp.llm_table_list_mapped_dedup.iloc[0])
print(tmp.hugging_table_list_dedup.iloc[0])
print(tmp.github_table_list_dedup.iloc[0])


# In[25]:


data_np=data


# In[ ]:


import numpy as np

# Scenario 1: structured ndarray with named fields
mask = (
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['github_table_list_dedup']]) &   ########
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['llm_table_list_mapped_dedup']]) &   ########
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['html_table_list_mapped_dedup']]) &   ########
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['hugging_table_list_dedup']])          ########
)
filtered = data_np[mask]
filtered



# In[27]:





# In[28]:


tmp = data[data['modelId']=='google/gemma-2-2b']
#tmp
print(tmp.html_table_list_mapped_dedup.iloc[0])
print(tmp.llm_table_list_mapped_dedup.iloc[0])
print(tmp.hugging_table_list_dedup.iloc[0])
print(tmp.github_table_list_dedup.iloc[0])


# In[54]:


# load s2orc_citations_cache.parquet
df12 = pd.read_parquet("data/processed/modelcard_citation_enriched.parquet")
df12.columns


# In[55]:


df12.head(5)


# In[57]:


# 标记哪些标题包含 "gemma"
mask = df12['title'].str.contains('gemma', case=False, na=False)  # ########
df_gemma = df12[mask]
df_gemma


# In[9]:


# load llm_markdown_table_results.parquet	
import pandas as pd
df1 = pd.read_parquet("data/processed/modelcard_step3_merged.parquet")
df1.columns


# In[10]:


import numpy as np

# Scenario 1: structured ndarray with named fields
data_np=df1
mask = (
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['github_table_list']]) &   ########
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['llm_table_list_mapped']]) &   ########
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['html_table_list_mapped']]) &   ########
    np.array([isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0 for x in data_np['hugging_table_list']])          ########
)
filtered = data_np[mask]
filtered


# In[11]:


df1[df1['modelId']=='google-bert/bert-base-uncased'].columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[98]:


df1[df1['modelId']=='google-bert/bert-base-uncased']['html_table_list_mapped'].iloc[0]


# In[100]:


df1[df1['modelId']=='google-bert/bert-base-uncased']['llm_table_list_mapped'].iloc[0]


# In[101]:


df1[df1['modelId']=='google-bert/bert-base-uncased']['github_table_list'].iloc[0]


# In[99]:


df1[df1['modelId']=='google-bert/bert-base-uncased']['hugging_table_list'].iloc[0]


# In[106]:


df1[df1['modelId']=='google-bert/bert-base-uncased']['github_link'].iloc[0]


# In[108]:


df1[df1['modelId']=='google-bert/bert-base-uncased']['pdf_link'].iloc[0]


# In[ ]:





# In[ ]:





# In[12]:


df1[df1['corpusid']==268857227][['extracted_tables', 'extracted_figures', 'llm_response_raw']]


# In[85]:


tmp = df1[df1['corpusid']==268857227]['extracted_tables'].iloc[0]
tmp[0]


# In[ ]:


#


# In[86]:


print(df1[df1['corpusid']==268857227]['llm_response_raw'].iloc[0])


# In[81]:


df1[df1['corpusid']==268379206]


# In[ ]:





# In[65]:


import json
print(json.dumps(json.loads(df_gemma[df_gemma['corpusId']==268379206].parsed_response.iloc[0]), indent=4))


# In[ ]:





# In[ ]:





# In[29]:


# load data/processed/modelcard_step4.parquet
import pandas as pd
data2 = pd.read_parquet("data/processed/modelcard_step3_dedup.parquet")
data2


# In[31]:


data2[data2['modelId']=='google/gemma-2-2b'].columns


# In[34]:


print(data2[data2['modelId']=='google/gemma-2-2b'][['github_link', 'pdf_link', 'title_arxiv', 'title_rxiv']].iloc[0])


# In[35]:


print(data2[data2['modelId']=='google/gemma-2-2b']['github_link'].iloc[0])


# In[41]:


print(data2[data2['modelId']=='google/gemma-2-2b']['pdf_link'].iloc[0])


# In[42]:


print(data2[data2['modelId']=='google/gemma-2-2b']['card'].iloc[0])


# In[45]:


# laod modelcard_citation_API.parquet   
df3 = pd.read_parquet("data/processed/modelcard_step3_merged.parquet")
df3


# In[ ]:





# In[46]:


df3[df3['modelId']=='google/gemma-2-2b'].columns


# In[47]:


# load modelcard_step1.parquet
df4 = pd.read_parquet("data/processed/modelcard_step1.parquet")
df4


# In[48]:


df4[df4['modelId']=='google/gemma-2-2b'].columns


# In[51]:


print(df4[df4['modelId']=='google/gemma-2-2b']['card'].iloc[0])


# In[143]:


import os
import pandas as pd
import json

# Set your prefix and base directory here
prefix = ""  # or "_429"
base_dir = "data/old_s2orc"

# Define all target files
files = {
    "TITLES_JSON_FILE": f"{base_dir}/modelcard_dedup_titles{prefix}.json",
    "TITLES_CACHE_FILE": f"{base_dir}/s2orc_titles2ids{prefix}.parquet",
    "BATCH_CACHE_FILE": f"{base_dir}/s2orc_batch_results{prefix}.parquet",
    "CITATIONS_CACHE_FILE": f"{base_dir}/s2orc_citations_cache{prefix}.parquet",
    "REFERENCES_CACHE_FILE": f"{base_dir}/s2orc_references_cache{prefix}.parquet",
    "MERGED_RESULTS_FILE": f"{base_dir}/s2orc_query_results{prefix}.parquet",
}

# Iterate and print schema
for name, path in files.items():
    print(f"\n📄 {name} — {path}")
    if not os.path.exists(path):
        print("❌ File not found.")
        continue
    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:  # Assume list of dicts
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
        else:
            print("⚠️ Unsupported file type.")
            continue

        print(f"✅ Loaded: {len(df)} rows")
        print("📌 Schema:")
        for col in df.columns:
            print(f"  - {col}: {df[col].dtype}")
    except Exception as e:
        print(f"❌ Error reading {name}: {e}")


# In[ ]:





# In[3]:


# load modelcard_step3_merged.parquet
import pandas as pd
df = pd.read_parquet("data/processed/modelcard_step3_merged.parquet")
df


# In[2]:


# 假设已有 DataFrame df
# 1. 构造布尔掩码，匹配关键词（不区分大小写）
mask = df['card_tags_x'].str.contains(r'(?i)(base_model)', na=False)  ########

# 2. 用掩码过滤出符合条件的行
filtered_df = df[mask]  ########

# 3. 查看筛选结果
print(filtered_df.card_tags_x.iloc[0])


# In[5]:


import re
import pandas as pd

# 假设已有 DataFrame df，且包含 'card_tags_x' 列

# 1. 筛出所有包含 base_model 的行
mask = df['card_tags_x'].str.contains(r'(?i)base_model', na=False)          ########
filtered_df = df[mask].copy()                                              ########

# 2. 从 card_tags_x 文本里用正则提取 base_model 和 base_model_relation
filtered_df['extracted_base_model'] = filtered_df['card_tags_x'] \
    .str.extract(r'base_model:\s*([^\s]+)', flags=re.IGNORECASE)            ########
filtered_df['extracted_relation'] = filtered_df['card_tags_x'] \
    .str.extract(r'base_model_relation:\s*([^\s]+)', flags=re.IGNORECASE)   ########

# 3. 统计：有提到 base_model_relation 的 vs. 没提到的数量
cnt_with_relation    = filtered_df['extracted_relation'].notna().sum()      ########
cnt_without_relation = filtered_df['extracted_relation'].isna().sum()       ########

print(f"总共提到 base_model 的行数：{len(filtered_df)}")
print(f"显式写了 base_model_relation 的：{cnt_with_relation}")
print(f"未写 base_model_relation 的：{cnt_without_relation}")

# 4. 查看每种关系出现的次数（包括 NaN）
print("\n各关系类型分布：")
print(filtered_df['extracted_relation'].value_counts(dropna=False))

# 5. （可选）看一下前几行提取结果
print("\n示例提取：")
print(filtered_df[['extracted_base_model', 'extracted_relation']].head())


# In[21]:


import pandas as pd
df = pd.read_parquet('data/processed/llm_markdown_table_results.parquet')
df.head(1)


# In[22]:


df[['combined_text', 'llm_response_raw']]


# In[ ]:




