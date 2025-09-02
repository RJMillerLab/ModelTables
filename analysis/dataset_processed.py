#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import os, time

#df1 = pd.read_parquet(os.path.join('./data/processed', "modelcard_ext_title.parquet"))
#df2 = pd.read_parquet(os.path.join('./data/processed',  "modelcard_step3_arxiv.parquet"))
df1 = pd.read_parquet("merged_df.parquet")
df2 = pd.read_parquet("extracted_annotations.parquet")
df2


# In[5]:


# drop df2 duplicates
df2_tmp = df2.drop_duplicates(subset=['filename', 'line_index'])


# In[4]:


len(df2), len(df1)


# In[17]:


df2.columns


# In[16]:


df2['extracted_tables'].iloc[3]


# In[18]:


df2['extracted_tablerefs'].iloc[3]


# In[ ]:


# {'end': 60035, 'extracted_text': 'Mean0.7940.5720.5660.8510.8330.7860.7930.658 0.7510.7430.8090.7020.7500.8250.3480.7260.764Median0.8150.5650.5830.8710.8490.7880.8020.720 0.7730.7530.8370.7180.7660.8520.3330.7240.768Std.0.1470.2380.2340.1340.1360.1680.1760.220 0.1800.1590.1880.1940.1690.1770.2210.1640.152Mean rank7.213.313.33.55.36.96.511.89.48.95.611.09.24.416.110.49.1Median rank7.014.014.03.05.06.56.013.09.09.04.012.010.03.017.011.09.0Wins880325326122010338859244606847271031533696113771593712Losses566112111212273755224849877637194159147503101375854734', 'start': 59510},
       


# In[15]:


import pandas as pd
from pprint import pprint

df2 = pd.read_parquet("extracted_annotations.parquet")
data_tables = df2['extracted_tablerefs'].iloc[3]
data_figures = df2['extracted_figures'].iloc[3]

def parse_tab_number(item):
    tab_id = item.get('id', '')
    if isinstance(tab_id, str) and tab_id.startswith('tab_'):
        parts = tab_id.split('_', 1)
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
    return float('inf')

#data_tables_filtered
data_figures_filtered = [
    f for f in data_figures 
    if f.get('id') and isinstance(f['id'], str) and f['id'].startswith('tab')
]
#merged_tabs = data_tables_filtered + data_figures_filtered
merged_tabs = data_figures_filtered
merged_tabs_sorted = sorted(merged_tabs, key=parse_tab_number)
pprint(merged_tabs_sorted)


# In[10]:


data_table_original


# In[12]:


#data_tableref_with_id


# In[ ]:


df2.to_json("extracted_annotations_pretty.json", 
            orient="records", 
            indent=2,
            force_ascii=False)


# In[ ]:





# In[35]:


# 提取目标位置的数据（假设是列表）
target_data = df2['extracted_figures'].iloc[3]
processed_data = [
    {
        'start': entry['start'],  # 确保start在最前面
        **{k: v for k, v in entry.items() if k != 'start'}
    }
    for entry in target_data  # 遍历列表中的每个字典
    if entry.get('id', '').startswith('tab_')  # 过滤条件
]
print(f"共处理 {len(processed_data)} 条有效数据：")
for i, item in enumerate(processed_data[:1]):  # 查看前5条
    print(f"[{i}] {df2['title'].iloc[3]} {item['id']} [{item['start']}: {item['end']}]")
    print(f"{item['extracted_text']}...\n")


# In[ ]:





# ## Analysis on the cache.parquet after querying the title. Understand elastic search better!

# In[7]:


print(df1.columns)
print(df2.columns)


# 

# In[9]:


#dup_df4 = df4[df4.duplicated(subset=['filename', 'line_index'], keep=False)]
#print("df4 duplicates:\n", dup_df4.head())
#len(df4), len(dup_df4)


# In[10]:


#df3[(df3['filename'] == 'step101_file') & (df3['line_index'] == 2737)].iloc[0]['extracted_tables'].tolist()


# In[11]:


#df4[(df4['filename'] == 'step101_file') & (df4['line_index'] == 2737)]


# In[28]:


df4 = pd.read_parquet("extracted_annotations.parquet")
df3 = pd.read_parquet("merged_df.parquet")
merged_df = pd.merge(df3, df4, on=['filename', 'line_index'], how='inner')
#merged_df = merged_df.drop(columns=['corpusid_y', 'corpusid', 'modelid'])
print(len(df3), len(df4), len(merged_df))
merged_df.head(1)


# In[21]:


import pandas as pd
import os, time
df1 = pd.read_parquet('query_cache.parquet')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)
df_rank1 = df1[df1['rank'] == 1].reset_index(drop=True)
df_sorted = df_rank1.sort_values(by='score', ascending=False).reset_index(drop=True)
# 找出query和retrieved_title完全一致的exact match
exact_matches = df_sorted[
    df_sorted['query'].str.lower().str.strip('" ') ==
    df_sorted['retrieved_title'].str.lower().str.strip('" ')
]

assert not exact_matches.empty, "No exact match found between query and retrieved title."

# 计算exact match的最小score
min_exact_match_score = exact_matches['score'].min()
print(f"\nMinimum score of exact match: {min_exact_match_score}")

# 找到并打印出score最接近exact match最小score的前5个条目
items_around_min_score = df_sorted.iloc[
    (df_sorted['score'] - min_exact_match_score).abs().argsort()[:5]
]
print("\nItems around minimum exact match score:")
print(items_around_min_score[['query', 'retrieved_title', 'score', 'rank']])

######## 新增部分：找到分数在exact match以上，但并非完全匹配的情况
above_min_score_non_exact = df_sorted[
    (df_sorted['score'] >= min_exact_match_score) &
    (df_sorted['query'].str.lower().str.strip('" ') != df_sorted['retrieved_title'].str.lower().str.strip('" '))
]

print("\nItems above or equal to min exact match score but NOT exactly matched:")
print(above_min_score_non_exact[['query', 'retrieved_title', 'score', 'rank']])

######## 新增：检查低于min_exact_match_score的条目，有多少条目仍然符合"retrieved_title所有单词都出现在query里"的要求
below_min_score = df_sorted[df_sorted['score'] < min_exact_match_score]

def is_eligible(row):
    retrieved_words = row['retrieved_title'].lower().split()
    query_text = row['query'].lower()
    return all(word in query_text for word in retrieved_words)

eligible_below_min = below_min_score[below_min_score.apply(is_eligible, axis=1)]

######## 新增部分：同时计算整体符合条件的条目数
retrieved_in_query = df_sorted[df_sorted.apply(is_eligible, axis=1)]

# 打印统计结果
print(f"\nMinimum exact match score: {min_exact_match_score}")
print(f"Eligible items (retrieved_title all words in query): {len(retrieved_in_query)} out of {len(df_sorted)} total items")
print(f"Items below minimum exact match score that can still be saved: {len(eligible_below_min)}")

# 如需进一步检查，这里可输出 eligible_below_min 中的部分内容：
print("\nEligible items below min exact match score (sample):")
print(eligible_below_min[['query', 'retrieved_title', 'score', 'rank']].head())



# In[76]:


### !!! tmr present
new_title_cols = ["title_arxiv", "title_rxiv", "title_github_readme", "title_github_html", "title_github_bibtex", "title_pdf", "title_other", "parsed_bibtex_tuple_list_github"]
for col in new_title_cols:
    non_empty = df1[col].apply(lambda x: isinstance(x, (list, np.ndarray, tuple)) and len(x) > 0)
    proportion = non_empty.mean() * 100
    print(f"{col}: {proportion:.2f}% 非空")


# In[85]:


col = "title_arxiv"
non_empty = df1[col].apply(lambda x: isinstance(x, (list, np.ndarray, tuple)) and len(x) > 0)
#print(df1[non_empty])
df1[non_empty].iloc[0][[col]]


# In[74]:


tmp = df1[non_empty].iloc[0]['title_github_bibtex']
tmp


# In[108]:


# load json tmp_dedup_titles
import json
with open('tmp_dedup_titles.json', 'r') as f:
    tmp_dedup_titles = json.load(f)
tmp_dedup_titles


# In[120]:


import requests
import time
import json

def batch_search_titles_get_paper_ids(titles):
    """
    For each title, perform a fuzzy search to get paperId.
    """
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    title_to_paper_id = {}

    for title in titles:
        params = {
            "query": title,
            "fields": "paperId,title",
            "limit": 1
        }
        print(f"🔍 Searching title: {title}")
        resp = requests.get(search_url, params=params)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                pid = data[0].get("paperId")
                print(f"✅ Found paperId: {pid}")
                title_to_paper_id[title] = pid
            else:
                print("❌ No result.")
        else:
            print(f"❌ Error: {resp.status_code}")
        time.sleep(3)  # small delay to avoid rate limit

    return title_to_paper_id

def batch_fetch_metadata(paper_ids):
    """
    Use batch API to retrieve paper metadata.
    """
    batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    batch_results = {}

    print(f"\n🚀 Fetching metadata for {len(paper_ids)} papers via batch...")
    payload = {
        "ids": paper_ids,
        "fields": "title,authors,year,venue,externalIds,isOpenAccess"
    }
    resp = requests.post(batch_url, json=payload)
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        for item in data:
            pid = item.get("paperId")
            batch_results[pid] = item
    else:
        print(f"❌ Batch fetch failed: {resp.status_code}")
    
    return batch_results

if __name__ == "__main__":
    # Toy titles to search
    titles = [
        'meta-transformer: a unified framework for multimodal learning',
        "Robust Speech Recognition via Large-Scale Weak Supervision",
        'searchqa: a new q&a dataset augmented with context from a search engine',
    ]

    # Step 1: Search each title → get paperId
    title_to_pid = batch_search_titles_get_paper_ids(titles)

    # Step 2: Batch fetch metadata
    paper_ids = list(title_to_pid.values())
    pid_to_metadata = batch_fetch_metadata(paper_ids)

    # Step 3: Map back title → metadata
    print("\n📄 Final Results:")
    final_mapping = {}
    for title, pid in title_to_pid.items():
        meta = pid_to_metadata.get(pid, {})
        final_mapping[title] = meta
        print(f"\n🔗 Title: {title}")
        print(json.dumps(meta, indent=2))

    # Optional: Save results for review
    with open("toy_batch_query_results.json", "w", encoding="utf-8") as f:
        json.dump(final_mapping, f, ensure_ascii=False, indent=2)


# In[106]:


# load parquet
tmp = pd.read_parquet('tmp.parquet')
print(tmp.columns)
col = "all_title_list"
non_empty = tmp[col].apply(lambda x: isinstance(x, (list, np.ndarray, tuple)) and len(x) > 0)
#print(df1[non_empty])
tmp[non_empty].iloc[0][[col]]


# In[ ]:


import json

# 加载 cache 的通用函数
def load_cache(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# 分析 cache，统计总查询数和非空结果数
def analyze_cache(cache, cache_name):
    total = len(cache)
    # 针对 GitHub extraction cache 中每个 value 是字典情况，检查 readme_title 和 html_title 是否有非空值
    sample_val = next(iter(cache.values()), None)
    if isinstance(sample_val, dict):
        non_empty = sum(1 for v in cache.values() 
                        if (v.get("readme_title") and str(v.get("readme_title")).strip()) or 
                           (v.get("html_title") and str(v.get("html_title")).strip()))
    else:
        non_empty = sum(1 for v in cache.values() if isinstance(v, str) and v.strip()) # , "| bioRxiv"
    print(f"{cache_name} - Total queries: {total}, Non-empty entries: {non_empty}")

# 定义各个 cache 的路径（根据实际情况修改路径）
ARXIV_CACHE_PATH = "data/processed/arxiv_titles_cache.json"
RXIV_CACHE_PATH = "data/processed/rxiv_titles_cache.json"
GITHUB_EXTRA_CACHE_PATH = "data/processed/github_extraction_cache.json"

# 加载各个 cache
arxiv_cache = load_cache(ARXIV_CACHE_PATH)
rxiv_cache = load_cache(RXIV_CACHE_PATH)
github_cache = load_cache(GITHUB_EXTRA_CACHE_PATH)

# 分析并输出各 cache 的统计信息
analyze_cache(arxiv_cache, "ArXiv Cache")
analyze_cache(rxiv_cache, "Rxiv Cache")
analyze_cache(github_cache, "GitHub Extraction Cache")

def analyze_github_cache(cache, cache_name="GitHub Cache"):
    total = len(cache)
    valid = 0
    for entry in cache.values():
        # 定义默认值：bibtex默认为 None 或 []，readme_title默认为 None，html_title默认为 None
        bibtex_valid = (entry.get("bibtex") not in [None, []])
        readme_valid = (entry.get("readme_title") not in [None, ""])
        html_valid   = (entry.get("html_title") not in [None, ""])
        if bibtex_valid or readme_valid or html_valid:
            valid += 1
    print(f"{cache_name} - Total entries: {total}, Valid entries: {valid}")
analyze_github_cache(github_cache)



# In[185]:


import pandas as pd
import numpy as np  ######## Ensure numpy is imported
from collections import Counter
import json  ######## Import json for converting dict to string

# Read the parquet file
parquet_file = "data/processed/modelcard_all_title_list.parquet"
df = pd.read_parquet(parquet_file)

# ----------------------------------------
# Step 1: Globally deduplicate bibtex JSON items

def deduplicate_bibtex_items(bibtex_series):  ########
    unique_items = set()  ########
    for bibtex_list in bibtex_series:  ########
        if isinstance(bibtex_list, (list, tuple, np.ndarray)):  ########
            for item in bibtex_list:  ########
                if isinstance(item, dict):  ########
                    # Convert the dict to a JSON string with sorted keys for consistency
                    item_str = json.dumps(item, sort_keys=True)  ########
                    unique_items.add(item_str)  ########
    return unique_items  ########

unique_bibtex_items = deduplicate_bibtex_items(df["parsed_bibtex_tuple_list"])  ########
unique_github_items = deduplicate_bibtex_items(df["parsed_bibtex_tuple_list_github"])  ########

print("Unique bibtex items count (parsed_bibtex_tuple_list):", len(unique_bibtex_items))  ########
print("Unique bibtex items count (parsed_bibtex_tuple_list_github):", len(unique_github_items))  ########

# ----------------------------------------
# Step 2: Globally extract and deduplicate valid titles

def extract_title_from_item(bibtex_item):  ########
    if isinstance(bibtex_item, dict) and bibtex_item.get("title"):  ########
        title = bibtex_item.get("title", "")  ########
        # Clean the title string
        title = title.replace("{", "").replace("}", "").lower().strip()  ########
        return title  ########
    return ""  ########

def extract_unique_titles(bibtex_series):  ########
    titles = []  ########
    for bibtex_list in bibtex_series:  ########
        if isinstance(bibtex_list, (list, tuple, np.ndarray)):  ########
            for item in bibtex_list:  ########
                title = extract_title_from_item(item)  ########
                if title:  ########
                    titles.append(title)  ########
    return set(titles)  ########

unique_titles_bibtex = extract_unique_titles(df["parsed_bibtex_tuple_list"])  ########
unique_titles_github = extract_unique_titles(df["parsed_bibtex_tuple_list_github"])  ########

print("Unique valid titles count (parsed_bibtex_tuple_list):", len(unique_titles_bibtex))  ########
print("Unique valid titles count (parsed_bibtex_tuple_list_github):", len(unique_titles_github))  ########

# ----------------------------------------
# Step 3: Per-row extraction of titles and count non-empty rows

def extract_titles_per_row(bibtex_list):  ########
    if not isinstance(bibtex_list, (list, tuple, np.ndarray)):  ########
        return []  ########
    titles = [extract_title_from_item(item) for item in bibtex_list if isinstance(item, dict)]  ########
    # Remove empty strings and deduplicate titles per row
    return list(set(t for t in titles if t))  ########

# Create new columns for extracted titles per row for each bibtex source
df["titles_bibtex"] = df["parsed_bibtex_tuple_list"].apply(extract_titles_per_row)  ########
df["titles_github"] = df["parsed_bibtex_tuple_list_github"].apply(extract_titles_per_row)  ########

# Count rows with non-empty title lists for each column
non_empty_titles_bibtex = df["titles_bibtex"].apply(lambda x: len(x) > 0).sum()  ########
non_empty_titles_github = df["titles_github"].apply(lambda x: len(x) > 0).sum()  ########

print("Rows with non-empty titles in parsed_bibtex_tuple_list:", non_empty_titles_bibtex, 
      f"({non_empty_titles_bibtex / len(df) * 100:.2f}%)")  ########
print("Rows with non-empty titles in parsed_bibtex_tuple_list_github:", non_empty_titles_github, 
      f"({non_empty_titles_github / len(df) * 100:.2f}%)")  ########

# Combine titles from both sources per row (deduplicated within the row)
df["combined_titles"] = df.apply(lambda row: list(set(row["titles_bibtex"] + row["titles_github"])), axis=1)  ########

# Count rows with non-empty combined title lists
non_empty_combined_titles = df["combined_titles"].apply(lambda x: len(x) > 0).sum()  ########
print("Rows with non-empty combined titles:", non_empty_combined_titles, 
      f"({non_empty_combined_titles / len(df) * 100:.2f}%)")  ########


# In[ ]:





# In[170]:


import pandas as pd
import numpy as np

# Uncomment to load data if needed:
# parquet_file = "data/processed/modelcard_step1.parquet"
# df_1 = pd.read_parquet(parquet_file)

print(df_1.columns)

# Filter rows where 'parsed_bibtex_tuple_list' is a list, tuple, or np.ndarray and its length > 0
filtered_df = df_1[df_1['parsed_bibtex_tuple_list'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)]
print(filtered_df['parsed_bibtex_tuple_list'])


"""non_empty_df = df["all_title_list"].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
non_empty_count = non_empty_df.sum()
total_count = len(df)
proportion = non_empty_count / total_count if total_count > 0 else 0

print(f"总行数: {total_count}")
print(f"all_title_list 非空的行数: {non_empty_count}")
print(f"占比: {proportion:.2%}")"""


# In[171]:


df


# In[217]:


import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Table 1: HuggingFace Card BibTeX Statistics
data_hf = [
    {
        "Category": "HuggingFace Readme BibTeX",
        "Unique BibTeX Items": 5782,
        "Valid Titles Count": 5303,
        "Card Total Count": 1108759,
        "Non-empty Count": 33699,
        "Percentage": f"{(33699/1108759*100):.2f}%"
    },
    {
        "Category": "HuggingFace GitHub URL BibTeX",
        "Unique BibTeX Items": 944,
        "Valid Titles Count": 883,
        "Card Total Count": 1108759,
        "Non-empty Count": 7517,
        "Percentage": f"{(7517/1108759*100):.2f}%"
    }
]

df_hf = pd.DataFrame(data_hf)
df_hf['Percentage_Value'] = df_hf["Non-empty Count"] / df_hf["Card Total Count"] * 100  ######## Compute numeric percentage

# -------------------------------
# Table 2: Cache Statistics (Updated)
data_cache = [
    {
        "Category": "Unique ArXiv URL",
        "Total Unique Items": 5921,
        "Titles Extracted": 5850,
        "Extraction Rate": f"{(5850/5921*100):.2f}%"
    },
    {
        "Category": "Unique Bio/MedRxiv URL",
        "Total Unique Items": 109,
        "Titles Extracted": 99,
        "Extraction Rate": f"{(99/109*100):.2f}%"
    },
    {
        "Category": "Unique GitHub URL",
        "Total Unique Items": 21190,
        "Titles Extracted": 16207,
        "Extraction Rate": f"{(16207/21190*100):.2f}%"
    },
    {
        "Category": "HuggingFace Card with any title",
        "Total Unique Items": 1108759,
        "Titles Extracted": 374360,
        "Extraction Rate": "33.76%"
    }
]

df_cache = pd.DataFrame(data_cache)
# Compute numeric extraction rate for plotting using new keys
df_cache['Extraction_Rate_Value'] = df_cache["Titles Extracted"] / df_cache["Total Unique Items"] * 100  ########

# -------------------------------
# Plot for HuggingFace Card BibTeX Statistics
plt.figure(figsize=(8, 6))
bars = plt.bar(df_hf["Category"], df_hf["Percentage_Value"])
plt.title("HuggingFace Card BibTeX Non-empty Percentage")
plt.ylabel("Percentage (%)")
plt.ylim(0, max(df_hf['Percentage_Value'])*1.5)
for i, bar in enumerate(bars):
    height = bar.get_height()
    text = (f"Unique: {df_hf.loc[i, 'Unique BibTeX Items']}\n"
            f"Valid: {df_hf.loc[i, 'Valid Titles Count']}\n"
            f"Non-empty: {df_hf.loc[i, 'Non-empty Count']}")
    plt.text(bar.get_x() + bar.get_width()/2, height, text, ha='center', va='bottom', fontsize=9)
plt.show()

# -------------------------------
# Plot for Cache (Extraction) Statistics
plt.figure(figsize=(10, 6))
bars = plt.bar(df_cache["Category"], df_cache["Extraction_Rate_Value"])
plt.title("Cache Extraction Rate")
plt.ylabel("Extraction Rate (%)")
plt.ylim(0, 110)
for i, bar in enumerate(bars):
    height = bar.get_height()
    text = (f"{df_cache.loc[i, 'Titles Extracted']} / {df_cache.loc[i, 'Total Unique Items']}\n"
            f"Rate: {df_cache.loc[i, 'Extraction Rate']}")
    plt.text(bar.get_x() + bar.get_width()/2, height, text, ha='center', va='bottom', fontsize=9)
plt.show()


# In[218]:


df_cache


# In[219]:


df_hf


# In[173]:


tmp = {i: rxiv_cache[i] for i in rxiv_cache if rxiv_cache[i] == '| bioRxiv' or rxiv_cache[i].isdigit()}
tmp


# In[174]:


len(rxiv_cache), len(tmp)


# In[ ]:





# In[ ]:





# In[175]:


import pandas as pd
import numpy as np

def is_valid(value):
    # 如果值是 numpy 数组，则只有当数组的 size > 0 时才有效
    if isinstance(value, np.ndarray):
        return value.size > 0

    # 如果值是列表，则只有当列表长度 > 0 时才有效
    if isinstance(value, list):
        return len(value) > 0

    # 如果值为 None 或 NaN，视为无效
    if pd.isnull(value):
        return False

    # 如果值是字符串，去除前后空格转小写后，以下情况视为无效：
    # 1. 空字符串
    # 2. "none"
    # 3. "[]"
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "" or s == "none" or s == "[]":
            return False
        return True

    # 其他类型默认视为有效
    return True

# 对 df1 中每一列统计有效值（非空）的个数
for col in df1.columns:
    valid_count = df1[col].apply(is_valid).sum()
    print(f"{col}: {valid_count}")



# In[176]:


df1[df1['successful_parse_count_github']>0]


# In[177]:


# write load json
ARXIV_CACHE_PATH = "data/processed/arxiv_titles_cache.json"
RXIV_CACHE_PATH = "data/processed/rxiv_titles_cache.json"
github_Cache = "data/processed/github_extraction_cache.json"
other_cache = "link2title_cache.json"

# load json
import json
with open(ARXIV_CACHE_PATH, "r") as f:
    arxiv_cache = json.load(f)
with open(RXIV_CACHE_PATH, "r") as f:
    rxiv_cache = json.load(f)
with open(github_Cache, "r") as f:
    github_cache = json.load(f)
with open(other_cache, "r") as f:
    other_cache = json.load(f)

print(len(arxiv_cache), len(rxiv_cache), len(github_cache))
print(arxiv_cache)
print(rxiv_cache)
print(github_cache)
print(other_cache)


# In[178]:


print(other_cache)


# In[179]:


print(other_cache)


# In[180]:


processed_base_path = os.path.join('data', 'processed')
data_type = 'modelcard'

start_time = time.time()
t1 = start_time
print("⚠️Step 1: Loading data...")
load_path = os.path.join(processed_base_path, f"{data_type}_step1.parquet")
df_step1 = pd.read_parquet(load_path, columns=['modelId', 'parsed_bibtex_tuple_list', 'downloads'])
ext_title_path = os.path.join(processed_base_path, f"{data_type}_ext_title.parquet")
df_ext = pd.read_parquet(ext_title_path)
df = pd.merge(df_step1, df_ext, on="modelId", how="left")


# In[181]:


df


# In[182]:


df.columns


# In[183]:


non_empty_df = df[df['biorxiv_title'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
print(non_empty_df)


# 

# In[84]:


#df0 = pd.read_parquet(os.path.join('./data/processed', "modelcard_step1.parquet"))
df0.columns


# In[24]:


print(df1['parsed_bibtex_tuple_list'].iloc[3])


# In[25]:


print(df1['title_list_from_bibtex'].iloc[3])


# In[114]:


pip install cloudscraper


# In[113]:


import requests
from bs4 import BeautifulSoup
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                  (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com"
}

def fetch_biorxiv_html_title(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 方法1：title 标签
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        # 方法2：meta 标签
        meta_title = soup.find('meta', {'name': 'DC.Title'})
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
        return ''
    except Exception as e:
        print(f"Error fetching bioRxiv HTML title from {url}: {e}")
        return ""

# ✅ 测试调用
title = fetch_biorxiv_html_title("https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2")
print("Title:", title)

def fetch_biorxiv_html_title(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # 方法1：用<title>标签
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        # 方法2：meta标签（更精准）
        meta_title = soup.find('meta', {'name': 'DC.Title'})
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
        return ''
    except Exception as e:
        print(f"Error fetching bioRxiv HTML title from {url}: {e}")
        return ""

fetch_biorxiv_html_title("https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2")


# In[120]:


import requests
import re

HEADERS = {"User-Agent": "Mozilla/5.0"}

def extract_biorxiv_id(url):
    """
    精确提取 bioRxiv DOI 格式 ID，包含完整路径，去掉 v2 版本号。
    示例：
    输入: https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2
    输出: 10.1101/2022.07.20.500902
    """
    pattern = r'biorxiv\.org/content/(10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+)(v\d+)?'
    m = re.search(pattern, url)
    if m:
        return m.group(1)  # 完整 DOI，无版本号
    return None

def fetch_biorxiv_title_via_api(url):
    try:
        bio_id = extract_biorxiv_id(url)
        if not bio_id:
            print("No valid ID found.")
            return ""
        print(bio_id)
        api_url = f"https://api.biorxiv.org/details/biorxiv/{bio_id}"
        resp = requests.get(api_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("collection"):
            title = data["collection"][0].get("title", "")
            return title.strip()
        return ""
    except Exception as e:
        print(f"API error for {url}: {e}")
        return ""

# 测试
url = "https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2"
title = fetch_biorxiv_title_via_api(url)
print("Title:", title)


# In[107]:


import numpy as np
first_idx = next(
    (
        i for i, links in enumerate(df0['pdf_link']) 
        if links is not None and isinstance(links, (list, np.ndarray)) 
        and any('biorxiv' in link.lower() for link in links)
    ), 
    None
)
print(df0.iloc[first_idx]['pdf_link'])


# In[121]:


import pandas as pd
tmp_df = pd.read_csv('all_links_with_category.csv')
tmp_df


# In[ ]:


tmp_df['link']


# In[135]:


# 筛选出 'link' 列中包含 'biorxiv' 的行
pdf_df = tmp_df[tmp_df['link'].str.contains('.pdf', case=False, na=False)]
pdf_df[pdf_df['domain']!='arxiv.org']['link'].iloc[0]


# In[138]:


# 筛选出 'link' 列中包含 'biorxiv' 的行
pdf_df = tmp_df[tmp_df['link'].str.contains('.pdf', case=False, na=False)]
pdf_df[pdf_df['domain']!='arxiv.org']['link'].iloc[1]


# In[149]:


import os
import re
import time
import requests
import hashlib
from bs4 import BeautifulSoup
import html2text

# 定义用于缓存README下载结果的全局cache
readme_cache = {}  ######## 用于避免重复下载

# 指定用于存储下载的README文件的文件夹，避免重复写入
README_FOLDER = "github_readme_output"
if not os.path.exists(README_FOLDER):
    os.makedirs(README_FOLDER)  ######## 新增文件夹创建

def clean_github_link(link):
    """
    清洗从Markdown中提取出来的GitHub链接，去掉多余的符号
    """
    cleaned = (
        link.split('{')[0]
            .split('}')[0]
            .split('[')[0]
            .split(']')[0]
            .split('(')[0]
            .split(')')[0]
            .split('<')[0]
            .split('>')[0]
            .split('*')[0]
            .split('`')[0]
            .split('"')[0]
            .split("'")[0]
            .split('!')[0]
            .strip()
    )
    return cleaned

def extract_bibtex_from_html(html_text):
    """
    尝试从HTML中提取BibTeX引用块
    """
    m = re.search(r'@[\w]+\{[\s\S]+?\}\s', html_text)
    if m:
        return m.group(0).strip()
    return None

def download_readme(github_url):
    """
    根据github_url下载README文件，先用cache避免重复下载，
    采用master/main分支和多种README文件名尝试下载。
    """
    if github_url in readme_cache:
        print(f"[Cache] 使用缓存的README: {github_url}")
        return readme_cache[github_url]  ######## 使用缓存

    # 简单解析URL，假设格式为：https://github.com/{user}/{repo}/...
    parts = github_url.strip('/').split('/')
    if len(parts) < 5:
        print("URL格式无法识别(非标准仓库地址):", github_url)
        readme_cache[github_url] = None
        return None
    # parts[0]为'https:'，parts[1]为空，parts[2]为'github.com'，parts[3]为user，parts[4]为repo
    user = parts[3]
    repo = parts[4]

    branches = ["master", "main"]  ######## 先尝试两个常见分支
    readme_variants = ["README.md", "README.rst", "README.txt", "README"]
    content = None
    for branch in branches:
        for variant in readme_variants:
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{variant}"
            print(f"[Download README] 尝试下载: {raw_url}")
            try:
                resp = requests.get(raw_url, timeout=10)
                if resp.status_code == 200 and resp.text.strip():
                    content = resp.text
                    print(f"[Download README] 成功下载: {raw_url}")
                    readme_cache[github_url] = content
                    # 保存到本地文件，便于调试
                    filename = hashlib.md5(github_url.encode('utf-8')).hexdigest() + ".md"
                    file_path = os.path.join(README_FOLDER, filename)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    return content
            except Exception as e:
                print(f"[Download README] 错误下载 {raw_url}: {e}")
    print(f"[Download README] 无法下载README: {github_url}")
    readme_cache[github_url] = None
    return None

def extract_title_from_readme(readme_content):
    """
    从README文本中提取标题，假设标题是第一个以"# "开头的行
    """
    lines = readme_content.splitlines()
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return None

def extract_html_title(github_url):
    """
    直接通过HTML抓取<title>标签作为备用方案，同时打印部分HTML用于调试
    """
    try:
        print(f"[HTML Fetch] 抓取HTML: {github_url}")
        resp = requests.get(github_url, timeout=10)
        html_text = resp.text
        print(f"[HTML Fetch] 抓取到的HTML前500字符:\n{html_text[:500]}\n")  ######## 打印部分HTML用于调试
        soup = BeautifulSoup(html_text, "html.parser")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception as e:
        print(f"[HTML Fetch] 错误抓取HTML: {github_url}, 错误: {e}")
    return None

def process_github_url(github_url):
    """
    对单个GitHub链接进行处理：
    1. 清洗链接
    2. 尝试从页面中提取BibTeX引用块
    3. 尝试下载README并提取其中的标题
    4. 备用方案：直接抓取HTML中的<title>
    返回一个包含三个不同来源title的字典。
    """
    print(f"\n[Process] 原始URL: {github_url}")
    cleaned_url = clean_github_link(github_url)
    print(f"[Process] 清洗后URL: {cleaned_url}")
    
    # Step 1: 尝试从仓库页面中提取BibTeX引用块
    try:
        resp = requests.get(cleaned_url, timeout=10)
        html_text = resp.text
        print(f"[Process] 用于BibTeX提取的HTML前500字符:\n{html_text[:500]}\n")  ######## 打印部分HTML用于调试
        bibtex = extract_bibtex_from_html(html_text)
    except Exception as e:
        print(f"[Process] 抓取BibTeX HTML错误: {e}")
        bibtex = None
    
    # Step 2: 尝试下载README并提取标题
    readme_content = download_readme(cleaned_url)
    readme_title = extract_title_from_readme(readme_content) if readme_content else None
    
    # Step 3: 备用方案，通过HTML<title>获取标题
    html_title = extract_html_title(cleaned_url)
    
    print(f"[Process] BibTeX: {bibtex}")
    print(f"[Process] README标题: {readme_title}")
    print(f"[Process] HTML标题: {html_title}")
    
    return {
        "bibtex": bibtex,
        "readme_title": readme_title,
        "html_title": html_title
    }

# toy example使用
if __name__ == "__main__":
    sample_url = "https://github.com/zhuang2002/PowerPaint"  ######## 这里可替换成你想测试的链接
    result = process_github_url(sample_url)
    print("\n[Final Result] 提取结果：")
    print(result)


# In[151]:


import json
print(json.dumps(result,indent=4))


# In[153]:


import html2text
import os
from pathlib import Path
import re

# 你的 BibTeX 提取类
class BibTeXExtractor:
    @staticmethod
    def extract(content: str):
        bibtex_entries = []
        bibtex_pattern = r"@(\w+)\{"
        current_entry = ""
        open_braces = 0
        inside_entry = False
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if not inside_entry and re.match(bibtex_pattern, line):
                inside_entry = True
                current_entry = line
                open_braces = line.count("{") - line.count("}")
            elif inside_entry:
                current_entry += " " + line
                open_braces += line.count("{") - line.count("}")

            if inside_entry and open_braces == 0:
                bibtex_entries.append(current_entry.strip())
                inside_entry = False
                current_entry = ""
        return bibtex_entries

# Step 1: 准备 HTML 示例
# load md file
with open('data/downloaded_github_readmes/0a56eee4feaa72bf9de689c28cdc1e3b.md', 'r') as f:
    html_example = f.read()

'''html_example = """
<p>If you find this repository helpful, feel free to cite our paper:</p>
<div data-snippet-clipboard-copy-content="@misc{wang2023hugnlp,
  doi       = {10.48550/ARXIV.2302.14286},
  url       = {https://arxiv.org/abs/2302.14286},
  author    = {Jianing Wang, Nuo Chen, Qiushi Sun, Wenkang Huang, Chengyu Wang, Ming Gao},
  title     = {HugNLP: A Unified and Comprehensive Library for Natural Language Processing},
  year      = {2023}
}">
<pre>
@misc{wang2023hugnlp,
  doi       = {10.48550/ARXIV.2302.14286},
  url       = {https://arxiv.org/abs/2302.14286},
  author    = {Jianing Wang, Nuo Chen, Qiushi Sun, Wenkang Huang, Chengyu Wang, Ming Gao},
  title     = {HugNLP: A Unified and Comprehensive Library for Natural Language Processing},
  year      = {2023}
}
</pre>
</div>
"""'''

# Step 2: 用 html2text 转为 Markdown 并保存
md_text = html2text.html2text(html_example)
save_path = Path("toy_readme.md")
with open(save_path, "w", encoding="utf-8") as f:
    f.write(md_text)
print(f"✅ 已保存 Markdown 到: {save_path.resolve()}")

# Step 3: 读取保存的 Markdown 并提取 BibTeX
with open(save_path, "r", encoding="utf-8") as f:
    content = f.read()

bibtex_entries = BibTeXExtractor.extract(content)
print(f"\n🍷 提取到 BibTeX 条数: {len(bibtex_entries)}")
for i, entry in enumerate(bibtex_entries):
    print(f"\n📌 BibTeX #{i+1}:\n{entry}")


# In[46]:


# Load a local markdown file and extract the title
with open("data/downloaded_github_readmes_processed/ffe99554161e13f8c563bee60c99c914.md", "r", encoding="utf-8") as f:
    content = f.read()

title = extract_title_from_readme(content)
print("Extracted title:", title)


# In[45]:


import re
from bs4 import BeautifulSoup

# Load a local markdown file and extract the title
with open("data/downloaded_github_readmes/ffe99554161e13f8c563bee60c99c914.md", "r", encoding="utf-8") as f:
    content = f.read()

title = parse_html_title(content)
print(title)


# In[50]:


import re

def process_html_title(title):
    """
    Process an HTML title string.
    If it matches a pattern like "GitHub - username/reponame: actual title", 
    remove the prefix ("GitHub - username/reponame:") and return the rest.
    Otherwise, if the title contains a colon, remove everything before the first colon.
    If no colon is present, return the title unchanged.
    """
    if not title:
        return title
    if title.startswith('GitHub'):
        m = re.match(r"GitHub\s*-\s*[^:]+:\s*(.*)", title)
        if m:
            return m.group(1).strip()
        if ":" in title:
            return title.split(":", 1)[1].strip()
    return title

# Example usage:
titles = [
    "GitHub - Tencent/HunyuanDiT: HunyuanDiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding",
    "Some other title: with extra info",
    "Regular Title Without Colon"
]
for t in titles:
    print("Processed title:", process_html_title(t))


# 

# In[1]:


import pandas as pd
df = pd.read_parquet("data/processed/github_readme_cache.parquet")
df.head()


# In[6]:


import requests
import xml.etree.ElementTree as ET
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 15
def batch_fetch_arxiv_titles(arxiv_ids, chunk_size=29999, delay_between_chunks=3):
    results = {}
    total = len(arxiv_ids)
    if total == 0:
        return results
    print(f"Batch fetching {total} arXiv IDs with chunk_size={chunk_size}")
    for i in range(0, total, chunk_size):
        chunk_ids = arxiv_ids[i:i+chunk_size]
        id_list_str = ",".join(chunk_ids)
        url = f'http://export.arxiv.org/api/query?id_list={id_list_str}'
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title_element = entry.find('{http://www.w3.org/2005/Atom}title')
                id_element = entry.find('{http://www.w3.org/2005/Atom}id')
                if title_element is not None and id_element is not None:
                    full_id = id_element.text.strip()
                    if full_id.startswith("http://arxiv.org/abs/"):
                        bare_id = full_id[len("http://arxiv.org/abs/"):]
                        bare_id = bare_id.split("v")[0]
                    else:
                        bare_id = full_id
                    results[bare_id] = title_element.text.strip()
        except Exception as e:
            print(f"Failed arXiv batch fetch chunk at i={i}: {e}")
        if i + chunk_size < total:
            time.sleep(delay_between_chunks)
    return results
batch_fetch_arxiv_titles(['2305.16023', '2305.13297'])


# In[9]:


import re  ########

def extract_arxiv_id(url):  ########
    if "arxiv" not in url.lower():  ########
        return None  ########
    m = re.search(r'(\d{4}\.\d{5})', url)  ########
    if m:  ########
        return m.group(1)  ########
    return None

extract_arxiv_id("https://arxiv.org/abs/2110.14168")


# In[ ]:


"""import json
ARXIV_CACHE_PATH = "arxiv_titles_cache.json"
RXIV_CACHE_PATH = "rxiv_titles_cache.json"

def load_cache(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
a1 = load_cache(ARXIV_CACHE_PATH)
a1"""


# In[21]:


non_empty_count = sum(1 for v in a1.values() if v not in [None, "", [], {}, ()])
non_empty_count


# In[ ]:


"""import json

ARXIV_CACHE_PATH = "arxiv_titles_cache.json"  ########

def load_cache(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(data, file_path):  ########
    with open(file_path, 'w') as f:  ########
        json.dump(data, f, ensure_ascii=False, indent=2)  ########

# 加载缓存
cache = load_cache(ARXIV_CACHE_PATH)

# 对每个 value 做替换：去除换行符和将双空格替换为单空格
for key, value in cache.items():  ########
    if isinstance(value, str):  ########
        new_value = value.replace('\n', '').replace('  ', ' ')  ########
        cache[key] = new_value  ########

# 保存回文件
save_cache(cache, ARXIV_CACHE_PATH)  ########

print("Cache updated.")  ########
"""


# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


import re
import pandas as pd

# Updated regex to detect arXiv IDs in the format: "arXiv:" followed by 4 digits, a dot, and 5 digits.
df2["new_arxiv_ids"] = df2["card"].apply(
    lambda x: re.findall(r'arXiv[:\s]*([0-9]{4}\.[0-9]{5})', x, flags=re.I) if pd.notna(x) else []
)

# Display the DataFrame to see the extracted arXiv IDs
df2


# In[48]:


print(df2[df2['modelId'] == 'sentence-transformers/all-mpnet-base-v2']['all_links'].iloc[0])


# In[49]:


'arxiv' in df2[df2['modelId'] == 'sentence-transformers/all-mpnet-base-v2']['all_links'].iloc[0]


# In[ ]:





# In[53]:


df2[df2['modelId'] == 'sentence-transformers/all-mpnet-base-v2']['card_tags'].iloc[0]


# In[33]:


non_empty_count = (df2['new_arxiv_ids'].apply(len) > 0)
print("Non-empty arxiv_ids count:", len(non_empty_count))
df2[non_empty_count]


# In[ ]:





# In[ ]:





# In[59]:


import pandas as pd
import re
import requests

def extract_links(df, col="all_links"):
    """
    从 DataFrame 指定列中提取所有链接，按逗号分割、strip 后全局去重
    """
    links = []
    for item in df[col]:
        if isinstance(item, str) and item.strip():
            links.extend([link.strip() for link in item.split(",")])
    return list(set(links))

def save_links_to_txt(links, filename="non_arxiv_links.txt"):
    """
    将链接列表保存为 txt 文件，每行一个链接
    """
    with open(filename, "w", encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")
    print(f"Saved {len(links)} unique non-arXiv links to {filename}")

def is_arxiv_link(url):
    """
    判断链接是否属于 arxiv 域，简单判断包含 "arxiv.org"（不区分大小写）
    """
    return "arxiv.org" in url.lower()

def fetch_domain_title(url):
    """
    根据 url 域名选择合适的方式查询文章标题。
    此处目前采用 HTML 方式抓取页面标题，后续可替换为各域名对应的 API 调用。
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        # 使用 BeautifulSoup 解析 HTML 获取 <title>
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return ""

def debug_process_non_arxiv_links(df, col="all_links"):
    """
    从 DataFrame 中提取链接，过滤掉 arXiv 链接，
    保存非 arXiv 链接到文件，并逐个调用 fetch_domain_title 调试查询标题
    """
    all_links = extract_links(df, col)
    print(f"Total unique links extracted: {len(all_links)}")
    non_arxiv_links = [link for link in all_links if not is_arxiv_link(link)]
    print(f"Non-arXiv links count: {len(non_arxiv_links)}")
    save_links_to_txt(non_arxiv_links, "non_arxiv_links.txt")
    
    for link in non_arxiv_links:
        title = fetch_domain_title(link)
        print(f"Link: {link}\n  -> Title: {title}\n")

if __name__ == '__main__':
    # 构造示例 DataFrame（请替换为你的实际数据）
    data = {
        "all_links": [
            "http://arxiv.org/abs/2304.12244v1, http://arxiv.org/pdf/1507.00123v2",
            "https://pubmed.ncbi.nlm.nih.gov/34567890/, https://dblp.uni-trier.de/rec/conf/icml/Smith2023",
            "https://nature.com/articles/s41586-023-04567-8, https://openaccess.thecvf.com/content_CVPR_2023/html/Chen_Sample_Paper_CVPR_2023_paper.html",
            ""
        ]
    }
    df = pd.DataFrame(data)
    
    # 调试处理非 arXiv 链接
    debug_process_non_arxiv_links(df, col="all_links")


# In[61]:


import pandas as pd
import re

def extract_links_from_df(df, col="all_links"):
    """
    从 DataFrame 的指定列中提取所有链接。
    假设每行内容是以逗号分隔的链接字符串，返回全局去重后的链接列表。
    """
    links = []
    for item in df[col]:
        if isinstance(item, str) and item.strip():
            # 按逗号分割后，去除每个链接的前后空白
            links.extend([link.strip() for link in item.split(",")])
    # 去重
    unique_links = list(set(links))
    return unique_links

def extract_domain(link):
    """
    从给定的 URL 中提取域名。
    使用正则表达式从 'http://' 或 'https://' 后面开始提取直到第一个 '/' 的部分。
    """
    try:
        m = re.search(r'https?://([^/]+)/', link)
        if m:
            return m.group(1)
    except Exception as e:
        print(f"Error extracting domain from {link}: {e}")
    return None

# 假设 df 是已加载的 DataFrame，其中包含 "all_links" 列
# 示例数据（实际使用时请替换为你的 DataFrame）

df = df2

# 提取所有链接并去重
all_links = extract_links_from_df(df, col="all_links")

# 构建新的 DataFrame，其中一列为链接，另一列为对应的域名
df_links = pd.DataFrame(all_links, columns=["link"])
df_links["domain"] = df_links["link"].apply(extract_domain)

# 输出结果以供调试
print("Extracted links and domains:")
print(df_links)

# 可选择将结果保存为 CSV 或 TXT
df_links.to_csv("all_links.csv", index=False)
print("Saved results to all_links.csv")


# In[64]:


df_links.domain.value_counts()[:20]


# In[65]:


df2


# In[80]:


print(df_links[df_links['domain']=="huggingface.co"]['link'].iloc[0]) #[df_links['domain'].str.contains('mlco2.github.io', na=False)]


# In[ ]:





# In[11]:


tmp = pd.read_parquet("src/final_extracted_data.parquet")
tmp


# In[4]:


import pandas as pd
tmp = pd.read_parquet("data/processed/modelcard_step3.parquet")
tmp


# In[12]:


import json
print(json.dumps(json.loads(tmp[tmp['citations_within_dataset'].notna()]['citations_within_dataset'].iloc[0]), indent=4))


# In[1]:


import pandas as pd
tmp = pd.read_parquet("data/processed/giturl_info.parquet")
tmp


# In[6]:


import pandas as pd
tmp = pd.read_parquet("data/processed/github_readmes_info.parquet")
tmp = pd.read_parquet("data/processed/github_readme_cache.parquet")
tmp


# In[39]:


from src.data_ingestion.citation_fetcher import AcademicAPIFactory, search_and_fetch_info
from src.data_ingestion.readme_parser import BibTeXExtractor, LinkExtractor, MarkdownHandler, ExtractionFactory
from src.data_ingestion.bibtex_parser import BibTeXFactory, ensure_string
import pandas as pd
import pyarrow.parquet as pq


# In[13]:


import pyarrow.parquet as pq
import pandas as pd
import json

# List of files to check
file_paths = [
    "data/processed/modelcard_step1.parquet",
    "data/processed/modelcard_step2.parquet",
    "data/processed/scilakeUnionBenchmark.pickle",
]

# Read schema of each file
schemas = {}
for file in file_paths:
    if file.endswith(".parquet"):
        try:
            schema = pq.read_schema(file)
            schemas[file] = set(schema.names)
        except Exception as e:
            pass
    elif file.endswith(".csv"):
        try:
            df = pd.read_csv(file, nrows=5)  # Read only first few rows for performance
            schemas[file] = set(df.columns)
        except Exception as e:
            pass
    elif file.endswith(".pickle"):
        pass

print(schemas)
"""# Compare new and old schema differences
comparison_results = {}
for file in schemas:
    if "old" in file:
        new_file = file.replace("old_", "")
        if new_file in schemas and isinstance(schemas[file], set) and isinstance(schemas[new_file], set):
            missing_columns = schemas[file] - schemas[new_file]  # Columns in old but not in new
            added_columns = schemas[new_file] - schemas[file]  # Columns in new but not in old
            comparison_results[file] = {
                "missing_columns": missing_columns,
                "added_columns": added_columns
            }

# Print results
for file, diff in comparison_results.items():
    print(f"\nComparison for {file}:")
    print(f"  Missing columns in new version: {diff['missing_columns']}")
    print(f"  Added columns in new version: {diff['added_columns']}")
"""


# In[39]:


print(json.dumps(df.head(5)['parsed_bibtex_tuple_list'].iloc[3][0],indent=4))
print(json.dumps(json.loads(df.head(5)['citations_within_dataset'].iloc[3]),indent=4))


# In[ ]:


schema = pq.read_schema(f"data/processed/modelcard_step1.parquet")
print(schema.names)
# ['modelId', 'author', 'last_modified', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'createdAt', 'card', 'card_tags', 'card_readme', 'pdf_link', 'github_link', 'all_links', 'extracted_bibtex', 'extracted_bibtex_tuple', 'parsed_bibtex_tuple_list', 'successful_parse_count']


# In[4]:


import json
json_str = res['references_within_dataset'].values[0]
parsed_data = json.loads(json_str)
print(json.dumps(parsed_data,indent=4))
print(parsed_data.keys())
print({k for item in parsed_data['data'] for k in item})


# This minimal code does the following:
# 1. Creates a mapping from unique paper keys to model_ids.
# 2. Iterates over each paper, parses both the JSON stored references ("citedPaper") and citations ("citingPaper").
# 3. Adds a link between the target paper’s model_id and any corresponding model_id found in each relationship.
# 4. Saves the final mapping (each model_id mapped to a list of linked model_ids) in a pickle file.

# In[58]:


df['parsed_bibtex_tuple_list'].iloc[3][0]['title'].replace('{', '').replace('}', '')


# In[13]:


# sort by downloads
df_sorted = unique_by_markdown[['modelId', 'csv_path', 'downloads']].sort_values(by='downloads', ascending=False)

# save file name for starmie loading
import pandas as pd
import os
def save_file_list(df):
    file_list = [os.path.basename(path) for path in df['csv_path'] if path]
    file_list_path = os.path.join("file_list.txt")
    with open(file_list_path, "w") as f:
        for file in file_list:
            f.write(file + "\n")
    print(f"Get list (to be used in starmie dataloader as file_list): {file_list_path}")
save_file_list(unique_by_markdown)


# In[25]:


unique_by_markdown[unique_by_markdown['csv_path'].notnull()]


# ### After the starmie search, analysis the results

# In[ ]:


from src.utils import save_analysis_results

os.makedirs("analysis", exist_ok=True)
final_df = save_analysis_results(df, returnResults, file_name="analysis/tmp.csv")
final_df


# In[4]:


#from utils import LinkExtractor, BibTeXExtractor, ExtractionFactory
import pandas as pd
df_split_temp['link_info'] = df_split_temp['card_readme'].apply(
    lambda x: LinkExtractor().extract_links(str(x)) if pd.notnull(x) else {"pdf_link": None, "github_link": None, "all_links": []}
)
df_split_temp['pdf_link'] = df_split_temp['link_info'].apply(lambda x: x['pdf_link'])
df_split_temp['github_link'] = df_split_temp['link_info'].apply(lambda x: x['github_link'])


# In[19]:


print(df_split_temp['csv_path'].notnull().sum())
df_split_temp[df_split_temp['csv_path'].notnull()]['csv_path']


# In[7]:


### double check the extracted markdown first


# In[ ]:


# check sample
print(df_split_temp[df_split_temp['modelId'].str.contains('0marr/distilbert-base-multilingual-cased-finetuned', na=False)]['card_readme'].values)
print('-'*10)


# In[ ]:


"""def inspect_column_values(column):
    """Inspect each value in the column and categorize its format."""
    for i, value in column.items():
        if value is None:
            print(f"Row {i}: None value detected")
        elif isinstance(value, list):
            print(f"Row {i}: List with {len(value)} entries")
        elif isinstance(value, str):
            print(f"Row {i}: Single string detected")
        else:
            print(f"Row {i}: Unexpected type {type(value)}")
import numpy as np

def inspect_and_convert(value):
    """Inspect and convert numpy arrays to lists or strings."""
    if isinstance(value, np.ndarray):
        print(f"Array detected: {value}")
        return value.tolist()  # Convert to Python list
    return value
# Apply the function and inspect the first few rows
df_split_temp["extracted_bibtex"] = df_split_temp["extracted_bibtex"].apply(inspect_and_convert)
# Recheck the types
#inspect_column_values(df_split_temp["extracted_bibtex"])
"""


# In[21]:


df_split_temp[df_split_temp['citations_within_dataset'].notna()]['citations_within_dataset'].value_counts()#[['references_within_dataset', 'citations_within_dataset','success_flag']]


# In[ ]:


# check sample
key = "extracted_bibtex"
filtered_df = df_split_temp[df_split_temp[key].notnull() & (df_split_temp[key].apply(lambda x: len(x) > 0))]
filtered_df[key].head(10)


# In[3]:


df_split_temp['contains_markdown_table'].value_counts()


# In[ ]:


df_split_temp['extracted_markdown_table'].value_counts()


# In[ ]:


df_split_temp['card_readme'].value_counts()


# In[5]:


value_counts = df_split_temp['card_readme'].value_counts()
unique_content = value_counts[value_counts == 1].index
count_unique_content = len(unique_content)
unique_indexes = df_split_temp[df_split_temp['card_readme'].isin(unique_content)].index
count_unique_indexes = len(unique_indexes)
count_remaining_indexes = len(df_split_temp) - count_unique_indexes
print("等于 1 的 content 数量:", count_unique_content)
print("等于 1 的 content 对应的 index 数量:", count_unique_indexes)
print("剩余的 index 数量:", count_remaining_indexes)


# In[6]:


value_counts = df_split_temp['card_readme'].value_counts()
content_with_two_counts = value_counts[value_counts == 2].index.tolist()
content_with_two_counts = content_with_two_counts[:5]
model_ids_dict = {
    content: df_split_temp[df_split_temp['card_readme'] == content]['modelId'].tolist()
    for content in content_with_two_counts
}
print("前 5 个出现次数等于 2 的内容及其 `modelId`:")
for content, model_ids in model_ids_dict.items():
    print(f"{model_ids[0]}, {model_ids[1]}")


# ### re-get the links
# 
# The PDF links can come from 
# - tags like `arxiv:id`
# - links from `card_readme` columns
# - arxiv/pdf links from github readme files
# - bibtex from `card_readme`
# - possible other card_tags (need to wait code runs and check manually, e.g. `papers` tags might contain the possible links)
# - Remember to remove the arxiv:1910.09700 or https://arxiv.org/abs/1910.09700 as it is the default PDF link exists from readme file
# - ...
# 

# In[7]:


'''import pandas as pd
import re

def extract_complete_bibtex_final(content: str):
    """
    Final version to extract complete BibTeX entries with balanced braces and multi-line support.
    """
    bibtex_entries = []
    bibtex_pattern = r"@(\w+)\{"
    current_entry = ""
    open_braces = 0
    inside_entry = False
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if not inside_entry and re.match(bibtex_pattern, line):
            inside_entry = True
            current_entry = line
            open_braces = line.count("{") - line.count("}")
        elif inside_entry:
            current_entry += " " + line
            open_braces += line.count("{") - line.count("}")
        if inside_entry and open_braces == 0:
            bibtex_entries.append(current_entry.strip())
            inside_entry = False
            current_entry = ""
    return bibtex_entries

def detect_bibtex_entries_final(card_content: str):
    """
    Final version for detecting BibTeX entries from content.
    """
    if not isinstance(card_content, str) or not card_content.strip():
        return (False, [])
    code_block_pattern = r"```(?:bibtex)?\n(.*?)(?:```|$)"
    code_blocks = re.findall(code_block_pattern, card_content, re.DOTALL)
    bibtex_entries = []
    for block in code_blocks:
        bibtex_entries += extract_complete_bibtex_final(block)
    
    ### Also detect ```
    # @xxxx{}``` cases
    inline_bibtex_pattern = r"```\s*@\w+\{.*?\}\s*```"
    inline_bibtex_matches = re.findall(inline_bibtex_pattern, card_content, re.DOTALL)
    for match in inline_bibtex_matches:
        bibtex_entries.append(match.strip("`"))
    
    # Also check for inline BibTeX entries outside of code blocks
    content_without_code_blocks = re.sub(code_block_pattern, "", card_content, flags=re.DOTALL)
    bibtex_entries += extract_complete_bibtex_final(content_without_code_blocks)
    if bibtex_entries:
        return (True, bibtex_entries)
    return (False, [])

df_split_temp['contains_bibtex'], df_split_temp['all_extracted_bibtex'] = zip(
    *df_split_temp['card_readme'].apply(detect_bibtex_entries_final)
)

def parse_bibtex_entries(entries):
    """
    Parse BibTeX entries into structured data (type, title, full entry).
    """
    parsed_entries = []
    for entry in entries:
        bibtex_type = re.match(r"^@(\w+)\{", entry)
        bibtex_type = bibtex_type.group(1) if bibtex_type else None
        bibtex_title = re.search(r"title\s*=\s*[{\"]([^{}\"]+)[}\"]", entry, re.IGNORECASE)
        bibtex_title = bibtex_title.group(1) if bibtex_title else None
        parsed_entries.append({
            "type": bibtex_type,
            "title": bibtex_title,
            "entry": entry
        })
    return parsed_entries

df_split_temp['parsed_bibtex_entries'] = df_split_temp['all_extracted_bibtex'].apply(parse_bibtex_entries)
df_split_temp['bibtex_count'] = df_split_temp['parsed_bibtex_entries'].apply(lambda x: len(x) if x else 0)
df_split_temp.to_csv("extracted_bibtex_entries_grouped.csv", index=False)'''


# In[40]:


'''# Compute statistics
bibtex_total_ratio = len(df_split_temp[df_split_temp['contains_bibtex']]) / len(df_split_temp)
print(f"BibTeX presence ratio: {bibtex_total_ratio:.6f}")

# Count occurrences for specific BibTeX counts
df_split_temp['bibtex_count'] = df_split_temp['parsed_bibtex_entries'].apply(lambda x: len(x) if x else 0)
bibtex_count_distribution = df_split_temp['bibtex_count'].value_counts()
print("BibTeX Count Distribution:")
print(bibtex_count_distribution)

# Check first five entries with exactly 2 BibTeX records
bibtex_2_entries = df_split_temp[df_split_temp['bibtex_count'] == 2][['parsed_bibtex_entries']].head(5)
print("First five entries with exactly 2 BibTeX records:")
print(bibtex_2_entries)

# Identify duplicate BibTeX entries
bibtex_flat_list = [entry['entry'] for entries in df_split_temp['parsed_bibtex_entries'] for entry in entries]
bibtex_duplicates = pd.Series(bibtex_flat_list).value_counts()
print("Duplicate BibTeX Entries:")
print(bibtex_duplicates[bibtex_duplicates > 1])

# Identify entries containing @ but without valid BibTeX
df_split_temp['contains_at_symbol'] = df_split_temp['card_readme'].apply(lambda x: '@' in x if isinstance(x, str) else False)
df_split_temp['invalid_bibtex'] = df_split_temp.apply(lambda x: x['contains_at_symbol'] and not x['contains_bibtex'], axis=1)
invalid_bibtex_entries = df_split_temp[df_split_temp['invalid_bibtex']]
print("Entries containing '@' but not valid BibTeX:")
print(invalid_bibtex_entries[['modelId', 'card_readme']].head())'''


# In[13]:


'''# Count the occurrences of each BibTeX type
bibtex_type_counts = pd.Series([entry['type'].lower() for entries in df_split_temp['parsed_bibtex_entries'] for entry in entries if entry['type']]).value_counts()
print("BibTeX Type Counts:")
print(bibtex_type_counts)'''


# In[10]:


# Compute statistics
bibtex_total_ratio = len(df_split_temp[df_split_temp['contains_bibtex']]) / len(df_split_temp)
print(f"BibTeX presence ratio: {bibtex_total_ratio:.6f}")
# Count occurrences for specific BibTeX counts
bibtex_count_distribution = df_split_temp['bibtex_count'].value_counts()
print("BibTeX Count Distribution:")
print(bibtex_count_distribution)
# Check first five entries with exactly 2 BibTeX records
bibtex_2_entries = df_split_temp[df_split_temp['bibtex_count'] == 2][['parsed_bibtex_entries']].head(5)
print("First five entries with exactly 2 BibTeX records:")
print(bibtex_2_entries)
# Identify duplicate BibTeX entries
bibtex_flat_list = [entry['entry'] for entries in df_split_temp['parsed_bibtex_entries'] for entry in entries]
bibtex_duplicates = pd.Series(bibtex_flat_list).value_counts()
print("Duplicate BibTeX Entries:")
print(bibtex_duplicates[bibtex_duplicates > 1])
# 0.0177297320698186


# In[30]:


# test extraction correctness
print(df_split_temp[df_split_temp['bibtex_count']==2]['all_extracted_bibtex'].iloc[0])
print(df_split_temp[df_split_temp['bibtex_count']==1]['parsed_bibtex_entries'].iloc[0])
print(df_split_temp[df_split_temp['bibtex_count']==1]['bibtex_count'].iloc[0])


# In[ ]:


# 
df_split_temp[df_split_temp['bibtex_count']==0].columns


# In[37]:


df_split_temp['contains_at_symbol'] = df_split_temp['card_readme'].apply(lambda x: '```bibtex' in x if isinstance(x, str) else False)
df_split_temp['invalid_bibtex'] = df_split_temp.apply(lambda x: x['contains_at_symbol'] and not x['contains_bibtex'], axis=1)
invalid_bibtex_entries = df_split_temp[df_split_temp['invalid_bibtex']]
print(invalid_bibtex_entries['card_readme'].value_counts().index)


# In[ ]:


# Define the priority of BibTeX types
bibtex_type_priority = {
    'article': 1,
    'inproceedings': 2,
    'InProceedings': 2,
    'INPROCEEDINGS': 2,
    'techreport': 3,
    'phdthesis': 3,
    'mastersthesis': 3,
    'thesis': 3,
    'misc': 4,
    'software': 4,
    'online': 4,
    'unpublished': 4,
    'dataset': 4,
    'data': 4,
    'model': 4,
    'book': 4,
    'artical': 5,  # Typo, lower priority
    'Paper': 5,
    'MISC': 5,
    'ARTICLE': 5,
    'unknown': 6
}

# Helper function to extract BibTeX type
def extract_bibtex_type(entry):
    match = re.match(r"@(\w+)", entry)
    return match.group(1).lower() if match else 'unknown'

# Function to check for PDF links
def has_pdf_link(entry):
    return '.pdf' in entry.lower()

# Process each row to sort the BibTeX entries
sorted_bibtex_entries = []
for row in bibtex_entries_df['all_extracted_bibtex']:
    # Extract type and check for PDF links
    processed_entries = []
    for entry in row:
        entry_type = extract_bibtex_type(entry)
        pdf_link = has_pdf_link(entry)
        priority = bibtex_type_priority.get(entry_type, 6)
        processed_entries.append((entry, entry_type, priority, pdf_link))
    # Sort based on: (PDF link, type priority, original order)
    processed_entries.sort(key=lambda x: (-x[3], x[2]))
    sorted_bibtex_entries.append([entry[0] for entry in processed_entries])
# Add sorted entries back to the DataFrame
bibtex_entries_df['sorted_bibtex'] = sorted_bibtex_entries
bibtex_entries_df['extracted_bibtex'] = bibtex_entries_df['sorted_bibtex'].apply(
    lambda x: x[0] if len(x) > 0 else None
)
filtered_bibtex_df = bibtex_entries_df.dropna(subset=['extracted_bibtex']).reset_index(drop=True)
filtered_bibtex_df


# In[145]:


import pandas as pd
import re

# Step 1: Check if entries start with valid BibTeX types
def get_bibtex_type(entry):
    match = re.match(r"^@(\w+){", entry.strip())
    return match.group(1) if match else None

# Apply function to the extracted_bibtex column
filtered_bibtex_df['bibtex_type'] = filtered_bibtex_df['extracted_bibtex'].apply(lambda x: get_bibtex_type(x) if isinstance(x, str) else None)

# Count occurrences of each BibTeX type
bibtex_counts = filtered_bibtex_df['bibtex_type'].value_counts()

# Display the counts
print("BibTeX Type Counts:")
print(bibtex_counts)

# Step 2: Extract titles from BibTeX
def extract_bibtex_title(entry):
    match = re.search(r"title\s*=\s*[{\"]([^{}\"]+)[}\"]", entry, re.IGNORECASE)
    return match.group(1) if match else None

filtered_bibtex_df['bibtex_title'] = filtered_bibtex_df['extracted_bibtex'].apply(lambda x: extract_bibtex_title(x) if isinstance(x, str) else None)

# Step 3: Remove default links and duplicates
default_link_pattern = r"(arxiv:1910\.09700|https://arxiv\.org/abs/1910\.09700)"
filtered_bibtex_df['is_default_link'] = filtered_bibtex_df['extracted_bibtex'].str.contains(default_link_pattern, na=False)

# Filter non-default links
df_cleaned = filtered_bibtex_df[~filtered_bibtex_df['is_default_link']]

# Show a preview of cleaned DataFrame
print("Cleaned DataFrame Preview:")
print(df_cleaned[['bibtex_type', 'bibtex_title', 'is_default_link']].head())


# In[29]:


import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pyarrow.parquet as pq

# 读取 modelcard_1.parquet 并提取 all_links
def load_filtered_links(file_path):
    print("⚠️ Loading data from Parquet file...")
    df = pq.read_table(file_path).to_pandas()

    # 解析 JSON 格式的 all_links 字段
    df["all_links"] = df["all_links"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])

    # 过滤掉 GitHub 和 PDF 链接
    def filter_links(links):
        return [link for link in links if "github.com" not in link and not link.lower().endswith(".pdf")]

    df["filtered_links"] = df["all_links"].apply(filter_links)

    # 展平成一个链接列表
    all_filtered_links = [link for links in df["filtered_links"] for link in links]
    print(f"✅ Extracted {len(all_filtered_links)} filtered links.")

    return all_filtered_links

# 检查 HTML 是否包含 <table> 标签
def check_html_for_tables(url):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200:
            return False

        soup = BeautifulSoup(response.text, "html.parser")
        return bool(soup.find("table"))
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return False

# 主函数：提取链接并检查是否包含表格
def main(parquet_path):
    links = load_filtered_links(parquet_path)

    results = []
    print("⚠️ Checking for tables in HTML...")
    for link in tqdm(links, desc="Processing links"):
        has_table = check_html_for_tables(link)
        results.append({"url": link, "has_table": has_table})

    # 转换为 DataFrame 并展示
    results_df = pd.DataFrame(results)
    
# 运行代码
if __name__ == "__main__":
    main("data/processed/modelcard_step1.parquet")


# In[ ]:





# - parse with several known domains and pre-given rules
# - parse from multiple possible bibtex in an order as arxiv >= several academic domains >= other with pdf > unknown, so that we know which one of the bibtex is the paper bibtex

# In[ ]:


import os
import re
from tqdm import tqdm
import pandas as pd

# Helper functions to extract fields
def extract_arxiv_id(bibtex):
    if isinstance(bibtex, str):
        # Updated regex to also capture generic arXiv ID formats (4 digits + '.' + 5 digits)
        match = re.search(r"(arxiv:\d+\.\d+|https?://(?:www\.)?arxiv\.org/(abs|pdf)/\d+\.\d+|\d{4}\.\d{5})", bibtex, re.IGNORECASE)
        if match:
            # Normalize the extracted ID by removing prefixes like 'arxiv:' or 'https://...'
            return re.sub(r"(arxiv:|https?://(?:www\.)?arxiv\.org/(abs|pdf)/)", "", match.group())
    return None

def extract_doi(bibtex):
    if isinstance(bibtex, str):
        match = re.search(r"(doi|DOI)\s*=\s*[{\"]([^{}\"]+)[}\"]", bibtex)
        return match.group(2) if match else None
    return None

def extract_url(bibtex):
    if isinstance(bibtex, str):
        match = re.search(r"(url|Url|URL)\s*=\s*[{\"]([^{}\"]+)[}\"]", bibtex)
        return match.group(2) if match else None
    return None

def classify_domain(url):
    if isinstance(url, str):
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)/", url)
        if domain_match:
            domain = domain_match.group(1).lower()
            if "arxiv.org" in domain:
                return "arxiv"
            elif "doi.org" in domain:
                return "doi"
            elif "openreview.net" in domain:
                return "openreview"
            elif "aclanthology.org" in domain:
                return "acl"
            elif "sciencedirect.com" in domain:
                return "sciencedirect"
            elif "springer.com" in domain:
                return "springer"
            elif "wiley.com" in domain:
                return "wiley"
            elif "tandfonline.com" in domain:
                return "tandfonline"
            elif "cambridge.org" in domain:
                return "cambridge"
            elif "nature.com" in domain:
                return "nature"
        if url.lower().endswith(".pdf"):
            return "others"
        return "unknown"
    return None

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
# Parse BibTeX entries using bibtexparser
def parse_bibtex_entry(entry):
    try:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        parser.ignore_nonstandard_types = False
        bib_database = bibtexparser.loads(entry, parser=parser)
        #bib_database = bibtexparser.loads(entry)
        parsed_data = []
        for item in bib_database.entries:
            parsed_data.append({
                "bibtex_title": item.get("title"),
                "bibtex_author": item.get("author"),
                "bibtex_journal": item.get("journal"),
                "bibtex_year": item.get("year"),
                "bibtex_url": item.get("url"),
                "bibtex_doi": item.get("doi"),
                "bibtex_arxiv_id": extract_arxiv_id(item.get("journal", "") + " " + item.get("eprint", ""))  # Combine journal and eprint fields
            })
        return parsed_data
    except Exception as e:
        return None

# Process BibTeX entries
processed_results = []
for index, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned)):
    bibtex_entry = row['extracted_bibtex']
    if not isinstance(bibtex_entry, str):
        processed_results.append({
            "modelId": row['modelId'],
            "regex_title": None,
            "regex_author": None,
            "regex_journal": None,
            "regex_year": None,
            "regex_url": None,
            "regex_doi": None,
            "regex_arxiv_id": None,
            "bibtex_parsed": None
        })
        continue

    # BibTeX parsing
    bibtex_parsed = parse_bibtex_entry(bibtex_entry)
    if bibtex_parsed:
        for parsed_item in bibtex_parsed:
            processed_results.append({
                "modelId": row['modelId'],
                "bibtex_title": parsed_item.get("bibtex_title"),
                "bibtex_author": parsed_item.get("bibtex_author"),
                "bibtex_journal": parsed_item.get("bibtex_journal"),
                "bibtex_year": parsed_item.get("bibtex_year"),
                "bibtex_url": parsed_item.get("bibtex_url"),
                "bibtex_doi": parsed_item.get("bibtex_doi"),
                "bibtex_arxiv_id": parsed_item.get("bibtex_arxiv_id"),

                # Regex parsing fallback
                "regex_title": row['bibtex_title'] or f"paper_{index}",
                "regex_author": None,  # Extend regex parsing for author if needed
                "regex_journal": None,  # Extend regex parsing for journal if needed
                "regex_year": None,  # Extend regex parsing for year if needed
                "regex_url": extract_url(bibtex_entry),
                "regex_doi": extract_doi(bibtex_entry),
                "regex_arxiv_id": extract_arxiv_id(bibtex_entry)
            })

# Convert to DataFrame for easy visualization
df_processed = pd.DataFrame(processed_results)

# Save processed results to CSV
df_processed.to_csv("processed_bibtex_links.csv", index=False)

# Join back with the original DataFrame using modelId
#df_joined = pd.merge(df_split_temp, df_processed, on='modelId', how='left', suffixes=('', '_processed'))
#df_joined.to_csv("processed_bibtex_links_with_join.csv", index=False)
#


# In[4]:


df_processed.info()


# In[6]:


import requests
from tqdm import tqdm

# Define the reliability ranking
reliability_sources = [
    "bibtex_doi", "bibtex_arxiv_id", "bibtex_url", 
    "regex_doi", "regex_url", "regex_arxiv_id"
]

# Helper function to construct links based on source
def construct_pdf_link(source, value):
    if source == "bibtex_doi" or source == "regex_doi":
        return f"https://doi.org/{value}"
    elif source == "bibtex_arxiv_id" or source == "regex_arxiv_id":
        return f"https://arxiv.org/pdf/{value}.pdf"
    elif source == "bibtex_url" or source == "regex_url":
        return value  # Already a URL
    return None

# Function to check if a URL is valid
def is_valid_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200 and "pdf" in response.headers.get('Content-Type', '')
    except:
        return False

# Process the DataFrame to get the final_url
final_urls = []
for _, row in tqdm(df_processed.iterrows(), total=len(df_processed)):
    final_url = None
    for source in reliability_sources:
        if pd.notna(row[source]):  # Check if the value is not null
            candidate_url = construct_pdf_link(source, row[source])
            #if candidate_url and is_valid_url(candidate_url):
            if candidate_url:
                final_url = candidate_url
                break
    final_urls.append(final_url)

# Add the final_url column to the DataFrame
df_processed["final_url"] = final_urls

# Save the processed DataFrame
df_processed.to_csv("processed_final_urls.csv", index=False)


# In[17]:


import matplotlib.pyplot as plt

# Assuming `df_processed['final_url'].value_counts()` is already calculated
# Get the value counts as a DataFrame
citation_counts = df_processed['final_url'].value_counts().reset_index()
citation_counts.columns = ['final_url', 'count']

# Plot the histogram of citation counts
plt.figure(figsize=(10, 6))
plt.hist(citation_counts['count'], bins=range(1, citation_counts['count'].max() + 1), edgecolor='black', log=True)
plt.title('Histogram of Citation Counts', fontsize=16)
plt.xlabel('Citation Count', fontsize=14)
plt.ylabel('Frequency (Log Scale)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[22]:


len(citation_counts[citation_counts['count']<10])/len(df_processed)


# In[ ]:


citation_counts[citation_counts['count']==1]


# In[ ]:


#df_processed['final_url'].value_counts()


# In[33]:


# then download the github readme files for all files
# first get github links



# In[17]:


#import pandas as pd
#df_split_temp = pd.read_parquet("data/processed/modelcard_step1.parquet")
df_split_temp.head(5)


# In[16]:


df_split_temp[~df_split_temp['github_link'].isna()]


# In[39]:


from src.data_preprocess import main_download, download_readme

base_output_dir = "github_readmes"
os.makedirs(base_output_dir, exist_ok=True)

df_split_temp = pd.read_csv('data/tmp_df_split_temp.csv')

download_info_df = main_download(df_split_temp, to_path="data/github_readmes_info.csv")
print(download_info_df.head())
print(f"Downloaded READMEs saved to '{to_path}'.")


# In[36]:


#df_split_temp[~df_split_temp['github_link'].isna()]


# In[15]:


df_processed[df_processed['final_url']=='https://doi.org/10.48550/ARXIV.2209.11055']['modelId']


# In[160]:


# Filter rows without pdf_type
missing_pdf_type = df_processed[df_processed['pdf_type'].isnull()]

# Print the number of rows without pdf_type
print(f"Entries without pdf_type: {len(missing_pdf_type)}")

# Print a sample row to inspect the bibtex_entry
sample_row = missing_pdf_type.sample(1)
print("Sample BibTeX Entry Without PDF Type:")
print(sample_row['bibtex_entry'].iloc[0])


# In[41]:


df_processed[(df_processed['pdf_type'].notna()) & (df_processed['pdf_type'].isin(['unknown']))]


# In[43]:


from urllib.parse import urlparse

df_processed_clean = df_processed.copy()
df_processed_clean['domain'] = df_processed_clean[df_processed_clean['pdf_type']=='unknown']['pdf_link'].apply(lambda x: urlparse(x).netloc if pd.notna(x) else None)
# Count the domains
domain_counts = df_processed_clean['domain'].value_counts()
domain_counts.index


# ### findings
# arxiv links can trust, doi links can trust, links starts with 

# In[ ]:


# check pdf link
df_split_temp['pdf_link_flat'] = df_split_temp['pdf_link'].apply(
    lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else []
)
df_split_temp_exploded = df_split_temp.explode('pdf_link_flat')
link_counts = df_split_temp_exploded['pdf_link_flat'].value_counts()
print(link_counts)


# In[ ]:


df_split_temp['extracted_markdown_table'].value_counts()

