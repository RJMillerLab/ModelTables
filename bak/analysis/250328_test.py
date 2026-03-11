#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 
import pandas as pd
df = pd.read_parquet('extracted_annotations.parquet')
df


# In[ ]:


df['extracted_tables'].iloc[0]


# In[ ]:


import pandas as pd

# 1. 读取 parquet 文件
df = pd.read_parquet('extracted_annotations.parquet')

# 2. 查看整个 DataFrame 的基本信息
print("DataFrame 行数与列数: ", df.shape)
print("DataFrame 列名: ", df.columns.tolist())

# 3. 看看前几行数据（可以只选择感兴趣的列）
#print(df[['extracted_tables', 'extracted_figures']].head(5))

# 4. 检查 extracted_tables 列中是否包含 '\begin{'
#    这里用正则表达式查找 '\\begin{'，注意需要转义
mask_tables = df['extracted_tables'].str.contains(r'begin{', na=False)
print("extracted_tables, 包含 'begin{' 的行数: ", mask_tables.sum())

# 5. 也可以查看具体哪些行有 '\begin{'，以及对应内容
df_with_latex = df[mask_tables]
print("包含 'begin{' 的前几行内容:")
#print(df_with_latex[['extracted_tables']].head(5))

# 6. 如果想看看 extracted_figures 列里是否也出现类似内容，可以再加一段：
mask_figures = df['extracted_figures'].str.contains(r'begin{', na=False)
print("extracted_figures 中包含 'begin{' 的行数: ", mask_figures.sum())
#print(df[mask_figures][['extracted_figures']].head(5))


# In[7]:


# lopad arxiv_html_cache.json
import json
with open('arxiv_html_cache.json') as f:
    data = json.load(f)
    


# In[ ]:





# In[ ]:


# 
import pandas as pd
df = pd.read_parquet('extracted_annotations.parquet')
df



# In[ ]:





# In[ ]:


# for all local files in arxiv_html_cache.json, please load them and judge whether they are metadata or fulltext
import json
import os
import shutil
from bs4 import BeautifulSoup
import pandas as pd

# Load the JSON file
with open('arxiv_html_cache.json') as f:
    data = json.load(f) # {'2109.13855v3': 'arxiv_fulltext_html/2109.13855v3.html'}

from bs4 import BeautifulSoup
import os

def classify_page(html_path):
    if not os.path.exists(html_path):
        #raise FileNotFoundError(f"File not found: {html_path}")
        return None
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    # Rule 1: Check for <meta> tags with the 'name' attribute starting with "citation_"
    meta_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.lower().startswith('citation_')})
    if meta_tags and len(meta_tags) >= 3:
        return 'metadata'
    # Rule 2: Check if there are multiple <section> or <article> tags present
    sections = soup.find_all(['section', 'article'])
    if len(sections) >= 2:
        return 'fulltext'
    # Rule 3: Check if the body text contains both "introduction" and "conclusion"
    body_text = soup.get_text(separator=' ').lower()
    if "introduction" in body_text and "conclusion" in body_text:
        return 'fulltext'
    # Default to 'metadata' if none of the above rules are met
    return 'metadata'

# Example usage: Call the function for testing
"""if __name__ == '__main__':
    test_path = '/Users/doradong/Repo/CitationLake/arxiv_fulltext_html/1309.1125v1.html'  # Replace with the actual file path
    try:
        page_type = classify_page(test_path)
        print(f"Page type: {page_type}")
    except Exception as e:
        print(e)"""

# turn the dict into a parquet, and use apply to classify the page, we already have classify_page
df = pd.DataFrame(data.items(), columns=['paper_id', 'html_path'])
df['page_type'] = df['html_path'].apply(classify_page)
df

df['page_type'].value_counts()


# In[ ]:


df['html_path']


# In[13]:


from urllib.parse import quote
import requests

def search_arxiv_title(title_query, max_results=5):
    base_url = "http://export.arxiv.org/api/query"
    # 用引号包裹标题，再编码（保留引号）
    encoded_query = quote(f'"{title_query}"')  
    params = {
        "search_query": f"ti:{encoded_query}",  # 精确搜索标题
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",  # 按相关性排序
        "sortOrder": "descending"
    }
    print(f"[DEBUG] Final arXiv API URL: {base_url}?{'&'.join(f'{k}={v}' for k,v in params.items())}")
    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[ERROR] arXiv API request failed: {e}")
        return None


# In[ ]:


import re

def preprocess_title(title):
    title = re.sub(r"[-:_*@&'\"]+", " ", title)
    return " ".join(title.split())

# 测试案例
test_cases = [
    ("fineas-financial: embedding_analysis of*sentiment", "fineas financial embedding analysis of sentiment"),
    ("Test@'\"symbols--combined::here", "Test symbols combined here"),
    ("Hello--World__2023*", "Hello World 2023"),
    ("'Quoted' \"Title\"", "Quoted Title"),
    ("arXiv@2023: New_Results-In*NLP", "arXiv 2023 New Results In NLP")
]

for original, expected in test_cases:
    processed = preprocess_title(original)
    print(f"原始: {original}\n处理: {processed}\n匹配: {processed == expected}\n---")


# In[ ]:


title = "ELEVATER A Benchmark and Toolkit for Evaluating Language Augmented Visual Models"
a = search_arxiv_title(title, max_results=5)
a


# In[ ]:


print(a)


# In[ ]:


max_results = 5
try:
    xml_text = search_arxiv_title(title, max_results=max_results)
    entries = parse_arxiv_atom(xml_text)
    print(entries)
    if not entries:
        print(f"[INFO] No Atom entries found for title: {title}")
except Exception as e:
    print(f"[ERROR] Atom feed error for '{title}': {e}")


# In[17]:


import os
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from urllib.parse import quote
import requests
import xml.etree.ElementTree as ET

def parse_arxiv_atom(xml_text):
    """
    Parse the XML (Atom feed) from arXiv to extract a list of entries.
    Each entry is a dict with keys: 'title', 'id', 'summary', 'updated', 'published'.
    """
    root = ET.fromstring(xml_text)
    ns = "{http://www.w3.org/2005/Atom}"
    entries_info = []
    for entry in root.findall(ns + 'entry'):
        entry_title = entry.find(ns + 'title')
        entry_id = entry.find(ns + 'id')
        entry_summary = entry.find(ns + 'summary')
        entry_updated = entry.find(ns + 'updated')
        entry_published = entry.find(ns + 'published')
        if entry_title is not None and entry_id is not None:
            info = {
                "title": entry_title.text.strip() if entry_title.text else "",
                "id": entry_id.text.strip() if entry_id.text else "",
                "summary": entry_summary.text.strip() if (entry_summary is not None and entry_summary.text) else "",
                "updated": entry_updated.text.strip() if (entry_updated is not None and entry_updated.text) else "",
                "published": entry_published.text.strip() if (entry_published is not None and entry_published.text) else ""
            }
            entries_info.append(info)
    return entries_info


# In[1]:


# load html_table.parquet

import pandas as pd
df = pd.read_parquet('final_integration.parquet')
df


# In[2]:


# load html_table.parquet

import pandas as pd
df = pd.read_parquet('extracted_annotations.parquet')
df


# In[7]:


df_1 = pd.read_parquet("final_integration.parquet")
df_1


# In[8]:


# python -m olmocr.pipeline ./localworkspace --pdfs tests/gnarly_pdfs/*.pdf
df_1['extracted_tables'].iloc[0]


# In[9]:


df_1['extracted_figures'].iloc[0]


# In[6]:


# load title2arxiv_new_cache.json
import json
with open('title2arxiv_new_cache.json') as f:
    data = json.load(f)
data


# In[5]:


# load pdf_download_cache.json
import json
with open('pdf_download_cache.json') as f:
    data_pdf = json.load(f)
data_pdf


# In[16]:


df = pd.read_csv("llm_outputs/llm_markdown_table_results.csv")
df


# In[18]:


print(df['llm_response_raw'].iloc[1])


# In[17]:


# deduplicate for df_read on 'html_path'
df_read_tmp = df_read.drop_duplicates(subset=['html_path'])
df_read_tmp


# In[6]:


df_read_tmp['page_type'].value_counts()


# In[9]:


df_read_tmp[['paper_number', 'paper_version']] = df_read_tmp['paper_id'].str.extract(r'(\d+\.\d+)(v\d+)?')  ########

df_read_tmp


# In[12]:


# check duplicate on paper_number
df_read_tmp[df_read_tmp['paper_number'].duplicated()]



# In[13]:


# check paper_number =xxxxxx
df_read_tmp[df_read_tmp['paper_number'] == '2410.03051']


# In[16]:


"""# load missing_titles_tmp.txt, turn each line into a key, withe value as "", save this json as tmp.json wiht indent=4

import json

def txt_to_json(input_txt, output_json):
    with open(input_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = {line.strip(): "" for line in lines if line.strip()}
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    input_txt = 'missing_titles_tmp.txt'   
    output_json = 'tmp.json'                 
    txt_to_json(input_txt, output_json)
    print(f"转换完成: {output_json}")
"""


# In[ ]:


# # load df4, remove 'hugging_table_list', 'hugging_table_list_sym', 'github_table_list', 'github_table_list_sym' and save back

import pandas as pd
df4 = pd.read_parquet("data/processed/modelcard_step2.parquet")

df4.drop(columns=['hugging_table_list', 'hugging_table_list_sym', 'github_table_list', 'github_table_list_sym'], inplace=True)

df4.to_parquet("data/processed/modelcard_step2_modified.parquet", compression='zstd', engine='pyarrow')



# In[ ]:


import pyarrow.parquet as pq

# 读取 Parquet 文件 schema 而不加载数据
parquet_file = pq.ParquetFile("data/processed/modelcard_step2.parquet")
schema = parquet_file.schema_arrow

print(schema)


# In[ ]:




