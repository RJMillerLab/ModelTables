#!/usr/bin/env python
# coding: utf-8

# In[11]:


df1 = pd.read_parquet("extracted_annotations.parquet")
df1


# In[17]:


df2 = pd.read_parquet("data/processed/modelcard_all_title_list.parquet")
df2


# In[ ]:


df = pd.read_parquet("final_integration_with_paths.parquet")
df
#'html_table_list', 'llm_table_list'


# In[22]:


df.columns


# In[ ]:





# In[ ]:





# In[7]:


df4[['hugging_table_list', 'github_table_list']]



# In[8]:


# test itesm with non [] github_table_list value counts
df4[df4['github_table_list'].apply(lambda x: len(x) > 0)]


# In[12]:


df4['github_table_list'].iloc[0]


# In[6]:


df4_non_empty = df4[df4['readme_path'].apply(lambda x: len(x) > 0)]
df4_non_empty


# In[24]:


df4.head()


# In[21]:


tmp2 ['query', 'html_table_list', 'saved_csv_paths'] #  'retrieved_title', 'html_html_path', 'pdf_pdf_path', 


# In[20]:


tmp1 = list(df1.columns)
tmp2 = list(df.columns)
# tmp1 in tmp2, check if all columns in df1 are in df
print(all([x in tmp2 for x in tmp1]))


# In[15]:





# In[14]:


import pandas as pd

file_path = "llm_markdown_table_results.csv"
df = pd.read_csv(file_path)
df


# In[24]:


# '9833371' in df['corpusid'] str

#id = '9027681'
id = '221802461'
df['corpusid'] = df['corpusid'].apply(str)
print(df[df['corpusid'].str.contains(id)].retrieved_title.iloc[0])
print(df[df['corpusid'].str.contains(id)].llm_response_raw.iloc[0])


# In[28]:


print(df[df['title']=='compositional semantic parsing on semi-structured tables']['llm_response_raw'].iloc[0])


# In[45]:


#df[df['corpusid'].str.contains(id)].html_table_list.iloc[0]
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置输出宽度自动扩展（避免折行）
pd.set_option('display.width', None)

# 如果列内容过长被截断了，也可以加这一句
pd.set_option('display.max_colwidth', None)


tmp = df[df['title']=='compositional semantic parsing on semi-structured tables']
tmp.drop(columns=['raw_json', 'combined_text', 'llm_prompt', 'extracted_tables', 'extracted_tablerefs', 'extracted_figures', 'extracted_figure_captions', 'extracted_figurerefs'])


# In[46]:


df['html_paper_id'].value_counts()


# In[4]:


import pandas as pd
import re

# 读取 parquet 文件
HTML_TABLE_PARQUET = "html_table.parquet"
df = pd.read_parquet(HTML_TABLE_PARQUET)

# 提取 arxiv_id_pure 和 arxiv_id_version
def parse_arxiv_id(paper_id):
    match = re.match(r"(\d{4}\.\d{5})(v(\d+))?", paper_id)
    if match:
        arxiv_id_pure = match.group(1)
        arxiv_id_version = int(match.group(3)) if match.group(3) else 1
        return pd.Series([arxiv_id_pure, arxiv_id_version])
    else:
        return pd.Series([paper_id, 1])  # fallback

df[['arxiv_id_pure', 'arxiv_id_version']] = df['paper_id'].apply(parse_arxiv_id)

# 保留每个 arxiv_id_pure 中 version 最大的记录
df_latest = df.sort_values('arxiv_id_version', ascending=False).drop_duplicates('arxiv_id_pure', keep='first')

# ✅ 结果保留在 df_latest
df_latest.head()


# In[8]:


TITLE2ARXIV_JSON = "title2arxiv_new_cache.json"
# load json
import json
def load_json_cache(json_file):
    with open(json_file, "r") as f:
        return json.load(f)
title2arxiv_map = load_json_cache(TITLE2ARXIV_JSON) # Example: { "Some paper title": "2301.12345v2", ... }
df_title2arxiv = pd.DataFrame(
    [(t, a) for t, a in title2arxiv_map.items()],
    columns=["retrieved_title", "arxiv_id"]
)
df_title2arxiv


# In[9]:


# 检查是否有版本号，比如 2311.12345v2
mask_has_version = df_title2arxiv['arxiv_id'].astype(str).str.contains('v', case=False)

# 打印结果
print("📌 有版本号 (含 'v') 的数量:", mask_has_version.sum())
print("📋 示例：")
print(df_title2arxiv[mask_has_version].head())


# In[10]:


# laod parquet
data_path = "data/processed/modelcard_all_title_list.parquet"
df = pd.read_parquet(data_path)
df


# In[ ]:





# In[ ]:





# In[23]:


print(df[~df['html_table_list'].isnull()].html_table_list.iloc[0])
print(df[~df['html_table_list'].isnull()].title.iloc[0])
print(df[~df['html_table_list'].isnull()].corpusid.iloc[0])


# In[ ]:


print(df[df['corpusid'].str.contains('9027681')].llm_response_raw.iloc[0])


# In[7]:


print(df[df['corpusid'].str.contains('9027681')].extracted_tables.iloc[0])
print(df[df['corpusid'].str.contains('9027681')].extracted_figures.iloc[0])


# In[8]:


print(df[df['corpusid'].str.contains('9027681')].combined_text.iloc[0])


# In[19]:


df.columns


# In[25]:


import json
json.dumps(df[df['corpusid'].str.contains('9027681')].extracted_tables.iloc[0],indent=4)


# In[29]:


import ast
import re
import json

s = df[df['corpusid'].str.contains('9027681')].extracted_tables.iloc[0]
s_fixed = s.replace("\n", "\\n")  ######## 将换行符转义
s_fixed = re.sub(r"}\s*{", "}, {", s_fixed)  ######## 在字典之间缺少逗号的地方插入逗号
data = ast.literal_eval(s_fixed)
formatted_json = json.dumps(data, indent=4)
print(formatted_json)


# In[30]:


s


# In[6]:


df.info()


# In[ ]:




