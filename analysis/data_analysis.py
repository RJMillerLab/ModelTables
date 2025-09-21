#!/usr/bin/env python
# coding: utf-8

# ### findings
# - This dataset was downloaded on January 14, 2025, from Hugging Face: [https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata](https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata).  
# - The earliest recorded time is March 2, 2022 ([GitHub documentation](https://github.com/huggingface/hub-docs/blob/main/docs/hub/api.md)), as any card created earlier is documented with this timestamp.  
# - Comparing the top 10 authors by download count with the top 10 authors by model count, we observe that some authors specialize in uploading processed datasets.  
# - On Hugging Face, if an arXiv link exists in the tags, it is always extracted and included in the tags.  

# In[2]:


import pandas as pd
import numpy as np
import os, json, re
import matplotlib.pyplot as plt
from src.utils import load_combined_data, get_statistics_card
from tqdm import tqdm
from joblib import Parallel, delayed

data_type = "modelcard" # or "datasetcard"
df = load_combined_data(data_type, file_path="~/Repo/CitationLake/data/")

stats = get_statistics_card(df)
print(json.dumps(stats, indent=4))


# In[3]:


author_downloads = df[df['card'] == 'Entry not found']['downloads'].value_counts().sort_index()

# 绘制直方图（优化颜色渐变）
plt.figure(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.8, 0.4, len(author_downloads)))
plt.bar(author_downloads.index, author_downloads.values, color=colors)

plt.xscale('log')  # 对 x 轴使用 log 缩放
plt.yscale('log')  # 对 y 轴使用 log 缩放
plt.xlabel('Downloads (log scale)', fontsize=12)
plt.ylabel('Frequency (log scale)', fontsize=12)
plt.title('Log-Scaled Histogram of Downloads for "Entry not found"', fontsize=14)
plt.tick_params(axis='both', labelsize=10)
plt.grid(False)
plt.show()


# In[ ]:


"""df['createdAt'] = pd.to_datetime(df['createdAt']).dt.date

# 统计频率
time_distribution = df[df['card'] == 'Entry not found']['createdAt'].value_counts().sort_index()

# 绘制时间分布图（优化颜色渐变）
plt.figure(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.8, 0.4, len(time_distribution)))
plt.bar(time_distribution.index, time_distribution.values, color=colors)

#plt.yscale('log')
plt.xlabel('Date (Year-Month-Day)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Time Distribution of "Entry not found" Cards', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)  # 调整 x 轴刻度显示
plt.tick_params(axis='both', labelsize=10)
plt.grid(False)
plt.show()"""


# In[ ]:


df['createdAt'] = pd.to_datetime(df['createdAt']).dt.date

# 统计每个日期总卡片数量和 'Entry not found' 数量
total_cards_per_date = df.groupby('createdAt').size()
entry_not_found_per_date = df[df['card'] == 'Entry not found'].groupby('createdAt').size()

# 计算归一化比例（占比）
normalized_proportion = (entry_not_found_per_date / total_cards_per_date).fillna(0)

# 绘制时间分布图（优化颜色渐变）
plt.figure(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.8, 0.4, len(normalized_proportion)))
plt.bar(normalized_proportion.index, normalized_proportion.values, color=colors)

plt.xlabel('Date (Year-Month-Day)', fontsize=12)
plt.ylabel('Proportion of "Entry not found"', fontsize=12)
plt.title('Proportion of "Entry not found" Cards Over Total Cards', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)  # 调整 x 轴刻度显示
plt.tick_params(axis='both', labelsize=10)
plt.grid(False)
plt.show()


# ### Another type: Invalid username or password.

# In[ ]:


# Entry not found stands for those model repo with no model cards, either because creating before 2022-03-02 or author just ignore model card editing randomly

# Invalid username or password, stands for repo readme which infringes the items, or this repo is private or disabled
# e.g. https://huggingface.co/RareConcepts/FurkinsWorld-SD35-LoKr
# e.g. https://huggingface.co/skyseven
# e.g. https://huggingface.co/coyotte508/datasets
# Abusive usage of gating (no model) (require user private information)
# https://huggingface.co/Anre3737/dreamgaussian/discussions/2
# malicious code
# https://huggingface.co/star23/baller8
# This repository has been marked as containing sensitive content and may contain potentially harmful and sensitive information.
# https://huggingface.co/CyberHarem/nino_fireemblem
# license-issue: CC BY-NC-SA 4.0
# https://huggingface.co/konohashinobi4/4xAnimesharp/discussions/1
# https://huggingface.co/deepinsight-unofficial/inswapper/discussions/2
# Repository disabled per company Stability AI’s request, infringing content matching their alleged intellectual property were found
# https://huggingface.co/ninjawick/LXDS_x9
# violating our content policy - see https://huggingface.co/content-guidelines

len(df[df['card'].isin(['Invalid username or password.'])]) #'Entry not found', 


# ### Another type: default template, or default tags

# In[ ]:


# split readme and tags
# this is processed in src.data_preprocess.step1 to get readme and tags
"""
from src.data_preprocess.step1 import extract_tags_and_readme_parallel

df_split = extract_tags_and_readme_parallel(df)"""


# In[29]:


#print(inconsistencies_df[inconsistencies_df['original_card']!='Entry not found'].loc[1]['original_card'])
#print(inconsistencies_df[inconsistencies_df['original_card']!='Entry not found'].loc[1]['restored_card'])
#inconsistencies_df[inconsistencies_df['original_card']!='Entry not found']


# ### Then parse card_tags to individual tags

# In[ ]:


# double check
card_value_counts = df_split[(df_split['card_tags'].isna()) & (df_split['card_readme'].isin(['Entry not found', 'None', None]))]['card'].value_counts()
print(card_value_counts)


# In[25]:


#df_split[df_split['modelId']=='kodonho/Solar-OrcaDPO-Solar-Instruct-SLERP']['card'].iloc[0]


# ### Parse

# In[ ]:


"""import yaml
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import re
from joblib import parallel_backend
from ruamel.yaml import YAML

# Initialize the YAML parser
yaml_parser = YAML(typ="safe")

# Function to clean YAML content
def clean_yaml_content(content):
    if content is None:
        return None
    # Replace tabs with spaces and normalize line endings
    content = content.replace('\t', ' ').replace('\r\n', '\n').replace('\r', '\n')
    return content

# Function to parse card_tags dynamically with error handling
def parse_card_tags_dynamic(card_tag, full_card_content=None, model_id=None):
    """
    解析 card_tag 字段的 YAML 内容，返回 (解析字典, 是否出错, 错误信息) 三元组
    """
    if not card_tag:
        return {}, False, None
    card_tag = clean_yaml_content(card_tag)
    try:
        # Parse YAML content using ruamel.yaml
        parsed_data = yaml_parser.load(card_tag.strip())
        # 如果解析结果不是 dict，就返回空 dict
        return parsed_data if isinstance(parsed_data, dict) else {}, False, None
    except Exception as e:
        # Capture error details for labeling
        error_message = f"Error parsing card_tags: {e}"
        print(error_message)
        print(f"Problematic card_tags content:\n{card_tag}")
        if model_id:
            print(f"Model ID: {model_id}")
        if full_card_content:
            print(f"Full card content:\n{full_card_content}")
        return {}, True, error_message  # Return error flag and message

def process_tags_and_combine_dynamic_parallel(df, n_jobs=4):
    """
    直接对整张表 df 进行并行处理，并返回最终包含解析结果的 DataFrame。
    - 新增功能：为 YAML 解析得到的所有字段添加 `card_tag_` 前缀。
    """

    def process_row(row):
        # 1. 解析 YAML
        parsed_tags, has_error, error_message = parse_card_tags_dynamic(
            row.card_tags, 
            full_card_content=row.card, 
            model_id=getattr(row, 'modelId', None)
        )

        # 2. 将所有键值对改为带 `card_tag_` 前缀
        prefixed_tags = {}
        for key, value in parsed_tags.items():
            prefixed_tags[f"card_tag_{key}"] = value

        # 3. 返回 (prefixed_tags, 所有新列名, 是否报错, 报错信息)
        return prefixed_tags, set(prefixed_tags.keys()), has_error, error_message

    total_rows = len(df)

    # 并行处理所有行
    with parallel_backend('loky'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_row)(row) 
            for row in tqdm(df.itertuples(), total=total_rows, desc=f"Processing {total_rows} Rows")
        )

    # 分解结果
    parsed_data_list, all_keys_list, error_flags, error_messages = zip(*results)

    # 聚合所有列名
    all_keys = set().union(*all_keys_list)

    # 构建包含所有新字段的 DataFrame
    parsed_results = {
        key: [data.get(key, None) for data in parsed_data_list]
        for key in all_keys
    }

    # 将错误信息加入 DataFrame
    parsed_results['error_flag'] = error_flags
    parsed_results['error_message'] = error_messages

    # 将解析结果转换为 DataFrame
    parsed_df = pd.DataFrame(parsed_results)

    # 将原 DataFrame 与解析结果合并
    combined_df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)
    return combined_df

# =====================
# 示例用法：
df_split_temp = process_tags_and_combine_dynamic_parallel(df_split, n_jobs=6)
print(df_split_temp.head())
# =====================
"""


# ### Use YAML to load

# In[ ]:


# stored in src/data_preprocess/step2.py
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from ruamel.yaml import YAML

from src.data_preprocess.step2 import chunked_parallel_processing
processed_chunks = chunked_parallel_processing(df_split, chunk_size=1000, n_jobs=6)
df_split_temp = pd.concat(processed_chunks, ignore_index=True)
print(df_split_temp.head())


# In[ ]:


"""# store the data
import pandas as pd
import json

def clean_and_save_parquet(df, file_path):
    # Iterate through all columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Handle mixed types by converting lists/dicts to JSON strings
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
            # Convert the column to string type for consistency
            df[col] = df[col].astype(str)
        # Fill NaN values with an empty string for object columns
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("")
    
    # Save the cleaned DataFrame as a Parquet file
    try:
        df.to_parquet(file_path, compression='zstd', engine='pyarrow', index=False)
        print(f"File saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving Parquet file: {e}")

# Usage
file_path = f"data/{data_type}_step2_data_split_tags.parquet"
clean_and_save_parquet(df_split_temp, file_path)"""


# In[ ]:


"""# save
import pandas as pd
import json

def save_parquet_lossless(df, file_path):
    """
    Save a DataFrame to a Parquet file with no data loss.
    
    Args:
    - df: The DataFrame to save.
    - file_path: Path to save the Parquet file.
    """
    # Serialize only columns with lists or dicts
    serialized_columns = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                # Serialize lists and dicts into JSON strings
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
                serialized_columns[col] = True
            else:
                serialized_columns[col] = False
        else:
            serialized_columns[col] = False

    # Save the DataFrame and serialized column metadata
    metadata = {"serialized_columns": serialized_columns}
    df.to_parquet(file_path, index=False, engine="pyarrow", metadata={"custom_metadata": json.dumps(metadata)})
    print(f"File saved successfully to {file_path}")
file_path = f"data/{data_type}_step2_data_split_tags.parquet"
save_parquet_lossless(df_split_temp, file_path)"""


# In[6]:


df_split_temp.to_csv(f"data/{data_type}_step2_data_split_tags.csv")


# In[ ]:


import pandas as pd
data_type = "modelcard"
df_restored = pd.read_csv(f"data/{data_type}_step2_data_split_tags.csv")


# In[ ]:


#assert df_split_temp.equals(df_restored), "Data mismatch after saving and loading!"
#print("Data loaded successfully with no loss!")


# In[ ]:


"""def load_parquet_lossless(file_path):
    """
    Load a Parquet file without data loss, restoring serialized columns.
    
    Args:
    - file_path: Path to the Parquet file.
    
    Returns:
    - Restored DataFrame.
    """
    # Load the DataFrame and metadata
    df = pd.read_parquet(file_path, engine="pyarrow", metadata=True)
    metadata = json.loads(df.attrs.get("custom_metadata", "{}"))

    # Restore serialized columns
    if "serialized_columns" in metadata:
        for col, is_serialized in metadata["serialized_columns"].items():
            if is_serialized and col in df.columns:
                # Deserialize JSON strings back into Python objects
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith(("{", "[")) else x
                )
    
    return df

df_restored = load_parquet_lossless(file_path)
assert df_split_temp.equals(df_restored), "Data mismatch after saving and loading!"
print("Data loaded successfully with no loss!")"""


# In[5]:


"""# load the data

import pandas as pd
import json
def load_and_restore_parquet(file_path, columns_to_restore):
    # Load the Parquet file
    df = pd.read_parquet(file_path)
    # Deserialize JSON strings back into Python objects
    for col in columns_to_restore:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith(("{", "[")) else x
            )
    return df
# Usage
file_path = f"data/{data_type}_step2_data_split_tags.parquet"
columns_to_restore = ["card_tags_widget"]  # Add all columns you serialized during saving
df_restored = load_and_restore_parquet(file_path, columns_to_restore)"""


# In[ ]:


df_tmp_value = df_restored.isnull().sum()  # 查看是否存在空值
df_tmp_value


# In[ ]:


df_tmp_value[df_tmp_value.values==0]


# In[ ]:


df_split_temp[['card_tags_metadata']].value_counts()


# In[ ]:


print((df_restored == "").sum())  # 检查有多少空字符串


# In[ ]:


# 


# In[ ]:


# Filter columns with 'card_tags_' in their names
card_tags_cols = [col for col in df_split_temp.columns if col.startswith('card_tags_')]

# Count the number of null values in these columns
card_tags_null_count = df_split_temp[card_tags_cols].isnull().sum()

# Create a DataFrame for better display
card_tags_null_count_df = pd.DataFrame({
    'Column': card_tags_null_count.index,
    'Null Count': card_tags_null_count.values
})
card_tags_null_count_df


# In[ ]:


df_split_temp.columns


# In[26]:


"""for i, chunk in enumerate(processed_chunks):
    duplicate_columns = chunk.columns[chunk.columns.duplicated()]
    if len(duplicate_columns) > 0:
        print(f"Chunk {i} has duplicate columns: {duplicate_columns.tolist()}")
"""


# In[ ]:


"""# 筛选出所有以 'card_tags' 开头的列
card_tags_columns = [col for col in df_split_temp.columns if col.startswith('card_tags')]

# 提取这些列的数据并逐列打印
for col in card_tags_columns:
    print(f"{col}: {df_split_temp[col].iloc[2]}")
    print('-')
"""


# In[93]:


# check the values from given value list
valid_licenses = [
    "afl-3.0", "agpl-3.0", "apache-2.0", "apple-ascl", "artistic-2.0",
    "bigcode-openrail-m", "bigscience-bloom-rail-1.0", "bigscience-openrail-m",
    "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "bsl-1.0",
    "c-uda", "cc", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "cc-by-nc-2.0", "cc-by-nc-3.0", "cc-by-nc-4.0", "cc-by-nc-nd-3.0",
    "cc-by-nc-nd-4.0", "cc-by-nc-sa-2.0", "cc-by-nc-sa-3.0", "cc-by-nc-sa-4.0",
    "cc-by-nd-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "cc0-1.0",
    "cdla-permissive-1.0", "cdla-permissive-2.0", "cdla-sharing-1.0",
    "creativeml-openrail-m", "deepfloyd-if-license", "ecl-2.0", "epl-1.0",
    "epl-2.0", "etalab-2.0", "eupl-1.1", "gemma", "gfdl", "gpl", "gpl-2.0",
    "gpl-3.0", "intel-research", "isc", "lgpl", "lgpl-2.1", "lgpl-3.0",
    "lgpl-lr", "llama2", "llama3", "llama3.1", "llama3.2", "lppl-1.3c",
    "mit", "mpl-2.0", "ms-pl", "ncsa", "odbl", "odc-by", "ofl-1.1",
    "openrail", "openrail++", "osl-3.0", "other", "pddl", "postgresql",
    "unknown", "unlicense", "wtfpl", "zlib"
]

df_split_temp['is_valid_license'] = df_split_temp['card_tags_license'].apply(
    lambda licenses: all(
        license.split('#')[0].replace(' ', '').replace('"', '').replace("'", "") in valid_licenses
        for license in licenses
    ) if isinstance(licenses, list) else False
)
# 输出不合法的 license 行
invalid_licenses_df = df_split_temp[~df_split_temp['is_valid_license']]


# In[ ]:


valid_licenses_lower = [i.lower() for i in valid_licenses]

def clean_and_validate(row):
    if not isinstance(row, str):
        return False
    # Remove comments starting with #
    row = row.split('#')[0].strip().lower()
    # Clean up brackets and quotes
    cleaned = row.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
    # Split by comma and strip each tag
    tags = [tag.strip() for tag in cleaned.split(",") if tag.strip()]
    return all(tag in valid_licenses_lower for tag in tags)

df_split_temp['all_valid'] = df_split_temp['card_tags_license'].apply(clean_and_validate)

non_default_values = df_split_temp[~df_split_temp['all_valid']]

print(f"不属于 valid_licenses 的记录数量: {len(non_default_values)}")
non_default_values['card_tags_license'].value_counts()


# In[ ]:


# 打印符合条件的行的全部内容
selected_rows = df_split_temp[
    df_split_temp['card_tags_license_cleaned'].isin([
        'other, License Rights and Redistribution., Subject to your compliance with this Agreement and the Documentation, Stability AI grants you a non-exclusive, worldwide, non-transferable, non-sublicensable, revocable, royalty free and limited license under Stability AI’s intellectual property or other rights owned by Stability AI embodied in the Software Products to reproduce, distribute, and create derivative works of the Software Products for purposes other than commercial or production use., You will not, and will not permit, assist or cause any third party to use, modify, copy, reproduce, create derivative works of, or distribute the Software Products (or any derivative works thereof, works incorporating the Software Products, or any data produced by the Software), in whole or in part, for any commercial or production purposes., If you distribute or make the Software Products, or any derivative works thereof, available to a third party, you shall (i) provide a copy of this Agreement to such third party, and (ii) retain the following attribution notice within a Notice text file distributed as a part of such copies: Japanese StableLM is licensed under the Japanese StableLM Research License, Copyright (c) Stability AI Ltd. All Rights Reserved.”, The licenses granted to you under this Agreement are conditioned upon your compliance with the Documentation and this Agreement, including the Acceptable Use Policy below and as may be updated from time to time in the future on stability.ai, which is hereby incorporated by reference into this Agreement., Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE SOFTWARE PRODUCTS  AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN AS IS BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE SOFTWARE PRODUCTS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE SOFTWARE PRODUCTS AND ANY OUTPUT AND RESULTS., Limitation of Liability. IN NO EVENT WILL STABILITY AI OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF STABILITY AI OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING., Intellectual Property., No trademark licenses are granted under this Agreement, and in connection with the Software Products, neither Stability AI nor Licensee may use any name or mark owned by or associated with the other or any of its affiliates, except as required for reasonable and customary use in describing and redistributing the Software Products., Subject to Stability AI’s ownership of the Software Products and derivatives made by or for Stability AI, with respect to any derivative works and modifications of the Software Products that are made by you, as between you and Stability AI, you are and will be the owner of such derivative works and modifications., If you institute litigation or other proceedings against Stability AI (including a cross-claim or counterclaim in a lawsuit) alleging that the Software Products or associated outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Stability AI from and against any claim by any third party arising out of or related to your use or distribution of the Software Products in violation of this Agreement., Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Software Products and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Stability AI may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the Software Products. Sections 2-4 shall survive the termination of this Agreement.', 'other, Yes, No'
    ])
]

# 打印选中的行
print(selected_rows['card'].to_string(index=False))


# In[ ]:


print(selected_rows.iloc[3]['card'])


# ### save

# In[ ]:


"""

import pandas as pd
import json

def save_parquet_lossless(df, file_path):
    """
    Save a DataFrame to a Parquet file with no data loss.
    
    Args:
    - df: The DataFrame to save.
    - file_path: Path to save the Parquet file.
    """
    # Serialize only columns with lists or dicts
    serialized_columns = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                # Serialize lists and dicts into JSON strings
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
                serialized_columns[col] = True
            else:
                serialized_columns[col] = False
        else:
            serialized_columns[col] = False

    # Save the DataFrame and serialized column metadata
    metadata = {"serialized_columns": serialized_columns}
    df.to_parquet(file_path, index=False, engine="pyarrow", metadata={"custom_metadata": json.dumps(metadata)})
    print(f"File saved successfully to {file_path}")
    
file_path = f"data/{data_type}_step2_data_split_tags.parquet"
save_parquet_lossless(df_split_temp, file_path)"""


# In[70]:


for col in df_split_temp.columns:
    if df_split_temp[col].apply(lambda x: isinstance(x, list)).any():
        df_split_temp[col] = df_split_temp[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)

df_split_temp.to_parquet("data/{data_type}_step2_data_split_tags.parquet", compression='zstd', engine='pyarrow', index=False)


# In[2]:


import pandas as pd

# 加载 Parquet 文件
df_split_temp_loaded = pd.read_parquet("data/{data_type}_step2_data_split_tags.parquet")

# 如果某些列需要还原为列表类型
columns_to_restore = [
    "card_tags_license",  # 示例列名
    "card_tags_language",  # 示例列名
    "card_tags_tags"  # 示例列名
]

# 将逗号分隔的字符串还原为列表
for col in columns_to_restore:
    if col in df_split_temp_loaded.columns:
        df_split_temp_loaded[col] = df_split_temp_loaded[col].apply(
            lambda x: x.split(", ") if isinstance(x, str) and x else []
        )


# In[3]:


df_split_temp = df_split_temp_loaded


# In[ ]:


df_split_temp.info()


# ### check default template

# In[ ]:


import re
from joblib import Parallel, delayed

# 提取默认模板的关键句
DEFAULT_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for {{ model_id | default("Model ID", true) }}

<!-- Provide a quick summary of what the model is/does. -->

{{ model_summary | default("", true) }}

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

{{ model_description | default("", true) }}

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **Funded by [optional]:** {{ funded_by | default("[More Information Needed]", true)}}
- **Shared by [optional]:** {{ shared_by | default("[More Information Needed]", true)}}
- **Model type:** {{ model_type | default("[More Information Needed]", true)}}
- **Language(s) (NLP):** {{ language | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}}

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

{{ direct_use | default("[More Information Needed]", true)}}

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

{{ downstream_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

{{ out_of_scope_use | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

{{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}}

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Model Card Authors [optional]

{{ model_card_authors | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}
"""
# 提取默认模板中所有 `<!-- -->` 包含的关键语句
default_keywords = [
    keyword.strip() for keyword in re.findall(r"<!--(.*?)-->", DEFAULT_TEMPLATE, re.DOTALL)
]

def normalize_text(text: str) -> str:
    """
    归一化文本：去除多余空白、换行、符号，转为小写。
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def is_default_card(card_content: str) -> bool:
    """
    判断 `card_content` 是否为默认模板：
    如果包含所有 default_keywords，则返回 True，否则 False。
    """
    if not isinstance(card_content, str):
        return False
    
    # 归一化卡片内容
    normalized_card = normalize_text(card_content)
    # 检查是否包含所有默认关键句
    return all(normalize_text(keyword) in normalized_card for keyword in default_keywords)

# 并行处理检查每个 card_readme 是否为默认模板
df_split_temp['is_default_card'] = Parallel(n_jobs=-1)(
    delayed(is_default_card)(row) for row in df_split_temp['card_readme']
)

# 汇总统计结果
default_count = df_split_temp['is_default_card'].sum()
total_count = len(df_split_temp)
non_default_count = total_count - default_count

# 打印结果
print(f"Default cards: {default_count}/{total_count} = {default_count / total_count:.2%}")
print(f"Non-default cards: {non_default_count}/{total_count} = {non_default_count / total_count:.2%}")

# 可选：查看哪些卡片被判定为默认模板
print(df_split_temp[df_split_temp['is_default_card']][['card_readme']])


# In[ ]:


df_tmptmp = df_split_temp[
    (~df_split_temp['is_default_card']) & 
    (~df_split_temp['card_readme'].str.contains("Entry not found", case=False, na=False))
][['modelId','card_readme']]
df_tmptmp.iloc[1]


# In[ ]:


print(df_tmptmp[df_tmptmp['modelId']=='Deci/DeciLM-6b']['card_readme'].iloc[0])


# ### TODO: Little issue about the extraction of markdown tables, can not process multiple-lines now

# In[ ]:


import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

markdown_tables_df = df_split_temp[df_split_temp['contains_markdown_table']]
print(f"Cards with Markdown tables: {len(markdown_tables_df)}")
print(markdown_tables_df[['card_readme', 'extracted_markdown_table']])


# In[ ]:


151088/1108759


# In[ ]:


markdown_tables_df[['modelId', 'extracted_markdown_table']]


# In[ ]:


print(markdown_tables_df[['extracted_markdown_table']].iloc[2]['extracted_markdown_table'])


# In[ ]:


markdown_tables_df['extracted_markdown_table'].value_counts()


# In[ ]:


#df_tmptmp.iloc[565482]['card_readme']


# ### check gated model
# 

# In[ ]:


import pandas as pd
import re

def analyze_keywords(tags):
    # 检测所有以 "extra_gated_" 开头的短语
    if not isinstance(tags, str):
        return []
    return re.findall(r'extra_gated_\w+', tags)

# 分析关键词并生成新的分析列
df_split_temp['detected_keywords'] = df_split_temp['card_tags'].apply(analyze_keywords)

# 展平所有关键词列表，并统计每种关键词的出现次数
all_keywords = df_split_temp['detected_keywords'].explode()
keyword_counts = all_keywords.value_counts()

# 创建一个新的DataFrame，用于存储包含目标关键词的记录
detected_rows = df_split_temp[df_split_temp['detected_keywords'].apply(len) > 0]

# 仅保留相关列，显示model_id和检测到的关键词
detected_results = detected_rows[['modelId', 'detected_keywords']]

# 打印关键词统计信息
print("Keyword counts:")
print(keyword_counts)

# 打印包含关键词的行及其model_id
print("\nDetected items with model_id:")
print(detected_results)


# In[49]:


df_split_temp.to_parquet("data/{data_type}_step3_markdown_gated.parquet", compression='zstd', engine='pyarrow', index=False)


# In[1]:


import pandas as pd
df_split_temp = pd.read_parquet("data/{data_type}_step3_markdown_gated.parquet")


# In[ ]:





# In[ ]:


df_split['card_readme'].value_counts()


# ### get BibTex

# In[ ]:


# get 
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def detect_bibtex_entry(card_content: str):
    """
    检测并提取 BibTeX entry 和 citation 信息。
    返回 (是否包含 BibTeX, 提取的 BibTeX entry)。
    """
    if not isinstance(card_content, str):
        return (False, None)
    # BibTeX entry 的正则表达式
    bibtex_pattern = (
        r"@(?P<type>\w+)\{(?P<key>[\w:-]+),\s*"  # 匹配 @type{key,
        r"(?:[^@]*?"                             # 匹配内容（非 @ 符号）
        r"(author|title|year|journal|url|archivePrefix|eprint|biburl|bibsource)"  # 匹配常见字段
        r"\s*=\s*{[^}]*},?)*"                    # 匹配字段值对
        r"\s*\}"                                 # 结束标记 }
    )
    # 搜索 BibTeX entry
    match = re.search(bibtex_pattern, card_content, re.DOTALL)
    if match:
        return (True, match.group(0).strip())
    # 没有 BibTeX entry
    return (False, None)

def process_row(row):
    return detect_bibtex_entry(row)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(process_row, df_split_temp['card_readme']), total=len(df_split_temp)))

# 将结果添加到 DataFrame
df_split_temp[['contains_bibtex', 'extracted_bibtex']] = pd.DataFrame(results, index=df_split_temp.index)

# 打印包含 BibTeX entry 的行
bibtex_entries_df = df_split_temp[df_split_temp['contains_bibtex']]
print(f"Cards with BibTeX entries: {len(bibtex_entries_df)}")
print(bibtex_entries_df[['card_readme', 'extracted_bibtex']])


# In[ ]:


bibtex_entries_df[['modelId', 'extracted_bibtex']]


# In[ ]:


bibtex_entries_df['extracted_bibtex'].value_counts()


# In[ ]:


19658/1108759


# ### Know the base_model, datasets, spaces

# In[ ]:


df.columns


# In[ ]:


# dataset in tags
# 
import pandas as pd
import ast
df_split_temp['tags_list'] = df_split_temp['tags']
non_empty_tags = df_split_temp['tags_list'].apply(lambda x: len(x) > 0)
print(f"Rows with non-empty tags: {non_empty_tags.sum()} out of {len(df_split_temp)}")
arxiv_count = df_split_temp['tags_list'].apply(lambda x: any('dataset:' in tag for tag in x))
proportion_with_arxiv = arxiv_count.sum() / len(df_split_temp)
numerator = arxiv_count.sum()
denominator = len(df_split_temp)
print(f"Proportion of items with dataset in tags: {numerator}/{denominator} = {numerator / denominator:.2%}")


# In[ ]:


import pandas as pd

# 假设 df_split_temp['card_tags'] 为字符串类型，包含每个模型的标签

# 检查非空 card_tags 的数量
non_empty_card_tags = df_split_temp['card_tags'].apply(lambda x: x is not None and len(x) > 0)
print(f"Rows with non-empty card_tags: {non_empty_card_tags.sum()} out of {len(df_split_temp)}")

# 检查 card_tags 是否包含 'datasets:'
datasets_in_card_tags_count = df_split_temp['card_tags'].apply(lambda x: 'datasets:' in str(x))
proportion_with_datasets_in_card_tags = datasets_in_card_tags_count.sum() / len(df_split_temp)

# 计算分子和分母
numerator = datasets_in_card_tags_count.sum()
denominator = len(df_split_temp)

print(f"Proportion of items with 'datasets:' in card_tags: {numerator}/{denominator} = {numerator / denominator:.2%}")

# 打印包含 'datasets:' 的模型 ID
models_with_datasets_in_card_tags = df_split_temp.loc[datasets_in_card_tags_count, 'modelId'].tolist()
print(f"Models with 'datasets:' in card_tags:\n{models_with_datasets_in_card_tags}")


# In[ ]:


import pandas as pd

# 假设 df_split_temp['card_readme'] 为字符串类型，包含每个模型的 readme 内容

# 检查非空 card_readme 的数量
non_empty_card_readme = df_split_temp['card_readme'].apply(lambda x: x is not None and len(x) > 0)
print(f"Rows with non-empty card_readme: {non_empty_card_readme.sum()} out of {len(df_split_temp)}")

# 检查 card_readme 是否包含 'huggingface.io/dataset/'
datasets_in_card_readme_count = df_split_temp['card_readme'].apply(lambda x: 'huggingface.co/datasets/' in str(x))
proportion_with_datasets_in_card_readme = datasets_in_card_readme_count.sum() / len(df_split_temp)

# 计算分子和分母
numerator = datasets_in_card_readme_count.sum()
denominator = len(df_split_temp)

print(f"Proportion of items with huggingface.co/datasets/' in card_readme: {numerator}/{denominator} = {numerator / denominator:.2%}")

# 打印包含 'huggingface.io/dataset/' 的模型 ID
models_with_datasets_in_card_readme = df_split_temp.loc[datasets_in_card_readme_count, 'modelId'].tolist()
print(f"Models with 'huggingface.co/datasets/' in card_readme:\n{models_with_datasets_in_card_readme}")


# In[ ]:


# single item
df_split_temp['tags'].iloc[0]


# In[ ]:


print(df_split_temp[df_split_temp['modelId']=='sentence-transformers/all-mpnet-base-v2']['card_tags'].iloc[0])


# In[ ]:


import requests
from bs4 import BeautifulSoup
import re

# 获取 arXiv 论文页面的 HTML
def get_arxiv_html(arxiv_url):
    response = requests.get(arxiv_url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve the page: {response.status_code}")
        return None

# 解析 HTML 并提取表格及相关信息
def extract_tables_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 提取所有的 <table> 标签
    tables = soup.find_all('table')
    
    if not tables:
        print("No tables found.")
        return []
    
    table_data = []
    
    # 遍历每一个表格
    for idx, table in enumerate(tables):
        table_info = {}
        
        # 提取表格的 caption（标题）
        caption = table.find('caption')
        table_info['caption'] = caption.text.strip() if caption else None
        
        # 提取表格的行（<tr> 标签）和列（<td> 标签）
        rows = table.find_all('tr')  # 提取表格行
        table_rows = []
        
        for row in rows:
            cols = row.find_all('td')  # 提取每一列
            cols = [col.text.strip() for col in cols]  # 获取每列的文本
            table_rows.append(cols)
        
        table_info['data'] = table_rows
        
        # 查找与表格相关的引用，通常包括“Table X”之类的文本
        text = soup.get_text()
        table_references = re.findall(r'Table\s+\d+', text)  # 匹配类似 "Table 1", "Table 2" 的模式
        table_info['references'] = [ref for ref in table_references if ref in text]
        
        table_data.append(table_info)
    
    return table_data

# 打印表格内容及其引用
def print_table_data(table_data):
    for idx, table in enumerate(table_data):
        print(f"\nTable {idx+1}:")
        print(f"Caption: {table['caption']}")
        print("Data:")
        for row in table['data']:
            print(row)
        print(f"References: {table['references']}")

# 示例：解析 arXiv 论文页面的表格
arxiv_url = 'https://arxiv.org/html/2402.11451v2'  # 替换为你感兴趣的论文页面链接
html_content = get_arxiv_html(arxiv_url)

if html_content:
    tables = extract_tables_from_html(html_content)
    print_table_data(tables)


# In[ ]:


# getting links from arxiv link
import requests
from bs4 import BeautifulSoup
import re

# 获取 arXiv 论文页面的 HTML
def get_arxiv_html(arxiv_url):
    response = requests.get(arxiv_url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve the page: {response.status_code}")
        return None

# 提取所有的链接
def extract_links_from_arxiv_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 提取所有的 <a> 标签
    links = soup.find_all('a', href=True)
    
    link_data = {
        'bibtex': [],
        'huggingface': [],
        'external': [],
        'other': []
    }
    
    # 分类链接
    for link in links:
        href = link['href']
        
        # Bibtex link
        if 'bibtex' in href:
            link_data['bibtex'].append(href)
        # Hugging Face link
        elif 'huggingface.co' in href:
            link_data['huggingface'].append(href)
        # 外部链接（外部平台、代码库等）
        elif re.match(r'http[s]?://', href):
            link_data['external'].append(href)
        else:
            link_data['other'].append(href)
    
    return link_data

# 示例：解析 arXiv 论文页面的所有链接
arxiv_url = 'https://arxiv.org/abs/2103.00020'  # 替换为你感兴趣的论文页面链接
html_content = get_arxiv_html(arxiv_url)

if html_content:
    links = extract_links_from_arxiv_html(html_content)
    print("Bibtex Links:", links['bibtex'])
    print("Huggingface Links:", links['huggingface'])
    print("External Links:", links['external'])
    print("Other Links:", links['other'])


# In[ ]:


from bs4 import BeautifulSoup


soup = BeautifulSoup(html_content, "html.parser")

# 找到包含多个选项卡（Tabs）的最外层 div
labstabs_div = soup.find("div", id="labstabs")

# 用来存储最终解析结果的结构
results = []

if labstabs_div:
    # arXivLabs 区块中，每个 tab 都是这样的结构：
    #   <input type="radio" name="tabs" id="xxx">
    #   <label for="xxx">TAB标题</label>
    #   <div class="tab"> ... 工具列表 ... </div>
    #
    # 注意：有时 label/tabs 的顺序略有差异，但大致类似
    #
    # 先逐个获取 "label" 元素，随后去找它的后续兄弟节点 <div class="tab">
    labels = labstabs_div.find_all("label")
    for label in labels:
        tab_title = label.get_text(strip=True)
        
        # 下一个兄弟节点可能是对应内容的 <div class="tab"> 块
        tab_div = label.find_next_sibling("div", class_="tab")
        if not tab_div:
            # 如果没有找到对应的内容块，则跳过
            continue
        
        # 如果是 "About arXivLabs" 这种纯文本的区块，处理方式略有不同
        # 我们先试着获取是否存在“工具列表”的结构
        lab_rows = tab_div.select(".columns.is-mobile.lab-row")
        
        # 如果没有任何lab-row，可能是About arXivLabs那种文本介绍
        if not lab_rows:
            # 把文本保存起来即可
            about_text = tab_div.get_text(strip=True)
            results.append({
                "tab_title": tab_title,
                "info": about_text,
                "tools": []
            })
            continue
        
        # 否则就是常规的（有开关、有工具名称和链接）的格式
        tab_info = {
            "tab_title": tab_title,
            "tools": []
        }
        
        # 遍历每一行（lab-row），解析工具名称和对应的 "What is ...?" 链接
        for row in lab_rows:
            # 第二列 .lab-name 里通常有： 工具名称 (What is XXX?)
            name_col = row.select_one(".lab-name")
            if name_col:
                raw_text = name_col.get_text(strip=True)
                # 例如 "Bibliographic Explorer (What is the Explorer?)"
                
                # 通常还会有一个 a 标签指向“what is ...”
                link_tag = name_col.find("a", href=True)
                link_href = link_tag["href"] if link_tag else None
                link_text = link_tag.get_text(strip=True) if link_tag else None
                
                # 去除多余符号，只保留前面的“工具名”
                # 方法之一：以左括号 "(" 为分割点（也可用别的方式精细处理）
                if "(" in raw_text:
                    tool_name = raw_text.split("(")[0].strip()
                else:
                    tool_name = raw_text
                
                tab_info["tools"].append({
                    "tool_name": tool_name,
                    "what_is_link_text": link_text,
                    "what_is_link": link_href
                })
        
        results.append(tab_info)

# 展示解析后结构
import json
print(json.dumps(results, indent=4, ensure_ascii=False))


# In[ ]:


print(html_content)


# In[ ]:


from bs4 import BeautifulSoup

def parse_arxiv_links(html_content):
    """
    解析 arXiv 页面中的 Bibliographic Tools, Code, Data, Media 等信息。
    
    参数:
        html_content (str): arXiv 页面 HTML 源代码。

    返回:
        dict: 包含解析结果的字典，如 Bibliographic Tools, Code, Data, Media 等链接。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    result = {
        "Bibliographic Tools": None,
        "Code": None,
        "Data": None,
        "Media": None
    }

    # 解析 Bibliographic Tools
    biblio_tools = soup.find('div', {'id': 'biblio-tools'})
    if biblio_tools:
        links = biblio_tools.find_all('a')
        result["Bibliographic Tools"] = [link['href'] for link in links if 'href' in link.attrs]
    
    # 解析 Code
    code_section = soup.find('div', {'id': 'code-links'})
    if code_section:
        links = code_section.find_all('a')
        result["Code"] = [link['href'] for link in links if 'href' in link.attrs]

    # 解析 Data
    data_section = soup.find('div', {'id': 'data-links'})
    if data_section:
        links = data_section.find_all('a')
        result["Data"] = [link['href'] for link in links if 'href' in link.attrs]

    # 解析 Media
    media_section = soup.find('div', {'id': 'media-links'})
    if media_section:
        links = media_section.find_all('a')
        result["Media"] = [link['href'] for link in links if 'href' in link.attrs]

    return result

# 测试解析
parsed_links = parse_arxiv_links(html_content)
print(parsed_links)


# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np

# 假设 df_split_temp['tags_list'] 为包含每个模型标签的列表

# 解析出所有包含 'datasets:' 的标签
def extract_datasets(card_tags):
    datasets = []
    # 如果 card_tags 是字符串类型（如逗号分隔的标签）
    if isinstance(card_tags, str):
        card_tags = card_tags.split(',')  # 将字符串按逗号分割成列表
        
    # 如果 card_tags 是 NumPy 数组类型，转换为列表
    elif isinstance(card_tags, np.ndarray):
        card_tags = card_tags.tolist()  # 将 NumPy 数组转换为列表
        
    # 如果 card_tags 是列表类型
    for tag in card_tags:
        if tag.strip().replace('"','').replace("'","").startswith('dataset:'):  # 判断是否包含 'datasets:' 标签
            datasets.append(tag.strip())
    return datasets

df_split_temp['datasets'] = df_split_temp['tags_list'].apply(lambda x: extract_datasets(x))

# 统计每个 dataset 的出现次数
all_datasets = [dataset for datasets_list in df_split_temp['datasets'] for dataset in datasets_list]
datasets_count = pd.Series(all_datasets).value_counts()

# 打印结果
print(f"Total number of datasets: {datasets_count.sum()}")
print(f"Number of unique datasets: {datasets_count.shape[0]}")
print("Dataset counts:\n", datasets_count)

# 如果您希望查看包含某个 dataset 的模型 ID，可以在此进行筛选
# dataset_to_check = 'datasets:example_dataset'  # 替换为您需要查询的 dataset
# models_with_specific_dataset = df_split_temp[df_split_temp['datasets'].apply(lambda x: dataset_to_check in x)]['modelId'].tolist()
# print(f"Models with '{dataset_to_check}':\n", models_with_specific_dataset)


# In[ ]:


df_split_temp['tags_list'].iloc[0]


# ### base_model
# 

# In[ ]:


import pandas as pd
import numpy as np

# 假设 df_split_temp['tags_list'] 为包含每个模型标签的列表

# 检测是否包含 'base_model' 或 'base_model_relation' 的标签
def extract_keywords(card_tags, keywords):
    if not card_tags:
        return []
    found_keywords = []
    if any(keyword in card_tags.strip().replace('"', '').replace("'", "") for keyword in keywords):
        found_keywords.append(card_tags.strip())
    return found_keywords

# 要检测的关键词
keywords = ['base_model', 'base_model_relation']

# 提取每行标签中的 base_model 和 base_model_relation
df_split_temp['base_model_keywords'] = df_split_temp['card_tags'].apply(lambda x: extract_keywords(x, keywords))

# 计算每个关键词的出现次数
base_model_count = df_split_temp['keywords'].apply(lambda x: any('base_model' in tag for tag in x)).sum()
base_model_relation_count = df_split_temp['keywords'].apply(lambda x: any('base_model_relation' in tag for tag in x)).sum()

# 计算总行数
total_items = len(df_split_temp)

# 计算占比
base_model_ratio = base_model_count / total_items
base_model_relation_ratio = base_model_relation_count / total_items

# 打印结果
print(f"Total number of items: {total_items}")
print(f"Number of items with 'base_model': {base_model_count} ({base_model_ratio:.2%})")
print(f"Number of items with 'base_model_relation': {base_model_relation_count} ({base_model_relation_ratio:.2%})")

# 如果您希望查看包含某个关键词的模型 ID，可以在此进行筛选
# base_model_to_check = 'base_model'  # 或者 'base_model_relation'
# models_with_specific_base_model = df_split_temp[df_split_temp['keywords'].apply(lambda x: any(base_model_to_check in tag for tag in x))]['modelId'].tolist()
# print(f"Models with '{base_model_to_check}':\n", models_with_specific_base_model)


# In[ ]:


import pandas as pd

# 假设 df_split_temp['card_tags'] 为字符串类型，包含每个模型的标签

# 检查是否包含 'base_model_relation'
def contains_base_model_relation(card_tags):
    if not card_tags:
        return False
    return 'base_model' in card_tags

# 统计包含 'base_model_relation' 的项目数量
base_model_relation_count = df_split_temp['card_tags'].apply(contains_base_model_relation).sum()

# 计算总行数
total_items = len(df_split_temp)

# 打印结果
print(f"Total number of items: {total_items}")
print(f"Number of items with 'base_model': {base_model_relation_count} ({base_model_relation_count / total_items:.2%})")


# In[ ]:


df_split_temp[df_split_temp['card_tags'].apply(contains_base_model_relation)]


# ### Github

# In[61]:


#restricted_entries = ['Entry not found', 'Invalid username or password.']

# check items not follow the \n---\n
#df_tmp = df[~df['card'].isin(restricted_entries)].reset_index(drop=True)
#df_tmp[~df_tmp['card'].str.contains('---', na=False)]


# In[62]:


#df[~df['card'].isin(['Entry not found', 'Invalid username or password.'])]['card'].value_counts()


# In[ ]:


# detect licenses
'''import re

valid_licenses = [
    "afl-3.0", "agpl-3.0", "apache-2.0", "apple-ascl", "artistic-2.0",
    "bigcode-openrail-m", "bigscience-bloom-rail-1.0", "bigscience-openrail-m",
    "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "bsl-1.0",
    "c-uda", "cc", "cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0",
    "cc-by-nc-2.0", "cc-by-nc-3.0", "cc-by-nc-4.0", "cc-by-nc-nd-3.0",
    "cc-by-nc-nd-4.0", "cc-by-nc-sa-2.0", "cc-by-nc-sa-3.0", "cc-by-nc-sa-4.0",
    "cc-by-nd-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "cc0-1.0",
    "cdla-permissive-1.0", "cdla-permissive-2.0", "cdla-sharing-1.0",
    "creativeml-openrail-m", "deepfloyd-if-license", "ecl-2.0", "epl-1.0",
    "epl-2.0", "etalab-2.0", "eupl-1.1", "gemma", "gfdl", "gpl", "gpl-2.0",
    "gpl-3.0", "intel-research", "isc", "lgpl", "lgpl-2.1", "lgpl-3.0",
    "lgpl-lr", "llama2", "llama3", "llama3.1", "llama3.2", "lppl-1.3c",
    "mit", "mpl-2.0", "ms-pl", "ncsa", "odbl", "odc-by", "ofl-1.1",
    "openrail", "openrail++", "osl-3.0", "other", "pddl", "postgresql",
    "unknown", "unlicense", "wtfpl", "zlib"
]

# Precompile the regex for valid licenses (reduces overhead of compilation during function calls)
license_pattern = re.compile(r"---\s*license\s*:\s*({})\s*---".format("|".join(re.escape(license) for license in valid_licenses)))

# Optimized function to check for valid templates
def is_matching_card_fast(card_content):
    # Quickly strip spaces and normalize the string (reduce heavy regex operations)
    card_content = card_content.strip().replace('\r', '').replace('\n', ' ')
    # Direct match with precompiled regex
    return bool(license_pattern.fullmatch(card_content))

# Use faster vectorized operations with Pandas
df["is_match"] = df["card"].map(is_matching_card_fast)  # Vectorized mapping
filtered_cards = df[df["is_match"]].drop(columns=["is_match"])  # Filter matched cards

# Display the filtered cards
filtered_cards.reset_index(drop=True, inplace=True)
filtered_cards['card']
'''


# In[ ]:





# ### Basic info

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TOP_K = 10

df['year'] = df['createdAt'].dt.year

for year in range(2018, 2025):
    df_year = df[df['year'] == year].sort_values(by='downloads', ascending=False).head(TOP_K)
    if not df_year.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(df_year['modelId'], df_year['downloads'], color=plt.cm.Blues(np.linspace(0.8, 0.4, len(df_year))))
        plt.xticks(rotation=90, fontsize=8)
        plt.title(f"Top {TOP_K} Models by Downloads in {year}")
        plt.ylabel("Downloads")
        plt.tight_layout()
        plt.savefig(f"top_{TOP_K}_models_by_downloads_{year}.png")
        plt.show()
        plt.close()

author_counts = df['author'].value_counts().head(TOP_K)

plt.figure(figsize=(10, 6))
plt.bar(author_counts.index, author_counts.values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(author_counts))))
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Top {TOP_K} Authors by Model Count")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig(f"top_{TOP_K}_authors_by_model_count.png")
plt.show()
plt.close()

author_downloads = df.groupby('author')['downloads'].sum().sort_values(ascending=False).head(TOP_K)

plt.figure(figsize=(10, 6))
plt.bar(author_downloads.index, author_downloads.values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(author_downloads))))
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Top {TOP_K} Authors by Download Count")
plt.ylabel("Total Downloads")
plt.tight_layout()
plt.savefig(f"top_{TOP_K}_authors_by_download_count.png")
plt.show()
plt.close()

task_category_counts = df['pipeline_tag'].value_counts().head(TOP_K)

plt.figure(figsize=(10, 6))
plt.bar(task_category_counts.index, task_category_counts.values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(task_category_counts))))
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Top {TOP_K} Task Categories")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig(f"top_{TOP_K}_task_categories.png")
plt.show()
plt.close()


# In[ ]:


print(df['card'][0])


# In[ ]:


import pandas as pd
import ast

# 假设 'tags' 列是字符串表示的数组，我们需要解析它
df_split_temp['tags_list'] = df_split_temp['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 去除 'arxiv:1910.09700' 标签，检查其他 'arxiv:' 标签
valid_arxiv_count = df_split_temp['tags_list'].apply(
    lambda x: any(tag.lower().startswith('arxiv:') and tag.lower().strip() != 'arxiv:1910.09700' for tag in x)
)

# 计算比例
proportion_with_arxiv = valid_arxiv_count.sum() / len(df_split_temp)
numerator = valid_arxiv_count.sum()
denominator = len(df_split_temp)

# 输出结果
print(f"Proportion of items with arXiv tags (excluding 'arxiv:1910.09700'): {numerator}/{denominator} = {numerator / denominator:.2%}")


# In[ ]:


print(df_split_temp[df_split_temp['modelId']=='sentence-transformers/all-mpnet-base-v2']['tags'].values)


# In[78]:


df_split_temp[arxiv_count]['tags_list'].value_counts()


# In[ ]:


from urllib.parse import urlparse
import pandas as pd
import re

# Valid PDF link domains
VALID_PDF_LINKS = [
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "dl.acm.org",
    "dblp.uni-trier.de",
    "scholar.google.com",
    "pubmed.ncbi.nlm.nih.gov",
    "frontiersin.org",
    "mdpi.com",
    "cvpr.thecvf.com",
    "nips.cc",
    "icml.cc",
    "ijcai.org",
    "webofscience.com",
    "journals.plos.org",
    "nature.com",
    "semanticscholar.org",
    "chemrxiv.org",
    "link.springer.com",
    "ieeexplore.ieee.org",
    "aaai.org",
    "openaccess.thecvf.com",
]

def extract_links(text):
    """Extract PDF and GitHub links from the text."""
    if pd.isna(text):
        return {"pdf_link": None, "github_link": None, "all_links": []}
    
    # Find all links (match https://, http://, and www.)
    all_links = [link.strip(".,)") for link in re.findall(r"(https?://\S+|www\.\S+)", text)]
    
    # Function to check if the link is a valid PDF link
    def is_valid_pdf_link(link):
        """
        Check if a link is a valid PDF link:
        1. Matches one of the predefined VALID_PDF_LINKS domains;
        2. Or ends with ".pdf".
        """
        try:
            parsed_url = urlparse(link)
            domain = parsed_url.netloc.lstrip("www.")  # Remove "www." prefix
        except Exception:
            return False  # Invalid link

        # Check if the domain is valid or the link ends with ".pdf"
        return domain in VALID_PDF_LINKS or link.lower().endswith(".pdf")
    
    # Filter PDF and GitHub links
    pdf_links = [link for link in all_links if is_valid_pdf_link(link)]
    github_links = [link for link in all_links if "github.com" in link]
    
    # Return results
    return {
        "pdf_link": pdf_links if pdf_links else None,
        "github_link": github_links if github_links else None,
        "all_links": all_links if all_links else None
    }

# Ensure the 'card_readme' column is filled with an empty string if it's NaN
df_split_temp['combined_text'] = df_split_temp['card_readme'].fillna('')

# Apply the link extraction function
results = df_split_temp['combined_text'].apply(extract_links)

# Extract the results into separate columns
df_split_temp['pdf_link'] = results.apply(lambda x: x["pdf_link"] if x["pdf_link"] else None)
df_split_temp['github_link'] = results.apply(lambda x: x["github_link"] if x["github_link"] else None)
df_split_temp['all_links'] = results.apply(lambda x: ', '.join(x["all_links"]) if x["all_links"] else None)

# Check if each link type column has a non-empty value (avoids empty strings, lists, or NaNs)
df_split_temp['all_links_non_empty'] = df_split_temp['all_links'].apply(lambda x: bool(x) and x != '[]' and x != '')
df_split_temp['pdf_link_non_empty'] = df_split_temp['pdf_link'].apply(lambda x: bool(x) and x != '[]' and x != '')
df_split_temp['github_link_non_empty'] = df_split_temp['github_link'].apply(lambda x: bool(x) and x != '[]' and x != '')

# Count the occurrences and calculate the ratios
pdf_link_count = df_split_temp['pdf_link_non_empty'].sum()
total_count = len(df_split_temp)
pdf_link_ratio = (pdf_link_count / total_count) * 100

# Count rows where 'all_links' is non-empty
all_link_count = df_split_temp['all_links_non_empty'].sum()
all_link_ratio = (all_link_count / total_count) * 100

github_link_count = df_split_temp['github_link_non_empty'].sum()
github_link_ratio = (github_link_count / total_count) * 100

# Count entries with no PDF but with GitHub links
no_pdf_has_github_count = df_split_temp[~df_split_temp['pdf_link_non_empty'] & df_split_temp['github_link_non_empty']].shape[0]
no_pdf_has_github_ratio = (no_pdf_has_github_count / total_count) * 100

# Output the results
print(f"Model cards with all links: {all_link_count}/{total_count} = {all_link_ratio:.2f}%")
print(f"Model cards with GitHub links: {github_link_count}/{total_count} = {github_link_ratio:.2f}%")
print(f"Model cards with PDF links: {pdf_link_count}/{total_count} = {pdf_link_ratio:.2f}%")
print(f"Model cards with NO PDF but HAS GitHub links: "
      f"{no_pdf_has_github_count}/{total_count} = {no_pdf_has_github_ratio:.2f}%")


# In[ ]:


from urllib.parse import urlparse
import pandas as pd
import re

# Valid PDF link domains
VALID_PDF_LINKS = [
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "dl.acm.org",
    "dblp.uni-trier.de",
    "scholar.google.com",
    "pubmed.ncbi.nlm.nih.gov",
    "frontiersin.org",
    "mdpi.com",
    "cvpr.thecvf.com",
    "nips.cc",
    "icml.cc",
    "ijcai.org",
    "webofscience.com",
    "journals.plos.org",
    "nature.com",
    "semanticscholar.org",
    "chemrxiv.org",
    "link.springer.com",
    "ieeexplore.ieee.org",
    "aaai.org",
    "openaccess.thecvf.com",
]

# Function to extract links from text
def extract_links(text):
    """Extract PDF and GitHub links from the text."""
    if pd.isna(text):
        return {"pdf_link": None, "github_link": None, "all_links": []}
    
    # Find all links (match https://, http://, and www.)
    all_links = [link.strip(".,)") for link in re.findall(r"(https?://\S+|www\.\S+)", text)]
    
    # Function to check if the link is a valid PDF link, excluding specific ones
    def is_valid_pdf_link(link):
        """
        Check if a link is a valid PDF link:
        1. Matches one of the predefined VALID_PDF_LINKS domains;
        2. Ends with ".pdf";
        3. Allows 'arxiv:1910.09700' but still requires other valid PDF links.
        """
        try:
            parsed_url = urlparse(link)
            domain = parsed_url.netloc.lstrip("www.")  # Remove "www." prefix
        except Exception:
            return False  # Invalid link

        # If the link is 'arxiv:1910.09700', allow it but still need another valid link
        return (domain in VALID_PDF_LINKS or link.lower().endswith(".pdf"))

    # Filter PDF and GitHub links
    pdf_links = [link for link in all_links if is_valid_pdf_link(link)]
    github_links = [link for link in all_links if "github.com" in link]
    
    # If 'arxiv:1910.09700' is found, ensure there are other valid PDF links
    has_arxiv_1910 = any("arxiv:1910.09700" in link for link in all_links)
    if has_arxiv_1910 and len(pdf_links) == 1 and "arxiv:1910.09700" in pdf_links:
        pdf_links = []  # Exclude arxiv:1910.09700 if no other valid link exists
    
    # Return results
    return {
        "pdf_link": pdf_links if pdf_links else None,
        "github_link": github_links if github_links else None,
        "all_links": all_links if all_links else None
    }

# Ensure the 'card_readme' column is filled with an empty string if it's NaN
df_split_temp['combined_text'] = df_split_temp['card_readme'].fillna('')

# Apply the link extraction function
results = df_split_temp['combined_text'].apply(extract_links)

# Extract the results into separate columns
df_split_temp['pdf_link'] = results.apply(lambda x: x["pdf_link"] if x["pdf_link"] else None)
df_split_temp['github_link'] = results.apply(lambda x: x["github_link"] if x["github_link"] else None)
df_split_temp['all_links'] = results.apply(lambda x: ', '.join(x["all_links"]) if x["all_links"] else None)

# Check if each link type column has a non-empty value (avoids empty strings, lists, or NaNs)
df_split_temp['all_links_non_empty'] = df_split_temp['all_links'].apply(lambda x: bool(x) and x != '[]' and x != '')
df_split_temp['pdf_link_non_empty'] = df_split_temp['pdf_link'].apply(lambda x: bool(x) and x != '[]' and x != '')
df_split_temp['github_link_non_empty'] = df_split_temp['github_link'].apply(lambda x: bool(x) and x != '[]' and x != '')

# Count the occurrences and calculate the ratios
pdf_link_count = df_split_temp['pdf_link_non_empty'].sum()
total_count = len(df_split_temp)
pdf_link_ratio = (pdf_link_count / total_count) * 100

# Count rows where 'all_links' is non-empty
all_link_count = df_split_temp['all_links_non_empty'].sum()
all_link_ratio = (all_link_count / total_count) * 100

github_link_count = df_split_temp['github_link_non_empty'].sum()
github_link_ratio = (github_link_count / total_count) * 100

# Count entries with no PDF but with GitHub links
no_pdf_has_github_count = df_split_temp[~df_split_temp['pdf_link_non_empty'] & df_split_temp['github_link_non_empty']].shape[0]
no_pdf_has_github_ratio = (no_pdf_has_github_count / total_count) * 100

# Output the results
print(f"Model cards with all links: {all_link_count}/{total_count} = {all_link_ratio:.2f}%")
print(f"Model cards with GitHub links: {github_link_count}/{total_count} = {github_link_ratio:.2f}%")
print(f"Model cards with PDF links: {pdf_link_count}/{total_count} = {pdf_link_ratio:.2f}%")
print(f"Model cards with NO PDF but HAS GitHub links: "
      f"{no_pdf_has_github_count}/{total_count} = {no_pdf_has_github_ratio:.2f}%")


# In[ ]:


# check pdf link
# 如果 pdf_link 是列表，逐个展开
df_split_temp['pdf_link_flat'] = df_split_temp['pdf_link'].apply(
    lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else []
)

# 展开 pdf_link 列（每个链接变成一行）
df_split_temp_exploded = df_split_temp.explode('pdf_link_flat')

# 统计每个链接的出现次数
link_counts = df_split_temp_exploded['pdf_link_flat'].value_counts()

# 输出链接频率统计
print(link_counts)


# In[ ]:


from urllib.parse import urlparse

# 提取 PDF 链接的域名前缀
def extract_domain(link):
    """
    从链接中提取域名部分。
    """
    if pd.isna(link) or not isinstance(link, str):
        return None
    parsed_url = urlparse(link)
    domain = parsed_url.netloc  # 获取域名部分
    return domain if domain else None

# 1. 如果 pdf_link 是列表，逐个提取域名
df_split_temp['pdf_link_domains'] = df_split_temp['pdf_link'].apply(
    lambda x: [extract_domain(link) for link in x] if isinstance(x, list) else None
)

# 2. 将域名列表展开（explode），便于统计频率
df_split_temp_exploded = df_split_temp.explode('pdf_link_domains')

# 3. 统计域名前缀的使用频率
domain_counts = df_split_temp_exploded['pdf_link_domains'].value_counts()

# 输出域名频率统计
print(domain_counts)


# In[ ]:


#print(df_split_temp[df_split_temp['pdf_link_non_empty']][['modelId', 'card_readme', 'pdf_link']].iloc[0]['card_readme'])
print(df_split_temp[df_split_temp['pdf_link_non_empty']][['modelId', 'pdf_link']])


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Expand the 'tags' column so that each tag is in its own row
df_exploded = df_split_temp.explode('tags')

# Count the occurrences of 'finetuned' and 'adapter' tags
finetuned_count = df_exploded[df_exploded['tags'].str.contains('finetuned', case=False, na=False)].shape[0]
adapter_count = df_exploded[df_exploded['tags'].str.contains('adapter', case=False, na=False)].shape[0]

# Output the counts of the tags
print(f"'finetuned' tag count: {finetuned_count}")
print(f"'adapter' tag count: {adapter_count}")

# Now, let's print out the rows that contain these tags
finetuned_items = df_exploded[df_exploded['tags'].str.contains('finetuned', case=False, na=False)]
adapter_items = df_exploded[df_exploded['tags'].str.contains('adapter', case=False, na=False)]

# Print the rows that contain 'finetuned' or 'adapter'
print("\nItems containing 'finetuned' tag:")
print(finetuned_items)

print("\nItems containing 'adapter' tag:")
print(adapter_items)


# In[ ]:





# In[ ]:





# In[ ]:


key = 'pdf_link'
df_exploded = df.explode(key)
df_exploded[key] = df_exploded[key].str.extract(r'(https?://[^/]+)')
prefix_counts = df_exploded[key].value_counts()
print(prefix_counts)


# In[ ]:


missing_in_pdf = df[df['tags_list'].apply(lambda x: any('arxiv:' in tag for tag in x)) & df['pdf_link'].isna()]
missing_in_pdf[['tags', 'combined_text', 'pdf_link', 'github_link', 'all_links']]


# In[9]:


# check all value_counts, see if there exist weird value


# In[ ]:





# In[ ]:


"""def normalize_text(text: str) -> str:
    """
    归一化文本，去除所有非字母数字字符并转小写。
    """
    return re.sub(r"[^a-zA-Z0-9]+", "", text.strip().lower())


def check_license_only(
    card_text: str,
    max_length: int = 400,
    allowed_licenses: list = None,
    license_templates: list = None
) -> bool:
    """
    判断当前卡片是否仅仅包含「合法 license」信息（无其它文本）。
    
    :param card_text: 原始卡片文本
    :param max_length: 允许的最大文本长度, 如果超过则说明不只是license
    :param allowed_licenses: 允许出现的license列表
    :param license_templates: 匹配license的正则模板列表
    :return: True / False
    """
    if allowed_licenses is None:
        allowed_licenses = ALLOW_LICENSES
    
    if license_templates is None:
        license_templates = LICENSE_TEMPLATES
    
    # 去除 \r 统一风格
    txt_stripped = card_text.replace("\r", "").strip()
    
    # 如果文本长度就已经超过我们定义的 max_length，基本就不是「license-only」了
    if len(txt_stripped) > max_length:
        return False
    
    # 逐条匹配
    for pattern in license_templates:
        match = re.match(pattern, txt_stripped, flags=re.IGNORECASE)
        if match:
            # 如果正则匹配成功，提取license值
            license_value = match.group(1).lower()
            if license_value in allowed_licenses:
                return True
    return False


def check_contains_default_template(
    card_text: str,
    default_template: str,
    fuzzy_threshold = None
) -> bool:
    """
    判断卡片文本是否包含默认模板内容。
    
    - 若未指定 fuzzy_threshold，则采用 "子串匹配" 方式（严格匹配）。
    - 若指定 fuzzy_threshold (0~1之间)，则可做一定程度的模糊匹配/相似度（这里仅给出思路，需自行实现相似度比较）。
    
    :param card_text: 原始卡片文本
    :param default_template: 默认模板的大段文本
    :param fuzzy_threshold: None 表示子串匹配；非 None 表示做相似度匹配
    :return: True / False
    """
    # 简单 normalize
    normalized_card = normalize_text(card_text)
    normalized_template = normalize_text(default_template)
    
    if fuzzy_threshold is None:
        # 子串匹配
        return normalized_template in normalized_card
    else:
        # 这里只是示例，你可以改用编辑距离或向量相似度
        # below is a trivial example of ratio = len_of_overlap / len_template
        overlap_length = 0
        # 简单地 linear-scan 对 template 中的 token 做计数
        # [这里你可以改用更专业的 fuzzy / difflib / rapidfuzz 库来做相似度处理]
        for i in range(len(normalized_template)):
            if i < len(normalized_card) and normalized_template[i] == normalized_card[i]:
                overlap_length += 1
            else:
                break
        ratio = float(overlap_length) / float(len(normalized_template))
        return (ratio >= fuzzy_threshold)


def count_more_information_needed(card_text: str) -> int:
    """
    统计 `[More Information Needed]` 或者类似格式出现的次数。
    """
    pattern = re.compile(r"\[More Information Needed\]", re.IGNORECASE)
    matches = pattern.findall(card_text)
    return len(matches)


def check_diff_ratio(card_text: str, reference_text: str) -> float:
    """
    示例：计算与参考文本的差异比率(diff ratio)。
    这里仅展示用 Python 的 difflib 来做一个简单文本 diff，相似度越高 => 差异度越低。
    
    :return: diff_ratio, 0~1, 越大表示差异越大。
    """
    import difflib
    
    card_lines = card_text.splitlines(keepends=False)
    ref_lines = reference_text.splitlines(keepends=False)
    differ = difflib.Differ()
    diff_result = list(differ.compare(ref_lines, card_lines))
    
    # 统计有哪些行是 + / - / ?
    # difflib 只是个演示，也可换成别的方法
    changed_lines = sum(
        1 for line in diff_result if line.startswith("+ ") or line.startswith("- ")
    )
    total_lines = max(len(ref_lines), len(card_lines))
    if total_lines == 0:
        return 0.0
    
    return float(changed_lines) / float(total_lines)


# ========== 主函数：判断是否为「default card」 ==========

def is_default_card(
    card_text: str,
    # == 基础条件 ==
    check_license: bool = True,        # 是否启用「license-only」判断
    check_template: bool = True,       # 是否启用「包含默认模板」判断
    
    # == 可选增强条件 ==
    max_license_len: int = 400,        # 仅license时的长度限制
    min_diff_ratio = None,  # 如果指定了，就基于 diff ratio 做进一步判断
    diff_reference_text: str = CONTAIN_DEFAULT_PATTERNS,  # diff 的对照模板
    
    # 如果想基于 "More Information Needed" 出现次数来辅助判断
    use_moreinfo_count: bool = False,
    moreinfo_count_threshold: int = 10,
    
    # 如果想基于卡片字数(字符数)来辅助判断
    max_card_length = None,
    
) -> bool:
    """
    返回 True 表示此 card_text 被认为是「default card」。
    """
    # (A) 如果只检测 license-only
    if check_license and check_license_only(
        card_text, max_length=max_license_len, allowed_licenses=ALLOW_LICENSES
    ):
        return True
    
    # (B) 如果需要检查是否「包含默认模板」（子串或模糊匹配都行）
    if check_template:
        # 这里先用严格子串匹配，如需模糊，可以在前面函数加参数 fuzzy_threshold=0.8
        if check_contains_default_template(card_text, CONTAIN_DEFAULT_PATTERNS, fuzzy_threshold=None):
            return True
    
    # ========== 以下是一系列可选条件，可根据需求灵活加 or 改成 and 关系 ==========

    # (C) diff ratio，如果指定了 min_diff_ratio，就对默认模板做 diff
    if min_diff_ratio is not None:
        ratio = check_diff_ratio(card_text, diff_reference_text)
        # diff_ratio 越大 => 差异越大，如果差异小于 X => 可以认为是大部分一样
        # 这里演示：当差异率 < 0.05 => 认为是默认卡
        if ratio < min_diff_ratio:
            return True

    # (D) `[More Information Needed]` 出现次数过多 => 可能是默认模板
    # 这里我们以「超过阈值」就认为是 default
    if use_moreinfo_count:
        cnt = count_more_information_needed(card_text)
        if cnt > moreinfo_count_threshold:
            return True

    # (E) 如果想要基于卡片总长度 <= X 也可能是 default
    if max_card_length is not None:
        # 这里也可以做一个非常低的字数判断
        # 例如：卡片文本长度 < 800 => 可能是基本没改动
        if len(card_text) <= max_card_length:
            return True

    # 如果没有任何条件触发 => 认为不是 default
    return False


# ======================= DEMO 测试 =======================

if __name__ == "__main__":
    # 示例：
    test_card_license_only = "---\nlicense: mit\n---\n"
    test_card_default_template = CONTAIN_DEFAULT_PATTERNS  # 直接就是大段文档
    test_card_custom = "This is a fully custom card with new content."

    print(
        "License-only =>",
        is_default_card(
            test_card_license_only,
            check_license=True,
            check_template=False,  # 只测试license
        )
    )

    print(
        "Default template =>",
        is_default_card(
            test_card_default_template,
            check_license=False,
            check_template=True
        )
    )

    print(
        "Custom =>",
        is_default_card(
            test_card_custom,
            check_license=True,
            check_template=True
        )
    )

    # 如果想测试 diff-based approach + [More Information Needed] 次数
    custom_card_with_some_mincnt = "I changed a bit of text. [More Information Needed]\n" * 15
    print(
        "Custom w/ 15 times '[More Information Needed]' =>",
        is_default_card(
            custom_card_with_some_mincnt,
            check_license=True,
            check_template=True,
            use_moreinfo_count=True,
            moreinfo_count_threshold=10  # 大于10就认为是default
        )
    )

    # 测试 diff ratio
    slightly_modified = CONTAIN_DEFAULT_PATTERNS.replace("## Model Details", "## My Personal Model").replace("More Information Needed", "ABC")
    # 仅少数行改动 => diff ratio 应该较小
    print(
        "Slightly modified =>",
        is_default_card(
            slightly_modified,
            check_license=False,
            check_template=False,   # 不用上面“模板子串匹配”
            min_diff_ratio=0.05     # 如果差异率 < 5%就认为是default
        )
    )"""


# In[ ]:


print(df[df['is_default_card']]['card'].value_counts())


# In[ ]:


print(df[~df['is_default_card']]['card'].value_counts().index[1])


# In[ ]:


# filtering rules
df['card'].value_counts()


# In[ ]:


# other special case, 
df_filtered = df[
    (df['card'].str.len() < 50) & 
    (~df['card'].str.contains('license', case=False, na=False)) &
    (~df['card'].str.contains('library_name', case=False, na=False))
]
df_filtered['card'].value_counts()


# In[ ]:


df_filtered = df[
    (df['card'] == 'Entry not found') & 
    df['tags'].apply(lambda x: 'arxiv' in x)
]
df_filtered


# In[93]:


df_filtered = df[
    (df['card'] == 'Entry not found')]
df_filtered['tags'].value_counts()


# In[ ]:




