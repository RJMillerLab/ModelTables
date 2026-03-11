#!/usr/bin/env python
# coding: utf-8

# ## Hugging Face Hub API Examples
# For more information, see the [Hugging Face Hub API documentation](https://huggingface.co/docs/hub/api).

# ## Basic Information of Models and Their MetaData

# In[1]:


import huggingface_hub


# In[10]:


import pandas as pd
import requests
from tqdm import tqdm  # 用于显示进度条

# Function to fetch models with pagination
def fetch_all_models():
    url = "https://huggingface.co/api/models"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"full": "true", "limit": 1000}  # 每页最多返回 1000 条
    all_models = []
    total_pages = None  # 初始值，之后动态估算页数

    with tqdm(total=total_pages, desc="Fetching models", unit="page") as pbar:
        while url:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break

            # 获取返回数据
            data = response.json()
            all_models.extend(data)
            
            # 动态更新进度条总数（只在第一次估算）
            if total_pages is None:
                total_pages = len(data)  # 初次抓取估算为每页的数量
                pbar.total = total_pages

            # 更新进度条
            pbar.update(1)

            # 检查 Link Header 是否包含下一页信息
            next_link = response.headers.get("Link", None)
            if next_link and 'rel="next"' in next_link:
                url = next_link.split(";")[0].strip("<>")
            else:
                url = None  # 没有更多数据，退出循环

    return all_models

# Fetch all models
models_data = fetch_all_models()

# Process models data into a DataFrame
models_list = []
for model in models_data:
    models_list.append({
        "modelId": model.get("modelId"),
        "author": model.get("author", "Unknown"),
        "creation_time": model.get("cardData", {}).get("creation_time", "Unknown"),
        "downloads": model.get("downloads", 0),
        "has_modelcard": "cardData" in model and model.get("cardData") is not None,
        "task_category": model.get("pipeline_tag", "Unknown"),
        "task_domain": model.get("tags", ["Unknown"])[0] if model.get("tags") else "Unknown"
    })

# 转换为 DataFrame
models_df = pd.DataFrame(models_list)

# 显示结果
print(f"Total models fetched: {len(models_df)}")

# 保存为 CSV
models_df.to_csv("huggingface_models.csv", index=False)
print("Data saved to huggingface_models.csv")


# In[12]:


len(models_df)


# In[13]:


models_df


# In[14]:


import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm

async def fetch_page(session, url):
    async with session.get(url) as response:
        if response.status == 413:  # Request too large
            print("Error 413: Request too large. Reducing page size...")
            return [], None
        elif response.status != 200:
            print(f"Error: {response.status}")
            return [], None
        data = await response.json()
        next_link = response.headers.get("Link", None)
        if next_link and 'rel="next"' in next_link:
            next_url = next_link.split(";")[0].strip("<>")
        else:
            next_url = None
        return data, next_url

async def fetch_all_models():
    base_url = "https://huggingface.co/api/models?full=true&limit=500"  # Reduced page size
    headers = {"User-Agent": "Mozilla/5.0"}
    all_models = []
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        next_url = base_url
        
        while next_url:
            print(f"Fetching: {next_url}")
            page_data, next_url = await fetch_page(session, next_url)
            all_models.extend(page_data)

    return all_models

def process_models(models_data):
    models_list = []
    for model in models_data:
        models_list.append({
            "modelId": model.get("modelId"),
            "author": model.get("author", "Unknown"),
            "creation_time": model.get("cardData", {}).get("creation_time", "Unknown"),
            "downloads": model.get("downloads", 0),
            "has_modelcard": "cardData" in model and model.get("cardData") is not None,
            "task_category": model.get("pipeline_tag", "Unknown"),
            "task_domain": model.get("tags", ["Unknown"])[0] if model.get("tags") else "Unknown"
        })
    return pd.DataFrame(models_list)

async def main():
    models_data = await fetch_all_models()
    models_df = process_models(models_data)
    models_df.to_csv("huggingface_models_async.csv", index=False)
    print("Data saved to huggingface_models_async.csv")

if __name__ == "__main__":
    asyncio.run(main())


# In[16]:


from huggingface_hub import list_models
import pandas as pd

# Function to fetch models using huggingface_hub.list_models
def fetch_all_models():
    print("Fetching all models...")
    models = list_models(full=True, cardData=True, sort='downloads', direction=-1)
    print(f"Total models fetched: {len(list(models))}")
    
    # Process models into a structured format
    models_list = []
    for model in models:
        models_list.append({
            "modelId": model.modelId,
            "author": model.author,
            "creation_time": model.cardData.get("creation_time", "Unknown") if model.cardData else "Unknown",
            "downloads": model.downloads or 0,
            "has_modelcard": model.cardData is not None,
            "task_category": model.pipeline_tag or "Unknown",
            "task_domain": model.tags[0] if model.tags else "Unknown"
        })
    
    # Convert to DataFrame
    models_df = pd.DataFrame(models_list)
    return models_df

# Main execution
if __name__ == "__main__":
    models_df = fetch_all_models()
    print(f"Saving {len(models_df)} models to 'huggingface_models.csv'...")
    models_df.to_csv("huggingface_models.csv", index=False)
    print("Data saved to huggingface_models.csv")


# In[22]:


from huggingface_hub import list_models, ModelCard
import pandas as pd

# 获取所有模型的列表
models = list(list_models(full=True))


# In[23]:


models


# ## Get Model Card

# In[11]:


from huggingface_hub import ModelCard


# ## Get Access to the Models Repo and Downloads It

# To avoid conflicts with the '/' character, which is commonly used as a path separator, we replace it with '_' when creating the filename for a model card. This allows us to split the author name and dataset name for better organization. For example, the repository for `jonatasgrosman/wav2vec2-large-xlsr-53-english` will be saved as `jonatasgrosman'wav2vec2-large-xlsr-53-english`.

# In[11]:


import git
username = '' # specify your huggingface username
password = '' # specify your huggingface password


# In[12]:


model_name = model_example.modelId
model_name


# In[19]:


name = model_name.replace('/', "'")
path = f'../models_repo/{name}' # specify your storage path


# In[17]:


git.Repo.clone_from(url=f'https://{username}:{password}@huggingface.co/{model_name}', to_path=f'{path}{name}')

