#!/usr/bin/env python
# coding: utf-8

# # Model and Model Cards Overview

# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


import pickle
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# ## Model Information Overview

# In[3]:


model_info = pd.read_parquet('data/model_info.parquet')
model_info


# In[4]:


# drop nan
model_info = model_info.dropna(subset=['creation_time'])
model_info['creation_time'] = model_info['creation_time'].apply(lambda x: x.date())
model_info


# In[ ]:





# In[ ]:





# ## Model Card Information Overview

# In[2]:


import pandas as pd

modelcard_info = pd.read_parquet('data/modelcard_info.parquet')
modelcard_info


# In[3]:


print(modelcard_info['model_card'][0])


# In[28]:


import pandas as pd
import re

# 定义无效链接列表
INVALID_LINKS = [
    "arxiv.org",
    "www.biorxiv.org",
    "www.medrxiv.org",
    "dl.acm.org",
    "dblp.uni-trier.de",
    "scholar.google.com",
    "pubmed.ncbi.nlm.nih.gov",
    "www.frontiersin.org",
    "www.mdpi.com",
    "cvpr.thecvf.com",
    "nips.cc",
    "icml.cc",
    "www.ijcai.org",
    "www.webofscience.com",
    "journals.plos.org",
    "www.nature.com",
    "www.semanticscholar.org",
    "chemrxiv.org",
    "link.springer.com",
    "ieeexplore.ieee.org",
    "aaai.org",
]

# 定义链接解析函数
def extract_links(text):
    if pd.isna(text):  # 如果是空值，返回空
        return {"pdf_link": None, "github_link": None, "all_links": []}

    # 匹配所有 http/https 链接并去除尾部标点符号
    all_links = [link.strip(".,)") for link in re.findall(r"https?://\S+", text)]

    # 筛选出有效 PDF 链接
    pdf_links = [
        link for link in all_links
        if any(prefix in link for prefix in INVALID_LINKS) and len(link.split('/')) > 3
    ]

    # 筛选出 GitHub 链接
    github_links = [link for link in all_links if "github.com" in link]

    # 返回结果
    return {
        "pdf_link": pdf_links if pdf_links else None,  # 第一个满足条件的 PDF 链接
        "github_link": github_links[0] if github_links else None,  # 第一个 GitHub 链接
        "all_links": all_links
    }

# 加载数据

# 处理 model_card 和 tags 字段
modelcard_info['combined_text'] = modelcard_info['model_card'].fillna('')

# 提取链接
results = modelcard_info['combined_text'].apply(extract_links)

# 解包结果到新列
modelcard_info['pdf_link'] = results.apply(lambda x: x["pdf_link"])
modelcard_info['github_link'] = results.apply(lambda x: x["github_link"])
modelcard_info['all_links'] = results.apply(lambda x: ', '.join(x["all_links"]))

# 筛选条件：pdf_link 列为空
filtered_df = modelcard_info[
    modelcard_info['pdf_link'].isna()
][['all_links', 'pdf_link', 'github_link']]

# 保存结果到 CSV
tmp_output_path = "tmp_modelcard_info_with_links.csv"
modelcard_info.to_csv(tmp_output_path, index=False, encoding="utf-8")
print(len(modelcard_info))
print(f"筛选结果已保存到: {tmp_output_path}")


# In[34]:


pdf_link_count = modelcard_info['pdf_link'].notna().sum()
total_count = len(modelcard_info)
pdf_link_ratio = (pdf_link_count / total_count) * 100

all_link_count = modelcard_info['all_links'].notna().sum()
total_count = len(modelcard_info)
all_link_ratio = (all_link_count / total_count) * 100

github_link_count = modelcard_info['github_link'].notna().sum()
total_count = len(modelcard_info)
github_link_ratio = (github_link_count / total_count) * 100

print(f"Model cards with all links: {all_link_count}/{total_count} = {all_link_ratio:.2f}%")
print(f"Model cards with github links: {github_link_count}/{total_count} = {github_link_ratio:.2f}%")
print(f"Model cards with PDF links: {pdf_link_count}/{total_count} = {pdf_link_ratio:.2f}%")


# In[ ]:





# In[26]:


modelcard_info['all_links']


# In[18]:


# Extract the domain part of the links to check their source
modelcard_info['pdf_link_domain'] = modelcard_info['pdf_link'].dropna().apply(
    lambda x: x.split("/")[2] if len(x.split("/")) > 2 else "No domain"
)

# Count the occurrences of each domain
domain_counts = modelcard_info['pdf_link_domain'].value_counts()

# Display the counts
domain_counts


# In[ ]:





# In[31]:


filtered_df = modelcard_info[(modelcard_info['arxiv_link'].isna()) & (modelcard_info['all_links'].notna())][['all_links', 'github_link', 'arxiv_link']]
len(filtered_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


"""import pandas as pd
import re
import requests
from tqdm import tqdm

# 筛选出 arxiv_link 或 github_link 不为空的记录
tmp_df = modelcard_info[
    (modelcard_info['github_link'].notna())
][['all_links', 'github_link', 'arxiv_link']].copy()

# 添加统计列
tmp_df['readme_fetched'] = False  # 标记是否成功获取 README 文件
tmp_df['extracted_arxiv_links'] = None  # 存储提取的 arXiv 链接
tmp_df['extracted_other_links'] = None  # 存储提取的其他链接

# 定义函数：从 GitHub 链接提取 README 内容
def fetch_readme(github_url):
    try:
        # 转换为原始文件链接
        if "github.com" in github_url:
            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").rstrip("/") + "/main/README.md"
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                return response.text  # 返回 README 文件内容
    except Exception as e:
        print(f"Error fetching README for {github_url}: {e}")
    return None  # 如果失败，返回 None

# 定义函数：从文本中提取链接
def extract_links_from_text(text):
    if text is None:
        return {"arxiv_links": [], "other_links": []}
    # 匹配所有 https 链接并去除尾部的标点符号
    all_links = [link.strip(".,)\"") for link in re.findall(r"https?://\S+", text)]
    # 筛选出 arXiv 链接和其他链接
    arxiv_links = [link for link in all_links if "arxiv.org" in link or link.endswith(".pdf")]
    other_links = [link for link in all_links if link not in arxiv_links]
    return {"arxiv_links": arxiv_links, "other_links": other_links}

# 遍历 tmp_df，逐步提取信息
for idx, row in tqdm(tmp_df.iterrows(), total=len(tmp_df)):
    github_url = row['github_link']
    if pd.notna(github_url):  # 如果 GitHub 链接存在
        readme_content = fetch_readme(github_url)  # 获取 README 内容
        if readme_content:  # 如果成功获取到 README 文件
            tmp_df.at[idx, 'readme_fetched'] = True  # 更新标记
            links = extract_links_from_text(readme_content)  # 提取链接
            tmp_df.at[idx, 'extracted_arxiv_links'] = ', '.join(links['arxiv_links']) if links['arxiv_links'] else None
            tmp_df.at[idx, 'extracted_other_links'] = ', '.join(links['other_links']) if links['other_links'] else None

# 保存结果到 CSV 文件
output_path = "modelcard_info_with_links.csv"
tmp_df.to_csv(output_path, index=False, encoding="utf-8")

# 打印统计信息
print("提取统计信息：")
print(f"总记录数: {len(tmp_df)}")
print(f"成功提取 README 文件: {tmp_df['readme_fetched'].sum()}")
print(f"提取到 arXiv 链接: {tmp_df['extracted_arxiv_links'].notna().sum()}")
print(f"提取到其他链接: {tmp_df['extracted_other_links'].notna().sum()}")

print(f"结果保存到: {output_path}")
"""


# In[41]:


import pandas as pd
import re
import requests
from tqdm import tqdm

# 筛选出 github_link 不为空的记录
tmp_df = modelcard_info[
    modelcard_info['github_link'].notna()
][['all_links', 'github_link', 'pdf_link']].copy()
tmp_df.to_csv('./tmp_df.csv', index=True)

# 添加统计列
tmp_df['readme_fetched'] = False  # 标记是否成功获取 README 文件
tmp_df['extracted_pdf_links'] = None  # 存储提取的 PDF 链接
tmp_df['extracted_other_links'] = None  # 存储提取的其他链接

# 定义函数：从 GitHub 链接提取 README 内容
def fetch_readme(github_url):
    try:
        # 转换为原始文件链接
        if "github.com" in github_url:
            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").rstrip("/") + "/main/README.md"
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                # 保存 README 内容
                with open(f"{github_url.split('/')[-1]}_README.md", "w", encoding="utf-8") as file:
                    file.write(response.text)
                return response.text  # 返回 README 文件内容
    except Exception as e:
        print(f"Error fetching README for {github_url}: {e}")
    return None  # 如果失败，返回 None

# 定义函数：从文本中提取链接
def extract_links_from_text(text):
    if text is None:
        return {"pdf_links": [], "other_links": []}
    # 匹配所有 https 链接并去除尾部的标点符号
    all_links = [link.strip(".,)") for link in re.findall(r"https?://\S+", text)]
    # 筛选出 PDF 链接和其他链接
    pdf_links = [
        link for link in all_links
        if any(prefix in link for prefix in ["arxiv.org", "biorxiv.org", "medrxiv.org", "dl.acm.org", "springer.com", "ieeexplore.ieee.org", "nips.cc", "icml.cc"]) or link.endswith(".pdf")
    ]
    other_links = [link for link in all_links if link not in pdf_links]
    return {"pdf_links": pdf_links, "other_links": other_links}

# 遍历 tmp_df，逐步提取信息
for idx, row in tqdm(tmp_df.iterrows(), total=len(tmp_df)):
    github_url = row['github_link']
    if pd.notna(github_url):  # 如果 GitHub 链接存在
        readme_content = fetch_readme(github_url)  # 获取 README 内容
        if readme_content:  # 如果成功获取到 README 文件
            tmp_df.at[idx, 'readme_fetched'] = True  # 更新标记
            links = extract_links_from_text(readme_content)  # 提取链接
            tmp_df.at[idx, 'extracted_pdf_links'] = ', '.join(links['pdf_links']) if links['pdf_links'] else None
            tmp_df.at[idx, 'extracted_other_links'] = ', '.join(links['other_links']) if links['other_links'] else None

# 保存结果到 CSV 文件
output_path = "modelcard_info_with_links.csv"
tmp_df.to_csv(output_path, index=False, encoding="utf-8")

# 打印统计信息
print("提取统计信息：")
print(f"总记录数: {len(tmp_df)}")
print(f"成功提取 README 文件: {tmp_df['readme_fetched'].sum()}")
print(f"提取到 PDF 链接: {tmp_df['extracted_pdf_links'].notna().sum()}")
print(f"提取到其他链接: {tmp_df['extracted_other_links'].notna().sum()}")

print(f"结果保存到: {output_path}")


# In[36]:


import pandas as pd

# Define input and output file paths
input_file = "modelcard_info_with_links.csv"
# Read the CSV file
df = pd.read_csv(input_file)


# In[37]:


# Analyze rows where arxiv_link is not empty or github_readme exists
filtered_rows = df[(df["arxiv_link"].notna()) | (df["extracted_arxiv_links"].notna())]
count = len(filtered_rows)

count


# In[38]:


df


# 

# In[37]:


import pandas as pd

# 读取 CSV 文件
input_file = "modelcard_info_with_links.csv"
output_file = "updated_modelcard_info_with_links.csv"
df = pd.read_csv(input_file)

# 定义一个函数来更新 arxiv_link，仅对每行的 extracted_arxiv_links 进行去重
def update_arxiv_link(row):
    # 仅在 arxiv_link 为空时尝试更新
    if pd.isna(row["arxiv_link"]) and pd.notna(row["extracted_arxiv_links"]):
        # 分割 extracted_arxiv_links，去重
        links = set(link.strip() for link in row["extracted_arxiv_links"].split(";") if link.strip())
        # 如果去重后的链接数量是 1，则更新到 arxiv_link
        if len(links) == 1:
            return list(links)[0]
    return row["arxiv_link"]

# 使用 apply 批量更新 arxiv_link 列，仅当 arxiv_link 为空时进行更新
df["arxiv_link"] = df.apply(update_arxiv_link, axis=1)

# 保存更新后的文件
df.to_csv(output_file, index=False)
print(f"更新后的文件已保存为 {output_file}")


# ## Model Cards Adoption and Downloads Traffic

# In[6]:


print('Number of Models:', len(model_info))
has_card = model_info[model_info['has_modelcard'] == True]
print('Number of Models with Model Cards:', len(has_card))
print('Percentage of Models with Model Cards:{:.2f}%'.format(len(has_card)/len(model_info)*100))


# In[ ]:





# In[7]:


total_downloads = model_info['downloads'].sum()
print('Total Downloads:', total_downloads)
total_downloads_has_card = has_card['downloads'].sum()
print('Total Downloads of Models with Model Cards:', total_downloads_has_card)
print('Percentage of Downloads of Models with Model Cards:{:.2f}%'.format(total_downloads_has_card/total_downloads*100))


# ## Model Number Growth

# In[9]:


min_time = min(model_info['creation_time'])
print('min_time', min_time)
max_time = max(model_info['creation_time'])
print('max_time', max_time)


# In[10]:


model_info = model_info.sort_values(by=['creation_time'])
model_info


# In[11]:


from dateutil.relativedelta import relativedelta

time_range = []
date_list = model_info['creation_time']
min_time = min(date_list)
max_time = max(date_list)
print(min_time, max_time)
time_delta = relativedelta(days=7)
start_time = datetime.datetime(min_time.year, min_time.month, 1).date()
end_time = (datetime.datetime(max_time.year, max_time.month+1, 1).date())
time_range.append(start_time)

while True:
    start_time += time_delta
    time_range.append(start_time)
    if start_time > end_time:
        break


# In[12]:


time_range_str = [i.strftime("%Y-%m-%d") for i in time_range]
model70k_number = []
modelcard_number = []
for time in time_range:
    model70k_number.append(len(model_info[model_info['creation_time'] < time]))
    modelcard_number.append(len(model_info[(model_info['creation_time'] < time) & (model_info['has_modelcard']==True)]))


# In[13]:


model_number = pd.DataFrame(columns=['model_time', 'model_number', 'model_type'])
model_number['model_time'] = time_range_str
model_number['model_number'] = model70k_number
model_number['model_type'] = 'all models'
model_number_card_pd = pd.DataFrame(columns=['model_time', 'model_number', 'model_type'])
model_number_card_pd['model_time'] = time_range_str
model_number_card_pd['model_number'] = modelcard_number
model_number_card_pd['model_type'] = 'models with model card'
model_number = pd.concat([model_number, model_number_card_pd])
model_number


# In[39]:


import plotly.graph_objs as go
import numpy as np
from scipy.optimize import curve_fit

# Filter the data to include only 'All model repositories'
all_models_data = model_number[model_number['model_type'] == 'all models']

# Filter the data to start from 2020-04-30
all_models_data = all_models_data[all_models_data['model_time'] >= '2020-04-30']

# Exponential function
def exponential(x, a, b):
    return a * np.exp(b * x)

# Fit the exponential curve
x_data = np.arange(len(all_models_data))
y_data = all_models_data['model_number']
popt, _ = curve_fit(exponential, x_data, y_data)

# Create the fitted curve data
x_fit = np.linspace(0, max(x_data), 100)
y_fit = exponential(x_fit, *popt)

# Exponential formula
exp_formula = f'y = {popt[0]:.2f} * exp({popt[1]:.2f} * x)'


# In[40]:


import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# Linear function
def linear_func(x, a, b):
    return a * x + b

# Fit the linear model on the logarithm of y_data
x_data = np.arange(len(y_data))
y_log_data = np.log(y_data)
popt, _ = curve_fit(linear_func, x_data, y_log_data)

# Calculate the fitted values
y_log_fit = linear_func(x_data, *popt)
y_fit = np.exp(y_log_fit)

# Exponential formula
exp_formula = f'y = {np.exp(popt[1]):.2f} * exp({popt[0]:.2f} * x)'

# Print the fitted exponential curve and the exponential formula
print("Fitted exponential curve:", y_fit)
print("Exponential formula:", exp_formula)


# In[38]:


import plotly.graph_objs as go
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# Filter the data to include only 'All model repositories'
all_models_data = model_number[model_number['model_type'] == 'all models']

# Filter the data to start from 2020-04-30
all_models_data = all_models_data[all_models_data['model_time'] >= '2020-04-30']

# Linear function
def linear_func(x, a, b):
    return a * x + b

# Fit the linear model on the logarithm of y_data
x_data = np.arange(len(all_models_data))
y_data = all_models_data['model_number']
y_log_data = np.log(y_data)
popt, _ = curve_fit(linear_func, x_data, y_log_data)

# Calculate the fitted values
y_log_fit = linear_func(x_data, *popt)
y_fit = np.exp(y_log_fit)

# Exponential formula
exp_formula = f'y = {np.exp(popt[1]):.2f} * exp({popt[0]:.2f} * x)'

# Create the bar plot
fig = go.Figure()

fig.add_trace(go.Bar(
    x=all_models_data['model_time'],
    y=all_models_data['model_number'],
    name="All model repositories",
))

# Add the exponential fit curve
fig.add_trace(go.Scatter(
    x=all_models_data['model_time'],
    y=y_fit,
    mode='lines',
    name='Exponential Fit',
    line=dict(color='red', dash='dash'),
))


# Calculate the weekly growth rate and doubling time
growth_rate = (np.exp(popt[0]) - 1) * 100
doubling_time = np.log(2) / popt[0]

# Print the growth rate and doubling time
print(f"Weekly growth rate: {growth_rate:.2f}%")
print(f"Doubling time: {doubling_time:.0f} weeks")

# Add the exponential formula annotation
fig.add_annotation(
    x=0.99,
    y=0.01,
    xref="paper",
    yref="paper",
    text=f"Weekly growth rate: {growth_rate:.2f}%",
    showarrow=False,
    font=dict(size=16, color="#000000"),
    bgcolor="#ffffff",
    opacity=0.8,
)


# Customize the layout
fig.update_layout(
    autosize=False,
    width=600,
    height=400,
    font_size=14,
    font_color="black",
    xaxis_title='',
    yaxis_title='Total Number of Models',
    legend_title_text='',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, itemsizing='constant'),
)

fig.show()


import plotly.io as pio

# Save the figure with a high resolution (dpi=300)
# pio.write_image(fig, 'Fig1_exp.jpeg', width=600, height=400, scale=10)


# In[ ]:





# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
csv_path = "huggingface_models_250108.csv"
df = pd.read_csv(csv_path)

# 确保 creation_time 是 datetime 类型
df['creation_time'] = pd.to_datetime(df['creation_time'])

# 提取年份
df['year'] = df['creation_time'].dt.year

# 设置 Top K
TOP_K = 20

# 1. 时间范围统计
creation_time_range = (df['creation_time'].min(), df['creation_time'].max())

# 2. 下载量分布 - 每年Top K模型
for year in range(2018, 2025):  # 从2018到2024年
    df_year = df[df['year'] == year].sort_values(by='downloads', ascending=False).head(TOP_K)
    if not df_year.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(df_year['modelId'], df_year['downloads'], color=plt.cm.Blues(np.linspace(0.4, 0.8, len(df_year))))
        plt.xticks(rotation=90, fontsize=8)
        plt.title(f"Top {TOP_K} Models by Downloads in {year}")
        plt.ylabel("Downloads")
        plt.tight_layout()
        plt.savefig(f"top_{TOP_K}_models_by_downloads_{year}.png")
        plt.show()
        plt.close()

# 3. 作者统计：不同模型数量排名前TOP_K
author_counts = df['author'].value_counts().head(TOP_K)

# 绘制作者模型数量排名
plt.figure(figsize=(10, 6))
plt.bar(author_counts.index, author_counts.values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(author_counts))))
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Top {TOP_K} Authors by Model Count")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig(f"top_{TOP_K}_authors_by_model_count.png")
plt.show()
plt.close()

# 4. 作者统计：模型下载数量排名前TOP_K
author_downloads = df.groupby('author')['downloads'].sum().sort_values(ascending=False).head(TOP_K)

# 绘制作者模型下载数量排名
plt.figure(figsize=(10, 6))
plt.bar(author_downloads.index, author_downloads.values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(author_downloads))))
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Top {TOP_K} Authors by Download Count")
plt.ylabel("Total Downloads")
plt.tight_layout()
plt.savefig(f"top_{TOP_K}_authors_by_download_count.png")
plt.show()
plt.close()

# 5. 任务类别统计
task_category_counts = df['task_category'].value_counts().head(TOP_K)

plt.figure(figsize=(10, 6))
plt.bar(task_category_counts.index, task_category_counts.values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(task_category_counts))))
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Top {TOP_K} Task Categories")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig(f"top_{TOP_K}_task_categories.png")
plt.show()
plt.close()

# 6. 任务领域统计
task_domain_counts = df['task_domain'].value_counts().head(TOP_K)

plt.figure(figsize=(10, 6))
plt.bar(task_domain_counts.index, task_domain_counts.values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(task_domain_counts))))
plt.xticks(rotation=90, fontsize=8)
plt.title(f"Top {TOP_K} Task Domains")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig(f"top_{TOP_K}_task_domains.png")
plt.show()
plt.close()

# 7. Rules-based filtering: At least 1 download and deduplication

# Step 1: 原始模型总数
total_models = len(df)

# Step 2: 过滤掉 downloads 为 0 的模型
df_filtered_downloads = df[df['downloads'] > 0]
models_with_downloads = len(df_filtered_downloads)

# Step 3: 去重处理：仅根据模型名称去重
df_filtered_downloads['base_model_name'] = df_filtered_downloads['modelId'].apply(lambda x: x.split('/')[-1])
df_unique_model_names = df_filtered_downloads.loc[
    df_filtered_downloads.groupby('base_model_name')['downloads'].idxmax()
]
models_unique_by_name_only = len(df_unique_model_names)
print(f"Models with downloads > 0: {models_with_downloads}")
print(f"Unique models by base name: {models_unique_by_name_only}")

# 绘制过滤步骤的柱状图
step_conditions = [
    "Total Models", 
    "With Downloads > 0", 
    "Unique by Name Only"
]
step_values = [
    total_models, 
    models_with_downloads, 
    models_unique_by_name_only
]

plt.figure(figsize=(10, 6))
plt.bar(step_conditions, step_values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(step_conditions))))
plt.xticks(rotation=45, fontsize=10)
plt.title("Step-by-Step Filtering Statistics")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig("step_by_step_filtering_statistics.png")
plt.show()

print("Step-by-step filtering chart has been saved.")


# In[2]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




