import pandas as pd
import numpy as np
import os, json, re
import matplotlib.pyplot as plt
from src.utils import load_combined_data, get_statistics_card
from tqdm import tqdm
from joblib import Parallel, delayed

data_type = "modelcard" # or "datasetcard"
df = load_combined_data(data_type, file_path="~/Repo/CitationLake/data/raw/")

stats = get_statistics_card(df)
print(json.dumps(stats, indent=4))

author_downloads = df[df['card'] == 'Entry not found']['downloads'].value_counts().sort_index()
print(author_downloads)

"""# 绘制直方图（优化颜色渐变）
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
"""

# 7. Rules-based filtering: At least 1 download and deduplication
# Rule 1: Filter out disabled models (this attribute is all None)
#df_enabled = df[df['disabled'] == False]
# Rule 2: Filter out private models (all public from API retrieving)
#df_public = df_enabled[df_enabled['private'] == False]

#df_non_empty_siblings = df[df['siblings_list'].apply(has_extra_siblings)]
#df_with_model_files = df_non_empty_siblings[df_non_empty_siblings['siblings_list'].apply(has_model_files)]


# Step 3: Rule 3 - Filter out models with 0 downloads
df_filtered_downloads = df_with_model_files[df_with_model_files['downloads'] > 0]
print(f"Number of repositories with downloads > 0: {len(df_filtered_downloads)}")

# Step 4: Rule 4 - Deduplication: Keep only the highest download per model base name
df_filtered_downloads['base_model_name'] = df_filtered_downloads['modelId'].apply(lambda x: x.split('/')[-1])
#df_unique_model_names = df_filtered_downloads.loc[
#    df_filtered_downloads.groupby('base_model_name')['downloads'].idxmax()
#]
#print(f"Number of unique repositories (based on highest downloads per model): {len(df_unique_model_names)}")

# Step 5: Calculate counts at each step
total_models = len(df)
models_with_siblings = len(df_non_empty_siblings)
models_with_model_files = len(df_with_model_files)
models_with_downloads = len(df_filtered_downloads)
#models_unique_by_name_only = len(df_unique_model_names)

# Step 6: Plot the step-by-step filtering statistics
step_conditions = [
    "Total repositories from huggingface",
    "Repositories with non-model files only",
    "Repositories with any model file",
    "Repositories with downloads > 0",
    #"Unique repositories by name",
]
step_values = [
    total_models,
    models_with_siblings,
    models_with_model_files,
    models_with_downloads,
    #models_unique_by_name_only,
]

plt.figure(figsize=(10, 6))
plt.bar(step_conditions, step_values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(step_conditions))))
plt.xticks(rotation=45, fontsize=10)
plt.title("Step-by-Step Filtering Statistics")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig("step_by_step_filtering_statistics_updated_rules.png")
plt.show()

print("Updated step-by-step filtering chart has been saved.")

# ===== Simplified step-by-step filtering figure =====

# Rule: downloads > 0
models_with_downloads = df[df['downloads'] > 0]

step_conditions = [
    "Total repositories on HF", 
    "Repositories with downloads > 0"
]
step_values = [
    len(df),
    len(models_with_downloads)
]

plt.figure(figsize=(8, 5))
plt.bar(step_conditions, step_values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(step_conditions))))
plt.xticks(rotation=20, fontsize=10)
plt.title("Step-by-Step Filtering Statistics")
plt.ylabel("Number of Models")
plt.tight_layout()
plt.savefig("step_by_step_filtering_statistics.png")
plt.close()
print("Saved step_by_step_filtering_statistics.png")

# ===== Added statistics & plotting section extracted from get_card_statistics.ipynb =====

# Ensure date column is datetime and add a `year` helper
if 'createdAt' in df.columns and not np.issubdtype(df['createdAt'].dtype, np.datetime64):
    df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')

df['year'] = df['createdAt'].dt.year

TOP_K = 10
out_dir = "analysis_figures"
os.makedirs(out_dir, exist_ok=True)

# 1) Top-K models by downloads each year
for year in range(int(df['year'].min()), int(df['year'].max()) + 1):
    df_year = df[df['year'] == year].sort_values(by='downloads', ascending=False).head(TOP_K)
    if df_year.empty:
        continue
    plt.figure(figsize=(10, 6))
    plt.bar(df_year['modelId'], df_year['downloads'], color=plt.cm.Blues(np.linspace(0.8, 0.4, len(df_year))))
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f"Top {TOP_K} Models by Downloads in {year}")
    plt.ylabel("Downloads")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"top_{TOP_K}_models_{year}.png"))
    plt.close()

# 2) Top-K authors by model count
if 'author' in df.columns:
    author_counts = df['author'].value_counts().head(TOP_K)
    plt.figure(figsize=(10, 6))
    plt.bar(author_counts.index, author_counts.values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(author_counts))))
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f"Top {TOP_K} Authors by Model Count")
    plt.ylabel("Number of Models")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"top_{TOP_K}_authors_by_model_count.png"))
    plt.close()

    # 3) Top-K authors by total downloads
    author_downloads = df.groupby('author')['downloads'].sum().sort_values(ascending=False).head(TOP_K)
    plt.figure(figsize=(10, 6))
    plt.bar(author_downloads.index, author_downloads.values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(author_downloads))))
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f"Top {TOP_K} Authors by Download Count")
    plt.ylabel("Total Downloads")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"top_{TOP_K}_authors_by_downloads.png"))
    plt.close()

# 4) Top-K pipeline tags (task categories) if available
if 'pipeline_tag' in df.columns:
    task_category_counts = df['pipeline_tag'].value_counts().head(TOP_K)
    plt.figure(figsize=(10, 6))
    plt.bar(task_category_counts.index, task_category_counts.values, color=plt.cm.Blues(np.linspace(0.8, 0.4, len(task_category_counts))))
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f"Top {TOP_K} Task Categories")
    plt.ylabel("Number of Models")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"top_{TOP_K}_task_categories.png"))
    plt.close()

print(f"Figures saved to {out_dir}/")