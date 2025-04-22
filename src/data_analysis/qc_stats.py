"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-07
Description:
  Get statistics of tables in CSV files from different resources (optimized with joblib)
  with additional model-level quality control. For each benchmark resource, two rows are generated:
  one with deduped (weighted) statistics and one (labeled "(sym)") with raw statistics computed by processing
  each CSV file instance (so if a file appears twice, its rows and columns are counted twice).

  2025‑04‑07 update:
  - Add non‑empty / valid‑title stats.
  - Compute a new column "all_title_list_valid" (using integration file) to filter valid titles.
  - Retain external resource hyperparameters.
  - Switch mode between "table_only" (grouped bar chart for #tables only)
    and "all_hyper" (grid of charts for all hyperparameters).
  - For grouped bar chart: Cluster 1 (Baseline/Fix) uses a green gradient;
    Clusters 2–5 (Sym, Dedup, w/ title, w/ valid title) use a blue gradient.
  - Additionally, save final results locally and plot #Cols, #Avg Rows, and Size(GB) for baseline as well.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.patches import Patch

# Configuration
INPUT_FILE = "data/processed/modelcard_step3_dedup.parquet"
INTEGRATION_FILE = "data/processed/final_integration_with_paths.parquet"
OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
VALID_TITLE_PARQUET = "data/processed/all_title_list_valid.parquet"

# Benchmark data (WDC removed)
benchmark_data = [
    ["SANTOS Small", 550, 6322, 6921, 0.45],
    ["TUS Small", 1530, 14810, 4466, 1.00],
    ["TUS Large", 5043, 54923, 1915, 1.50],
    ["SANTOS Large", 11090, 123477, 7675, 11.00],
    #["WDC", 50000000, 250000000, 14, 500.00]
]

RESOURCES = {
    'hugging': ['hugging_table_list_dedup'],
    'github': ['github_table_list_dedup'],
    'html': ['html_table_list_mapped_dedup'],
    'llm': ['llm_table_list_mapped_dedup']
}

BENCHMARK_NAMES = [x[0] for x in benchmark_data]  # For legend


def process_csv_file(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return {
            'path': csv_file,
            'rows': df.shape[0],
            'cols': df.shape[1],
            'size': os.path.getsize(csv_file)/(1024**3),
            'status': 'valid'
        }, None
    except Exception as e:
        return None, str(e)


def compute_resource_stats(df, resource):
    col = RESOURCES[resource][0]
    paths = df[col].explode()
    valid_paths = paths[paths.apply(lambda x: isinstance(x, str) and os.path.exists(x))]
    unique_paths = valid_paths.unique().tolist()

    dup_results = Parallel(n_jobs=-1)(
        delayed(process_csv_file)(p)
        for p in tqdm(valid_paths.tolist(), desc=f"[DUPLICATED] Processing {resource} files")
    )
    dup_valid_files = [r[0] for r in dup_results if r[0] and r[0]['status'] == 'valid']

    dedup_results = Parallel(n_jobs=-1)(
        delayed(process_csv_file)(p)
        for p in tqdm(unique_paths, desc=f"[DEDUP] Processing {resource} files")
    )
    dedup_valid_files = [r[0] for r in dedup_results if r[0] and r[0]['status'] == 'valid']

    def calculate_metrics(file_list):
        if not file_list:
            return [0, 0, 0, 0]
        total_cols = sum(f['cols'] for f in file_list)
        total_rows = sum(f['rows'] for f in file_list)
        avg_rows = int(total_rows / len(file_list))
        total_size = sum(f['size'] for f in file_list)
        return [len(file_list), total_cols, avg_rows, total_size]

    dup_metrics = calculate_metrics(dup_valid_files)
    dedup_metrics = calculate_metrics(dedup_valid_files)

    title_paths = list()
    valid_title_paths = list()
    # iterate over rows
    for p_list, ht, hvt in zip(df[col], df['has_title'], df['has_valid_title']):
        #if isinstance(p_list, (list, tuple, np.ndarray)):
        if ht:
            title_paths.extend([p for p in p_list if isinstance(p, str)])
        if hvt:
            valid_title_paths.extend([p for p in p_list if isinstance(p, str)])
    title_paths_set = set(title_paths)
    valid_title_paths_set = set(valid_title_paths)

    #title_count = sum(1 for p in unique_paths if p in title_paths_set)
    #valid_title_count = sum(1 for p in unique_paths if p in valid_title_paths_set)
    title_count_dedup = len(title_paths_set & set(unique_paths))
    valid_title_count_dedup = len(valid_title_paths_set & set(unique_paths))

    title_valid_files = [f for f in dedup_valid_files if f['path'] in title_paths_set]  ########
    valid_title_valid_files = [f for f in dedup_valid_files if f['path'] in valid_title_paths_set]  ########
    title_valid_metrics = calculate_metrics(title_valid_files)
    valid_title_valid_metrics = calculate_metrics(valid_title_valid_files)

    # save valid title list to local txt files
    title_valid_paths = [f['path'] for f in title_valid_files]
    valid_title_valid_paths = [f['path'] for f in valid_title_valid_files]
    title_valid_paths_set = set(title_valid_paths)
    valid_title_valid_paths_set = set(valid_title_valid_paths)
    print(f"Found {len(title_valid_paths_set)} valid titles in {resource} files")
    print(f"Found {len(valid_title_valid_paths_set)} valid titles in {resource} files")
    # save to txt files
    title_valid_file = os.path.join(OUTPUT_DIR, f"{resource}_title_valid.txt")
    valid_title_valid_file = os.path.join(OUTPUT_DIR, f"{resource}_valid_title_valid.txt")
    with open(title_valid_file, 'w') as f:
        for path in title_valid_paths_set:
            f.write(f"{path}\n")
    with open(valid_title_valid_file, 'w') as f:
        for path in valid_title_valid_paths_set:
            f.write(f"{path}\n")
    print(f"Saved valid title list to {title_valid_file}")
    print(f"Saved valid title list to {valid_title_valid_file}")

    return {
        f"{resource}-dup": dup_metrics,
        f"{resource}-dedup": dedup_metrics,
        f"{resource}-title_metrics": title_valid_metrics,
        f"{resource}-valid_metrics": valid_title_valid_metrics,
        #f"{resource}-title": title_count,
        f"{resource}-title-dedup": title_count_dedup,
        #f"{resource}-valid": valid_title_count,
        f"{resource}-valid-dedup": valid_title_count_dedup
    }

def create_combined_results(benchmark_data, resource_stats):
    columns = ["Benchmark", "# Tables", "# Cols", "Avg # Rows", "Size (GB)"]
    df = pd.DataFrame(benchmark_data, columns=columns)
    for resource in RESOURCES:
        unique_row = pd.DataFrame([[f"scilake-{resource}"] + list(resource_stats[f"{resource}-dedup"])], columns=columns)
        symlink_row = pd.DataFrame([[f"scilake-{resource} (duplicated)"] + list(resource_stats[f"{resource}-dup"])], columns=columns)
        w_title_row = pd.DataFrame([[f"scilake-{resource}-title-dedup"] + list(resource_stats[f"{resource}-title_metrics"])], columns=columns)
        w_valid_row = pd.DataFrame([[f"scilake-{resource}-valid-dedup"] + list(resource_stats[f"{resource}-valid_metrics"])], columns=columns)
        agg_values = []
        for i in range(4):
            val = (resource_stats[f"{resource}-dup"][i] +
                   resource_stats[f"{resource}-dedup"][i] +
                   resource_stats[f"{resource}-title_metrics"][i] +
                   resource_stats[f"{resource}-valid_metrics"][i])
            agg_values.append(val)
        #all_row = pd.DataFrame([[f"scilake-{resource}-all"] + agg_values], columns=columns)
        #df = pd.concat([df, unique_row, symlink_row, w_title_row, w_valid_row, all_row], ignore_index=True)
        df = pd.concat([df, unique_row, symlink_row, w_title_row, w_valid_row], ignore_index=True)
    return df

def annotate_bars(ax, fontsize=16):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=fontsize, rotation=0)

def plot_metric(df, metric, filename):
    from matplotlib.patches import Patch
    fontsize=12
    plt.rcParams.update({
        'font.size': 18,           
        'axes.titlesize': 18,      
        'axes.labelsize': 18,   
        'xtick.labelsize': 18,    
        'ytick.labelsize': 18,     
        'legend.fontsize': 18,     
        'figure.titlesize': 18     
    })
    figsize=(12, 4)
    
    palette_baseline = ["#8b2e2e", "#b74a3c", "#d96e44", "#f29e4c", "#FFBE5F"]
    palette_resource = ["#486f90", "#4e8094", "#50a89d", "#a5d2bc"]

    bar_width = 0.15
    gap = 0.4
    group_width = len(RESOURCES) * bar_width + gap
    clusters = ['baseline', 'duplicated', 'dedup', 'w/ title', 'w/ valid title']
    resources = list(RESOURCES.keys())

    cluster_key_map = {
        'duplicated': " (duplicated)",
        'dedup': "",              
        'w/ title': "-title-dedup",
        'w/ valid title': "-valid-dedup"
    }

    heights = []
    colors = []
    positions = []
    for i, val in enumerate(df.iloc[:4][metric]):
        positions.append(i * bar_width)
        heights.append(val)
        colors.append(palette_baseline[i])
    # duplicated, dedup, w/ title, w/ valid title
    for ci, cluster in enumerate(clusters[1:], start=1):
        for ri, resource in enumerate(resources):
            suffix = cluster_key_map[cluster]
            idx = f"scilake-{resource}{suffix}"
            val = df[df['Benchmark'] == idx][metric].values
            if len(val):
                positions.append(ci * group_width + ri * bar_width)
                heights.append(val[0])
                colors.append(palette_resource[ri])

    xtick_positions = [0 + (4 - 1) * bar_width / 2] + [
        i * group_width + (len(resources) - 1) * bar_width / 2 for i in range(1, len(clusters))
    ]
    xtick_labels = clusters

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.08, 0.1, 0.7, 0.8])

    ax.bar(positions, heights, width=bar_width, color=colors)
    ax.set_yscale('log')
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels) #, fontsize=12
    ax.set_ylabel(f"{metric} (log scale)")
    ax.set_title(f"{metric}")
    annotate_bars(ax, fontsize=fontsize)

    handles_baseline = [
        Patch(facecolor=palette_baseline[i], label=BENCHMARK_NAMES[i])
        for i in range(len(BENCHMARK_NAMES))
    ]
    labels_baseline = [f"Baseline: {n}" for n in BENCHMARK_NAMES]

    handles_resource = [
        Patch(facecolor=palette_resource[i], label=resources[i])
        for i in range(len(resources))
    ]
    labels_resource = [f"Resource: {res}" for res in resources]

    fig.legend(
        handles_baseline, labels_baseline,
        loc="upper left",           
        bbox_to_anchor=(0.80, 0.90),
        title="Baseline" 
    )
    fig.legend(
        handles_resource, labels_resource,
        loc="upper left",
        bbox_to_anchor=(0.80, 0.50),
        title="Resource"
    )

    # avoid using tight_layout()
    # avoid bbox_inches='tight'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def main():
    df = pd.read_parquet(INPUT_FILE)
    df_integration = pd.read_parquet(INTEGRATION_FILE)

    valid_titles = set(df_integration['query'].dropna().str.strip())
    df['all_title_list_valid'] = df['all_title_list'].apply(
        lambda x: [t for t in x if t in valid_titles] if isinstance(x, (list, tuple, np.ndarray)) else []
    )
    df['has_title'] = df['all_title_list'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
    df['has_valid_title'] = df['all_title_list_valid'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
    df.to_parquet(VALID_TITLE_PARQUET, index=False)
    print(f"Saved valid‑title list to {VALID_TITLE_PARQUET}")

    resource_stats = {}
    for resource in RESOURCES:
        print(f"\nProcessing {resource}...")
        stats = compute_resource_stats(df, resource)
        resource_stats.update(stats)

    results_df = create_combined_results(benchmark_data, resource_stats)
    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.parquet")
    results_df.to_parquet(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    plot_metric(results_df, "# Tables", "benchmark_tables.pdf")
    plot_metric(results_df, "# Cols", "benchmark_cols.pdf")
    plot_metric(results_df, "Avg # Rows", "benchmark_avg_rows.pdf")
    print(f"Saved figure to {OUTPUT_DIR}/benchmark_tables.pdf")
    print(f"Saved figure to {OUTPUT_DIR}/benchmark_cols.pdf")
    print(f"Saved figure to {OUTPUT_DIR}/benchmark_avg_rows.pdf")

if __name__ == "__main__":
    main()
