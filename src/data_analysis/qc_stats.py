"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-07
Description: Get statistics of tables in CSV files from different resources with optimized binary reading for ~15x performance improvement.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from src.utils import to_parquet
import csv

# Configuration
INPUT_FILE = "data/processed/modelcard_step3_merged.parquet"
INPUT_FILE_DEDUP = "data/processed/modelcard_step3_dedup.parquet"
INTEGRATION_FILE = "data/processed/final_integration_with_paths.parquet"
OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
VALID_TITLE_PARQUET = "data/processed/all_title_list_valid.parquet"

# V2 mode configuration
V2_MODE = True  # Set to True to use v2 versions of CSV files
V2_SUFFIX = "_v2"  # Suffix for v2 output files

# Benchmark data (WDC removed)
benchmark_data = [
    ["SANTOS Small", 550, 6322, 6921, 0.45],
    ["TUS Small", 1530, 14810, 4466, 1.00],
    ["TUS Large", 5043, 54923, 1915, 1.50],
    ["SANTOS Large", 11090, 123477, 7675, 11.00],
    ["WDC", 50000000, 250000000, 14, 500.00]
]

RESOURCES = {
    'hugging': ['hugging_table_list_dedup'],
    'github': ['github_table_list_dedup'],
    'html': ['html_table_list_mapped_dedup'],
    'llm': ['llm_table_list_mapped_dedup']
}

BENCHMARK_NAMES = [x[0] for x in benchmark_data]  # For legend


def find_v2_csv_path(original_path):
    """Find v2 version of CSV file if it exists, otherwise return original path.
    
    Args:
        original_path: Original CSV file path
        
    Returns:
        Path to v2 version if exists, otherwise original path
    """
    if not V2_MODE:
        return original_path
    
    # Check if file exists
    if not os.path.exists(original_path):
        return original_path
    
    # Get directory and filename
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    
    # Look for v2 directory
    v2_dir = dir_path.replace('deduped_hugging_csvs', 'deduped_hugging_csvs_v2')
    v2_dir = v2_dir.replace('deduped_github_csvs', 'deduped_github_csvs_v2')
    v2_dir = v2_dir.replace('tables_output', 'tables_output_v2')
    
    # Check if v2 directory exists
    if not os.path.exists(v2_dir):
        return original_path
    
    # Look for v2 file
    v2_path = os.path.join(v2_dir, filename)
    if os.path.exists(v2_path):
        return v2_path
    
    return original_path


def count_rows_fast(csv_path, chunk_size=8 * 1024 * 1024, head_flag=False):
    """Count rows quickly by counting newlines in binary chunks.
    
    Args:
        csv_path: Path to CSV file
        chunk_size: Size of chunks to read
        head_flag: If True, includes header in count (total lines)
                  If False, excludes header (data rows only, like pandas)
    
    - Counts b"\n" occurrences across the file
    - If file is non-empty and does not end with a newline, adds 1
    """
    try:
        file_size = os.path.getsize(csv_path)
        if file_size == 0:
            return 0
        newline_count = 0
        last_byte_newline = False
        with open(csv_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                newline_count += data.count(b'\n')
                last_byte_newline = data.endswith(b'\n')
        # If file doesn't end with a newline, there's one more line
        total_lines = newline_count if last_byte_newline else newline_count + 1
        
        if head_flag:
            return total_lines  # Include header
        else:
            return max(0, total_lines - 1) if total_lines > 0 else 0  # Exclude header
    except Exception:
        return 0


def count_columns_from_header_fast(csv_path, max_scan_bytes=8 * 1024 * 1024):
    """Read up to the first newline only and parse that header row with csv.reader.
    
    This avoids scanning the entire file for malformed quoting elsewhere.
    """
    try:
        header_bytes = bytearray()
        with open(csv_path, 'rb') as f:
            while True:
                # Read moderately sized chunks to find first newline quickly
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                nl_pos = chunk.find(b'\n')
                if nl_pos != -1:
                    header_bytes.extend(chunk[:nl_pos])
                    break
                header_bytes.extend(chunk)
                if len(header_bytes) >= max_scan_bytes:
                    break
        if not header_bytes:
            return 0
        header_str = header_bytes.decode('utf-8', errors='ignore')
        row = next(csv.reader([header_str]), None)
        return len(row) if row is not None else 0
    except Exception:
        return 0


def process_csv_file(csv_file):
    """Optimized CSV processing using binary reading for better performance."""
    try:
        # Find v2 version if V2_MODE is enabled
        actual_csv_file = find_v2_csv_path(csv_file)
        
        # df = pd.read_csv(actual_csv_file, dtype=str, keep_default_na=False)
        # Use optimized binary reading methods with head_flag=False to match pandas behavior
        rows = count_rows_fast(actual_csv_file, head_flag=False)  # Exclude header to match pandas
        cols = count_columns_from_header_fast(actual_csv_file)
        return {
            'path': actual_csv_file,  # Store actual path used (v2 if available)
            'original_path': csv_file,  # Store original path for reference
            #'rows': df.shape[0],
            #'cols': df.shape[1],
            'rows': rows,
            'cols': cols,
            'size': os.path.getsize(actual_csv_file)/(1024**3),
            'status': 'valid'
        }, None
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
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
        avg_rows = total_rows / len(file_list)  # 保持小数，不使用 int()
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

def annotate_bars(ax, fontsize=16, baseline_count=0, metric="", bar_width=0.15, group_width=0.4):
    """Annotate bars with different formatting for baseline vs scilake data.
    
    Args:
        ax: matplotlib axis
        fontsize: font size for annotations
        baseline_count: number of baseline bars (to distinguish from scilake bars)
        metric: metric name to determine special formatting rules
        bar_width: width of individual bars
        group_width: width of group spacing
    """
    # Reduce font size to minimize overlap
    annotation_fontsize = max(8, fontsize - 4)
    
    # Get all bar heights for smart positioning
    heights = [p.get_height() for p in ax.patches if p.get_height() > 0]
    if not heights:
        return
    
    # Calculate dynamic vertical offset based on data range
    min_height = min(heights)
    max_height = max(heights)
    height_range = max_height - min_height
    
    # Base offset - smaller for better spacing
    base_offset = 2
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if height > 0:
            # Determine if this is a baseline bar or scilake bar
            is_baseline = i < baseline_count
            
            # Special formatting for Avg # Rows
            if metric == "Avg # Rows":
                if is_baseline:
                    # Baseline: keep as integer
                    display_text = f'{int(height)}'
                else:
                    # Scilake: use 1 decimal place
                    display_text = f'{height:.1f}'
            else:
                # For other metrics: integers show as int, decimals show 1 decimal place
                if height == int(height):
                    display_text = f'{int(height)}'
                else:
                    display_text = f'{height:.1f}'
            
            # Smart vertical positioning to reduce overlap
            # Alternate between top and bottom positioning for nearby bars
            if i % 2 == 0:
                # Even bars: position above
                va = 'bottom'
                y_offset = base_offset + (height / max_height) * 2  # Reduced dynamic offset
            else:
                # Odd bars: position below (if there's space)
                va = 'top'
                y_offset = -(base_offset + 1)
            
            # Always keep horizontal centering - no horizontal offset
            x_offset = 0
            
            ax.annotate(display_text,
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va=va, fontsize=annotation_fontsize, rotation=0,
                        xytext=(x_offset, y_offset), 
                        textcoords='offset points')

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
    df = pd.read_parquet(INPUT_FILE, columns=['modelId', 'all_title_list'])
    df_integration = pd.read_parquet(INTEGRATION_FILE, columns=['query'])
    # read data/processed/modelcard_step3_dedup.parquet and get modelId and 4 resources keys
    df_dedup = pd.read_parquet(INPUT_FILE_DEDUP, columns=['modelId', 'hugging_table_list_dedup', 'github_table_list_dedup', 'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup'])
    # merge df and df_dedup by modelId
    df = df.merge(df_dedup, on='modelId', how='left')

    valid_titles = set(df_integration['query'].dropna().str.strip())
    df['all_title_list_valid'] = df['all_title_list'].apply(
        lambda x: [t for t in x if t in valid_titles] if isinstance(x, (list, tuple, np.ndarray)) else []
    )
    df['has_title'] = df['all_title_list'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
    df['has_valid_title'] = df['all_title_list_valid'].apply(lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0)
    
    # Only save modelId and the 3 new attributes to reduce file size
    df_optimized = df[['modelId', 'all_title_list', 'all_title_list_valid', 'has_title', 'has_valid_title']].copy()
    to_parquet(df_optimized, VALID_TITLE_PARQUET)
    print(f"Saved valid‑title list to {VALID_TITLE_PARQUET}")
    del df_optimized

    resource_stats = {}
    for resource in RESOURCES:
        print(f"\nProcessing {resource}...")
        stats = compute_resource_stats(df, resource)
        resource_stats.update(stats)

    results_df = create_combined_results(benchmark_data, resource_stats)
    
    # Add v2 suffix to output files if V2_MODE is enabled
    if V2_MODE:
        results_path = os.path.join(OUTPUT_DIR, f"benchmark_results{V2_SUFFIX}.parquet")
        print(f"\n🔧 V2 Mode enabled - using v2 CSV files when available")
    else:
        results_path = os.path.join(OUTPUT_DIR, "benchmark_results.parquet")
    
    to_parquet(results_df, results_path)
    print(f"\nSaved results to {results_path}")

if __name__ == "__main__":
    main()
