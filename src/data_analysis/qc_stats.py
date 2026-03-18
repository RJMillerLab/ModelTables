"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-09-30
Description: Get statistics of tables in CSV files from different resources with optimized binary reading for ~15x performance improvement.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from src.utils import to_parquet, load_config, is_list_like, to_list_safe
import csv

# Filter configuration for tables that are too long or too wide
MAX_COLS = 100  # Maximum number of columns
MAX_ROWS = 200  # Maximum number of rows

# ====== Skip generic CSV sets ======
GENERIC_TABLE_PATTERNS = [
    "1910.09700_table",
    "204823751_table"
]

# Benchmark data (WDC removed)
# Note: # Cols is total columns across all tables (avg_cols × #tables)
#       Avg # Rows is average rows per table
#       Size (GB) is total benchmark size
benchmark_data = [
    ["SANTOS Small", 550, 6322, 6921, 0.45],
    ["TUS Small", 1530, 14810, 4466, 1.00],
    ["TUS Large", 5043, 54923, 1915, 1.50],
    ["SANTOS Large", 11090, 123477, 7675, 11.00],
    ["WDC", 50000000, 250000000, 14, 500.00],
    ["GitTable", 1000000, 12000000, 142, 0.0],  # 1M tables × 12 avg cols, avg 142 rows, size unknown
    ["WikiTables", 1400000, 7500000, 14, 0.0],  # 1.4M tables, 7.5M total cols (avg ~5.36 cols/table), avg ~14 rows (from sample data), size unknown
    ["UGEN-V1", 1050, 10550, 8, 0.004],  # 1050 tables: 50 query (8×11) + 1000 datalake (8×10). Weighted avg rows: (50×8+1000×8)/1050=8. Total size: 205KB+4MB≈4.2MB
    ["UGEN-V2", 1050, 13650, 23, 0.01],  # 1050 tables: 50 query (107×13) + 1000 datalake (19×13). Weighted avg rows: (50×107+1000×19)/1050≈23. Total size: 2MB+8MB=10MB
]

RESOURCES = {
    'hugging': ['hugging_table_list_dedup'],
    'github': ['github_table_list_dedup'],
    'html': ['html_table_list_mapped_dedup'],
    'llm': ['llm_table_list_mapped_dedup']
}

BENCHMARK_NAMES = [x[0] for x in benchmark_data]  # For legend


def _infer_resource_from_path(path):
    """Infer resource from path (prefix match so tagged dirs like deduped_hugging_csvs_v2_251117 are recognized)."""
    if "/deduped_hugging_csvs" in path or "/hugging" in path:
        return "hugging"
    if "/deduped_github_csvs" in path or "/github" in path:
        return "github"
    if "/tables_output" in path or "/html" in path:
        return "html"
    if "/llm_tables" in path or "/llm" in path:
        return "llm"
    return None


def resolve_to_tagged_csv_path(path, tag):
    """
    Resolve a parquet path to the CSV under the tag-specific directory.
    - hugging/github/html: dirs deduped_hugging_csvs_<tag>, deduped_github_csvs_<tag>, tables_output_<tag>.
    - llm_tables is not versioned: path unchanged.
    """
    if not path or not isinstance(path, str):
        return path
    res = _infer_resource_from_path(path)
    if res is None:
        return path
    base = os.path.basename(path)
    if res == "llm":
        dir_name = "llm_tables"
    else:
        dir_name = {"hugging": "deduped_hugging_csvs", "github": "deduped_github_csvs", "html": "tables_output"}[res] + "_" + tag
    target_dir = os.path.join('data', 'processed', dir_name)
    current_dir = os.path.dirname(path)
    if os.path.normpath(current_dir) == os.path.normpath(target_dir):
        return path
    candidate = os.path.join(target_dir, base)
    return candidate if os.path.exists(candidate) else path


def find_v2_csv_path(original_path):
    """Legacy: If a v2 counterpart exists (deduped_*_v2, tables_output_v2), return it; else return original_path.
    Prefer resolve_to_tagged_csv_path(path, tag) when tag is set."""
    if not os.path.exists(original_path):
        return original_path
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    v2_dir = dir_path.replace('deduped_hugging_csvs', 'deduped_hugging_csvs_v2')
    v2_dir = v2_dir.replace('deduped_github_csvs', 'deduped_github_csvs_v2')
    v2_dir = v2_dir.replace('tables_output', 'tables_output_v2')
    if not os.path.exists(v2_dir):
        return original_path
    v2_path = os.path.join(v2_dir, filename)
    return v2_path if os.path.exists(v2_path) else original_path


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


def should_filter_table_by_size_from_data(rows, cols):
    """Check if table should be filtered based on dimensions (using already computed data).
    
    Args:
        rows: Number of rows (already computed)
        cols: Number of columns (already computed)
        
    Returns:
        True if table should be filtered (too long/wide), False otherwise
    """
    if rows is None or cols is None:
        return False
    
    # Filter if too many columns
    if cols >= MAX_COLS:
        return True
    # Filter if too many rows
    if rows >= MAX_ROWS:
        return True
    
    return False


def process_csv_file(csv_file, tag=None):
    """Optimized CSV processing using binary reading for better performance.
    When tag is set, resolve path to tag-specific dir (tag is the full suffix, e.g. v2 or v2_251117).
    Otherwise use path as-is (no-tag run = v1 paths in parquet).
    """
    try:
        if tag:
            actual_csv_file = resolve_to_tagged_csv_path(csv_file, tag)
        else:
            actual_csv_file = csv_file

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

def compute_resource_stats(df, resource, tag):
    col = RESOURCES[resource][0]
    paths = df[col].explode()
    
    # Determine suffix for output files (use tag if provided)
    suffix = f"_{tag}" if tag else ""
    
    # Debug: Check path existence
    total_paths = len(paths.dropna())
    existing_paths = paths[paths.apply(lambda x: isinstance(x, str) and os.path.exists(x))]
    non_existing_count = total_paths - len(existing_paths)
    
    if total_paths > 0:
        print(f"📊 {resource}: Total paths: {total_paths:,}, Existing: {len(existing_paths):,}, Missing: {non_existing_count:,}")
        if non_existing_count > 0:
            # Show sample missing paths (up to 3)
            missing_paths = paths[paths.apply(lambda x: isinstance(x, str) and not os.path.exists(x))].head(3)
            if len(missing_paths) > 0:
                print(f"   Sample missing paths:")
                for p in list(missing_paths)[:3]:
                    print(f"      - {p}")
    
    valid_paths = existing_paths
    # Filter out generic / too-general tables
    def is_generic_table(path):
        filename = os.path.basename(path)
        return any(pattern in filename for pattern in GENERIC_TABLE_PATTERNS)
    
    before_generic_filter = len(valid_paths)
    valid_paths = valid_paths[~valid_paths.apply(is_generic_table)]
    generic_filtered = before_generic_filter - len(valid_paths)
    print(f"Filtered generic tables for {resource}: removed {generic_filtered:,} files.")
    
    unique_paths = valid_paths.unique().tolist()

    dup_results = Parallel(n_jobs=-1)(
        delayed(process_csv_file)(p, tag)
        for p in tqdm(valid_paths.tolist(), desc=f"[DUPLICATED] Processing {resource} files")
    )
    dup_valid_files = [r[0] for r in dup_results if r[0] and r[0]['status'] == 'valid']

    dedup_results = Parallel(n_jobs=-1)(
        delayed(process_csv_file)(p, tag)
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
        # Skip non list-like entries (including NaN / None)
        if not is_list_like(p_list):
            continue

        # Convert to list safely (supports numpy arrays / NaN, etc.)
        p_list = to_list_safe(p_list)
        
        # Extract valid string paths
        if ht:
            title_paths.extend([p for p in p_list if isinstance(p, str) and pd.notna(p)])
        if hvt:
            valid_title_paths.extend([p for p in p_list if isinstance(p, str) and pd.notna(p)])
    title_paths_set = set(title_paths)
    valid_title_paths_set = set(valid_title_paths)
    
    print(f"  📋 {resource}: Collected {len(title_paths)} title paths ({len(title_paths_set)} unique)")
    print(f"  📋 {resource}: Collected {len(valid_title_paths)} valid_title paths ({len(valid_title_paths_set)} unique)")

    #title_count = sum(1 for p in unique_paths if p in title_paths_set)
    #valid_title_count = sum(1 for p in unique_paths if p in valid_title_paths_set)
    title_count_dedup = len(title_paths_set & set(unique_paths))
    valid_title_count_dedup = len(valid_title_paths_set & set(unique_paths))
    
    print(f"  📋 {resource}: After dedup matching - title: {title_count_dedup}, valid_title: {valid_title_count_dedup}")

    # Normalize paths for comparison (handle both absolute and relative paths)
    # Create a mapping from normalized paths to original paths
    def normalize_path(p):
        """Normalize path for comparison (resolve to absolute if possible)"""
        if not isinstance(p, str):
            return None
        try:
            # Try to resolve to absolute path, fallback to original if fails
            abs_path = os.path.abspath(p) if os.path.exists(p) else p
            return os.path.normpath(abs_path)
        except:
            return os.path.normpath(p)
    
    # Normalize all paths in the sets for comparison
    title_paths_normalized = {normalize_path(p): p for p in title_paths_set if normalize_path(p)}
    valid_title_paths_normalized = {normalize_path(p): p for p in valid_title_paths_set if normalize_path(p)}
    
    # Match files using normalized paths
    title_valid_files = []
    for f in dedup_valid_files:
        normalized = normalize_path(f['path'])
        if normalized and normalized in title_paths_normalized:
            title_valid_files.append(f)
    
    valid_title_valid_files = []
    for f in dedup_valid_files:
        normalized = normalize_path(f['path'])
        if normalized and normalized in valid_title_paths_normalized:
            valid_title_valid_files.append(f)
    
    print(f"  📋 {resource}: Matched {len(title_valid_files)} title_valid files from {len(dedup_valid_files)} dedup files")
    print(f"  📋 {resource}: Matched {len(valid_title_valid_files)} valid_title_valid files from {len(dedup_valid_files)} dedup files")
    
    # Filter out tables that are too long or too wide (v2 filtering) - using already computed data
    original_valid_count = len(valid_title_valid_files)
    valid_title_valid_files = [
        f for f in valid_title_valid_files 
        if not should_filter_table_by_size_from_data(f['rows'], f['cols'])
    ]
    filtered_count = original_valid_count - len(valid_title_valid_files)
    if filtered_count > 0:
        print(f"  Filtered {filtered_count} tables (too long/wide) from {resource}_valid_title_valid{v2_suffix}{suffix}.txt")
    
    title_valid_metrics = calculate_metrics(title_valid_files)
    valid_title_valid_metrics = calculate_metrics(valid_title_valid_files)

    # save valid title list to local txt files
    title_valid_paths = [f['path'] for f in title_valid_files]
    valid_title_valid_paths = [f['path'] for f in valid_title_valid_files]
    title_valid_paths_set = set(title_valid_paths)
    valid_title_valid_paths_set = set(valid_title_valid_paths)
    print(f"Found {len(title_valid_paths_set)} valid titles in {resource} files")
    print(f"Found {len(valid_title_valid_paths_set)} valid titles in {resource} files")
    
    # Instead of writing per‑resource txt files, return the valid_title_valid path set
    # so the caller can aggregate everything into a single all_valid_title_valid*.txt.
    return (
        {
            f"{resource}-dup": dup_metrics,
            f"{resource}-dedup": dedup_metrics,
            f"{resource}-title_metrics": title_valid_metrics,
            f"{resource}-valid_metrics": valid_title_valid_metrics,
            #f"{resource}-title": title_count,
            f"{resource}-title-dedup": title_count_dedup,
            #f"{resource}-valid": valid_title_count,
            f"{resource}-valid-dedup": valid_title_count_dedup
        },
        valid_title_valid_paths_set,
    )

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
    Uses intelligent label placement to avoid overlaps.
    
    Args:
        ax: matplotlib axis
        fontsize: font size for annotations
        baseline_count: number of baseline bars (to distinguish from scilake bars)
        metric: metric name to determine special formatting rules
        bar_width: width of individual bars
        group_width: width of group spacing
    """
    # Reduce font size to minimize overlap
    annotation_fontsize = max(7, fontsize - 6)
    
    # Get all patches and their properties
    patches = [p for p in ax.patches if p.get_height() > 0]
    if not patches:
        return
    
    # Calculate positions and heights
    positions = []
    heights = []
    for i, p in enumerate(patches):
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        positions.append((x, y))
        heights.append(y)
    
    max_height = max(heights) if heights else 1
    min_height = min(heights) if heights else 0
    
    for i, (p, (x, height)) in enumerate(zip(patches, positions)):
        if height <= 0:
            continue
            
        # Determine if this is a baseline bar or scilake bar
        is_baseline = i < baseline_count
        
        # Special formatting for Avg # Rows
        if metric == "Avg # Rows":
            if is_baseline:
                display_text = f'{int(height)}'
            else:
                display_text = f'{height:.1f}'
        else:
            # For other metrics: integers show as int, decimals show 1 decimal place
            if height == int(height):
                display_text = f'{int(height)}'
            else:
                display_text = f'{height:.1f}'
        
        # Intelligent placement to avoid overlaps
        # For baseline bars (dense group), use staggered vertical offsets (closer to bars)
        if is_baseline:
            # Use pattern: every 3rd bar uses larger offset to stagger labels
            pattern = i % 3
            if pattern == 0:
                # Standard above (close to bar)
                va = 'bottom'
                y_offset = 2 + (height / max_height) * 1
            elif pattern == 1:
                # Lower offset (closer to bar)
                va = 'bottom'
                y_offset = 3 + (height / max_height) * 1.5
            else:
                # Slightly higher offset (still close)
                va = 'bottom'
                y_offset = 4 + (height / max_height) * 2
        else:
            # For scilake bars (more spaced), use simple alternating (close to bars)
            if i % 2 == 0:
                va = 'bottom'
                y_offset = 2
            else:
                va = 'bottom'
                y_offset = 3
        
        # Place annotation with smaller padding to reduce overlap
        ax.annotate(display_text,
                  (x, height),
                  ha='center', va=va, fontsize=annotation_fontsize, rotation=0,
                  xytext=(0, y_offset), 
                  textcoords='offset points',
                  bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

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
    
    # Extended palette for 9 baseline benchmarks (red shades from dark to light)
    palette_baseline = [
        "#8b2e2e",  # Dark red
        "#a03a35",  # Dark red-orange
        "#b74a3c",  # Red-brown
        "#c85a45",  # Medium red-orange
        "#d96e44",  # Orange-red
        "#e6864c",  # Light orange-red
        "#f29e4c",  # Orange
        "#FFB55A",  # Light orange
        "#FFBE5F"   # Pale orange-yellow
    ]
    palette_resource = ["#486f90", "#4e8094", "#50a89d", "#a5d2bc"]

    bar_width = 0.12  # Reduced bar width for tighter spacing
    gap = 0.25  # Reduced gap between clusters for tighter layout
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
    # Find all baseline benchmarks (not starting with "scilake-")
    baseline_mask = ~df['Benchmark'].str.startswith('scilake-')
    baseline_df = df[baseline_mask]
    num_baselines = len(baseline_df)
    
    # Ensure we have enough colors
    if num_baselines > len(palette_baseline):
        # Extend palette if needed by creating more shades
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        # Generate additional colors by interpolation from last color to a lighter shade
        cmap = LinearSegmentedColormap.from_list('reds', [palette_baseline[-1], '#FFF4E6'])
        n_needed = num_baselines - len(palette_baseline)
        additional_colors = []
        for i in np.linspace(0.2, 1.0, n_needed):
            rgb = cmap(i)
            # Convert to hex (rgb is already in [0,1] range)
            hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            additional_colors.append(hex_color)
        palette_baseline = palette_baseline + additional_colors
    
    for i, val in enumerate(baseline_df[metric]):
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

    xtick_positions = [0 + (num_baselines - 1) * bar_width / 2] + [
        i * group_width + (len(resources) - 1) * bar_width / 2 for i in range(1, len(clusters))
    ]
    xtick_labels = clusters

    fig = plt.figure(figsize=figsize)

    # Adjust axes to leave more space at bottom for two-row legend
    ax = fig.add_axes([0.08, 0.15, 0.7, 0.75])

    ax.bar(positions, heights, width=bar_width, color=colors)
    ax.set_yscale('log')
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels) #, fontsize=12
    ax.set_ylabel(f"{metric} (log scale)")
    ax.set_title(f"{metric}")
    annotate_bars(ax, fontsize=fontsize, baseline_count=num_baselines, metric=metric, bar_width=bar_width, group_width=group_width)

    handles_baseline = [
        Patch(facecolor=palette_baseline[i], label=BENCHMARK_NAMES[i])
        for i in range(len(BENCHMARK_NAMES))
    ]
    labels_baseline = BENCHMARK_NAMES

    handles_resource = [
        Patch(facecolor=palette_resource[i], label=resources[i])
        for i in range(len(resources))
    ]
    labels_resource = resources

    # Create two-row legend for baseline benchmarks to avoid overlap
    num_baseline = len(BENCHMARK_NAMES)
    ncol_baseline = (num_baseline + 1) // 2  # Ceiling division for 2 rows
    
    fig.legend(
        handles_baseline, labels_baseline,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol_baseline,
        fontsize=11,
        columnspacing=1.0,
        handletextpad=0.5,
        title="Baseline Benchmarks"
    )
    fig.legend(
        handles_resource, labels_resource,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=11,
        columnspacing=1.0,
        handletextpad=0.5,
        title="Resources"
    )

    # avoid using tight_layout()
    # avoid bbox_inches='tight'
    plt.savefig(os.path.join('data', 'analysis', filename), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get statistics of tables in CSV files from different resources")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')   
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    os.makedirs("data/analysis", exist_ok=True)

    # Determine input/output paths based on tag (tag is full suffix like v2 or v2_251117; no hardcoded _v2).
    input_file_dedup = os.path.join(base_path, 'processed', f"modelcard_step3_dedup{v2_suffix}{suffix}.parquet")
    # Use titles2ids as canonical source of query/retrieved titles
    query_file = os.path.join(base_path, 'processed', f"s2orc_titles2ids{suffix}.parquet")
    modelid2titles_path = os.path.join(base_path, 'processed', f"modelcard_all_title_list{suffix}.parquet")
    
    print("📁 Paths in use:")
    print(f"   Input dedup:    {input_file_dedup}")
    print(f"   Input query:    {query_file}")
    print(f"   Input modelid2titles: {modelid2titles_path}")

    df = pd.read_parquet(modelid2titles_path, columns=['modelId', 'all_title_list'])
    # Only keep entries where both query_title and retrieved_title are non-null,
    # then drop retrieved_title for downstream stats.
    df_integration = pd.read_parquet(query_file, columns=['query_title', 'retrieved_title', 'corpusId'])
    df_integration = df_integration[df_integration['query_title'].notna() & df_integration['retrieved_title'].notna()].copy()
    print(f"df_integration shape after dropping null query_title and retrieved_title: {df_integration.shape}")
    df_integration = df_integration[df_integration['corpusId'].notna()].copy()
    print(f"df_integration shape after dropping null corpusId: {df_integration.shape}")
    df_integration = df_integration.drop(columns=['retrieved_title', 'corpusId'])
    df_integration.rename(columns={'query_title': 'query'}, inplace=True)
    # read data/processed/modelcard_step3_dedup.parquet and get modelId and 4 resources keys
    df_dedup = pd.read_parquet(input_file_dedup, columns=['modelId', 'hugging_table_list_dedup', 'github_table_list_dedup', 'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup'])
    # merge df and df_dedup by modelId
    df = df.merge(df_dedup, on='modelId', how='left')

    valid_titles = set(df_integration['query'].dropna().str.strip())
    df['all_title_list_valid'] = df['all_title_list'].apply(lambda x: [t for t in to_list_safe(x) if t in valid_titles] if is_list_like(x) else [])
    df['has_title'] = df['all_title_list'].apply(lambda x: is_list_like(x) and len(to_list_safe(x)) > 0)
    df['has_valid_title'] = df['all_title_list_valid'].apply(lambda x: is_list_like(x) and len(to_list_safe(x)) > 0)
    
    # Only save modelId and the 3 new attributes to reduce file size
    df_optimized = df[['modelId', 'all_title_list', 'all_title_list_valid', 'has_title', 'has_valid_title']].copy()
    VALID_TITLE_PARQUET = os.path.join('data', 'processed', f"all_title_list_valid{v2_suffix}{suffix}.parquet")
    to_parquet(df_optimized, VALID_TITLE_PARQUET)
    print(f"Saved valid‑title list to {VALID_TITLE_PARQUET}")
    del df_optimized

    resource_stats = {}
    combined_paths = set()
    for resource in RESOURCES:
        print(f"\nProcessing {resource}...")
        stats, valid_paths_set = compute_resource_stats(df, resource, tag=args.tag)
        resource_stats.update(stats)
        combined_paths.update(valid_paths_set)
    results_df = create_combined_results(benchmark_data, resource_stats)
    results_path = os.path.join('data', 'analysis', f"benchmark_results{v2_suffix}{suffix}.parquet")
    to_parquet(results_df, results_path)
    print(f"\nSaved results to {results_path}")
    
    # Write a single global valid‑title list (tables already filtered in compute_resource_stats)
    all_valid_title_valid_file = os.path.join('data', 'analysis', f"all_valid_title_valid{v2_suffix}{suffix}.txt")
    with open(all_valid_title_valid_file, 'w') as f:
        for path in sorted(combined_paths):
            f.write(path + "\n")
    print(f"Saved concatenated valid-title list to {all_valid_title_valid_file} ({len(combined_paths)})")
    print(f"  (Tables already filtered by size thresholds: max_cols={MAX_COLS}, max_rows={MAX_ROWS})")
