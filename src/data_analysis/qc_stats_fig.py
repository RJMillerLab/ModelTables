"""
Author: Zhengyuan Dong
Created: 2025-04-07
Last Modified: 2025-09-30
Description: Plot benchmark results for number of tables, columns, and average rows per table.
"""

#from src.data_analysis.qc_stats import plot_metric
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V2 mode configuration (should match qc_stats.py)
V2_MODE = True  # Set to True to use v2 versions of CSV files
V2_SUFFIX = "_v2"  # Suffix for v2 output files

RESOURCES = {
    'hugging': ['hugging_table_list_dedup'],
    'github': ['github_table_list_dedup'],
    'html': ['html_table_list_mapped_dedup'],
    'llm': ['llm_table_list_mapped_dedup']
}

RESOURCE_LABELS = {
    'hugging': 'ModelCard',
    'github': 'GitHub',
    'html': 'arXiv',
    'llm': 'S2ORC'
}

# Define benchmark names that should be treated as baseline (not scilake)
BASELINE_BENCHMARKS = ["SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large"]

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
            
            ax.annotate(
                display_text,
                (p.get_x() + p.get_width() / 2, height),
                ha='center', va=va,
                fontsize=annotation_fontsize,
                xytext=(x_offset, y_offset), 
                textcoords='offset points'
            )

def plot_metrics_grid(df, include_wdc=True): 
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    metrics = ["# Tables", "Avg # Cols", "Avg # Rows"]
    # Extended palette to support up to 5 baseline benchmarks (including WDC)
    palette_baseline = ["#8b2e2e", "#b74a3c", "#d96e44", "#f29e4c", "#FFBE5F"]
    palette_resource = ["#486f90", "#4e8094", "#50a89d", "#a5d2bc"]

    bar_width = 0.15
    gap = 0.4
    group_width = len(RESOURCES) * bar_width + gap
    clusters = ['Benchmarks', 'All', 'Dedup', 'Title', 'Valid-title']
    resources = list(RESOURCES.keys())

    cluster_key_map = {
        'All': " (duplicated)",
        'Dedup': "",
        'Title': "-title-dedup",
        'Valid-title': "-valid-dedup"
    }

    # Define baseline benchmarks locally to ensure WDC is included
    local_baseline_benchmarks = ["SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large", "WDC"]
    
    # Filter WDC if not included
    if not include_wdc:
        local_baseline_benchmarks = [b for b in local_baseline_benchmarks if b != "WDC"]
        print(f"⚠️ WDC excluded from plotting (include_wdc={include_wdc})")
    
    # Dynamically identify baseline benchmarks from the data
    baseline_mask = df['Benchmark'].isin(local_baseline_benchmarks)
    baseline_df = df[baseline_mask]
    scilake_df = df[~baseline_mask]
    
    print(f"Found {len(baseline_df)} baseline benchmarks: {baseline_df['Benchmark'].tolist()}")
    print(f"Found {len(scilake_df)} scilake entries")
    
    # Check if WDC is present
    wdc_present = 'WDC' in baseline_df['Benchmark'].values
    if wdc_present:
        print("✅ WDC data found and included in baseline benchmarks")
    else:
        if include_wdc:
            print("⚠️ WDC data not found in the dataset (will be skipped)")
        else:
            print("✅ WDC data excluded as requested")
        
    # Debug: Show the classification process
    print(f"\nDebug - Classification process:")
    print(f"Local baseline benchmarks: {local_baseline_benchmarks}")
    print(f"DataFrame Benchmark values: {df['Benchmark'].tolist()}")
    print(f"Baseline mask result:")
    for i, (idx, row) in enumerate(df.iterrows()):
        is_baseline = row['Benchmark'] in local_baseline_benchmarks
        print(f"  Row {i}: {row['Benchmark']} -> {'Baseline' if is_baseline else 'Scilake'}")
        
    # Verify the classification is correct
    print(f"\nBaseline benchmarks verification:")
    for name in local_baseline_benchmarks:
        if name in df['Benchmark'].values:
            print(f"  ✅ {name}: Found in baseline")
        else:
            print(f"  ❌ {name}: Not found")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=False, constrained_layout=True)
    fig.align_ylabels(axes)
    fig.suptitle("Log-scale statistic across different benchmarks", fontsize=22)

    for ax, metric in zip(axes, metrics):
        heights = []
        colors = []
        positions = []
        
        # Plot baseline benchmarks
        for i, (_, row) in enumerate(baseline_df.iterrows()):
            positions.append(i * bar_width)
            heights.append(row[metric])
            colors.append(palette_baseline[i % len(palette_baseline)])
        
        # Plot scilake data
        for ci, cluster in enumerate(clusters[1:], start=1):
            for ri, resource in enumerate(resources):
                suffix = cluster_key_map[cluster]
                idx = f"scilake-{resource}{suffix}"
                val = scilake_df[scilake_df['Benchmark'] == idx][metric].values
                if len(val):
                    positions.append(ci * group_width + ri * bar_width)
                    heights.append(val[0])
                    colors.append(palette_resource[ri])

        xtick_positions = [len(baseline_df) * bar_width / 2 - bar_width/2] + [
            i * group_width + (len(resources) - 1) * bar_width / 2 for i in range(1, len(clusters))
        ]
        xtick_labels = clusters
        ax.bar(positions, heights, width=bar_width, color=colors)
        ax.set_yscale('log')
        ax.margins(y=0.1)
        annotate_bars(ax, fontsize=12, baseline_count=len(baseline_df), metric=metric, bar_width=bar_width, group_width=group_width)
        ax.set_ylabel(f"{metric}", fontsize=24)

    axes[-1].set_xticks(xtick_positions)
    axes[-1].set_xticklabels(xtick_labels, rotation=0, fontsize=17)

    # Create legends based on actual data
    handles_baseline = [
        Patch(facecolor=palette_baseline[i % len(palette_baseline)], label=name)
        for i, name in enumerate(baseline_df['Benchmark'])
    ]
    handles_resource = [
        Patch(facecolor=palette_resource[i], label=RESOURCE_LABELS[res])
        for i, res in enumerate(resources)
    ]

    # Create two separate legends
    legend1 = fig.legend(
        handles_baseline,
        [h.get_label() for h in handles_baseline],
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=len(baseline_df),
        fontsize=13,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles_resource,
        [h.get_label() for h in handles_resource],
        loc="lower center", bbox_to_anchor=(0.5, -0.11),
        ncol=4,
        fontsize=13,
    )

    # Add v2 suffix to output files if V2_MODE is enabled
    if V2_MODE:
        pdf_path = os.path.join(OUTPUT_DIR, f"benchmark_metrics_vertical{V2_SUFFIX}.pdf")
        png_path = os.path.join(OUTPUT_DIR, f"benchmark_metrics_vertical{V2_SUFFIX}.png")
    else:
        pdf_path = os.path.join(OUTPUT_DIR, "benchmark_metrics_vertical.pdf")
        png_path = os.path.join(OUTPUT_DIR, "benchmark_metrics_vertical.png")
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {pdf_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    # Use v2 file if V2_MODE is enabled
    if V2_MODE:
        results_path = os.path.join(OUTPUT_DIR, f"benchmark_results{V2_SUFFIX}.parquet")
        print(f"🔧 V2 Mode enabled - reading from {results_path}")
    else:
        results_path = os.path.join(OUTPUT_DIR, "benchmark_results.parquet")
    
    results_df = pd.read_parquet(results_path)
    
    # Calculate average columns
    results_df["Avg # Cols"] = results_df["# Cols"] / results_df["# Tables"]
    
    print("DataFrame info:")
    print(f"Shape: {results_df.shape}")
    print(f"Columns: {list(results_df.columns)}")
    
    # Use the same classification logic as plot_metrics_grid
    baseline_mask = results_df['Benchmark'].isin(BASELINE_BENCHMARKS)
    baseline_df = results_df[baseline_mask]
    scilake_df = results_df[~baseline_mask]
    
    print("\nBaseline benchmarks found:")
    print(baseline_df[['Benchmark', '# Tables', '# Cols', 'Avg # Rows']])
    
    print("\nScilake entries found:")
    print(scilake_df[['Benchmark', '# Tables', '# Cols', 'Avg # Rows']])
    
    # Check for WDC specifically
    wdc_mask = results_df['Benchmark'] == 'WDC'
    if wdc_mask.any():
        print("\n✅ WDC data found:")
        print(results_df[wdc_mask][['Benchmark', '# Tables', '# Cols', 'Avg # Rows']])
    else:
        print("\n⚠️ WDC data not found in the dataset")
        print("If you want to include WDC, you may need to:")
        print("1. Run src.data_analysis.qc_stats to regenerate the data with WDC")
        print("2. Or manually add WDC data to the parquet file")
    
    # Control whether to include WDC in the plot
    include_wdc_in_plot = True  # control whether to include WDC in the plot: True=include WDC, False=exclude WDC
    
    print(f"\n🎨 Plotting configuration:")
    print(f"Include WDC in plot: {include_wdc_in_plot}")
    
    if include_wdc_in_plot:
        print("📊 Generating plot WITH WDC data")
    else:
        print("📊 Generating plot WITHOUT WDC data (WDC will be excluded)")
    
    plot_metrics_grid(results_df, include_wdc=include_wdc_in_plot)