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
from src.data_analysis.qc_stats import annotate_bars

OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V2 mode configuration (should match qc_stats.py)
V2_MODE = False  # Set to True to use v2 versions of CSV files
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
    'llm': 'Semantic Scholar'
}

# Define benchmark names that should be treated as baseline (not scilake)
BASELINE_BENCHMARKS = [
    "SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large", "WDC",
    "GitTable", "WikiTables", "UGEN-V1", "UGEN-V2"
]

def plot_metrics_grid(df, include_wdc=True): 
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    metrics = ["# Tables", "Avg # Cols", "Avg # Rows"]
    # Extended palette to support 9 baseline benchmarks (red shades from dark to light)
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
    clusters = ['Benchmarks', 'All', 'Dedup', 'Title', 'Valid-title']
    resources = list(RESOURCES.keys())

    cluster_key_map = {
        'All': " (duplicated)",
        'Dedup': "",
        'Title': "-title-dedup",
        'Valid-title': "-valid-dedup"
    }

    # Define baseline benchmarks locally (all benchmarks except scilake-*)
    local_baseline_benchmarks = [
        "SANTOS Small", "TUS Small", "TUS Large", "SANTOS Large", "WDC",
        "GitTable", "WikiTables", "UGEN-V1", "UGEN-V2"
    ]
    
    # Filter WDC if not included
    if not include_wdc:
        local_baseline_benchmarks = [b for b in local_baseline_benchmarks if b != "WDC"]
        print(f"⚠️ WDC excluded from plotting (include_wdc={include_wdc})")
    
    # Dynamically identify baseline benchmarks from the data
    baseline_mask = df['Benchmark'].isin(local_baseline_benchmarks)
    baseline_df = df[baseline_mask]
    scilake_df = df[~baseline_mask]
    
    # Calculate cluster widths: baseline cluster and scilake clusters
    num_baselines = len(baseline_df)
    baseline_cluster_width = num_baselines * bar_width  # Width of baseline cluster
    scilake_cluster_width = len(resources) * bar_width  # Width of each scilake cluster
    
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

    # Adjust figure for compact layout
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=False, constrained_layout=False)
    # Compact spacing: reduce top (title-to-plot), bottom (legend-to-plot), and hspace (between subplots)
    # Increase bottom significantly to leave space for xlabel + legend without overlap
    fig.subplots_adjust(top=0.94, bottom=0.25, hspace=0.25)  # More space at bottom for xlabel and legend
    fig.align_ylabels(axes)
    fig.suptitle("Log-scale statistic across different benchmarks", fontsize=22, y=0.98)  # Position title closer to plots

    for ax, metric in zip(axes, metrics):
        heights = []
        colors = []
        positions = []
        
        # Extend palette if needed (similar to qc_stats.py)
        num_baselines = len(baseline_df)
        current_palette = palette_baseline.copy()
        if num_baselines > len(current_palette):
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np
            # Generate additional colors by interpolation
            cmap = LinearSegmentedColormap.from_list('reds', [current_palette[-1], '#FFF4E6'])
            n_needed = num_baselines - len(current_palette)
            additional_colors = []
            for i in np.linspace(0.2, 1.0, n_needed):
                rgb = cmap(i)
                hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                additional_colors.append(hex_color)
            current_palette = current_palette + additional_colors
        
        # Plot baseline benchmarks (cluster 0: starting at position 0)
        baseline_start = 0
        for i, (_, row) in enumerate(baseline_df.iterrows()):
            positions.append(baseline_start + i * bar_width)
            heights.append(row[metric])
            colors.append(current_palette[i])
        
        # Plot scilake data (clusters 1, 2, 3, 4)
        # Each cluster starts after previous cluster + gap
        cluster_start_positions = []
        cluster_start_positions.append(baseline_start + baseline_cluster_width + gap)  # First scilake cluster
        
        for ci in range(1, len(clusters) - 1):
            cluster_start_positions.append(cluster_start_positions[-1] + scilake_cluster_width + gap)
        
        for ci, cluster in enumerate(clusters[1:], start=1):
            cluster_start = cluster_start_positions[ci - 1]
            for ri, resource in enumerate(resources):
                suffix = cluster_key_map[cluster]
                idx = f"scilake-{resource}{suffix}"
                val = scilake_df[scilake_df['Benchmark'] == idx][metric].values
                if len(val):
                    positions.append(cluster_start + ri * bar_width)
                    heights.append(val[0])
                    colors.append(palette_resource[ri])

        # Calculate xtick positions at the center of each cluster
        xtick_positions = []
        # Baseline cluster center
        xtick_positions.append(baseline_start + baseline_cluster_width / 2 - bar_width / 2)
        # Scilake clusters centers
        for cluster_start in cluster_start_positions:
            xtick_positions.append(cluster_start + scilake_cluster_width / 2 - bar_width / 2)
        xtick_labels = clusters
        ax.bar(positions, heights, width=bar_width, color=colors)
        ax.set_yscale('log')
        ax.margins(y=0.1)
        annotate_bars(ax, fontsize=12, baseline_count=len(baseline_df), metric=metric, bar_width=bar_width, group_width=scilake_cluster_width + gap)
        ax.set_ylabel(f"{metric}", fontsize=24)

    axes[-1].set_xticks(xtick_positions)
    # Set xlabels to horizontal (left-to-right) for better readability
    axes[-1].set_xticklabels(xtick_labels, rotation=0, fontsize=17, ha='center')

    # Create legends based on actual data (use extended palette if needed)
    # Reuse the same palette extension logic from the plot loop
    num_baselines = len(baseline_df)
    legend_palette = palette_baseline.copy()
    if num_baselines > len(legend_palette):
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        cmap = LinearSegmentedColormap.from_list('reds', [legend_palette[-1], '#FFF4E6'])
        n_needed = num_baselines - len(legend_palette)
        additional_colors = []
        for i in np.linspace(0.2, 1.0, n_needed):
            rgb = cmap(i)
            hex_color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            additional_colors.append(hex_color)
        legend_palette = legend_palette + additional_colors
    
    handles_baseline = [
        Patch(facecolor=legend_palette[i], label=name)
        for i, name in enumerate(baseline_df['Benchmark'])
    ]
    handles_resource = [
        Patch(facecolor=palette_resource[i], label=RESOURCE_LABELS[res])
        for i, res in enumerate(resources)
    ]

    # Create two separate legends in two rows to avoid overlap
    # Calculate number of columns for baseline legend (split into 2 rows)
    num_baseline = len(baseline_df)
    ncol_baseline = (num_baseline + 1) // 2  # Ceiling division for 2 rows
    
    # Manually position legends - adjust y values to move up (closer to 0 or positive)
    legend1 = fig.legend(
        handles_baseline,
        [h.get_label() for h in handles_baseline],
        loc="upper center", bbox_to_anchor=(0.5, 0.22),
        ncol=ncol_baseline,
        fontsize=12,
        columnspacing=1.0,
        handletextpad=0.5,
        frameon=True,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles_resource,
        [h.get_label() for h in handles_resource],
        loc="upper center", bbox_to_anchor=(0.5, 0.15),
        ncol=4,
        fontsize=12,
        columnspacing=1.0,
        handletextpad=0.5,
        frameon=True,
    )

    # Add v2 suffix to output files if V2_MODE is enabled
    if V2_MODE:
        pdf_path = os.path.join(OUTPUT_DIR, f"benchmark_metrics_vertical{V2_SUFFIX}.pdf")
        png_path = os.path.join(OUTPUT_DIR, f"benchmark_metrics_vertical{V2_SUFFIX}.png")
    else:
        pdf_path = os.path.join(OUTPUT_DIR, "benchmark_metrics_vertical.pdf")
        png_path = os.path.join(OUTPUT_DIR, "benchmark_metrics_vertical.png")
    
    # Use bbox_inches='tight' carefully - it may crop legends, so we set explicit bottom space
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
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