"""
Author: Zhengyuan Dong
Created: 2025-04-07
Last Modified: 2025-04-22
Description: Plot benchmark results for number of tables, columns, and average rows per table.
"""

#from src.data_analysis.qc_stats import plot_metric
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESOURCES = {
    'hugging': ['hugging_table_list_dedup'],
    'github': ['github_table_list_dedup'],
    'html': ['html_table_list_mapped_dedup'],
    'llm': ['llm_table_list_mapped_dedup']
}

RESOURCE_LABELS = {
    'hugging': 'Hugging Face',
    'github': 'GitHub',
    'html': 'HTML',
    'llm': 'S2ORC+LLM'
}

# Benchmark data (WDC removed)
benchmark_data = [
    ["SANTOS Small", 550, 6322, 6921, 0.45],
    ["TUS Small", 1530, 14810, 4466, 1.00],
    ["TUS Large", 5043, 54923, 1915, 1.50],
    ["SANTOS Large", 11090, 123477, 7675, 11.00],
    #["WDC", 50000000, 250000000, 14, 500.00]
]

BENCHMARK_NAMES = [x[0] for x in benchmark_data]

def annotate_bars(ax, fontsize=16):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2, height*0.9),
                        ha='center', va='bottom', fontsize=fontsize, rotation=0)

def plot_metrics_grid(df): 
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    metrics = ["# Tables", "# Cols", "Avg # Rows"]
    palette_baseline = ["#8b2e2e", "#b74a3c", "#d96e44", "#f29e4c", "#FFBE5F"]
    palette_resource = ["#486f90", "#4e8094", "#50a89d", "#a5d2bc"]

    bar_width = 0.15
    gap = 0.4
    group_width = len(RESOURCES) * bar_width + gap
    clusters = ['Benchmarks', 'Ours-dup', 'Ours-dedup', 'Ours-w/ \n title', 'Ours-w/ \n valid title']
    resources = list(RESOURCES.keys())

    cluster_key_map = {
        'Ours-dup': " (duplicated)",
        'Ours-dedup': "",
        'Ours-w/ \n title': "-title-dedup",
        'Ours-w/ \n valid title': "-valid-dedup"
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=False, constrained_layout=True)
    fig.suptitle("Statistic across different benchmarks", fontsize=22)  ########

    for ax, metric in zip(axes, metrics):
        heights = []
        colors = []
        positions = []
        for i, val in enumerate(df.iloc[:4][metric]):
            positions.append(i * bar_width)
            heights.append(val)
            colors.append(palette_baseline[i])
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
        ax.bar(positions, heights, width=bar_width, color=colors)
        ax.set_yscale('log')
        ax.margins(y=0.1)
        annotate_bars(ax, fontsize=12)
        ax.set_ylabel(f"{metric}", fontsize=24) 

    axes[-1].set_xticks(xtick_positions)
    axes[-1].set_xticklabels(xtick_labels, rotation=0, fontsize=17)

    handles_baseline = [
        Patch(facecolor=palette_baseline[i], label=BENCHMARK_NAMES[i])
        for i in range(len(BENCHMARK_NAMES))
    ]
    handles_resource = [
        Patch(facecolor=palette_resource[i], label=RESOURCE_LABELS[res])
        for i, res in enumerate(resources)
    ]

    fig.legend(
        handles_baseline + handles_resource,
        [h.get_label() for h in handles_baseline + handles_resource],
        loc="lower center", bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        fontsize=15,
    )

    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_metrics_vertical.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.parquet")
    results_df = pd.read_parquet(results_path)
    # remove rows with WDC
    results_df = results_df[~results_df["Benchmark"].str.contains("WDC")]
    #results_df.to_parquet(results_path, index=False)
    print(results_df)
    plot_metrics_grid(results_df)
    print(f"Saved figure to {OUTPUT_DIR}/benchmark_metrics_vertical.pdf")