#!/usr/bin/env python3
"""
Horizontal 2×4 heatmap visualization for 8-way combinations
4 rows × 2 columns layout
"""
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def create_horizontal_2x4_heatmap(stats, output_path, fig_dir="data/analysis"):
    """Create a horizontal 2×4 heatmap: 4 rows (ModelCard×Dataset) × 2 cols (Paper)"""
    os.makedirs(fig_dir, exist_ok=True)
    
    combination_counts = stats['combination_counts']
    total = stats['total']
    max_count = max(combination_counts.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Sampling Distribution over Ground Truth\n(Paper (P) × ModelCard (MC) × Dataset (D))', 
             fontsize=22, fontweight='bold', y=0.98)
    
    colors = ["#a5d2bc", "#50a89d", "#4e8094", "#486f90"]
    cmap = LinearSegmentedColormap.from_list("teal_gradient", colors)
    
    # Build 2×4 matrix (transposed from original 4×2)
    # Rows: Paper (No Paper, Paper)
    # Cols: ModelCard × Dataset (4 combinations)
    matrix = []
    annotations = []
    col_labels = []
    
    for p_val in [0, 1]:
        row = []
        row_annot = []
        for (m_val, d_val) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            count = combination_counts.get(str((p_val, m_val, d_val)), 0)
            row.append(count)
            row_annot.append(f"{count}\n({count/total*100:.1f}%)")
        matrix.append(row)
        annotations.append(row_annot)
    
    row_labels = ["No Paper (P=F)", "Paper (P=T)"]
    
    # Column labels
    for (m_val, d_val) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mc_label = "MC=T" if m_val else "MC=F"
        d_label = "D=T" if d_val else "D=F"
        col_labels.append(f"{mc_label}, {d_label}")
    
    sns.heatmap(
        np.array(matrix),
        annot=annotations,
        fmt='s',
        cmap=cmap,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=2,
        linecolor='white',
        vmin=0,
        vmax=max_count,
        square=True,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('ModelCard (MC) × Dataset (D)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Paper (P)', fontsize=20, fontweight='bold')
    
    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = os.path.join(fig_dir, output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-file", type=str)
    parser.add_argument("--output", type=str, default="sampling_2x4_horizontal.pdf")
    parser.add_argument("--fig-dir", type=str, default="fig")
    args = parser.parse_args()
    
    if args.stats_file:
        with open(args.stats_file, 'r') as f:
            data = json.load(f)
        stats = {
            "combination_counts": data.get("combination_counts", {}),
            "total": data.get("final_unique_pairs", 200)
        }
    else:
        stats = {
            "combination_counts": {
                "(0, 0, 0)": 21, "(0, 0, 1)": 11, "(0, 1, 0)": 21, "(0, 1, 1)": 12,
                "(1, 0, 0)": 35, "(1, 0, 1)": 24, "(1, 1, 0)": 24, "(1, 1, 1)": 50
            },
            "total": 198
        }
    
    create_horizontal_2x4_heatmap(stats, args.output, args.fig_dir)

