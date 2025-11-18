#!/usr/bin/env python3
"""
Visualize GT levels agreement (3-way correlation heatmap)
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def visualize_gt_agreement(jsonl_path, output_path, fig_dir="data/analysis"):
    """Create heatmap showing agreement between 3 GT levels"""
    import os
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Build 2x2x2 agreement matrix
    # Count combinations of (paper, modelcard, dataset)
    combo_counts = {}
    for r in results:
        gt = r.get('gt_labels', {})
        combo = (gt.get('paper', 0), gt.get('modelcard', 0), gt.get('dataset', 0))
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    
    # Create 4x4 matrix showing Paper×ModelCard with Dataset colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ground Truth Level Agreement\n(Paper × ModelCard × Dataset)', 
                 fontsize=22, fontweight='bold')
    
    colors = ["#a5d2bc", "#50a89d", "#4e8094", "#486f90"]
    cmap = LinearSegmentedColormap.from_list("teal_gradient",colors)
    
    # Left: Dataset=0, Right: Dataset=1
    for d_idx, d_val in enumerate([0, 1]):
        ax = axes[d_idx]
        
        matrix = []
        annotations = []
        for p_val in [0, 1]:
            row = []
            row_annot = []
            for mc_val in [0, 1]:
                count = combo_counts.get((p_val, mc_val, d_val), 0)
                row.append(count)
                row_annot.append(f"{count}")
            matrix.append(row)
            annotations.append(row_annot)
        
        sns.heatmap(np.array(matrix), annot=annotations, fmt='s', cmap=cmap, ax=ax,
                   xticklabels=["MC=F", "MC=T"], yticklabels=["P=F", "P=T"],
                   linewidths=2, linecolor='white', cbar=False,
                   annot_kws={'size': 14}, square=True)
        
        ax.set_title(f'Dataset={d_val}', fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('ModelCard (MC)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Paper (P)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_file = f"{fig_dir}/{output_path}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    visualize_gt_agreement('output/gpt_evaluation/step2_full_198.jsonl', 
                          'gt_agreement.pdf', fig_dir="data/analysis")
