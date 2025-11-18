#!/usr/bin/env python3
"""
Visualize LLM-LLM agreement and LLM-GT agreement
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap


def visualize_llm_agreement(jsonl_path, output_path, fig_dir="data/analysis"):
    """Create visualizations for LLM agreement"""
    import os
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Model names
    models = ['anthropic/claude-3.5-sonnet', 'deepseek/deepseek-chat', 
              'meta-llama/llama-3-70b-instruct', 'openai/gpt-3.5-turbo']
    model_short = ['Claude', 'DeepSeek', 'Llama', 'GPT-3.5']
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('LLM Agreement Analysis', fontsize=22, fontweight='bold')
    
    colors = ["#a5d2bc", "#50a89d", "#4e8094", "#486f90"]
    cmap = LinearSegmentedColormap.from_list("teal_gradient", colors)
    
    # 1. Inter-model agreement matrix
    ax1 = fig.add_subplot(gs[0, 0])
    agreement_matrix = np.zeros((4, 4))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                agree = 0; total = 0
                for r in results:
                    resp = r.get('model_responses', {})
                    if m1 in resp and m2 in resp:
                        total += 1
                        if resp[m1].get('related') == resp[m2].get('related'):
                            agree += 1
                if total > 0:
                    agreement_matrix[i, j] = agree / total
                    agreement_matrix[j, i] = agree / total
    
    sns.heatmap(np.flipud(agreement_matrix), annot=True, fmt='.2f', cmap=cmap,
               xticklabels=model_short, yticklabels=list(reversed(model_short)), ax=ax1,
               cbar=True, square=True, linewidths=1, vmin=0, vmax=1)
    ax1.set_title('Inter-Model Agreement', fontsize=14, fontweight='bold')
    
    # 2-4. LLM vs GT for each level
    levels_data = []
    for level_name, level_key, signal_key in [('Paper', 'paper', 'paper_level'),
                                               ('ModelCard', 'modelcard', 'model_level'),
                                               ('Dataset', 'dataset', 'dataset_level')]:
        llm_vs_gt = []
        for m in models:
            correct = 0; total = 0
            for r in results:
                gt = r.get('gt_labels', {}).get(level_key, 0)
                resp = r.get('model_responses', {}).get(m, {})
                level_sig = resp.get('level_signals', {}).get(signal_key, False)
                if resp and 'level_signals' in resp:
                    total += 1
                    if level_sig == (gt == 1):
                        correct += 1
            llm_vs_gt.append(correct/max(1, total))
        levels_data.append((level_name, llm_vs_gt))
    
    # Plot 3 level-specific bars
    for idx, (level_name, data) in enumerate(levels_data):
        ax = fig.add_subplot(gs[0 if idx < 2 else 1, 1 if idx == 0 else (2 if idx == 1 else 0)])
        bars = ax.bar(model_short, data, color=colors[:4])
        ax.set_ylim(0, 1)
        ax.set_title(f'LLM vs GT ({level_name})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    # Overall LLM performance
    ax5 = fig.add_subplot(gs[1, 1:])
    overall = (np.array(levels_data[0][1]) + np.array(levels_data[1][1]) + np.array(levels_data[2][1])) / 3
    bars = ax5.bar(model_short, overall, color=colors[:4])
    ax5.set_ylim(0, 1)
    ax5.set_title('Overall LLM vs GT (Average)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Average Accuracy', fontsize=12)
    for bar, val in zip(bars, overall):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    output_file = f"{fig_dir}/{output_path}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    visualize_llm_agreement('output/gpt_evaluation/step2_full_198.jsonl', 
                          'llm_agreement.pdf', fig_dir="data/analysis")
