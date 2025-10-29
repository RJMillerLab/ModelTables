import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib.gridspec import GridSpec

def cohens_kappa(labels1, labels2):
    confusion = np.zeros((2, 2))
    for l1, l2 in zip(labels1, labels2):
        confusion[int(l1), int(l2)] += 1
    P_o = (confusion[0, 0] + confusion[1, 1]) / np.sum(confusion)
    P_e = ((confusion[0, :].sum() * confusion[:, 0].sum()) + 
           (confusion[1, :].sum() * confusion[:, 1].sum())) / (np.sum(confusion)**2)
    return (P_o - P_e) / (1 - P_e) if P_e != 1 else 1.0

def fleiss_kappa(M):
    M = np.array(M)
    N, k = M.shape
    unique_vals = np.unique(M)
    p_j = np.array([np.sum(M == val) / (N * k) for val in unique_vals])
    P_e = np.sum(p_j ** 2)
    P_i = (np.sum(M * M, axis=1) - k) / (k * (k - 1))
    P_bar = np.mean(P_i)
    return (P_bar - P_e) / (1 - P_e) if P_e != 1 else 1.0

def main():
    with open('output/gpt_evaluation/step2_full_198.jsonl', 'r') as f:
        results = [json.loads(line) for line in f if line.strip()]

    models_full = ['anthropic/claude-3.5-sonnet', 'deepseek/deepseek-chat',
                   'meta-llama/llama-3-70b-instruct', 'openai/gpt-3.5-turbo']
    models = ['Claude', 'DeepSeek', 'Llama', 'GPT-3.5']
    gt_lvls = ['paper', 'modelcard', 'dataset']
    gt_names = ['Paper', 'ModelCard', 'Dataset']

    # ------- (1) GT-GT Consistency -------
    gt_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            l1 = [r.get('gt_labels', {}).get(gt_lvls[i], 0) for r in results]
            l2 = [r.get('gt_labels', {}).get(gt_lvls[j], 0) for r in results]
            gt_matrix[i, j] = cohens_kappa(l1, l2)

    # ------- (2) LLM-LLM Consistency -------
    label_map = {'YES': 1, 'NO': 0, 'UNSURE': 0.5}
    llm_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            l1 = [label_map.get(r.get('model_responses', {}).get(models_full[i], {}).get('related', 'NO'), 0)
                  for r in results]
            l2 = [label_map.get(r.get('model_responses', {}).get(models_full[j], {}).get('related', 'NO'), 0)
                  for r in results]
            llm_matrix[i, j] = cohens_kappa(l1, l2)

    # ------- (3) LLM-GT Consistency -------
    llm_gt_matrix = np.zeros((4, 3))
    for i in range(4):
        for j in range(3):
            l1 = [label_map.get(r.get('model_responses', {}).get(models_full[i], {}).get('related', 'NO'), 0)
                  for r in results]
            l2 = [r.get('gt_labels', {}).get(gt_lvls[j], 0) for r in results]
            llm_gt_matrix[i, j] = cohens_kappa(l1, l2)

    # ------- (4) Fleiss' Kappa -------
    M = [[label_map.get(r.get('model_responses', {}).get(m, {}).get('related', 'NO'), 0)
          for m in models_full] for r in results]
    fleiss_k = fleiss_kappa(M)
    
    # ------- Plot -------
    # Use GridSpec with width ratios matching column counts to ensure equal cell sizes
    fig = plt.figure(figsize=(16, 5.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[3, 4, 3], wspace=0.12)
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Plot heatmaps without individual colorbars
    sns.heatmap(gt_matrix, annot=True, fmt=".2f", xticklabels=gt_names, yticklabels=gt_names,
                ax=ax0, cmap="Blues", square=True, cbar=False, vmin=0, vmax=1)
    ax0.set_xlabel("GT-GT Consistency (Kappa)", fontsize=13, labelpad=10)
    
    sns.heatmap(llm_matrix, annot=True, fmt=".2f", xticklabels=models, yticklabels=models,
                ax=ax1, cmap="Blues", square=True, cbar=False, vmin=0, vmax=1)
    ax1.set_xlabel("LLM-LLM Consistency (Kappa)", fontsize=13, labelpad=10)
    
    im = sns.heatmap(llm_gt_matrix, annot=True, fmt=".2f", xticklabels=gt_names, yticklabels=models,
                     ax=ax2, cmap="Blues", square=True, cbar=False, vmin=0, vmax=1)
    ax2.set_xlabel("LLM-GT Consistency (Kappa)", fontsize=13, labelpad=10)
    
    fig.suptitle("Consistency Metrics (Cohen's Kappa)", fontsize=16, y=0.99)
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.05, right=0.88)
    
    # Add a single shared colorbar on the right with explicit positioning
    # Create an axes for the colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.012, 0.65])  # [left, bottom, width, height]
    cbar = fig.colorbar(im.get_children()[0], cax=cbar_ax)
    plt.savefig("fig/consistency_heatmaps.pdf")
    print("✓ Saved heatmaps to fig/consistency_heatmaps.pdf")
    plt.close()

if __name__ == "__main__":
    main()