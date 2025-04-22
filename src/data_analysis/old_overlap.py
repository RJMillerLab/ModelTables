"""
Author: Zhengyuan Dong
Created: 2025-04-04
Description: Analyze saved overlap scores and direct citation links to evaluate threshold decisions.
"""

import pickle
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# === Paths ===
SCORE_PICKLE = "data/processed/modelcard_citation_overlap_by_paperId_score.pickle"
DIRECT_PICKLE = "data/processed/modelcard_citation_direct_relation.pickle"
RESULT_DIR = "data/analysis"
FIG_PATH = os.path.join(RESULT_DIR, "score_distribution.pdf")

os.makedirs(RESULT_DIR, exist_ok=True)

# === Load data ===
with open(SCORE_PICKLE, "rb") as f:
    score_map = pickle.load(f)

with open(DIRECT_PICKLE, "rb") as f:
    direct_map = pickle.load(f)

print("✅ Files loaded successfully\n")

# === Overlap Score Stats ===
all_scores = list(score_map.values())
max_score = max(all_scores)
min_score = min(all_scores)
avg_score = sum(all_scores) / len(all_scores)

# === Direct Citation Stats ===
direct_edges = sum(len(v) for v in direct_map.values())
direct_nodes = len(direct_map)
direct_degrees = [len(v) for v in direct_map.values()]

# === Full Summary Table ===
summary_all = {
    "Category": [
        "Overlap Score", "Overlap Score", "Overlap Score",
        "Direct Citation", "Direct Citation"
    ],
    "Metric": [
        "# Paper Pairs with Scores",
        "Max Overlap Score",
        "Average Overlap Score",
        "# Papers with Direct Citations",
        "# Direct Citation Links"
    ],
    "Value": [
        len(all_scores),
        f"{max_score:.2f}",
        f"{avg_score:.4f}",
        direct_nodes,
        direct_edges
    ]
}
df_summary_all = pd.DataFrame(summary_all)

print("\n📊 Full Analysis Summary:")
print(df_summary_all.to_string(index=False))

# === Create Set of Direct Pairs ===
direct_pairs = set()
for k, v in direct_map.items():
    for neighbor in v:
        direct_pairs.add(tuple(sorted([k, neighbor])))

# === Label all scored pairs ===
score_labels = []
score_values = []
for (pid1, pid2), score in score_map.items():
    pair = tuple(sorted([pid1, pid2]))
    label = "Direct" if pair in direct_pairs else "None"
    score_labels.append(label)
    score_values.append(score)

# === Plot histogram split by direct vs. non-direct ===
colors = {"Direct": "orange", "None": "gray"}
plt.figure(figsize=(8, 5))

for label in ["Direct", "None"]:
    values = [s for s, l in zip(score_values, score_labels) if l == label]
    plt.hist(values, bins=100, alpha=0.6, label=label, color=colors[label])

plt.axvline(0.6, color='red', linestyle='--', linewidth=1.5, label='Threshold = 0.6')
plt.title("Overlap Score Distribution: Direct vs. Non-Direct")
plt.xlabel("Overlap Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig2_path = os.path.join(RESULT_DIR, "score_direct_vs_nondirect.pdf")
plt.savefig(fig2_path, format='pdf')
print(f"📉 Histogram (Direct vs. Non-Direct) saved to: {fig2_path}")


import seaborn as sns
# KDE plot
plt.figure(figsize=(8, 5))
for label in ["Direct", "None"]:
    values = [s for s, l in zip(score_values, score_labels) if l == label]
    sns.kdeplot(values, label=label, color=colors[label], fill=True, alpha=0.4, linewidth=2)

plt.axvline(0.6, color='red', linestyle='--', linewidth=1.5, label='Threshold = 0.6')
plt.title("KDE of Overlap Scores: Direct vs. Non-Direct")
plt.xlabel("Overlap Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig_kde_path = os.path.join(RESULT_DIR, "score_direct_vs_nondirect_KDE.pdf")
plt.savefig(fig_kde_path, format='pdf')
print(f"📈 KDE plot saved to: {fig_kde_path}")
