"""
Author: Zhengyuan Dong
Created: 2025-04-04
Last Modified: 2025-04-06
Description: Analyze saved overlap scores and direct citation links to evaluate threshold decisions using precision-recall and KDE valley method, then visualize KDE distribution and histogram by class with multiple thresholds.
"""

import os, json, gzip, pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve
from scipy.stats import ttest_ind
from sklearn.metrics import roc_curve

from scipy.stats import mannwhitneyu, ks_2samp

# === Paths ===
COMBINED_PATH = "data/processed/modelcard_citation_all_matrices.pkl.gz"  ######## updated ########
RESULT_DIR = "data/analysis"
os.makedirs(RESULT_DIR, exist_ok=True)

def otsu_threshold(values):
    """(NEW) Compute Otsu's threshold for 0-1 overlap scores."""
    hist, bin_edges = np.histogram(values, bins=256, range=(0, 1))
    total = values.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground, weight_background, weight_foreground = 0, 0, 0, 0

    for i in range(len(hist)):
        sum_total += i * hist[i]

    for i in range(len(hist)):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_foreground += i * hist[i]
        mean_background = float(sum_foreground) / weight_background
        mean_foreground = float(sum_total - sum_foreground) / weight_foreground
        between_class_var = (weight_background * weight_foreground *
                             (mean_background - mean_foreground) ** 2)
        if between_class_var > current_max:
            current_max = between_class_var
            # For bins: threshold is mid of bin i and i+1
            threshold = (bin_edges[i] + bin_edges[i+1]) / 2
    return threshold

def load_direct_pairs_from_combined(combined):
    score_matrix = combined["direct_label"]
    paper_index = combined["paper_index"]
    rows, cols = score_matrix.nonzero()
    direct_pairs = set()
    for i, j in zip(rows, cols):
        if i >= j:  # avoid duplicates (i,j) and (j,i)
            continue
        if score_matrix[i, j] == 1.0:
            pair = tuple(sorted([paper_index[i], paper_index[j]]))
            direct_pairs.add(pair)
    return direct_pairs

# === Load data ===
with gzip.open(COMBINED_PATH, "rb") as f:
    combined = pickle.load(f)

print("✅ Combined matrix file loaded successfully\n")

# === Create set of direct citation pairs ===
direct_pairs = load_direct_pairs_from_combined(combined)

# === Prepare labels and scores ===
y_true = []
y_scores = []
score_labels = []

score_matrix = combined["max_pr"]
paper_index = combined["paper_index"]

rows, cols = score_matrix.nonzero()
for i, j in zip(rows, cols):
    if i >= j:
        continue
    pair = tuple(sorted([paper_index[i], paper_index[j]]))
    score = score_matrix[i, j]
    is_direct = (pair in direct_pairs)
    y_true.append(1 if is_direct else 0)
    y_scores.append(score)
    score_labels.append("Direct" if is_direct else "None")

# === Compute precision-recall for F1-based threshold ===
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx_pr = f1.argmax()
best_threshold_f1 = thresholds_pr[best_idx_pr]
best_f1 = f1[best_idx_pr]

print(f"🎯 Best Threshold (F1-based) = {best_threshold_f1:.4f}  (F1 = {best_f1:.4f})")

# === Prepare DataFrame for distribution analysis (KDE, histogram) ===
df_scores = pd.DataFrame({"Score": y_scores, "Label": score_labels})
vals_direct = df_scores[df_scores.Label == "Direct"]["Score"].values
vals_none = df_scores[df_scores.Label == "None"]["Score"].values

# === Statistical test (p-value) using t-test ===
p_value = ttest_ind(vals_direct, vals_none, equal_var=False).pvalue
print(f"📊 t-test p-value (Direct vs None): {p_value:.4e}")

########
# 新增：Mann-Whitney U 检验（非参数检验，不要求正态分布）
u_stat, p_value_mwu = mannwhitneyu(vals_direct, vals_none, alternative='two-sided')
print(f"📊 Mann-Whitney U test p-value: {p_value_mwu:.4e}")

# 新增：KS 检验，比较两个累积分布函数
ks_stat, p_value_ks = ks_2samp(vals_direct, vals_none)
print(f"📊 KS test p-value: {p_value_ks:.4e}")

########
# 根据上述假设检验结果，如果各 p-value 均不显著（例如 p > 0.05），说明 Direct 与 None 之间的分布差异不足，
# 即使选取一个阈值也无法明显区分这两类样本，从而证明基于阈值的划分方法缺乏统计依据。
########

# === Compute KDE "valley" threshold ===
kde_direct = sns.kdeplot(vals_direct, bw_adjust=0.7).get_lines()[0].get_data()
plt.clf()  # Clear
kde_none = sns.kdeplot(vals_none, bw_adjust=0.7).get_lines()[0].get_data()
plt.clf()  # Clear

x1, y1 = kde_direct
x2, y2 = kde_none
x_common = np.linspace(0, 1, 1000)
interp_y1 = np.interp(x_common, x1, y1)
interp_y2 = np.interp(x_common, x2, y2)
diff = np.abs(interp_y1 - interp_y2)
valley_threshold = x_common[diff.argmin()]
print(f"🟡 KDE Valley Threshold = {valley_threshold:.4f} (minimum density difference)")

########
# === Compute Youden's J threshold (via ROC) ===
fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
youden = tpr - fpr
best_idx_roc = np.argmax(youden)
best_threshold_youden = thresholds_roc[best_idx_roc]
best_youden_score = youden[best_idx_roc]
print(f"🟢 Youden's J = {best_youden_score:.4f}; Threshold = {best_threshold_youden:.4f}")
########

########
# === Compute Otsu’s threshold ===
#otsu_thresh = otsu_threshold(np.array(y_scores))
#print(f"🟣 Otsu Threshold = {otsu_thresh:.4f}")
########

# === Plot KDE with multiple thresholds ===
plt.figure(figsize=(8, 5))

colors = {"Direct": "orange", "None": "gray"}
for label in ["Direct", "None"]:
    sns.kdeplot(
        df_scores[df_scores["Label"] == label]["Score"],
        label=label,
        color=colors[label],
        fill=True,
        alpha=0.4,
        linewidth=2
    )

# Mark vertical lines for each threshold
plt.axvline(best_threshold_f1, color='red', linestyle='--', linewidth=1.5,
            label=f'F1 = {best_threshold_f1:.2f}')
plt.axvline(valley_threshold, color='blue', linestyle='--', linewidth=1.5,
            label=f'KDE valley = {valley_threshold:.2f}')
plt.axvline(best_threshold_youden, color='green', linestyle='--', linewidth=1.5,
            label=f'Youden = {best_threshold_youden:.2f}')
#plt.axvline(otsu_thresh, color='purple', linestyle='--', linewidth=1.5,
#            label=f'Otsu = {otsu_thresh:.2f}')

plt.title("KDE of Overlap Scores by Label (Multiple Threshold Methods)")
plt.xlabel("Overlap Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
kde_fig_path = os.path.join(RESULT_DIR, "score_direct_vs_nondirect_KDE_multi_thresholds.pdf")
plt.savefig(kde_fig_path, format='pdf')
plt.close()
print(f"📈 KDE with multiple thresholds saved to: {kde_fig_path}")

# === Histogram (Distribution Plot) with multiple thresholds (3-row subplot) ===
fig, axes = plt.subplots(3, 1, figsize=(8, 15))  # 三行子图

# --- (1) 第一行: None + Direct 重叠 ---
axes[0].hist(
    vals_none,
    bins=200,
    alpha=0.5,
    color="gray",
    label="None",
    zorder=1,
    log=True
)
axes[0].hist(
    vals_direct,
    bins=200,
    alpha=0.6,
    color="orange",
    label="Direct",
    zorder=2,
    log=True
)

axes[0].axvline(best_threshold_f1, color='red', linestyle='--', linewidth=1.5,
                label=f'F1 = {best_threshold_f1:.2f}')
axes[0].axvline(valley_threshold, color='blue', linestyle='--', linewidth=1.5,
                label=f'KDE valley = {valley_threshold:.2f}')
axes[0].axvline(best_threshold_youden, color='green', linestyle='--', linewidth=1.5,
                label=f'Youden = {best_threshold_youden:.2f}')
#axes[0].axvline(otsu_thresh, color='purple', linestyle='--', linewidth=1.5,
#                label=f'Otsu = {otsu_thresh:.2f}')

axes[0].set_title("Histogram (None + Direct)")
axes[0].set_xlabel("Overlap Score")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# --- (2) 第二行: 只看 Direct ---
axes[1].hist(
    vals_direct,
    bins=200,
    alpha=0.6,
    color="orange",
    label="Direct",
    zorder=2,
    log=True
)
axes[1].axvline(best_threshold_f1, color='red', linestyle='--', linewidth=1.5,
                label=f'F1 = {best_threshold_f1:.2f}')
axes[1].axvline(valley_threshold, color='blue', linestyle='--', linewidth=1.5,
                label=f'KDE valley = {valley_threshold:.2f}')
axes[1].axvline(best_threshold_youden, color='green', linestyle='--', linewidth=1.5,
                label=f'Youden = {best_threshold_youden:.2f}')
#axes[1].axvline(otsu_thresh, color='purple', linestyle='--', linewidth=1.5,
#                label=f'Otsu = {otsu_thresh:.2f}')
axes[1].set_title("Histogram (Direct Only)")
axes[1].set_xlabel("Overlap Score")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# --- (3) 第三行: 只看 None ---
axes[2].hist(
    vals_none,
    bins=200,
    alpha=0.5,
    color="gray",
    label="None",
    zorder=1,
    log=True
)
axes[2].axvline(best_threshold_f1, color='red', linestyle='--', linewidth=1.5,
                label=f'F1 = {best_threshold_f1:.2f}')
axes[2].axvline(valley_threshold, color='blue', linestyle='--', linewidth=1.5,
                label=f'KDE valley = {valley_threshold:.2f}')
axes[2].axvline(best_threshold_youden, color='green', linestyle='--', linewidth=1.5,
                label=f'Youden = {best_threshold_youden:.2f}')
#axes[2].axvline(otsu_thresh, color='purple', linestyle='--', linewidth=1.5,
#                label=f'Otsu = {otsu_thresh:.2f}')
axes[2].set_title("Histogram (None Only)")
axes[2].set_xlabel("Overlap Score")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
hist_fig_path = os.path.join(RESULT_DIR, "score_histogram_3row_comparison.pdf")
plt.savefig(hist_fig_path, format='pdf')
plt.close()
print(f"📊 3-row comparison histogram saved to: {hist_fig_path}")
