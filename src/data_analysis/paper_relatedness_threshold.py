"""
Author: Zhengyuan Dong
Created: 2025-04-04
Last Modified: 2025-04-06
Description: Analyze saved overlap scores and direct citation links to evaluate threshold decisions using precision-recall and KDE valley method, then visualize KDE distribution and histogram by class with multiple thresholds.
"""

import os, json, gzip, pickle
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from src.utils import load_config
from sklearn.metrics import (
    precision_recall_curve, roc_curve, balanced_accuracy_score, 
    matthews_corrcoef, confusion_matrix, f1_score, precision_score, recall_score
)
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest

# === Paths ===
COMBINED_PATH = "data/processed/modelcard_citation_all_matrices.pkl.gz"  ######## updated ########
RESULT_DIR = "data/analysis"

def find_balanced_accuracy_threshold(y_true, y_scores):
    """Find threshold that maximizes balanced accuracy."""
    thresholds = np.linspace(0, 1, 100)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold, best_score

def find_mcc_threshold(y_true, y_scores):
    """Find threshold that maximizes Matthews Correlation Coefficient."""
    thresholds = np.linspace(0, 1, 100)
    best_score = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold, best_score

def find_gmean_threshold(y_true, y_scores):
    """Find threshold that maximizes geometric mean of sensitivity and specificity."""
    thresholds = np.linspace(0, 1, 100)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = np.sqrt(sensitivity * specificity)
        if gmean > best_score:
            best_score = gmean
            best_threshold = threshold
            
    return best_threshold, best_score

def find_iba_threshold(y_true, y_scores):
    """Find threshold that maximizes Index of Balanced Accuracy (IBA)."""
    thresholds = np.linspace(0, 1, 100)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        dominance = abs(sensitivity - specificity)
        iba = balanced_acc * (1 + dominance)
        if iba > best_score:
            best_score = iba
            best_threshold = threshold
            
    return best_threshold, best_score

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

def find_percentile_threshold(y_true, y_scores, percentile=95):
    """Find threshold at a specific percentile of non-direct scores"""
    non_direct_scores = y_scores[y_true == 0]
    threshold = np.percentile(non_direct_scores, percentile)
    return threshold

def find_isolation_forest_threshold(y_scores):
    """Use Isolation Forest to find threshold"""
    clf = IsolationForest(contamination=0.1, random_state=42)
    scores_2d = y_scores.reshape(-1, 1)
    clf.fit(scores_2d)
    scores = clf.score_samples(scores_2d)
    threshold = np.percentile(scores, 90)
    return threshold

def find_hybrid_threshold(y_true, y_scores):
    """Combine direct and non-direct information"""
    # 1. 确保direct citations的召回率
    direct_scores = y_scores[y_true == 1]
    min_direct_threshold = np.min(direct_scores)
    
    # 2. 从non-direct中找出异常值
    non_direct_scores = y_scores[y_true == 0]
    non_direct_threshold = np.percentile(non_direct_scores, 95)
    
    # 3. 取两者的最小值作为最终阈值
    return min(min_direct_threshold, non_direct_threshold)

# === Main execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine overlap thresholds")
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--input', dest='input', default=None,
                        help='Path to modelcard_citation_all_matrices pkl.gz (default: auto-detect from tag)')
    parser.add_argument('--output-dir', dest='output_dir', default=None,
                        help='Directory for output files (default: data/analysis)')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    tag = args.tag
    suffix = f"_{tag}" if tag else ""
    
    # Determine input/output paths based on tag
    combined_path = args.input or os.path.join(processed_base_path, f"modelcard_citation_all_matrices{suffix}.pkl.gz")
    result_dir = args.output_dir or os.path.join(base_path, 'analysis')
    os.makedirs(result_dir, exist_ok=True)
    
    print("📁 Paths in use:")
    print(f"   Input matrices:      {combined_path}")
    print(f"   Output directory:    {result_dir}")

    # === Load data ===
    with gzip.open(combined_path, "rb") as f:
        combined = pickle.load(f)

    print("✅ Combined matrix file loaded successfully")
    print("📊 Available keys in the data:", list(combined.keys()), "\n")

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

    # Add data validation
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n_direct = np.sum(y_true == 1)
    n_nondirect = np.sum(y_true == 0)
    n_total = len(y_true)

    print(f"\n📊 Data Distribution:")
    print(f"Total pairs: {n_total}")
    print(f"Direct pairs: {n_direct} ({n_direct/n_total*100:.2f}%)")
    print(f"Non-direct pairs: {n_nondirect} ({n_nondirect/n_total*100:.2f}%)")
    assert n_direct + n_nondirect == n_total, "Data validation failed: direct + non-direct != total"

# === Compute precision-recall for F1-based threshold ===
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx_pr = f1.argmax()
    best_threshold_f1 = thresholds_pr[best_idx_pr]
    best_f1 = f1[best_idx_pr]
    
    print(f"🎯 Best Threshold (F1-based) = {best_threshold_f1:.4f}  (F1 = {best_f1:.4f})")
    
    # === Compute Youden's J threshold (via ROC) ===
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    youden = tpr - fpr
    best_idx_roc = np.argmax(youden)
    best_threshold_youden = thresholds_roc[best_idx_roc]
    best_youden_score = youden[best_idx_roc]
    print(f"🟢 Youden's J = {best_youden_score:.4f}; Threshold = {best_threshold_youden:.4f}")
    
    # === Compute other thresholds ===
    balanced_threshold, balanced_score = find_balanced_accuracy_threshold(y_true, y_scores)
    mcc_threshold, mcc_score = find_mcc_threshold(y_true, y_scores)
    gmean_threshold, gmean_score = find_gmean_threshold(y_true, y_scores)
    iba_threshold, iba_score = find_iba_threshold(y_true, y_scores)
    
    print(f"Balanced Accuracy Threshold = {balanced_threshold:.4f} (score = {balanced_score:.4f})")
    print(f"MCC Threshold = {mcc_threshold:.4f} (score = {mcc_score:.4f})")
    print(f"G-mean Threshold = {gmean_threshold:.4f} (score = {gmean_score:.4f})")
    print(f"IBA Threshold = {iba_threshold:.4f} (score = {iba_score:.4f})")
    
    # === Prepare DataFrame for distribution analysis (KDE, histogram) ===
    df_scores = pd.DataFrame({"Score": y_scores, "Label": score_labels})
    vals_direct = df_scores[df_scores.Label == "Direct"]["Score"].values
    vals_none = df_scores[df_scores.Label == "None"]["Score"].values
    
    # === Statistical test (p-value) using t-test ===
    p_value = ttest_ind(vals_direct, vals_none, equal_var=False).pvalue
    print(f"📊 t-test p-value (Direct vs None): {p_value:.4e}")
    
    # Mann-Whitney U test
    u_stat, p_value_mwu = mannwhitneyu(vals_direct, vals_none, alternative='two-sided')
    print(f"📊 Mann-Whitney U test p-value: {p_value_mwu:.4e}")
    
    # KS test
    ks_stat, p_value_ks = ks_2samp(vals_direct, vals_none)
    print(f"📊 KS test p-value: {p_value_ks:.4e}")
    
    # === Add cross-validation for threshold stability ===
def evaluate_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    return {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }

# 5-fold cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize results storage
cv_results = {
    'f1': {'scores': [], 'thresholds': []},
    'youden': {'scores': [], 'thresholds': []},
    'balanced': {'scores': [], 'thresholds': []},
    'mcc': {'scores': [], 'thresholds': []},
    'gmean': {'scores': [], 'thresholds': []},
    'iba': {'scores': [], 'thresholds': []}
}

for train_idx, val_idx in kf.split(y_scores):
    y_train = np.array(y_true)[train_idx]
    y_val = np.array(y_true)[val_idx]
    scores_train = np.array(y_scores)[train_idx]
    scores_val = np.array(y_scores)[val_idx]
    
    # Compute thresholds on training set
    precision_train, recall_train, thresholds_pr_train = precision_recall_curve(y_train, scores_train)
    f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train + 1e-8)
    best_idx_train = f1_train.argmax()
    best_threshold_f1_train = thresholds_pr_train[best_idx_train]
    
    fpr_train, tpr_train, thresholds_roc_train = roc_curve(y_train, scores_train)
    youden_train = tpr_train - fpr_train
    best_idx_roc_train = np.argmax(youden_train)
    best_threshold_youden_train = thresholds_roc_train[best_idx_roc_train]
    
    # Compute other thresholds on training set
    balanced_threshold_train, _ = find_balanced_accuracy_threshold(y_train, scores_train)
    mcc_threshold_train, _ = find_mcc_threshold(y_train, scores_train)
    gmean_threshold_train, _ = find_gmean_threshold(y_train, scores_train)
    iba_threshold_train, _ = find_iba_threshold(y_train, scores_train)
    
    # Store results for each method
    for method, threshold in [
        ('f1', best_threshold_f1_train),
        ('youden', best_threshold_youden_train),
        ('balanced', balanced_threshold_train),
        ('mcc', mcc_threshold_train),
        ('gmean', gmean_threshold_train),
        ('iba', iba_threshold_train)
    ]:
        metrics = evaluate_threshold(y_val, scores_val, threshold)
        cv_results[method]['scores'].append(metrics)
        cv_results[method]['thresholds'].append(threshold)

# Print cross-validation results
print("\n📊 Cross-validation Results:")
for method in ['f1', 'youden', 'balanced', 'mcc', 'gmean', 'iba']:
    scores = cv_results[method]['scores']
    thresholds = cv_results[method]['thresholds']
    
    avg_f1 = np.mean([m['f1'] for m in scores])
    avg_precision = np.mean([m['precision'] for m in scores])
    avg_recall = np.mean([m['recall'] for m in scores])
    std_f1 = np.std([m['f1'] for m in scores])
    avg_threshold = np.mean(thresholds)
    
    print(f"\n{method.upper()} Method:")
    print(f"  Average Threshold: {avg_threshold:.4f}")
    print(f"  F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")

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
plt.axvline(best_threshold_youden, color='green', linestyle='--', linewidth=1.5,
            label=f'Youden = {best_threshold_youden:.2f}')
plt.axvline(balanced_threshold, color='purple', linestyle='--', linewidth=1.5,
            label=f'Balanced = {balanced_threshold:.2f}')
plt.axvline(mcc_threshold, color='blue', linestyle='--', linewidth=1.5,
            label=f'MCC = {mcc_threshold:.2f}')
plt.axvline(gmean_threshold, color='brown', linestyle='--', linewidth=1.5,
            label=f'G-mean = {gmean_threshold:.2f}')
plt.axvline(iba_threshold, color='pink', linestyle='--', linewidth=1.5,
            label=f'IBA = {iba_threshold:.2f}')

plt.title("KDE of Overlap Scores by Label (Multiple Threshold Methods)")
plt.xlabel("Overlap Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
kde_fig_path = os.path.join(result_dir, "score_direct_vs_nondirect_KDE_multi_thresholds.pdf")
plt.savefig(kde_fig_path, format='pdf')
plt.close()
print(f"📈 KDE with multiple thresholds saved to: {kde_fig_path}")

# === Multi-score analysis and visualization (3-row, each row for a score) ===
score_keys = ['max_pr', 'jaccard', 'dice']
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

for idx, score_key in enumerate(score_keys):
    print(f"\n{'='*30}\nAnalyzing score: {score_key}\n{'='*30}")
    score_matrix = combined[score_key]
    paper_index = combined["paper_index"]
    y_true = []
    y_scores = []
    score_labels = []
    rows, cols = score_matrix.nonzero()
    for i, j in zip(rows, cols):
        if i == j:
            continue  # skip self-loop
        if i > j:
            continue  # avoid duplicates
        pair = tuple(sorted([paper_index[i], paper_index[j]]))
        score = score_matrix[i, j]
        is_direct = (pair in direct_pairs)
        y_true.append(1 if is_direct else 0)
        y_scores.append(score)
        score_labels.append("Direct" if is_direct else "None")
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n_direct = np.sum(y_true == 1)
    n_nondirect = np.sum(y_true == 0)
    n_total = len(y_true)
    print(f"Total pairs: {n_total}")
    print(f"Direct pairs: {n_direct} ({n_direct/n_total*100:.2f}%)")
    print(f"Non-direct pairs: {n_nondirect} ({n_nondirect/n_total*100:.2f}%)")
    assert n_direct + n_nondirect == n_total, "Data validation failed: direct + non-direct != total"
    
    # Compute thresholds
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx_pr = f1.argmax()
    best_threshold_f1 = thresholds_pr[best_idx_pr]
    best_f1 = f1[best_idx_pr]
    
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    youden = tpr - fpr
    best_idx_roc = np.argmax(youden)
    best_threshold_youden = thresholds_roc[best_idx_roc]
    
    # Compute other thresholds
    balanced_threshold, balanced_score = find_balanced_accuracy_threshold(y_true, y_scores)
    mcc_threshold, mcc_score = find_mcc_threshold(y_true, y_scores)
    gmean_threshold, gmean_score = find_gmean_threshold(y_true, y_scores)
    iba_threshold, iba_score = find_iba_threshold(y_true, y_scores)
    
    # Print thresholds
    print(f"F1-based Threshold = {best_threshold_f1:.4f} (F1 = {best_f1:.4f})")
    print(f"Youden's J Threshold = {best_threshold_youden:.4f}")
    print(f"Balanced Accuracy Threshold = {balanced_threshold:.4f} (score = {balanced_score:.4f})")
    print(f"MCC Threshold = {mcc_threshold:.4f} (score = {mcc_score:.4f})")
    print(f"G-mean Threshold = {gmean_threshold:.4f} (score = {gmean_score:.4f})")
    print(f"IBA Threshold = {iba_threshold:.4f} (score = {iba_score:.4f})")
    
    # Plot
    ax = axes[idx]
    colors = {"Direct": "orange", "None": "gray"}
    for label in ["Direct", "None"]:
        ax.hist(
            [s for s, l in zip(y_scores, score_labels) if l == label],
            bins=200,
            alpha=0.5 if label == "None" else 0.6,
            color=colors[label],
            label=label,
            zorder=1 if label == "None" else 2,
            log=True
        )
    
    # Threshold lines
    ax.axvline(best_threshold_f1, color='red', linestyle='--', linewidth=1.5, label=f'F1 = {best_threshold_f1:.2f}')
    ax.axvline(best_threshold_youden, color='green', linestyle='--', linewidth=1.5, label=f'Youden = {best_threshold_youden:.2f}')
    ax.axvline(balanced_threshold, color='purple', linestyle='--', linewidth=1.5, label=f'Balanced = {balanced_threshold:.2f}')
    ax.axvline(mcc_threshold, color='blue', linestyle='--', linewidth=1.5, label=f'MCC = {mcc_threshold:.2f}')
    ax.axvline(gmean_threshold, color='brown', linestyle='--', linewidth=1.5, label=f'G-mean = {gmean_threshold:.2f}')
    ax.axvline(iba_threshold, color='pink', linestyle='--', linewidth=1.5, label=f'IBA = {iba_threshold:.2f}')
    
    ax.set_title(f"Histogram ({score_key})")
    ax.set_xlabel("Overlap Score")
    ax.set_ylabel("Frequency")
    ax.legend()
plt.tight_layout()
main_fig_path = os.path.join(result_dir, "score_histogram_3row_multi_score.pdf")
plt.savefig(main_fig_path, format='pdf')
plt.close()
print(f"\n📈 Multi-score 3-row histogram saved to: {main_fig_path}")

# === Compute new thresholds ===
percentile_threshold = find_percentile_threshold(y_true, y_scores)
isolation_threshold = find_isolation_forest_threshold(y_scores)
hybrid_threshold = find_hybrid_threshold(y_true, y_scores)

print(f"\n📊 New Threshold Methods:")
print(f"Percentile (95th) Threshold = {percentile_threshold:.4f}")
print(f"Isolation Forest Threshold = {isolation_threshold:.4f}")
print(f"Hybrid Threshold = {hybrid_threshold:.4f}")

# === Evaluate new thresholds ===
def evaluate_threshold_with_metrics(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算direct citations的召回率
    direct_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 计算从non-direct中找出的潜在重要关系比例
    non_direct_total = tn + fp
    potential_important = fp / non_direct_total if non_direct_total > 0 else 0
    
    return {
        'direct_recall': direct_recall,
        'potential_important_ratio': potential_important,
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }

# 评估新方法
new_methods = {
    'percentile': percentile_threshold,
    'isolation': isolation_threshold,
    'hybrid': hybrid_threshold
}

print("\n📊 New Methods Evaluation:")
for method, threshold in new_methods.items():
    metrics = evaluate_threshold_with_metrics(y_true, y_scores, threshold)
    print(f"\n{method.upper()} Method:")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Direct Recall: {metrics['direct_recall']:.4f}")
    print(f"  Potential Important Ratio: {metrics['potential_important_ratio']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

# === Plot distribution analysis ===
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_scores, x="Score", hue="Label", common_norm=False)
plt.axvline(percentile_threshold, color='red', linestyle='--', label='Percentile')
plt.axvline(isolation_threshold, color='green', linestyle='--', label='Isolation Forest')
plt.axvline(hybrid_threshold, color='blue', linestyle='--', label='Hybrid')
plt.title("Distribution of Scores with New Thresholds")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(result_dir, "score_distribution_new_thresholds.pdf"))
plt.close()
