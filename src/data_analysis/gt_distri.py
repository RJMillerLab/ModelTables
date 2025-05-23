"""
Author: Zhengyuan Dong
Created: 2025-04-11
Last Modified: 2025-04-21
Description: Distribution analysis of GT lengths in pickle files using a loader class and reusable functions.
Usage:
    python -m src.data_analysis.gt_distri
"""

import os
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'figure.titlesize': 28,
    'legend.fontsize': 22
})

OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {                                            ########
    "paper":              "#4e8094",                   ########
    "model":              "#50a89d",                   ########
    "dataset":            "#a5d2bc",                   ########
    "santos_large":       "#d96e44",                   ########
    "santos_small":       "#f29e4c",                   ########
    "tus_large":          "#8b2e2e",                   ########
    "tus_small":          "#b74a3c",                   ########
}
# "#6f4e37","#c0c0c0",   

# -------- Loader --------
class GTLengthLoader:
    def __init__(self, name: str, path: str, key: str | None = None):
        self.name, self.path, self.key = name, path, key

    def _load(self):
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rb") as f:
            data = pickle.load(f)
        return data[self.key] if self.key else data

    def lengths(self):
        data = self._load()
        return [len(v) for v in data.values() if isinstance(v, list)]

# -------- Helper --------
def load_lengths(path_map):
    out = {}
    for src, info in path_map.items():
        path, key = info if isinstance(info, tuple) else (info, None)
        out[src] = GTLengthLoader(src, path, key).lengths()
    return out

def plot_kde(length_data, title, prefix):
    plt.figure(figsize=(8, 8))
    total = sum(len(v) for v in length_data.values())
    for raw, lens in length_data.items():
        if not lens: continue
        w     = np.ones_like(lens) / total
        kde   = gaussian_kde(lens, weights=w)
        xs    = np.linspace(min(lens) - 1, max(lens) + 1, 200)
        ys    = np.clip(kde(xs), 1e-12, None)         ########
        label = raw
        plt.fill_between(xs, ys, alpha=0.4, color=PALETTE[label], label=label)
        plt.plot(xs, ys, color=PALETTE[label], linewidth=2)
    plt.title(title)
    plt.xlabel("List Length")
    plt.ylabel("Proportion Density")
    plt.xscale("log")                                 ########
    plt.yscale("log")                                 ########
    plt.legend(title="Source", fontsize=22)           ########
    plt.grid(False); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_kde.pdf"))

def plot_histogram(length_data, palette, title, prefix):
    plt.figure(figsize=(8, 8))
    total = sum(len(v) for v in length_data.values())        ########
    max_val = max(max(v) for v in length_data.values() if v)
    bins = np.arange(0, max_val + 2)
    for src, lengths in length_data.items():
        if not lengths: continue
        weights = np.ones_like(lengths) / total              ########
        plt.hist(lengths, bins=bins, weights=weights,        ########
                 alpha=0.4, label=src, color=palette[src],
                 edgecolor=None)
    plt.title(title); plt.xlabel("List Length"); plt.ylabel("Proportion") ########
    plt.xscale("log"); plt.legend(); plt.grid(False); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_hist.pdf"))

def plot_log_boxplot(length_data, palette, title, prefix):
    import matplotlib.patches as mpatches
    plt.figure(figsize=(8, 8))

    labels = []
    data = []
    colors = []
    for src in length_data:
        if not length_data[src]:
            continue
        labels.append(src)
        data.append(length_data[src])
        colors.append(palette[src])

    box = plt.boxplot(data, patch_artist=True, showfliers=True)

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.xticks(range(1, len(labels)+1), labels, rotation=30, fontsize=16)

    plt.yscale('log')
    plt.ylabel('List Length (log scale)', fontsize=20)
    plt.title(title, fontsize=22)
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

    patches = [mpatches.Patch(color=palette[l], label=l) for l in labels]
    plt.legend(handles=patches, title='Source', fontsize=16,
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(labels))  # <--- 这里改了

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_boxplot.pdf"))
    plt.show()

GT_DIR = "data/gt"
PATHS = {
    "paper":  os.path.join(GT_DIR, "csv_pair_adj_direct_label_processed.pkl"),
    "model":         os.path.join(GT_DIR, "scilake_gt_modellink_model_adj_processed.pkl"),
    "dataset":       os.path.join(GT_DIR, "scilake_gt_modellink_dataset_adj_processed.pkl"),
    "santos_small":  "/Users/doradong/Repo/santos/groundtruth/santosUnionBenchmark.pickle",
    "santos_large": "/Users/doradong/Repo/santos/groundtruth/real_tablesUnionBenchmark.pickle",
    "tus_small":     "/Users/doradong/Repo/table-union-search-benchmark/tus_small_query_candidate.pkl",
    "tus_large":     "/Users/doradong/Repo/table-union-search-benchmark/tus_large_query_candidate.pkl",
}

lengths = load_lengths(PATHS)

# plot_histogram(lengths, "GT Length (All Sources)", "gt_all")   ########
#plot_kde(lengths, "GT Length Distribution (All Sources)", "gt_all")
plot_log_boxplot(lengths, PALETTE, "GT Length Distribution (Boxplot, Log Scale)", "gt_boxplot")

print("Plots saved →", OUTPUT_DIR)
