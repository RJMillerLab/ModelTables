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
from scipy.sparse import load_npz
from tqdm import tqdm


plt.rcParams.update({
    'font.size': 28,
    'axes.titlesize': 28,
    'axes.labelsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 10,
    'figure.titlesize': 28,
    'legend.fontsize': 18
})

OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "Paper GT":           "#486f90",
    "Model GT":           "#4e8094",
    "Dataset GT":         "#50a89d",
    "SANTOS Large":       "#f29e4c",
    "SANTOS Small":       "#8b2e2e",
    "TUS Large":          "#d96e44",
    "TUS Small":          "#b74a3c",
    "Union GT":           "#a5d2bc",
}

# -------- Loader --------
class GTLengthLoader:
    def __init__(self, name: str, path: str, key: str | None = None):
        self.name, self.path, self.key = name, path, key

    def _load_pkl(self):
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rb") as f:
            data = pickle.load(f)
        return data[self.key] if self.key else data
    
    # def _load(self):
    #     if self.path.endswith(".npz"):
    #         # delegate to our new helper
    #         return None  # signal to use lengths_from_npz
    #     else:
    #         # old pickle path
    #         return self._load_pkl()

    #def lengths(self):
    #    data = self._load()
    #    return [len(v) for v in data.values() if isinstance(v, list)]
    def lengths(self):
        if self.path.endswith(".npz"):
            M = load_npz(self.path).tocsr()
            return M.getnnz(axis=1).tolist()
        else:
            data = self._load_pkl()
            return [len(v) for v in data.values() if isinstance(v, list)]

# -------- Helper --------
def load_lengths(path_map):
    out = {}
    for src, info in path_map.items():
        path, key = info if isinstance(info, tuple) else (info, None)
        out[src] = GTLengthLoader(src, path, key).lengths()
    return out

def plot_kde(length_data, title, prefix):
    plt.figure(figsize=(8, 4))
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
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(title="Source", fontsize=22)
    plt.grid(False); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_kde.pdf"))

def plot_histogram(length_data, palette, title, prefix):
    plt.figure(figsize=(6, 3))
    total = sum(len(v) for v in length_data.values())
    max_val = max(max(v) for v in length_data.values() if v)
    bins = np.arange(0, max_val + 2)
    for src, lengths in length_data.items():
        if not lengths: continue
        weights = np.ones_like(lengths) / total
        plt.hist(lengths, bins=bins, weights=weights,
                 alpha=0.4, label=src, color=palette[src],
                 edgecolor=None)
    plt.title(title); plt.xlabel("List Length"); plt.ylabel("Proportion")
    plt.xscale("log"); plt.legend(); plt.grid(False); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_hist.pdf"))

def plot_log_boxplot(length_data, palette, title, prefix):
    plt.figure(figsize=(10, 4))
    labels = []
    data = []
    colors = []
    for src in length_data:
        if not length_data[src]:
            continue
        labels.append(src)
        data.append(length_data[src])
        colors.append(palette[src])

    box = plt.boxplot(data, patch_artist=True, showmeans=False, showbox=True, showcaps=True, showfliers=True, medianprops={'visible': False})  

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)

    split_labels = [l.replace(' ', '\n') for l in labels]
    plt.xticks(range(1, len(split_labels) + 1), split_labels, rotation=0, fontsize=17)

    plt.yscale('log')
    plt.ylabel('GT Length', fontsize=24)
    plt.title(title, fontsize=22)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_boxplot.pdf"))
    print("Boxplot saved →", os.path.join(OUTPUT_DIR, f"{prefix}_boxplot.pdf"))

def plot_violin(length_data, palette, title, prefix):
    plt.figure(figsize=(10, 4))
    labels = []
    data = []
    colors = []
    for src in length_data:
        if not length_data[src]:
            continue
        labels.append(src)
        data.append(length_data[src])
        colors.append(palette[src])

    violin = plt.violinplot(data, showmeans=False, showmedians=False)
    
    # Customize violin plot colors
    for i, pc in enumerate(violin['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(1.0)
    
    # Customize median and mean lines
    #violin['cmedians'].set_color('black')
    #violin['cmeans'].set_color('red')

    split_labels = [l.replace(' ', '\n') for l in labels]
    plt.xticks(range(1, len(split_labels) + 1), split_labels, rotation=0, fontsize=17)

    plt.yscale('log')
    plt.ylabel('GT Length', fontsize=24)
    plt.title(title, fontsize=22)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_violin.pdf"))
    print("Violin plot saved →", os.path.join(OUTPUT_DIR, f"{prefix}_violin.pdf"))

if __name__ == "__main__":
    GT_DIR = "data/gt"
    ROOT_DIR = "/Users/doradong/Repo"
    PATHS = {
        "SANTOS Small": os.path.join(ROOT_DIR, "santos/groundtruth/santosUnionBenchmark.pickle"),
        "TUS Small":    os.path.join(ROOT_DIR, "table-union-search-benchmark/tus_small_query_candidate.pkl"),
        "TUS Large":    os.path.join(ROOT_DIR, "table-union-search-benchmark/tus_large_query_candidate.pkl"),
        "SANTOS Large": os.path.join(ROOT_DIR, "santos/groundtruth/real_tablesUnionBenchmark.pickle"),
        #"Paper GT":     os.path.join(GT_DIR, "csv_pair_adj_direct_label_processed.pkl"),
        #"Model GT":     os.path.join(GT_DIR, "scilake_gt_modellink_model_adj_processed.pkl"),
        #"Dataset GT":   os.path.join(GT_DIR, "scilake_gt_modellink_dataset_adj_processed.pkl"),
        "Paper GT":     os.path.join(GT_DIR, "csv_pair_matrix_direct_label.npz"),
        "Model GT":     os.path.join(GT_DIR, "scilake_gt_modellink_model_adj.npz"),
        "Dataset GT":   os.path.join(GT_DIR, "scilake_gt_modellink_dataset_adj.npz"),
        "Union GT":        os.path.join(GT_DIR, "csv_pair_union_direct.npz"),
    }

    lengths = load_lengths(PATHS)

    # Debug prints: each dataset's count, min, max
    for src, vals in lengths.items():
        if vals:
            print(f"{src}: count={len(vals)}, min={min(vals)}, max={max(vals)}")
        else:
            print(f"{src}: no data")

    # plot_histogram(lengths, "GT Length (All Sources)", "gt_all")   ########
    #plot_kde(lengths, "GT Length Distribution (All Sources)", "gt_all")
    plot_log_boxplot(lengths, PALETTE, "Log-scale GT length distribution across benchmarks", "gt_boxplot")
    plot_violin(lengths, PALETTE, "Log-scale GT length distribution across benchmarks", "gt_violin")
