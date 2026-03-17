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
from typing import Optional
from src.data_gt.step3_gt import get_npz_path


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
    "Paper Links":           "#486f90",
    "Model Links":           "#4e8094",
    "Dataset Links":         "#50a89d",
    "SANTOS Large":       "#f29e4c",
    "SANTOS Small":       "#8b2e2e",
    "TUS Large":          "#d96e44",
    "TUS Small":          "#b74a3c",
    "UGEN-V1":            "#FFBE5F",
    "UGEN-V2":            "#FFB55A",
    "All Links":           "#a5d2bc",
    # "TUS Others":          "#8b2e2e",
    # "TUS Santos":          "#8b2e2e",
}

def get_baseline_gt_paths(v2_suffix, suffix, REPO_DIR="../"):
    LEVEL_NPZ, _ = get_npz_path(v2_suffix, suffix, f"{REPO_DIR}/ModelTables/data/gt{v2_suffix}{suffix}")
    PATHS = {
        "SANTOS Small": os.path.join(REPO_DIR, "santos/groundtruth/santosUnionBenchmark.pickle"),
        "TUS Small":    os.path.join(REPO_DIR, "table-union-search-benchmark/tus_small_query_candidate.pkl"),
        "TUS Large":    os.path.join(REPO_DIR, "table-union-search-benchmark/tus_large_query_candidate.pkl"),
        "SANTOS Large": os.path.join(REPO_DIR, "santos/groundtruth/real_tablesUnionBenchmark.pickle"),
        "UGEN-V1":      os.path.join(REPO_DIR, "gen/evaluation/groundtruth/ugen_v1UnionBenchmark.pickle"),
        "UGEN-V2":      os.path.join(REPO_DIR, "gen/evaluation/groundtruth/ugen_v2UnionBenchmark.pickle"),
        # "TUS Others":    os.path.join(REPO_DIR, "santos/groundtruth/tusUnionBenchmark.pickle"),
        # "TUS Santos":    os.path.join(REPO_DIR, "table-union-search-benchmark/tus_query_candidate.pkl"),
        "Paper Links":     LEVEL_NPZ["direct"],
        "Model Links":     LEVEL_NPZ["model"],
        "Dataset Links":   LEVEL_NPZ["dataset"],
        "All Links":     LEVEL_NPZ["union"],
    }
    return PATHS

GENERIC_TABLE_PATTERNS = ["1910.09700_table", "204823751_table"]

# -------- Loader --------
class GTLengthLoader:
    def __init__(self, name: str, path: str):
        self.name, self.path = name, path

    def _load_pkl(self, path: str):
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rb") as f:
            data = pickle.load(f)
        return data
    
    def _load_csvlist(self, idx_path: str):
        opener = gzip.open if idx_path.endswith(".gz") else open
        with opener(idx_path, 'rb') as f:
            lst = pickle.load(f)
        return [os.path.basename(str(x)) for x in lst]

    def _length_npz(self, path: str, keep_mask):
        M = load_npz(self.path).tocsr()
        
        map_key = {"Paper Links": "direct", "All Links": "union"}
        idx_names = _load_csvlist(LEVEL_CSVLIST[map_key[self.name]])
        if idx_names and len(idx_names) == M.shape[0]:
            name2idx = {n: i for i, n in enumerate(idx_names)}
            keep_mask = np.ones(len(idx_names), dtype=bool)
            for n, i in name2idx.items():
                if any(p in n for p in GENERIC_TABLE_PATTERNS):
                    keep_mask[i] = False
        if keep_mask is None:
            return [n for n in M.getnnz(axis=1).tolist() if n > 0]

        # Fast masked counting without building a sliced submatrix
        indptr, indices = M.indptr, M.indices
        out = []
        for i in range(M.shape[0]):
            if not keep_mask[i]:
                continue
            start, end = indptr[i], indptr[i+1]
            if end <= start:
                continue
            cnt = int(np.count_nonzero(keep_mask[indices[start:end]]))
            if cnt > 0:
                out.append(cnt)
        return out

    def lengths(self):
        """
        Return per-row link counts (>0) with optional filtering for csv-level GTs
        to exclude generic CSV sets. Filtering rules apply to csv-level matrices only.
        """
        if self.path.endswith(".npz"):
            data = self._length_npz(self.path, keep_mask)
        else:
            data = self._load_pkl(self.path)
        return [l for l in (len(v) for v in data.values() if isinstance(v, list)) if l > 0]

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
    plt.ylabel('# Links', fontsize=22)
    plt.title(title, fontsize=22)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_boxplot.pdf"))
    print("Boxplot saved →", os.path.join(OUTPUT_DIR, f"{prefix}_boxplot.pdf"))

def plot_violin(length_data, palette, title, prefix):
    # Increase figure width to accommodate more labels
    n_labels = sum(1 for src in length_data if length_data[src])
    # Reduce figure width since we're making violins closer together
    fig_width = max(10, n_labels * 0.9)  # Reduced multiplier from 1.2 to 0.9
    plt.figure(figsize=(fig_width, 4))
    
    labels = []
    data = []
    colors = []
    for src in length_data:
        if not length_data[src]:
            continue
        labels.append(src)
        data.append(length_data[src])
        colors.append(palette[src])

    # Use tighter positions to reduce spacing between violins
    # Reduce width to 0.5 (default is 0.8) to make violins narrower and closer together
    positions = list(range(1, len(labels) + 1))
    violin = plt.violinplot(data, positions=positions, widths=0.5, showmeans=False, showmedians=False)
    
    # Customize violin plot colors
    for i, pc in enumerate(violin['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(1.0)
    
    # Customize median and mean lines
    #violin['cmedians'].set_color('black')
    #violin['cmeans'].set_color('red')

    # Use smaller font size and split labels with newlines to prevent overlap
    split_labels = [l.replace(' ', '\n') for l in labels]
    plt.xticks(range(1, len(split_labels) + 1), split_labels, 
               rotation=0, fontsize=12)

    plt.yscale('log')
    plt.ylabel('# Links', fontsize=22)
    plt.title(title, fontsize=22)

    # Reduce margins by adjusting subplot parameters - tighter margins
    plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.12)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_violin.pdf"), 
                bbox_inches='tight', pad_inches=0.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_violin.png"), 
                bbox_inches='tight', pad_inches=0.02)
    print("Violin plot saved →", os.path.join(OUTPUT_DIR, f"{prefix}_violin.pdf"))
    print("Violin plot saved →", os.path.join(OUTPUT_DIR, f"{prefix}_violin.png"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot GT length distribution")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode for GT files.')
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()
    
    ROOT_DIR = "/Users/doradong/Repo"
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    GT_PATHS = get_baseline_gt_paths(v2_suffix, suffix)

    lengths = {}
    for src, path in GT_PATHS.items():
        lengths[src] = GTLengthLoader(src, path).lengths()
        if vals:
            print(f"{src}: count={len(vals)}, min={min(vals)}, max={max(vals)}")
        else:
            print(f"{src}: no data")

    # plot_histogram(lengths, "GT Length (All Sources)", f"gt_all{suffix}")
    #plot_kde(lengths, "GT Length Distribution (All Sources)", f"gt_all{suffix}")
    #plot_log_boxplot(lengths, PALETTE, "Log-scale GT link count distribution across benchmarks", f"gt_boxplot{suffix}")
    plot_violin(lengths, PALETTE, "Log-scale links count distribution across benchmarks", f"gt_violin{v2_suffix}{suffix}")

"""
SANTOS Small: count=50, min=11, max=31
TUS Small: count=1327, min=4, max=235
TUS Large: count=4296, min=4, max=735
SANTOS Large: count=82, min=20, max=20
Paper Links: count=92963, min=1, max=66522
Model Links: count=92874, min=2, max=15327
Dataset Links: count=92856, min=2, max=12052
All Links: count=92964, min=2, max=80345
Violin plot saved → data/analysis/gt_violin_violin.pdf
"""