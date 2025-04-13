"""""
Author: Zhengyuan Dong
Created: 2025-04-11
Last Modified: 2025-04-11
Description: Distribution analysis of GT lengths in pickle files using a loader class and reusable functions.
Usage:
    python -m src.data_analysis.gt_distri
"""""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

# Output directory for saving figures
OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Teal gradient-inspired color palette
TEAL_PALETTE = {
    "scilake": "#a5d2bc",
    "santos": "#50a89d",
    "tus": "#4e8094"
}
NEW_PALETTE = {
    "scilake_direct": "#8b2e2e",
    "scilake_rate": "#d96e44",
}
# ================= Loader Class =================
class GTLengthLoader:
    """Class to load pickle files and extract the length of list-type ground truth entries."""
    def __init__(self, source_name: str, file_path: str):
        self.source_name = source_name
        self.file_path = file_path

    def load_lengths(self):
        print(f"[INFO] Loading {self.source_name} from {self.file_path} ...")
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"[INFO] {self.source_name} contains {len(data)} entries. Processing lengths...")

        lengths = []
        for k in tqdm(data, desc=f"[{self.source_name}] Processing"):
            v = data[k]
            if isinstance(v, list):
                lengths.append(len(v))
        return lengths
    
    def get_large_entries(self, threshold=1000):
        """Return all keys whose value list length exceeds threshold."""
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)

        large_keys = []
        for k, v in data.items():
            if isinstance(v, list) and len(v) > threshold:
                large_keys.append((k, len(v)))
        return large_keys

def print_large_keys(pickle_paths, threshold=1000):
    print(f"\n=== Entries with GT list length > {threshold} ===")
    for source, path in pickle_paths.items():
        loader = GTLengthLoader(source_name=source, file_path=path)
        large_entries = loader.get_large_entries(threshold=threshold)
        print(f"[{source}] Found {len(large_entries)} entries > {threshold}")
        for k, l in large_entries:
            print(f"  - {k} ({l})")

def load_all_lengths(pickle_paths):
    """Load lengths from multiple sources."""
    length_data = {}
    for source, path in pickle_paths.items():
        loader = GTLengthLoader(source_name=source, file_path=path)
        length_data[source] = loader.load_lengths()
    return length_data

def plot_histogram(length_data, palette, title, output_prefix):
    """Plot histogram with transparent bars and log scale."""
    plt.figure(figsize=(10, 6))
    # 修改：处理空列表，若数据为空则默认最大值为 0 ########
    max_val = 0  ########
    for v in length_data.values():  ########
        if len(v) > 0:  ########
            max_val = max(max_val, max(v))  ########
    bins = np.arange(0, max_val + 2)  ########
    for source, lengths in length_data.items():
        if len(lengths) > 0:  ########
            plt.hist(lengths, bins=bins, alpha=0.4, label=source,
                     color=palette[source], edgecolor=None)
    plt.title(title)
    plt.xlabel("List Length")
    plt.ylabel("Frequency")
    plt.xscale("log")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_histogram.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_histogram.pdf"))
    # plt.show()

def plot_kde(length_data, palette, title, output_prefix):
    """Plot KDE smoothed distribution with fill and log scale."""
    plt.figure(figsize=(10, 6))

    def plot_kde_manual(data, label, color):
        if len(data) == 0:  ########
            return  ########
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data) - 1, max(data) + 1, 200)
        y_vals = kde(x_vals)
        plt.fill_between(x_vals, y_vals, alpha=0.4, label=label, color=color)
        plt.plot(x_vals, y_vals, linewidth=2, color=color)

    for source, lengths in length_data.items():
        plot_kde_manual(lengths, source, palette[source])

    plt.title(title)
    plt.xlabel("List Length")
    plt.ylabel("Density")
    plt.xscale("log")
    plt.legend(title="Source")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_kde.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}_kde.pdf"))
    # plt.show()

PICKLE_PATHS_1 = {  ########
    "scilake": "/Users/doradong/Repo/CitationLake/data/gt/scilake_large_gt__direct_label.pickle",  ########
    "santos": "/Users/doradong/Repo/santos/groundtruth/santosUnionBenchmark.pickle",  ########
    "tus": "/Users/doradong/Repo/santos/groundtruth/tusUnionBenchmark.pickle"  ########
}
print_large_keys(PICKLE_PATHS_1, threshold=1000)
length_data_1 = load_all_lengths(PICKLE_PATHS_1)
plot_histogram(length_data_1, TEAL_PALETTE, "Distribution of GT Lengths (Histogram)", "gt1")
plot_kde(length_data_1, TEAL_PALETTE, "Distribution of GT Lengths (KDE)", "gt1")

PICKLE_PATHS_2 = {  ########
    "scilake_direct": "/Users/doradong/Repo/CitationLake/data/gt/scilake_large_gt__direct_label.pickle",  ########
    "scilake_rate": "/Users/doradong/Repo/CitationLake/data/gt/scilake_large_gt__overlap_rate.pickle",  ########
}
length_data_2 = load_all_lengths(PICKLE_PATHS_2)
plot_histogram(length_data_2, NEW_PALETTE, "Distribution of GT Lengths (Histogram)", "gt2")
plot_kde(length_data_2, NEW_PALETTE, "Distribution of GT Lengths (KDE)", "gt2")
print('saved to ', OUTPUT_DIR)

PICKLE_PATHS_3 = {  ########
    "santos_union": "/Users/doradong/Repo/santos/groundtruth/santosUnionBenchmark.pickle",  ########
    "santos_joinable": "/Users/doradong/Repo/santos/groundtruth/santosIntentColumnBenchmark.pickle",  ########
    "tus_union": "/Users/doradong/Repo/santos/groundtruth/tusUnionBenchmark.pickle",  ########
    "tus_joinable": "/Users/doradong/Repo/santos/groundtruth/tusIntentColumnBenchmark.pickle",  ########
    "realtables_union": "/Users/doradong/Repo/santos/groundtruth/real_tablesUnionBenchmark.pickle",  ########
    "realtables_joinable": "/Users/doradong/Repo/santos/groundtruth/real_tablesIntentColumnBenchmark.pickle",  ########
}
SIX_PALETTE = {  ########
    "santos_union": "#e41a1c",       ########
    "santos_joinable": "#377eb8",    ########
    "tus_union": "#4daf4a",          ########
    "tus_joinable": "#984ea3",       ########
    "realtables_union": "#ff7f00",   ########
    "realtables_joinable": "#a65628" ########
}
length_data_3 = load_all_lengths(PICKLE_PATHS_3)  ########
plot_histogram(length_data_3, SIX_PALETTE, "GT Length Distribution: Union vs Joinable", "gt3")  ########
plot_kde(length_data_3, SIX_PALETTE, "GT Length Distribution: Union vs Joinable", "gt3")  ########
print('Saved union vs joinable comparison to', OUTPUT_DIR)
