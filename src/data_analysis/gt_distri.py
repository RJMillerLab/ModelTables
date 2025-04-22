# Updated `gt_distri` script with combined GT file support

"""
Author: Zhengyuan Dong
Created: 2025-04-11
Last Modified: 2025-04-21
Description: Distribution analysis of GT lengths in pickle files using a loader class and reusable functions.
Usage:
    python -m src.data_analysis.gt_distri
"""

import os
import gzip                              ########
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

plt.rcParams.update({
    'font.size': 50,     
    'axes.titlesize': 50,
    'axes.labelsize': 50,
    'xtick.labelsize': 40,
    'ytick.labelsize': 40,       
    'legend.fontsize': 40,       
    'figure.titlesize': 50     
})

# Output directory for saving figures
OUTPUT_DIR = "data/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Teal gradient-inspired color palette
TEAL_PALETTE = {
    "scilake": "#4e8094",
    "santos": "#50a89d",
    "tus": "#a5d2bc"
}
NEW_PALETTE = {
    "scilake_direct": "#8b2e2e",
    "scilake_rate": "#d96e44",
}

# ================= Loader Class =================
class GTLengthLoader:
    """Class to load pickle files and extract the length of list-type ground truth entries."""
    def __init__(self, source_name: str, file_path: str, gt_key: str = None):  ########
        self.source_name = source_name
        self.file_path = file_path
        self.gt_key = gt_key                                                  ########

    def _load_data(self):
        # Choose gzip or normal open based on file extension
        if self.file_path.endswith(".gz"):                                    ########
            with gzip.open(self.file_path, 'rb') as f:                        ########
                data = pickle.load(f)
        else:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
        # Extract sub-dictionary if key provided
        if self.gt_key:
            data = data[self.gt_key]                                          ########
        return data

    def load_lengths(self):
        print(f"[INFO] Loading {self.source_name} from {self.file_path} ...")
        data = self._load_data()                                              ########
        print(f"[INFO] {self.source_name} contains {len(data)} entries. Processing lengths...")
        lengths = [len(v) for v in data.values() if isinstance(v, list)]
        return lengths
    
    def get_large_entries(self, threshold=1000):
        data = self._load_data()                                              ########
        return [(k, len(v)) for k, v in data.items() if isinstance(v, list) and len(v) > threshold]

# ================= Helper functions =================
def print_large_keys(pickle_info, threshold=1000):
    print(f"\n=== Entries with GT list length > {threshold} ===")
    for source, info in pickle_info.items():
        # info can be a path or (path, key)
        if isinstance(info, tuple):                                         ########
            path, key = info                                                ########
        else:
            path, key = info, None                                           ########
        loader = GTLengthLoader(source, path, key)                          ########
        large = loader.get_large_entries(threshold)
        print(f"[{source}] Found {len(large)} entries > {threshold}")
        for k, l in large:
            print(f"  - {k} ({l})")

def load_all_lengths(pickle_info):
    length_data = {}
    for source, info in pickle_info.items():
        if isinstance(info, tuple):                                         ########
            path, key = info                                                ########
        else:
            path, key = info, None                                           ########
        loader = GTLengthLoader(source, path, key)                          ########
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

# ================= Original single-file pickles =================
GT_FILE = "data/gt/scilake_gt_all_matrices__overlap_rate.pkl.gz"            ########
PICKLE_PATHS_1 = {
    "scilake": (GT_FILE, "csv_real_gt"),
    "santos": "/Users/doradong/Repo/santos/groundtruth/santosUnionBenchmark.pickle",
    "tus": "/Users/doradong/Repo/santos/groundtruth/tusUnionBenchmark.pickle"
}
print_large_keys(PICKLE_PATHS_1, threshold=1000)
length_data_1 = load_all_lengths(PICKLE_PATHS_1)
plot_histogram(length_data_1, TEAL_PALETTE, "Distribution of GT Lengths (Histogram)", "gt1")
plot_kde(length_data_1, TEAL_PALETTE, "Distribution of GT Lengths (KDE)", "gt1")

# ================ Combined GT file support ================
SCILAKE_PICKLE_PATHS = {                                                     ########
    "scilake_direct": (GT_FILE, "csv_real_gt"),                  ########
    "scilake_rate":  (GT_FILE, "csv_real_gt")                    ########
}

print_large_keys(SCILAKE_PICKLE_PATHS, threshold=1000)                       ########
length_data_2 = load_all_lengths(SCILAKE_PICKLE_PATHS)                       ########
# plots using NEW_PALETTE...
plot_histogram(length_data_2, NEW_PALETTE, "Distribution of GT Lengths (Direct vs Rate)", "gt2")  ########
plot_kde(length_data_2, NEW_PALETTE, "Distribution of GT Lengths (Direct vs Rate)", "gt2")        ########
print("Saved to", OUTPUT_DIR)

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
