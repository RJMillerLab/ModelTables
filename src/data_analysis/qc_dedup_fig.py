#!/usr/bin/env python
# run_heatmap.py
"""
Author: Zhengyuan Dong
Created: 2025-04-11
Description: Split saving heatmap from qc_dedup.py to a separate script.
Usage: 
    python -m src.data_analysis.qc_dedup_fig
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.data_analysis.qc_dedup import save_heatmap

OUTPUT_DIR = "data/deduped"
dup_matrix_file = os.path.join(OUTPUT_DIR, "dup_matrix.pkl")
stats_file = os.path.join(OUTPUT_DIR, "stats.json")

if __name__ == "__main__":
    if not os.path.exists(dup_matrix_file):
        raise FileNotFoundError(f"Dup matrix file not found: {dup_matrix_file}")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    dup_matrix = pd.read_pickle(dup_matrix_file)
    with open(stats_file, "r") as f:
        stats = json.load(f)

    save_heatmap(dup_matrix, stats["cross_unique_counts"], OUTPUT_DIR)
