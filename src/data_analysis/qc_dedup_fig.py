#!/usr/bin/env python
# run_heatmap.py
"""
Author: Zhengyuan Dong
Created: 2025-04-11
Description: Split saving heatmap from step2_dedup_tables.py to a separate script.
Usage: 
    python -m src.data_analysis.qc_dedup_fig --tag 251117
"""

import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.data_preprocess.step2_dedup_tables import save_heatmap
from src.utils import load_config

OUTPUT_DIR = "data/deduped"
FIG_DIR = "data/analysis"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps from step2_dedup_tables results")
    parser.add_argument('--tag', dest='tag', default=None, help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--v2_mode', dest='v2_mode', action='store_true', help='Use v2 mode.')
    args = parser.parse_args()
    
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    suffix = f"_{args.tag}" if args.tag else ""
    v2_suffix = "_v2" if args.v2_mode else ""

    output_dir = os.path.join(base_path, f"deduped{v2_suffix}{suffix}")
    fig_dir = os.path.join(base_path, 'analysis')
    table_parquet_path = os.path.join(base_path, 'processed', f"modelcard_step3_merged{v2_suffix}{suffix}.parquet")
    
    dup_matrix_file = os.path.join(output_dir, f"dup_matrix{suffix}.pkl")
    stats_file = os.path.join(output_dir, f"stats{suffix}.json")
    
    print("📁 Paths in use:")
    print(f"   Input directory:     {output_dir}")
    print(f"   Figure directory:    {fig_dir}")
    print(f"   Dup matrix file:     {dup_matrix_file}")
    print(f"   Stats file:          {stats_file}")
    
    if not os.path.exists(dup_matrix_file):
        raise FileNotFoundError(f"Dup matrix file not found: {dup_matrix_file}")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    dup_matrix = pd.read_pickle(dup_matrix_file)
    with open(stats_file, "r") as f:
        stats = json.load(f)
    
    save_heatmap(dup_matrix, stats["cross_unique_counts"], table_parquet_path, fig_dir, file_suffix=suffix, v2_suffix=v2_suffix)
    save_heatmap(dup_matrix, stats["cross_unique_counts"], table_parquet_path, fig_dir, is_percentage=True, file_suffix=suffix, v2_suffix=v2_suffix)
    print(f"✅ Heatmaps saved to {fig_dir}")
