#!/usr/bin/env python
"""
Author: Zhengyuan Dong
Date: 2025-09-28
Last Edited: 2025-10-15
Description: Compare two folders of CSVs: histogram of per-table columns and rows (by basename).

Usage example:
python -m src.data_analysis.qc_anomaly --recursive --anomaly-min-cols 500 --anomaly-min-rows 1000 --anomaly-ratio 2.0
"""
import argparse
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from threading import Lock
try:
    from batch_process_tables import build_modelid_sql_query
except Exception:
    build_modelid_sql_query = None
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.data_analysis.qc_stats import count_rows_fast, count_columns_from_header_fast


def count_columns_from_header_ultra_simple(csv_path, max_scan_bytes=8 * 1024 * 1024):
    # Deprecated path kept for compatibility; not used in simplified mode.
    return count_columns_from_header_fast(csv_path, max_scan_bytes=max_scan_bytes)


def list_csv_files(paths, recursive=False):
    files = []
    for path in paths:
        if os.path.isdir(path):
            if recursive:
                for root, _, fnames in os.walk(path):
                    for fname in fnames:
                        if fname.lower().endswith('.csv'):
                            files.append(os.path.join(root, fname))
            else:
                for fname in os.listdir(path):
                    if fname.lower().endswith('.csv'):
                        files.append(os.path.join(path, fname))
    return files


CACHE_DB_PATH = os.path.join("data", "cache", "qc_counts.sqlite")


def _ensure_cache_db():
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    con = sqlite3.connect(CACHE_DB_PATH)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS counts(
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                rows_with_header INTEGER NOT NULL,
                cols INTEGER NOT NULL
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_counts_mtime ON counts(mtime)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_counts_size ON counts(size)")
        con.commit()
    finally:
        con.close()


def _load_cache_map():
    """Load entire cache table into memory as a dict[path] -> (mtime, size, rows, cols)."""
    _ensure_cache_db()
    con = sqlite3.connect(CACHE_DB_PATH)
    try:
        cur = con.execute("SELECT path, mtime, size, rows_with_header, cols FROM counts")
        return {row[0]: (row[1], row[2], row[3], row[4]) for row in cur.fetchall()}
    finally:
        con.close()


def _write_cache_updates(updates_items):
    if not updates_items:
        return
    _ensure_cache_db()
    con = sqlite3.connect(CACHE_DB_PATH)
    try:
        con.executemany(
            "INSERT OR REPLACE INTO counts(path, mtime, size, rows_with_header, cols) VALUES(?,?,?,?,?)",
            updates_items,
        )
        con.commit()
    finally:
        con.close()


def index_csv_basenames(directory, recursive=False):
    basenames = set()
    if not os.path.isdir(directory):
        return basenames
    if recursive:
        for root, _, fnames in os.walk(directory):
            for fname in fnames:
                if fname.lower().endswith('.csv'):
                    basenames.add(fname)
    else:
        for fname in os.listdir(directory):
            if fname.lower().endswith('.csv'):
                basenames.add(fname)
    return basenames


def count_columns_from_header(csv_path):
    """Count columns by reading only the first row with CSV parsing (handles quoted commas)."""
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                return 0
            return len(first_row)
    except Exception:
        return 0


def count_rows_including_header(csv_path):
    """Count rows efficiently, including the header row."""
    try:
        count = 0
        with open(csv_path, 'rb') as f:
            for _ in f:
                count += 1
        return count
    except Exception:
        return 0

def resolve_resource_dirs(resource):
    if resource == "hugging":
        return "data/processed/deduped_hugging_csvs", "data/processed/deduped_hugging_csvs_v2"
    if resource == "github":
        return "data/processed/deduped_github_csvs", "data/processed/deduped_github_csvs_v2"
    return "data/processed/tables_output", "data/processed/tables_output_v2"

def prepare_file_maps(v1_dir, v2_dir, recursive):
    v1_files = list_csv_files([v1_dir], recursive=recursive)
    if not v1_files:
        print("No CSV files found in v1_dir.")
        return [], {}, {}, []
    
    # Build v2_file_map first
    v2_file_map = {}
    if recursive:
        for root, _, fnames in os.walk(v2_dir):
            for fname in fnames:
                if fname.lower().endswith('.csv'):
                    v2_file_map[fname] = os.path.join(root, fname)
    else:
        for fname in os.listdir(v2_dir):
            if fname.lower().endswith('.csv'):
                v2_file_map[fname] = os.path.join(v2_dir, fname)
    
    v1_file_map = {os.path.basename(p): p for p in v1_files}
    
    # Always include all files (union of v1 and v2)
    basenames = sorted(set(v1_file_map.keys()) | set(v2_file_map.keys()))
    
    return v1_files, v2_file_map, v1_file_map, basenames

def build_resource_mappings(resource):
    csv_to_modelid = {}
    csv_to_sourcepath = {}
    if resource == "hugging":
        csv_to_modelid = build_hugging_modelid_map()
    elif resource == "github":
        csv_to_sourcepath = build_github_source_map()
    else:
        csv_to_sourcepath = build_arxiv_source_map()
    return csv_to_modelid, csv_to_sourcepath

def build_hugging_modelid_map():
    mapping = {}
    try:
        import duckdb
        from batch_process_tables import build_modelid_sql_query
        sql = build_modelid_sql_query() if build_modelid_sql_query else None
        if sql:
            con = duckdb.connect()
            rows = con.execute(sql).fetchall()
            con.close()
            for csv_name, model_ids in rows:
                if not csv_name:
                    continue
                base = os.path.basename(str(csv_name))
                first_mid = str(model_ids).split(';')[0].strip() if model_ids else ''
                if base and first_mid and base not in mapping:
                    mapping[base] = first_mid
        else:
            rel_path = "data/processed/modelcard_step3_merged.parquet"
            if os.path.exists(rel_path):
                import pandas as pd
                df_rel = pd.read_parquet(rel_path)
                cols = df_rel.columns.tolist()
                for col in ["hugging_table_list_dedup", "hugging_table_list", "csv_paths"]:
                    if col not in cols:
                        continue
                    for _, row in df_rel.iterrows():
                        mid = row.get("modelId")
                        vals = row.get(col)
                        if pd.isna(mid) or vals is None:
                            continue
                        if not isinstance(vals, (list, tuple)):
                            vals = [vals]
                        for v in vals:
                            try:
                                base = os.path.basename(str(v))
                            except Exception:
                                continue
                            if base and base not in mapping:
                                mapping[base] = mid
                    if mapping:
                        break
        
        # Also load v2 mapping (for v2_only files)
        import pandas as pd
        import json
        v2_mapping_json = "data/processed/hugging_deduped_mapping_v2.json"
        step2_parquet = "data/processed/modelcard_step2.parquet"
        
        if os.path.exists(v2_mapping_json) and os.path.exists(step2_parquet):
            # Load hash -> csv_paths mapping
            with open(v2_mapping_json, 'r') as f:
                hash_to_csvs = json.load(f)
            
            # Load modelId -> readme_hash mapping
            df_step2 = pd.read_parquet(step2_parquet, columns=['modelId', 'readme_hash'])
            hash_to_modelid = dict(zip(df_step2['readme_hash'], df_step2['modelId']))
            
            # Build csv -> modelId mapping
            for readme_hash, csv_list in hash_to_csvs.items():
                model_id = hash_to_modelid.get(readme_hash)
                if model_id and csv_list:
                    for csv_path in csv_list:
                        base = os.path.basename(str(csv_path))
                        if base and base not in mapping:  # Don't overwrite v1 mapping
                            mapping[base] = model_id
            
    except Exception:
        pass
    return mapping

def build_github_source_map():
    mapping = {}
    try:
        import pandas as pd
        import json
        map_paths = [
            "data/processed/csv_to_readme_mapping.parquet",
            "data/processed/processed_paths.parquet",
            "data/processed/raw_csv_to_text_mapping.parquet",
        ]
        for mp in map_paths:
            if not os.path.exists(mp):
                continue
            df_map = pd.read_parquet(mp)
            cols = df_map.columns.tolist()
            if "csv_paths" in cols and "readme_path" in cols:
                for _, row in df_map.iterrows():
                    rp = row.get("readme_path")
                    if isinstance(rp, (list, tuple)):
                        rp = next((x for x in rp if isinstance(x, str) and x), None)
                    cps = row.get("csv_paths")
                    if cps is None:
                        continue
                    if not isinstance(cps, (list, tuple)):
                        cps = [cps]
                    for pth in cps:
                        try:
                            base = os.path.basename(str(pth))
                        except Exception:
                            continue
                        if base and base not in mapping and isinstance(rp, str):
                            mapping[base] = rp
            elif "csv_path" in cols and "readme_path" in cols:
                for _, row in df_map.iterrows():
                    cp = row.get("csv_path")
                    rp = row.get("readme_path")
                    if isinstance(rp, (list, tuple)):
                        rp = next((x for x in rp if isinstance(x, str) and x), None)
                    if not isinstance(cp, str) or not isinstance(rp, str):
                        continue
                    base = os.path.basename(cp)
                    if base and base not in mapping:
                        mapping[base] = rp
            if len(mapping) > 0:
                break
        
        # Also load v2 mapping (md_to_csv_mapping.json in deduped_github_csvs_v2)
        v2_mapping_json = "data/processed/deduped_github_csvs_v2/md_to_csv_mapping.json"
        if os.path.exists(v2_mapping_json):
            with open(v2_mapping_json, 'r') as f:
                md_to_csv = json.load(f)
            # md_to_csv maps: md_basename -> [list of csv_basenames]
            # We reverse it to csv_basename -> md_basename
            for md_file, csv_list in md_to_csv.items():
                if not csv_list or csv_list is None:
                    continue
                readme_path = f"data/downloaded_github_readmes/{md_file}.md"
                if isinstance(csv_list, list):
                    for csv_basename in csv_list:
                        if csv_basename and csv_basename not in mapping:
                            mapping[csv_basename] = readme_path
        
    except Exception:
        pass
    return mapping

def build_arxiv_source_map():
    mapping = {}
    try:
        import pandas as pd
        html_maps = [
            "data/processed/html_table.parquet",
            "data/processed/final_integration_with_paths.parquet",
        ]
        for hp in html_maps:
            if not os.path.exists(hp):
                continue
            df_html = pd.read_parquet(hp)
            cols = df_html.columns.tolist()
            if "table_list" in cols and "html_path" in cols:
                for _, row in df_html.iterrows():
                    hpv = row.get("html_path")
                    tl = row.get("table_list")
                    if tl is None:
                        continue
                    if not isinstance(tl, (list, tuple)):
                        tl = [tl]
                    for pth in tl:
                        try:
                            base = os.path.basename(str(pth))
                        except Exception:
                            continue
                        if base and base not in mapping and isinstance(hpv, str):
                            mapping[base] = hpv
            elif "html_table_list" in cols and "html_html_path" in cols:
                for _, row in df_html.iterrows():
                    hpv = row.get("html_html_path")
                    tl = row.get("html_table_list")
                    if tl is None:
                        continue
                    if not isinstance(tl, (list, tuple)):
                        tl = [tl]
                    for pth in tl:
                        try:
                            base = os.path.basename(str(pth))
                        except Exception:
                            continue
                        if base and base not in mapping and isinstance(hpv, str):
                            mapping[base] = hpv
            if len(mapping) > 0:
                break
        
        # Also load v2 mapping (html_parsing_results_v2.parquet)
        v2_parquet = "data/processed/html_parsing_results_v2.parquet"
        if os.path.exists(v2_parquet):
            df_v2 = pd.read_parquet(v2_parquet)
            if "csv_paths" in df_v2.columns and "html_path" in df_v2.columns:
                for _, row in df_v2.iterrows():
                    html_path = row.get("html_path")
                    csv_paths = row.get("csv_paths")
                    if csv_paths is None or not isinstance(html_path, str):
                        continue
                    if not isinstance(csv_paths, (list, tuple)):
                        csv_paths = [csv_paths]
                    for csv_path in csv_paths:
                        try:
                            base = os.path.basename(str(csv_path))
                        except Exception:
                            continue
                        if base and base not in mapping:  # Don't overwrite v1 mapping
                            mapping[base] = html_path
        
    except Exception:
        pass
    return mapping

def print_overlap_summary(v1_file_map, v2_file_map):
    v1_bases = set(v1_file_map.keys())
    v2_bases = set(v2_file_map.keys())
    inter_bases = v1_bases & v2_bases
    union_bases = v1_bases | v2_bases
    only_v1 = v1_bases - v2_bases
    only_v2 = v2_bases - v1_bases
    print("=== qc_anomaly: Dataset summary ===")
    print(f"V1 CSV basenames: {len(v1_bases)}")
    print(f"V2 CSV basenames: {len(v2_bases)}")
    print(f"Overlap (intersection): {len(inter_bases)}")
    print(f"Only in V1: {len(only_v1)}")
    print(f"Only in V2: {len(only_v2)}")
    print(f"Union: {len(union_bases)}")
    if only_v1:
        examples = list(sorted(only_v1))[:5]
        print("Only V1 examples:", ", ".join(examples))
    if only_v2:
        examples = list(sorted(only_v2))[:5]
        print("Only V2 examples:", ", ".join(examples))

def process_basenames(basenames, v1_file_map, v2_file_map, workers):
    v1_orig, v1_trans, v2_orig, v2_trans = [], [], [], []
    results_rows = []
    csv_rows = []
    col_counter = count_columns_from_header_fast

    # Load cache into memory once; collect updates from threads safely
    cache_map = _load_cache_map()
    updates = []  # list of tuples (path, mtime, size, rows_with_header, cols)
    updates_lock = Lock()

    def _get_stats(path):
        if not path:
            return None, None
        ap = os.path.abspath(path)
        try:
            st = os.stat(ap)
        except Exception:
            return None, None
        mtime = st.st_mtime
        size = st.st_size
        cached = cache_map.get(ap)
        if cached and cached[0] == mtime and cached[1] == size:
            # rows_with_header, cols from cache
            return cached[2], cached[3]
        # compute fresh
        cols = col_counter(ap)
        rows_with_header = count_rows_fast(ap, head_flag=True)
        with updates_lock:
            updates.append((ap, mtime, size, rows_with_header, cols))
            cache_map[ap] = (mtime, size, rows_with_header, cols)
        return rows_with_header, cols

    def _proc_base(base):
        p1 = v1_file_map.get(base)
        p2 = v2_file_map.get(base)
        v1_t, v1_o = (None, None)
        v2_t, v2_o = (None, None)
        if p1:
            v1_t, v1_o = _get_stats(p1)
        if p2:
            v2_t, v2_o = _get_stats(p2)
        return base, v1_o, v1_t, v2_o, v2_t
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_proc_base, b): b for b in basenames}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing CSV basenames"):
            try:
                base, v1_o, v1_t, v2_o, v2_t = fut.result()
                if v1_o is not None and v1_t is not None and v2_o is not None and v2_t is not None:
                    v1_orig.append(v1_o)
                    v1_trans.append(v1_t)
                    v2_orig.append(v2_o)
                    v2_trans.append(v2_t)
                    results_rows.append((base, v1_o, v1_t, v2_o, v2_t))
                # Always include all rows (both, v1_only, v2_only)
                csv_rows.append((base, v1_o, v1_t, v2_o, v2_t))
            except Exception:
                pass
    # persist cache updates in one batch
    _write_cache_updates(updates)
    return results_rows, csv_rows, v1_orig, v1_trans, v2_orig, v2_trans

def plot_density_iqr(v1_orig, v2_orig, v1_trans, v2_trans, bins, png_out):
    import numpy as np
    cols_combined = v1_orig + v2_orig
    rows_combined = v1_trans + v2_trans
    if not cols_combined and not rows_combined:
        print("No data to plot.")
        return
    def make_log_bins(values, num_bins):
        vals = [v for v in values if v is not None and v > 0]
        if not vals:
            return num_bins
        vmin = max(1, min(vals))
        vmax = max(vals)
        if vmax <= vmin:
            vmax = vmin + 1
        return np.logspace(np.log10(vmin), np.log10(vmax), num_bins)
    bins_cols = make_log_bins(cols_combined, bins)
    bins_rows = make_log_bins(rows_combined, bins)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    axc, axr = axes
    axc.hist(v1_orig, bins=bins_cols, density=True, histtype='step', linewidth=1.8, label='V1 columns', color='#1f77b4')
    axc.hist(v2_orig, bins=bins_cols, density=True, histtype='step', linewidth=1.8, label='V2 columns', color='#2ca02c')
    axc.set_xscale('log')
    axc.set_ylabel("Density")
    axc.set_xlabel("Columns (count)")
    axc.set_title("Columns: density (log-x)")
    axc.legend()
    axr.hist(v1_trans, bins=bins_rows, density=True, histtype='step', linewidth=1.8, label='V1 rows', color='#ff7f0e')
    axr.hist(v2_trans, bins=bins_rows, density=True, histtype='step', linewidth=1.8, label='V2 rows', color='#9467bd')
    axr.set_xscale('log')
    axr.set_xlabel("Rows (count)")
    axr.set_title("Rows: density (log-x)")
    axr.legend()
    def iqr_band(ax, data, color, label):
        arr = np.array([v for v in data if v is not None and v > 0])
        if arr.size == 0:
            return
        q1, q3 = np.percentile(arr, [25, 75])
        ax.axvspan(q1, q3, color=color, alpha=0.10, lw=0, label=f"{label} IQR")
    iqr_band(axc, v1_orig, '#1f77b4', 'V1')
    iqr_band(axc, v2_orig, '#2ca02c', 'V2')
    iqr_band(axr, v1_trans, '#ff7f0e', 'V1')
    iqr_band(axr, v2_trans, '#9467bd', 'V2')
    fig.suptitle("V1 vs V2: density with IQR bands (narrower bands => narrower distribution)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(png_out), exist_ok=True)
    fig.savefig(png_out.replace('.png', '_density_iqr.png'), dpi=220)
    plt.close(fig)

def plot_ecdf(v1_orig, v2_orig, v1_trans, v2_trans, out_path="data/analysis/combined_ecdf.png"):
    import numpy as np
    def ecdf(arr):
        a = np.array([v for v in arr if v is not None and v > 0])
        if a.size == 0:
            return np.array([]), np.array([])
        a = np.sort(a)
        y = np.linspace(0, 1, a.size, endpoint=True)
        return a, y
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x1, y1 = ecdf(v1_orig); x2, y2 = ecdf(v2_orig)
    if x1.size:
        ax1.plot(x1, y1, color='#1f77b4', lw=1.8, label='V1')
    if x2.size:
        ax1.plot(x2, y2, color='#2ca02c', lw=1.8, label='V2')
    ax1.set_xscale('log')
    ax1.set_xlabel('Columns (count)')
    ax1.set_ylabel('ECDF')
    ax1.set_title('Columns: ECDF (log-x)')
    ax1.grid(alpha=0.3, ls='--')
    ax1.legend()
    x1, y1 = ecdf(v1_trans); x2, y2 = ecdf(v2_trans)
    if x1.size:
        ax2.plot(x1, y1, color='#ff7f0e', lw=1.8, label='V1')
    if x2.size:
        ax2.plot(x2, y2, color='#9467bd', lw=1.8, label='V2')
    ax2.set_xscale('log')
    ax2.set_xlabel('Rows (count)')
    ax2.set_ylabel('ECDF')
    ax2.set_title('Rows: ECDF (log-x)')
    ax2.grid(alpha=0.3, ls='--')
    ax2.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_change_maps(csv_rows, csv_to_modelid, out_prefix="data/analysis/change_maps"):
    import numpy as np
    # Build arrays (overlapping only)
    cols_v1, cols_v2, rows_v1, rows_v2, bases = [], [], [], [], []
    for base, v1_o, v1_t, v2_o, v2_t in csv_rows:
        if v1_o is not None and v2_o is not None and v1_t is not None and v2_t is not None:
            cols_v1.append(v1_o)
            cols_v2.append(v2_o)
            rows_v1.append(v1_t)
            rows_v2.append(v2_t)
            bases.append(base)
    if not bases:
        return
    cols_v1 = np.array(cols_v1); cols_v2 = np.array(cols_v2)
    rows_v1 = np.array(rows_v1); rows_v2 = np.array(rows_v2)
    # Deltas
    d_cols = cols_v2 - cols_v1
    d_rows = rows_v2 - rows_v1
    # Identify anomaly-like large shrink cases (heuristic): big negative diffs
    k = min(50, len(bases))
    idx_cols = np.argsort(d_cols)[:k]
    idx_rows = np.argsort(d_rows)[:k]

    # 2x2: scatter-hexbin cols/rows + diff hist cols/rows
    import matplotlib.pyplot as plt
    import os
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Cols hexbin with diagonal
    hb1 = ax1.hexbin(cols_v1 + 1e-9, cols_v2 + 1e-9, gridsize=60, bins='log', cmap='Blues')
    lim_c = [max(1, min(cols_v1.min(), cols_v2.min())), max(cols_v1.max(), cols_v2.max())]
    ax1.plot(lim_c, lim_c, ls='--', c='#888', lw=1)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('V1 columns'); ax1.set_ylabel('V2 columns'); ax1.set_title('Columns: V1 vs V2 (hexbin)')
    # Overlay top-k shrink points in red
    ax1.scatter(cols_v1[idx_cols], cols_v2[idx_cols], s=10, c='crimson', alpha=0.9, label='top shrink')
    ax1.legend()
    fig.colorbar(hb1, ax=ax1, label='log10(count)')

    # Rows hexbin with diagonal
    hb2 = ax2.hexbin(rows_v1 + 1e-9, rows_v2 + 1e-9, gridsize=60, bins='log', cmap='Greens')
    lim_r = [max(1, min(rows_v1.min(), rows_v2.min())), max(rows_v1.max(), rows_v2.max())]
    ax2.plot(lim_r, lim_r, ls='--', c='#888', lw=1)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('V1 rows'); ax2.set_ylabel('V2 rows'); ax2.set_title('Rows: V1 vs V2 (hexbin)')
    ax2.scatter(rows_v1[idx_rows], rows_v2[idx_rows], s=10, c='crimson', alpha=0.9, label='top shrink')
    ax2.legend()
    fig.colorbar(hb2, ax=ax2, label='log10(count)')

    # Diff histograms with vertical zero line
    ax3.hist(d_cols, bins=100, color='#1f77b4', alpha=0.8)
    ax3.axvline(0, ls='--', c='#444')
    ax3.set_title('Columns difference (V2 - V1)')
    ax3.set_xlabel('Δ cols'); ax3.set_ylabel('Frequency')

    ax4.hist(d_rows, bins=100, color='#ff7f0e', alpha=0.8)
    ax4.axvline(0, ls='--', c='#444')
    ax4.set_title('Rows difference (V2 - V1)')
    ax4.set_xlabel('Δ rows'); ax4.set_ylabel('Frequency')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    fig.savefig(out_prefix + '.png', dpi=220)
    plt.close(fig)

def plot_density_iqr_multi(density_data, bins, out_path):
    import numpy as np
    import matplotlib.pyplot as plt
    resources = list(density_data.keys())
    n = len(resources)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), sharey=False)
    for idx, res in enumerate(resources):
        v1_orig, v2_orig, v1_trans, v2_trans = density_data[res]
        def make_log_bins(values, num_bins):
            vals = [v for v in values if v is not None and v > 0]
            if not vals:
                return num_bins
            vmin = max(1, min(vals))
            vmax = max(vals)
            if vmax <= vmin:
                vmax = vmin + 1
            return np.logspace(np.log10(vmin), np.log10(vmax), num_bins)
        bins_cols = make_log_bins(v1_orig + v2_orig, bins)
        bins_rows = make_log_bins(v1_trans + v2_trans, bins)
        axc = axes[0, idx]
        axr = axes[1, idx]
        axc.hist(v1_orig, bins=bins_cols, density=True, histtype='step', linewidth=1.6, label='V1', color='#1f77b4')
        axc.hist(v2_orig, bins=bins_cols, density=True, histtype='step', linewidth=1.6, label='V2', color='#2ca02c')
        axc.set_xscale('log'); axc.set_title(f"{res} cols")
        axc.legend(fontsize=8)
        axr.hist(v1_trans, bins=bins_rows, density=True, histtype='step', linewidth=1.6, label='V1', color='#ff7f0e')
        axr.hist(v2_trans, bins=bins_rows, density=True, histtype='step', linewidth=1.6, label='V2', color='#9467bd')
        axr.set_xscale('log'); axr.set_title(f"{res} rows")
        axr.legend(fontsize=8)
        def iqr_band(ax, data, color):
            arr = np.array([v for v in data if v is not None and v > 0])
            if arr.size == 0:
                return
            q1, q3 = np.percentile(arr, [25, 75])
            ax.axvspan(q1, q3, color=color, alpha=0.10, lw=0)
        iqr_band(axc, v1_orig, '#1f77b4'); iqr_band(axc, v2_orig, '#2ca02c')
        iqr_band(axr, v1_trans, '#ff7f0e'); iqr_band(axr, v2_trans, '#9467bd')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_ecdf_multi(ecdf_data, out_path="data/analysis/combined_ecdf_multi.png"):
    import numpy as np
    import matplotlib.pyplot as plt
    resources = list(ecdf_data.keys())
    n = len(resources)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    def ecdf(arr):
        a = np.array([v for v in arr if v is not None and v > 0])
        if a.size == 0:
            return np.array([]), np.array([])
        a = np.sort(a)
        y = np.linspace(0, 1, a.size, endpoint=True)
        return a, y
    for idx, res in enumerate(resources):
        v1_orig, v2_orig, v1_trans, v2_trans = ecdf_data[res]
        ax = axes[idx]
        x1, y1 = ecdf(v1_orig); x2, y2 = ecdf(v2_orig)
        if x1.size: ax.plot(x1, y1, color='#1f77b4', lw=1.6, label='V1')
        if x2.size: ax.plot(x2, y2, color='#2ca02c', lw=1.6, label='V2')
        ax.set_xscale('log'); ax.set_title(f"{res} cols ECDF"); ax.grid(alpha=0.3, ls='--'); ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_scatter_grid_multi(scatter_data, out_path="data/analysis/scatter_grid_multi.png", top_k=200):
    """Make a 2xN grid: top row V1 (rows vs cols), bottom row V2; columns are resources.
    Axes share global limits to keep scales identical across subplots.
    Highlights top shrink points (largest (v1/v2) ratio) in red on both rows.
    scatter_data[res] = (v1_rows, v1_cols, v2_rows, v2_cols)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    resources = list(scatter_data.keys())
    n = len(resources)
    if n == 0:
        return
    # Build global limits (log)
    all_rows = []
    all_cols = []
    for res in resources:
        v1_r, v1_c, v2_r, v2_c = scatter_data[res]
        all_rows.extend([x for x in (list(v1_r)+list(v2_r)) if x is not None and x > 0])
        all_cols.extend([y for y in (list(v1_c)+list(v2_c)) if y is not None and y > 0])
    if not all_rows or not all_cols:
        return
    rmin, rmax = max(1, min(all_rows)), max(all_rows)
    cmin, cmax = max(1, min(all_cols)), max(all_cols)

    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), sharex=False, sharey=False)
    for j, res in enumerate(resources):
        v1_r, v1_c, v2_r, v2_c = scatter_data[res]
        # Compute top shrink based on cols and rows ratios separately, then union
        v1_r_a = np.array([x for x in v1_r])
        v1_c_a = np.array([y for y in v1_c])
        v2_r_a = np.array([x for x in v2_r])
        v2_c_a = np.array([y for y in v2_c])
        valid = (v1_r_a > 0) & (v2_r_a > 0) & (v1_c_a > 0) & (v2_c_a > 0)
        shrink_cols = np.log2((v1_c_a[valid]+1)/(v2_c_a[valid]+1))
        shrink_rows = np.log2((v1_r_a[valid]+1)/(v2_r_a[valid]+1))
        score = shrink_cols + shrink_rows
        top_idx = np.argsort(-score)[:min(top_k, score.size)]  # highest shrink
        # Top row: V1
        ax1 = axes[0, j]
        ax1.scatter(v1_r, v1_c, s=6, alpha=0.35, edgecolors='none', c='#1f77b4')
        ax1.plot([rmin, rmax], [cmin, cmax], ls='--', c='#888', lw=1)
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_xlim(rmin, rmax); ax1.set_ylim(cmin, cmax)
        ax1.set_title(f"{res} V1 rows vs cols")
        # Bottom row: V2 (+ highlight top shrink in red)
        ax2 = axes[1, j]
        ax2.scatter(v2_r, v2_c, s=6, alpha=0.35, edgecolors='none', c='#2ca02c')
        if top_idx.size > 0:
            ax2.scatter(v2_r_a[valid][top_idx], v2_c_a[valid][top_idx], s=12, c='crimson', alpha=0.9, label='top shrink')
            ax2.legend(fontsize=8)
        ax2.plot([rmin, rmax], [cmin, cmax], ls='--', c='#888', lw=1)
        ax2.set_xscale('log'); ax2.set_yscale('log')
        ax2.set_xlim(rmin, rmax); ax2.set_ylim(cmin, cmax)
        ax2.set_title(f"{res} V2 rows vs cols")
        for ax in (ax1, ax2):
            ax.grid(True, linestyle='--', alpha=0.25)
            ax.set_xlabel('rows'); ax.set_ylabel('cols')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def _print_aligned_table(headers, rows):
    cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))
    def fmt(row):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
    print(fmt(headers))
    for r in rows:
        print(fmt(r))

def write_overlay_rows(csv_writer, resource, csv_rows, csv_to_modelid, csv_to_sourcepath, anomaly_min_rows, anomaly_min_cols, anomaly_ratio):
    """Write only unequal rows using provided csv writer; return (kept, total)."""
    total = 0
    kept = 0
    for base, v1_o, v1_t, v2_o, v2_t in sorted(csv_rows):
        mode = 'both' if (v1_o is not None and v2_o is not None) else ('v1_only' if v1_o is not None else 'v2_only')
        cols_equal = (v1_o == v2_o) if (v1_o is not None and v2_o is not None) else ''
        rows_equal = (v1_t == v2_t) if (v1_t is not None and v2_t is not None) else ''
        cols_diff = (v2_o - v1_o) if (v1_o is not None and v2_o is not None) else ''
        rows_diff = (v2_t - v1_t) if (v1_t is not None and v2_t is not None) else ''
        # Anomaly detection: v2 has high rows/cols ratio OR v2 has many cols OR v2 has many rows
        anomaly_v2 = False
        if v2_o and v2_t:
            # High rows/cols ratio
            ratio_anomaly = (v2_t >= anomaly_min_rows) and (v2_t / v2_o >= anomaly_ratio)
            # Many columns
            cols_anomaly = (v2_o >= anomaly_min_cols)
            # Many rows  
            rows_anomaly = (v2_t >= anomaly_min_rows)
            anomaly_v2 = ratio_anomaly or cols_anomaly or rows_anomaly
        
        anomaly = True if anomaly_v2 else False if (mode == 'both') else ''
        model_id = csv_to_modelid.get(base, '')
        source_path = csv_to_sourcepath.get(base, '')
        total += 1
        is_rows_diff = (rows_equal not in ('', 'True', True))
        is_cols_diff = (cols_equal not in ('', 'True', True))
        is_missing = (mode != 'both')
        if not (is_rows_diff or is_cols_diff or is_missing):
            continue
        kept += 1
        csv_writer.writerow([
            resource,
            base,
            model_id,
            source_path,
            mode,
            v1_o if v1_o is not None else '',
            v2_o if v2_o is not None else '',
            cols_equal,
            cols_diff,
            v1_t if v1_t is not None else '',
            v2_t if v2_t is not None else '',
            rows_equal,
            rows_diff,
            anomaly,
        ])
    return kept, total

def write_overlay_csv(csv_out, resource, csv_rows, csv_to_modelid, csv_to_sourcepath, anomaly_min_rows, anomaly_min_cols, anomaly_ratio):
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['resource', 'basename', 'modelId', 'source_path', 'mode', 'v1_cols', 'v2_cols', 'cols_equal', 'cols_diff', 'v1_rows', 'v2_rows', 'rows_equal', 'rows_diff', 'anomaly'])
        kept, total = write_overlay_rows(writer, resource, csv_rows, csv_to_modelid, csv_to_sourcepath, anomaly_min_rows, anomaly_min_cols, anomaly_ratio)

    print(f"Saved unequal items to {csv_out} ({kept}/{total}).")
### updated ###
def main():
    ap = argparse.ArgumentParser()
    # Run all resources; no single-resource flag
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories of the provided directories")
    ap.add_argument("--png-out", default="data/analysis/v1v2_overlay.png", help="Path to save the overlay histogram PNG")
    ap.add_argument("--bins", type=int, default=50, help="Number of bins for histograms")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 4) * 2), help="Max worker threads for parallel file processing")
    ap.add_argument("--top", type=int, default=0, help="Print top-N examples where rows >> columns (by ratio)")
    ap.add_argument("--csv-out", default="data/analysis/v1v2_overlay.csv", help="CSV with per-basename stats, equality diffs, and anomaly flag")
    ap.add_argument("--anomaly-ratio", type=float, default=2.0, help="Rows/cols ratio threshold to flag anomalies")
    ap.add_argument("--anomaly-min-rows", type=int, default=1000, help="Minimum rows to consider when flagging anomalies")
    ap.add_argument("--anomaly-min-cols", type=int, default=500, help="Minimum columns to consider when flagging anomalies")
    args = ap.parse_args()

    # Run all resources and aggregate
    resources = ["hugging", "github", "arxiv"]
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['resource', 'basename', 'modelId', 'source_path', 'mode', 'v1_cols', 'v2_cols', 'cols_equal', 'cols_diff', 'v1_rows', 'v2_rows', 'rows_equal', 'rows_diff', 'anomaly'])
        density_data = {}
        ecdf_data = {}
        scatter_data = {}
        per_summary = []
        for res in resources:
            v1_dir, v2_dir = resolve_resource_dirs(res)
            v1_files, v2_file_map, v1_file_map, basenames = prepare_file_maps(v1_dir, v2_dir, args.recursive)
            if not basenames:
                print(f"[WARN] No basenames for resource {res}; skipping")
                continue
            csv_to_modelid, csv_to_sourcepath = build_resource_mappings(res)
            print(f"=== {res}: qc_anomaly summary ===")
            print_overlap_summary(v1_file_map, v2_file_map)
            results_rows, csv_rows, v1_orig, v1_trans, v2_orig, v2_trans = process_basenames(
                basenames, v1_file_map, v2_file_map, args.workers
            )
            kept, total = write_overlay_rows(writer, res, csv_rows, csv_to_modelid, csv_to_sourcepath, args.anomaly_min_rows, args.anomaly_min_cols, args.anomaly_ratio)
            # Count mode categories and equal/changed
            mode_counts = {"both": 0, "v1_only": 0, "v2_only": 0}
            changed = 0
            for base, v1_o, v1_t, v2_o, v2_t in csv_rows:
                mode = 'both' if (v1_o is not None and v2_o is not None) else ('v1_only' if v1_o is not None else 'v2_only')
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                if mode == 'both' and not (v1_o == v2_o and v1_t == v2_t):
                    changed += 1
            print(f"{res}: total={len(csv_rows)} | both={mode_counts['both']}, v1_only={mode_counts['v1_only']}, v2_only={mode_counts['v2_only']} | changed(both)={changed}")
            print(f"Saved unequal items for {res} to {args.csv_out} ({kept}/{total}).")
            per_summary.append((res, kept, total, mode_counts['both'], mode_counts['v1_only'], mode_counts['v2_only'], changed))
            density_data[res] = (v1_orig, v2_orig, v1_trans, v2_trans)
            ecdf_data[res] = (v1_orig, v2_orig, v1_trans, v2_trans)
            scatter_data[res] = (v1_trans, v1_orig, v2_trans, v2_orig)
        # Combined plots
        plot_density_iqr_multi(density_data, args.bins, args.png_out.replace('.png', '_multi.png'))
        plot_ecdf_multi(ecdf_data, out_path="data/analysis/combined_ecdf_multi.png")
        plot_scatter_grid_multi(scatter_data, out_path="data/analysis/scatter_grid_multi.png", top_k=200)
        # Print summary table
        print("=== Summary by resource ===")
        headers = ["resource", "total_in_csv_rows", "both", "v1_only", "v2_only", "changed(both)", "unequal_saved"]
        rows = []
        for res, kept, total, both_c, v1_c, v2_c, changed in per_summary:
            rows.append([res, total, both_c, v1_c, v2_c, changed, kept])
        _print_aligned_table(headers, rows)

if __name__ == "__main__":
    main()
