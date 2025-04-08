"""
Author: Zhengyuan Dong
Created: 2025-04-03
Last Modified: 2025-04-04
Description:
    Compute the overlap rate of citation and citing papers based on paperId
    with flexible “modes” controlled by a tiny factory layer.

    Supported relationship modes
    ----------------------------
    • overlap_rate   : uses a (pid → {related_pids}) mapping with weights
    • direct_label  : uses a (pid → {cited_by_pids}) mapping as GT
    • overlap_label  : uses a (pid → {related_pids}) mapping with weights

    Supported table source modes
    ----------------------------
    • step4_symlink  : use step‑4 parquet + *_sym columns
    • step3_merged   : use step‑3 parquet + raw columns

    Outputs
    -------
    • pairwise overlap scores
    • thresholded related paper pairs
    • direct citation links (if paperId appears in references of another)
    • ground‑truth benchmark table paths (pickle)

Usage:
    python -m src.data_gt.step3_gt --rel_mode [overlap_rate, overlap_label, direct_label] --tbl_mode [step4_symlink, step3_merged] (require step4.parquet by create_symlink.py)
    python -m src.data_gt.step3_gt --rel_mode direct_label --tbl_mode step3_merged
"""

import os
import json
import pandas as pd
import pickle
from tqdm import tqdm
from enum import Enum
from collections import defaultdict, Counter
from typing import Dict, Iterable, Set
from joblib import Parallel, delayed

# === Configuration ===
DATA_DIR = "data/processed"
GT_DIR = "data/gt"

# ---- default files (can be overridden by CLI/kwargs) ----
FILES = {
    "overlap_rate": f"{DATA_DIR}/modelcard_citation_overlap_rate.pickle",
    "overlap_label": f"{DATA_DIR}/modelcard_citation_overlap_label.pickle",
    "direct_label": f"{DATA_DIR}/modelcard_citation_direct_label.pickle",
    "step4_symlink": f"{DATA_DIR}/modelcard_step4.parquet",
    "step3_merged": f"{DATA_DIR}/modelcard_step3_merged.parquet",
    "integration": f"{DATA_DIR}/final_integration_with_paths.parquet",
    "title_list": f"{DATA_DIR}/modelcard_all_title_list.parquet",
    "symlink_mapping": f"{DATA_DIR}/symlink_mapping.pickle"
}

MODEL_GT_PATH = f"{GT_DIR}/groundtruth_model_pairs.pickle"  ########
MAPPED_GT_PATH = f"{GT_DIR}/groundtruth_mapped_pairs.pickle"  ########
CSV_PAIR_FREQ_PATH = f"{GT_DIR}/csv_pair_frequency.pickle"  ########

OUTPUT_GT_PATH = f"{GT_DIR}/scilakeUnionBenchmark_by_ids.pickle"
DISCOUNT_RATE = 0.5
IS_STRICT_MODE = True
THRESHOLD = 0.2 # for overlap_rate only


# ===== ENUMS =============================================================== #
class RelationshipMode(str, Enum):
    OVERLAP_RATE = "overlap_rate"
    DIRECTED_CITE = "direct_label"
    OVERLAP_LABEL = "overlap_label"


class TableSourceMode(str, Enum):
    STEP4_SYMLINK = "step4_symlink"
    STEP3_MERGED = "step3_merged"


# ===== FACTORIES =========================================================== #
def load_relationships(mode: RelationshipMode):
    """Factory loader for paperId‑level relationship graphs."""
    path = FILES[mode.value]
    print(f"Loading relationships ({mode.value}) from: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_table_source(mode: TableSourceMode):
    """Factory loader for table‑list metadata."""
    path = FILES[mode.value]
    print(f"Loading table source ({mode.value}) from: {path}")
    df = pd.read_parquet(path)
    return df.set_index("modelId", drop=False)

def load_symlink_mapping():
    path = FILES["symlink_mapping"]
    print(f"Loading symlink mapping from: {path}")
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    return mapping

# ===== HELPER FUNCTIONS FOR PARALLEL PROCESSING ============================
def process_model_pair(pair, df_tables, use_symlink):
    """Process a single model pair to extract CSV pair edges.
    Returns a tuple: ((m1, m2), set_of_csv_edges)
    """
    m1, m2 = pair
    csv_list_m1 = []
    csv_list_m2 = []
    if m1 in df_tables.index:
        csv_list_m1 = extract_csv_list(df_tables.loc[m1], use_symlink)
    if m2 in df_tables.index:
        csv_list_m2 = extract_csv_list(df_tables.loc[m2], use_symlink)
    union_csv = sorted(set(csv_list_m1).union(set(csv_list_m2)))
    csv_edges = set()
    for i in range(len(union_csv)):
        for j in range(i + 1, len(union_csv)):
            edge = (union_csv[i], union_csv[j])
            csv_edges.add(edge)
    return (pair, csv_edges)

def map_csv_edges(result, symlink_map):
    """Map CSV symlink edges back to their real paths.
    result: ((m1, m2), csv_edges)
    Returns: ((m1, m2), new_csv_edges)
    """
    pair, csv_edges = result
    new_csv_edges = set()
    for csv_edge in csv_edges:
        csv1, csv2 = csv_edge
        real_csv1 = symlink_map.get(csv1, csv1)
        real_csv2 = symlink_map.get(csv2, csv2)
        new_csv_edges.add((real_csv1, real_csv2))
    return (pair, new_csv_edges)

# ===== BUSINESS LOGIC ====================================================== #
def extract_csv_list(
    row: pd.Series,
    use_symlink: bool,
):
    if use_symlink:
        hugging_col = "hugging_table_list_sym"
        github_col = "github_table_list_sym"
        html_col = "html_table_list_sym"
        llm_col = "llm_table_list_sym"
    else:
        hugging_col = "hugging_table_list_dedup"
        github_col = "github_table_list_dedup"
        html_col = "html_table_list_mapped_dedup"
        llm_col = "llm_table_list_mapped_dedup"
    csv_list = []
    csv_list.extend(row[hugging_col])
    csv_list.extend(row[github_col])
    csv_list.extend(row[html_col])
    csv_list.extend(row[llm_col])
    seen = set()
    result = []
    for csv in csv_list:
        if csv not in seen:
            seen.add(csv)
            result.append(csv)
    return result

def build_ground_truth(
    rel_mode: RelationshipMode = RelationshipMode.OVERLAP_RATE,
    tbl_mode: TableSourceMode = TableSourceMode.STEP4_SYMLINK,
    use_symlink: bool | None = True,
):
    """High‑level orchestration for building GT tables."""
    if use_symlink is None:  # auto‑infer from table source mode
        use_symlink = tbl_mode == TableSourceMode.STEP4_SYMLINK

    tmp_relationships = load_relationships(rel_mode)

    new_relationships = defaultdict(set)
    for pair, score in tqdm(tmp_relationships.items(), desc="Filtering relationships"):
        p1, p2 = pair
        #if p1!=p2: # we could include self-loop
        if rel_mode in ['direct_label', 'overlap_label']:
            if score >= 1.0: # avoid there are integer
                new_relationships[p1].add(p2)
                #new_relationships[p2].add(p1) # edge only compute once
        elif rel_mode == 'overlap_rate':
            if score >= THRESHOLD:
                new_relationships[p1].add(p2)
                #new_relationships[p2].add(p1)
    paperid_relationships = new_relationships
    # add self-loop
    for pid in paperid_relationships.keys():
        paperid_relationships[pid].add(pid)

    # --- metadata & mappings -------------------------------------------------
    df_metadata = pd.read_parquet(FILES["integration"], columns=["corpusid", "query"])
    df_titles = pd.read_parquet(FILES["title_list"], columns=["modelId", "all_title_list"])
    df_titles_exploded = df_titles.explode("all_title_list")

    # title → modelIds
    title_to_modelIds = defaultdict(set)
    for _, row in tqdm(df_titles_exploded.iterrows(), total=len(df_titles_exploded), desc="Mapping titles to modelIds"):  ######## Added tqdm here
        title = row["all_title_list"]
        if pd.notna(title):
            title_to_modelIds[title.strip()].add(row["modelId"])

    # paperId → modelIds
    paperId_to_modelIds = defaultdict(set)
    for _, row in tqdm(df_metadata.iterrows(), total=len(df_metadata), desc="Mapping paperIds to modelIds"):  ######## Added tqdm here
        pid = row["corpusid"] # actually this is paperId
        title = row["query"]
        if pd.notna(pid) and pd.notna(title):
            model_ids = title_to_modelIds[title.strip()]
            for mid in model_ids:
                paperId_to_modelIds[str(pid)].add(mid)

    # --- table source --------------------------------------------------------
    df_tables = load_table_source(tbl_mode)

    # ==== Step 1: Build model pair set without weighting ====
    model_pairs = set()
    # 1.1 Intra-paper: Build model pairs within the same paper
    for pid, models in tqdm(paperId_to_modelIds.items(), desc="Building intra-paper model pairs"):  ######## Added tqdm here
        models = list(models)
        n = len(models)
        # self-loop of model level
        for m in models:
            model_pairs.add((m, m))
        # pairwise combinations
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    pair_key = tuple(sorted((models[i], models[j])))
                    model_pairs.add(pair_key)
    # 1.2 Inter-paper: Build model pairs between different papers based on paper relationships
    for p1, related_papers in tqdm(paperid_relationships.items(), desc="Building inter-paper model pairs"):  ######## Added tqdm here
        models_p1 = paperId_to_modelIds.get(p1, set())
        for p2 in related_papers:
            models_p2 = paperId_to_modelIds.get(p2, set())
            for m1 in models_p1:
                for m2 in models_p2:
                    pair_key = tuple(sorted((m1, m2)))
                    model_pairs.add(pair_key)
    print(f"Total model pairs to process: {len(model_pairs)}")

    # ==== Step 2: For each model pair, compute CSV weight based on table information ====
    """groundtruth_pairs = set()
    for m1, m2 in model_pairs:
        csv_list_m1 = []
        csv_list_m2 = []
        if m1 in df_tables.index:
            csv_list_m1 = extract_csv_list(df_tables.loc[m1], use_symlink)
        if m2 in df_tables.index:
            csv_list_m2 = extract_csv_list(df_tables.loc[m2], use_symlink)
        union_csv = sorted(set(csv_list_m1).union(set(csv_list_m2))) # only for this step we can union
        csv_edges = set()
        for i in range(len(union_csv)):
            for j in range(i + 1, len(union_csv)):
                edge = (union_csv[i], union_csv[j])
                csv_edges.add(edge)
        groundtruth_pairs.add(((m1, m2), csv_edges))"""
    model_pair_results = Parallel(n_jobs=-1)(
        delayed(process_model_pair)(pair, df_tables, use_symlink)
        for pair in tqdm(list(model_pairs), desc="Processing model pairs", leave=True)  ######## Added tqdm with joblib
    )
    # Save intermediate model-level groundtruth pairs ########
    with open(MODEL_GT_PATH, "wb") as f:
        pickle.dump(model_pair_results, f)
    print(f"Intermediate model-level groundtruth pairs saved to {MODEL_GT_PATH}")
    # because we use set, we can only use symlink above
    # ==== TODO: Additional Checks ====
    
    # ==== Step 2.5: Map symlink paths back to real CSV paths (if using symlink) ====
    if use_symlink:
        symlink_map = load_symlink_mapping()
        """new_groundtruth_pairs = []
        for (m1, m2), csv_edges in tqdm(groundtruth_pairs, desc="Mapping symlinks to real paths"):
            new_csv_edges = set()
            for csv_edge in csv_edges:
                csv1, csv2 = csv_edge
                real_csv1 = symlink_map.get(csv1, csv1)
                real_csv2 = symlink_map.get(csv2, csv2)
                new_csv_edges.add((real_csv1, real_csv2))
            new_groundtruth_pairs.append(((m1, m2), new_csv_edges))
        groundtruth_pairs = new_groundtruth_pairs"""
        mapped_results = Parallel(n_jobs=-1)(
            delayed(map_csv_edges)(result, symlink_map)
            for result in tqdm(model_pair_results, desc="Mapping CSV edges", leave=True)  ########
        )
    else:
        mapped_results = model_pair_results
    # Save intermediate mapped groundtruth pairs ########
    with open(MAPPED_GT_PATH, "wb") as f:
        pickle.dump(mapped_results, f)
    print(f"Intermediate mapped groundtruth pairs saved to {MAPPED_GT_PATH}")

    # ==== Step 2.6: Count CSV pair frequencies at CSV pair level ====
    csv_pair_counter = Counter()
    for _, csv_edges in mapped_results:
        for edge in csv_edges:
            csv_pair_counter[edge] += 1
    # Save CSV pair frequency counts ########
    with open(CSV_PAIR_FREQ_PATH, "wb") as f:
        pickle.dump(csv_pair_counter, f)
    print(f"CSV pair frequency counts saved to {CSV_PAIR_FREQ_PATH}")
    print(f"Total unique CSV pairs: {len(csv_pair_counter)}")  ########

    #groundtruth_pairs = list(groundtruth_pairs)
    # ==== Step 3: Save the groundtruth pairs ====
    """os.makedirs(os.path.dirname(OUTPUT_GT_PATH), exist_ok=True)
    with open(OUTPUT_GT_PATH, "wb") as f:
        pickle.dump(groundtruth_pairs, f)
    print(f"✅ Groundtruth saved to {OUTPUT_GT_PATH}")"""
    # ==== Step 3: Save the final groundtruth pairs (model pair level with CSV edges) ====
    # Here we save the mapped_results as the final groundtruth
    os.makedirs(os.path.dirname(OUTPUT_GT_PATH), exist_ok=True)
    with open(OUTPUT_GT_PATH, "wb") as f:
        pickle.dump(mapped_results, f)
    print(f"✅ Final groundtruth saved to {OUTPUT_GT_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SciLake union benchmark tables.")
    parser.add_argument(
        "--rel_mode",
        choices=[m.value for m in RelationshipMode],
        default=RelationshipMode.OVERLAP_RATE.value,
        help="Relationship graph mode.",
    )
    parser.add_argument(
        "--tbl_mode",
        choices=[m.value for m in TableSourceMode],
        default=TableSourceMode.STEP4_SYMLINK.value,
        help="Table source mode.",
    )
    args = parser.parse_args()

    build_ground_truth(
        rel_mode=RelationshipMode(args.rel_mode),
        tbl_mode=TableSourceMode(args.tbl_mode),
        use_symlink=True
    )
