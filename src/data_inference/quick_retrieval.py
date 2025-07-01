"""
Author: Zhengyuan Dong
Date: 2025-06-17

Usage:

# Use overlap level
python src/data_inference/quick_retrieval.py --gt_dir data/gt --link "google/vit-base-patch16-224-in21k" --level overlap

# Use model level and methodology_or_result intent
python src/data_inference/quick_retrieval.py --gt_dir data/gt --link "2102.07033" --level model --intent methodology_or_result

# Use dataset level and influential
python src/data_inference/quick_retrieval.py --gt_dir data/gt --link "215768677" --level dataset --influential

# Use inferred results
python src/data_inference/quick_retrieval.py --gt_dir data/gt --link "google/vit-base-patch16-224-in21k" --use_inferred --inferred_file test_hnsw_search_shuffle_col_tfidf_entity_full.json --top_k 5

# Use ground truth
python src/data_inference/quick_retrieval.py --gt_dir data/gt --link "google/vit-base-patch16-224-in21k" --use_gt --intent methodology_or_result --influential

# Use inferred results
python src/data_inference/quick_retrieval.py --gt_dir data/gt --link "google/vit-base-patch16-224-in21k" --use_inferred --augmentation shuffle_col
"""

import os
import argparse
import pickle
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import re
import json

# Ground truth file patterns
INTENTS = ["", "methodology_or_result"]
INFLUENTIALS = ["", "influential"]
LEVELS = ["direct", "overlap", "max_pr", "union", "model", "dataset"]
AUGMENTATIONS = ["shuffle_col", "shuffle_row", "drop_cell"]

def get_gt_files(level: str, intent: str = "", influential: bool = False) -> Tuple[str, str]:
    """Get ground truth file names based on level, intent and influential flag."""
    if level == "direct":
        prefix = "csv_pair_matrix_direct_label"
        csvlist_prefix = "csv_list_direct_label"
    elif level == "overlap":
        prefix = "csv_pair_adj_overlap_rate"
        csvlist_prefix = "csv_list"
    elif level == "max_pr":
        prefix = "csv_pair_matrix_max_pr"
        csvlist_prefix = "csv_list_max_pr"
    elif level == "union":
        prefix = "csv_pair_union_direct"
        csvlist_prefix = "csv_pair_union_direct_csv_list"
    elif level == "model":
        prefix = "scilake_gt_modellink_model_adj"
        csvlist_prefix = "scilake_gt_modellink_model_adj_csv_list"
    elif level == "dataset":
        prefix = "scilake_gt_modellink_dataset_adj"
        csvlist_prefix = "scilake_gt_modellink_dataset_adj_csv_list"
    else:
        raise ValueError(f"Unknown level: {level}")

    suffix = ""
    if intent:
        suffix += f"_{intent}"
    if influential:
        suffix += "_influential"
    
    matrix_file = f"{prefix}{suffix}.npz"
    csvlist_file = f"{csvlist_prefix}{suffix}.pkl"
    
    return matrix_file, csvlist_file

def get_table_source(basename: str) -> str:
    """Determine the source of a table based on its basename pattern."""
    # Check for arXiv pattern (e.g., 2102.07033_table1.csv)
    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?_table\d+\.csv$', basename):
        return "arxiv"
    
    # Check for Semantic Scholar pattern (e.g., 215768677_table2.csv)
    if re.match(r'^\d{9}_table\d+\.csv$', basename):
        return "semantic_scholar"
    
    # Check for hash pattern (e.g., 0cf63c386b_table2.csv)
    if re.match(r'^[a-f0-9]{10}_table\d+\.csv$', basename):
        return "hugging"
    
    # Check for GitHub hash pattern (e.g., ba1acdb60d_table1.csv)
    if re.match(r'^[a-f0-9]{10}_table\d+\.csv$', basename):
        return "github"
    
    return "unknown"

def parse_link(link: str, df: pd.DataFrame) -> List[str]:
    """Parse different types of links to get related CSV files."""
    # Case 1: Direct CSV file
    if link.endswith('.csv'):
        return [os.path.basename(link)]
    
    # Case 2: Model ID
    if '/' in link:  # e.g., "google/vit-base-patch16-224-in21k"
        model_row = df[df['modelId'] == link]
        if not model_row.empty:
            tables = []
            for col in ['html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup', 
                       'hugging_table_list_dedup', 'github_table_list_dedup']:
                table_list = model_row[col].iloc[0]
                if isinstance(table_list, str):
                    table_list = eval(table_list)
                tables.extend([os.path.basename(t) for t in table_list])
            return tables
    
    # Case 3: arXiv ID
    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', link):
        return [os.path.basename(f) for f in df['all_table_list_dedup'].explode() 
                if f.startswith(f"{link}_table")]
    
    # Case 4: Semantic Scholar ID
    if re.match(r'^\d{9}$', link):
        return [os.path.basename(f) for f in df['all_table_list_dedup'].explode() 
                if f.startswith(f"{link}_table")]
    
    return []

def load_ground_truth(gt_dir: str, level: str, intent: str = "", influential: bool = False) -> Tuple[np.ndarray, List[str]]:
    """Load ground truth data based on level, intent and influential flag."""
    matrix_file, csvlist_file = get_gt_files(level, intent, influential)
    
    npz_path = os.path.join(gt_dir, matrix_file)
    csvlist_path = os.path.join(gt_dir, csvlist_file)
    
    if not (os.path.isfile(npz_path) and os.path.isfile(csvlist_path)):
        raise FileNotFoundError(f"Missing files for level={level}, intent={intent}, influential={influential}")
        
    with open(csvlist_path, 'rb') as f:
        csv_list = pickle.load(f)
    matrix = load_npz(npz_path)
    
    return matrix, csv_list

def load_inferred_results(augmentation: str = "shuffle_col") -> Dict[str, List[str]]:
    """Load inferred results based on augmentation type."""
    # Default to shuffle_col if not specified
    if not augmentation:
        augmentation = "shuffle_col"
    
    # Construct the filename based on augmentation
    filename = f"test_hnsw_search_{augmentation}_tfidf_entity_full.json"
    
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def find_related_tables_gt(csv_basenames: List[str], matrix: np.ndarray, csv_list: List[str], 
                          top_k: int = 5) -> List[Dict]:
    """Find related tables based on ground truth matrix."""
    related_tables = []
    
    for csv_basename in csv_basenames:
        try:
            idx = csv_list.index(csv_basename)
        except ValueError:
            print(f"Warning: CSV {csv_basename} not found in csv_list")
            continue
        
        # Get related indices from matrix
        related_indices = np.argsort(matrix[idx].toarray().flatten())[-top_k-1:-1][::-1]
        
        # Get related tables
        for rel_idx in related_indices:
            if matrix[idx, rel_idx] > 0:  # Only include if there is a relationship
                table_info = {
                    "table_basename": csv_list[rel_idx],
                    "source": get_table_source(csv_list[rel_idx])
                }
                if table_info not in related_tables:
                    related_tables.append(table_info)
    
    return related_tables

def find_related_tables_inferred(csv_basenames: List[str], inferred_results: Dict[str, List[str]], 
                               top_k: int = 5) -> List[Dict]:
    """Find related tables based on inferred results."""
    related_tables = []
    
    # Only use the first CSV file to get top_k results
    if csv_basenames and csv_basenames[0] in inferred_results:
        # Get top-k related tables
        for related_csv in inferred_results[csv_basenames[0]][:top_k]:
            table_info = {
                "table_basename": related_csv,
                "source": get_table_source(related_csv)
            }
            if table_info not in related_tables:
                related_tables.append(table_info)
    
    return related_tables

def main():
    parser = argparse.ArgumentParser(description="Quick table retrieval based on ground truth or inferred results")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth files")
    parser.add_argument("--link", type=str, required=True, help="Link to find related tables for")
    parser.add_argument("--use_gt", action="store_true", help="Use ground truth")
    parser.add_argument("--use_inferred", action="store_true", help="Use inferred results")
    parser.add_argument("--level", type=str, default="direct", choices=LEVELS,
                       help="Ground truth level to use")
    parser.add_argument("--intent", type=str, default="", choices=INTENTS,
                       help="Intent type for ground truth")
    parser.add_argument("--influential", action="store_true",
                       help="Use influential ground truth")
    parser.add_argument("--augmentation", type=str, default="shuffle_col", choices=AUGMENTATIONS,
                       help="Augmentation type for inferred results")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top related tables to return")
    
    args = parser.parse_args()
    
    if not (args.use_gt or args.use_inferred):
        raise ValueError("Must specify either --use_gt or --use_inferred")
    if args.use_gt and args.use_inferred:
        raise ValueError("Cannot use both --use_gt and --use_inferred")
    
    # Load model card data
    df = pd.read_parquet("data/processed/modelcard_step3_dedup.parquet")
    
    # Parse link to get CSV basenames
    csv_basenames = parse_link(args.link, df)
    if not csv_basenames:
        print(f"No CSV files found for link: {args.link}")
        return
    
    # Find related tables
    if args.use_gt:
        matrix, csv_list = load_ground_truth(args.gt_dir, args.level, args.intent, args.influential)
        related_tables = find_related_tables_gt(csv_basenames, matrix, csv_list, args.top_k)
        result_type = f"ground truth (level={args.level}, intent={args.intent}, influential={args.influential})"
    else:
        inferred_results = load_inferred_results(args.augmentation)
        related_tables = find_related_tables_inferred(csv_basenames, inferred_results, args.top_k)
        result_type = f"inferred results (augmentation={args.augmentation})"
    
    # Print results
    print(f"\nInput link: {args.link}")
    print(f"Found {len(csv_basenames)} CSV files:")
    for csv in csv_basenames:
        print(f"  - {csv} (source: {get_table_source(csv)})")
    
    print(f"\nRelated tables (using {result_type}):")
    for i, table in enumerate(related_tables, 1):
        print(f"\n{i}. Table: {table['table_basename']}")
        print(f"   Source: {table['source']}")

if __name__ == "__main__":
    main() 