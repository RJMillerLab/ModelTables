#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-14
Description: Improved smart sampling for GPT evaluation based on ground truth construction logic.
             This script addresses the specific requirements:
             1. Table Relatedness - based on final ground truth matrices
             2. Model Relatedness - based on model card metadata used to infer table relatedness
             3. Smart filtering - ensure samples have sufficient information (not empty)
             4. Balanced sampling across different levels and attributes

Key improvements:
- Uses actual ground truth construction logic from step3_gt.py and modelcard_matrix.py
- Filters for nonempty paper & csv lists (like ground truth construction)
- Samples from model-based and dataset-based relationships
- Ensures sufficient information for meaningful evaluation
"""

import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
from scipy.sparse import load_npz
import pickle

# Configuration
GT_DIR = "data/gt"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"

class ImprovedSmartSampler:
    def __init__(self, gt_dir: str = GT_DIR, processed_dir: str = PROCESSED_DIR):
        self.gt_dir = gt_dir
        self.processed_dir = processed_dir
        self.ground_truth_matrices = {}
        self.csv_lists = {}
        self.model_data = None
        self.model_metadata = None
        
    def load_ground_truth(self):
        """Load ground truth matrices and CSV lists"""
        print("Loading ground truth matrices...")
        
        # Load different levels of ground truth
        levels = {
            "direct": "csv_pair_matrix_direct_label.npz",
            "max_pr": "csv_pair_matrix_max_pr.npz", 
            "model": "scilake_gt_modellink_model_adj_processed.npz",
            "dataset": "scilake_gt_modellink_dataset_adj_processed.npz"
        }
        
        for level, filename in levels.items():
            npz_path = os.path.join(self.gt_dir, filename)
            csvlist_path = os.path.join(self.gt_dir, f"csv_list_{level}.pkl")
            
            if os.path.exists(npz_path) and os.path.exists(csvlist_path):
                try:
                    matrix = load_npz(npz_path)
                    with open(csvlist_path, 'rb') as f:
                        csv_list = pickle.load(f)
                    
                    self.ground_truth_matrices[level] = matrix
                    self.csv_lists[level] = csv_list
                    print(f"  ✓ Loaded {level}: {matrix.shape}, {len(csv_list)} CSVs")
                except Exception as e:
                    print(f"  ✗ Failed to load {level}: {e}")
            else:
                print(f"  ✗ Missing files for {level}")
    
    def load_model_data(self):
        """Load model data with proper filtering (like ground truth construction)"""
        print("Loading model data with filtering...")
        
        # Load model metadata (step1)
        model_metadata_file = os.path.join(self.processed_dir, "modelcard_step1.parquet")
        if os.path.exists(model_metadata_file):
            self.model_metadata = pd.read_parquet(model_metadata_file)
            print(f"  ✓ Loaded {len(self.model_metadata)} model metadata records")
        else:
            print(f"  ✗ Missing model metadata file: {model_metadata_file}")
        
        # Load deduplicated model data (step3_dedup)
        model_file = os.path.join(self.processed_dir, "modelcard_step3_dedup.parquet")
        if os.path.exists(model_file):
            self.model_data = pd.read_parquet(model_file)
            print(f"  ✓ Loaded {len(self.model_data)} deduplicated model records")
        else:
            print(f"  ✗ Missing model file: {model_file}")
    
    def apply_ground_truth_filters(self):
        """Apply the same filtering conditions used in ground truth construction"""
        print("Applying ground truth construction filters...")
        
        if self.model_data is None:
            print("  ✗ No model data to filter")
            return
        
        # Filter 1: Nonempty paper & csv lists (from step3_gt.py line 192-197)
        print("  Filtering for nonempty paper & csv lists...")
        
        # Check what columns are available
        print(f"  Available columns: {self.model_data.columns.tolist()}")
        
        # Find the correct column names (they might vary)
        paper_col = None
        csv_col = None
        
        for col in self.model_data.columns:
            if 'title' in col.lower() and 'list' in col.lower():
                paper_col = col
            elif 'table' in col.lower() and 'list' in col.lower():
                csv_col = col
        
        if paper_col is None or csv_col is None:
            print(f"  ✗ Could not find required columns. Available: {self.model_data.columns.tolist()}")
            return
        
        print(f"  Using paper column: {paper_col}")
        print(f"  Using csv column: {csv_col}")
        
        # Apply filtering
        initial_count = len(self.model_data)
        
        # Filter for nonempty lists
        filtered_data = []
        for _, row in self.model_data.iterrows():
            papers = row[paper_col] if isinstance(row[paper_col], list) else []
            csvs = row[csv_col] if isinstance(row[csv_col], list) else []
            
            # Keep rows with both nonempty paper and csv lists
            if papers and csvs:
                filtered_data.append(row)
        
        self.model_data = pd.DataFrame(filtered_data)
        print(f"  ✓ Filtered from {initial_count} to {len(self.model_data)} models with nonempty paper & csv lists")
        
        # Filter 2: Model-based relationships (from modelcard_matrix.py line 304)
        if self.model_metadata is not None:
            print("  Filtering for model-based relationships...")
            
            # Keep rows with at least one base/model link
            model_filtered = self.model_metadata[
                self.model_metadata.apply(
                    lambda r: bool(r.get('tag_base_model_list', []) or r.get('readme_modelid_list', [])), 
                    axis=1
                )
            ]
            print(f"  ✓ Found {len(model_filtered)} models with base/model relationships")
    
    def sample_table_pairs_with_ground_truth(self, 
                                            num_pairs: int = 100,
                                            positive_ratio: float = 0.5,
                                            level_distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Sample table pairs based on actual ground truth matrices"""
        
        if level_distribution is None:
            level_distribution = {
                "direct": 0.3,
                "max_pr": 0.2, 
                "model": 0.25,
                "dataset": 0.25
            }
        
        print(f"Sampling {num_pairs} table pairs from ground truth...")
        print(f"  Positive ratio: {positive_ratio}")
        print(f"  Level distribution: {level_distribution}")
        
        pairs = []
        num_positive = int(num_pairs * positive_ratio)
        num_negative = num_pairs - num_positive
        
        # Sample positive pairs from ground truth matrices
        positive_pairs = self._sample_positive_pairs_from_gt(num_positive, level_distribution)
        pairs.extend(positive_pairs)
        
        # Sample negative pairs (not in ground truth)
        negative_pairs = self._sample_negative_pairs_from_gt(num_negative)
        pairs.extend(negative_pairs)
        
        # Shuffle and add metadata
        random.shuffle(pairs)
        
        for i, pair in enumerate(pairs):
            pair["id"] = f"gt-pair-{i+1}"
            pair["evaluation_type"] = "table_relatedness"
        
        print(f"  ✓ Generated {len(pairs)} pairs ({len(positive_pairs)} positive, {len(negative_pairs)} negative)")
        return pairs
    
    def _sample_positive_pairs_from_gt(self, num_pairs: int, level_distribution: Dict[str, float]) -> List[Dict[str, Any]]:
        """Sample positive pairs from ground truth matrices"""
        pairs = []
        
        for level, ratio in level_distribution.items():
            if level not in self.ground_truth_matrices:
                continue
                
            num_level_pairs = int(num_pairs * ratio)
            if num_level_pairs == 0:
                continue
            
            matrix = self.ground_truth_matrices[level]
            csv_list = self.csv_lists[level]
            
            # Get positive pairs (non-zero entries in matrix)
            coo_matrix = matrix.tocoo()
            positive_indices = list(zip(coo_matrix.row, coo_matrix.col))
            
            if len(positive_indices) == 0:
                continue
            
            # Sample random positive pairs
            sampled_indices = random.sample(positive_indices, 
                                          min(num_level_pairs, len(positive_indices)))
            
            for row_idx, col_idx in sampled_indices:
                csv_a = csv_list[row_idx]
                csv_b = csv_list[col_idx]
                
                if csv_a != csv_b:  # Avoid self-pairs
                    pairs.append({
                        "table_a_path": csv_a,
                        "table_b_path": csv_b,
                        "ground_truth_level": level,
                        "is_positive": True,
                        "relation_strength": 1.0  # Boolean matrix
                    })
        
        return pairs
    
    def _sample_negative_pairs_from_gt(self, num_pairs: int) -> List[Dict[str, Any]]:
        """Sample negative pairs (not in any ground truth)"""
        pairs = []
        
        # Get all CSV files from ground truth lists
        all_csvs = set()
        for csv_list in self.csv_lists.values():
            all_csvs.update(csv_list)
        
        if len(all_csvs) < 2:
            return pairs
        
        all_csvs = list(all_csvs)
        
        for _ in range(num_pairs):
            csv_a, csv_b = random.sample(all_csvs, 2)
            
            # Check if this pair is actually negative (not in any ground truth)
            if self._is_negative_pair(csv_a, csv_b):
                pairs.append({
                    "table_a_path": csv_a,
                    "table_b_path": csv_b,
                    "ground_truth_level": "negative",
                    "is_positive": False,
                    "relation_strength": 0.0
                })
        
        return pairs
    
    def _is_negative_pair(self, csv_a: str, csv_b: str) -> bool:
        """Check if a pair is truly negative (not in any ground truth)"""
        for level, csv_list in self.csv_lists.items():
            if csv_a in csv_list and csv_b in csv_list:
                idx_a = csv_list.index(csv_a)
                idx_b = csv_list.index(csv_b)
                
                matrix = self.ground_truth_matrices[level]
                if matrix[idx_a, idx_b] > 0:
                    return False
        
        return True
    
    def sample_model_pairs_with_metadata(self, num_pairs: int = 50) -> List[Dict[str, Any]]:
        """Sample model pairs based on model metadata (like modelcard_matrix.py)"""
        print(f"Sampling {num_pairs} model pairs from metadata...")
        
        if self.model_metadata is None:
            print("  ✗ No model metadata available")
            return []
        
        # Apply the same filtering as in modelcard_matrix.py
        # Keep rows with at least one base/model link
        filtered_models = self.model_metadata[
            self.model_metadata.apply(
                lambda r: bool(r.get('tag_base_model_list', []) or r.get('readme_modelid_list', [])), 
                axis=1
            )
        ]
        
        if len(filtered_models) < 2:
            print("  ✗ Not enough models with base/model relationships")
            return []
        
        pairs = []
        model_ids = filtered_models['modelId'].tolist()
        
        for i in range(num_pairs):
            model_a_id, model_b_id = random.sample(model_ids, 2)
            
            model_a_row = filtered_models[filtered_models['modelId'] == model_a_id].iloc[0]
            model_b_row = filtered_models[filtered_models['modelId'] == model_b_id].iloc[0]
            
            pairs.append({
                "id": f"model-pair-{i+1}",
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "model_a_metadata": self._extract_model_metadata(model_a_row),
                "model_b_metadata": self._extract_model_metadata(model_b_row),
                "evaluation_type": "model_relatedness"
            })
        
        print(f"  ✓ Generated {len(pairs)} model pairs")
        return pairs
    
    def _extract_model_metadata(self, model_row: pd.Series) -> Dict[str, Any]:
        """Extract relevant metadata from model row"""
        metadata = {}
        
        # Basic info
        metadata['modelId'] = model_row.get('modelId', '')
        metadata['tags'] = model_row.get('tags', [])
        metadata['pipeline_tag'] = model_row.get('pipeline_tag', '')
        metadata['library_name'] = model_row.get('library_name', '')
        
        # Downloads and likes
        metadata['downloads'] = model_row.get('downloads', 0)
        metadata['likes'] = model_row.get('likes', 0)
        
        # Card content (first 500 chars)
        card_content = model_row.get('card', '')
        if isinstance(card_content, str):
            metadata['card_content'] = card_content[:500] + "..." if len(card_content) > 500 else card_content
        else:
            metadata['card_content'] = str(card_content)[:500]
        
        # Base model relationships
        metadata['tag_base_model_list'] = model_row.get('tag_base_model_list', [])
        metadata['readme_modelid_list'] = model_row.get('readme_modelid_list', [])
        
        return metadata
    
    def convert_to_evaluation_format(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert sampled pairs to evaluation format"""
        evaluation_pairs = []
        
        for pair in pairs:
            if pair["evaluation_type"] == "table_relatedness":
                # Load table content
                try:
                    table_a_md = self._csv_to_markdown(pair["table_a_path"])
                    table_b_md = self._csv_to_markdown(pair["table_b_path"])
                    
                    evaluation_pairs.append({
                        "id": pair["id"],
                        "table_a_md": table_a_md,
                        "table_b_md": table_b_md,
                        "ground_truth_level": pair["ground_truth_level"],
                        "is_positive": pair["is_positive"],
                        "relation_strength": pair["relation_strength"],
                        "table_a_path": pair["table_a_path"],
                        "table_b_path": pair["table_b_path"]
                    })
                except Exception as e:
                    print(f"  ✗ Failed to load tables for {pair['id']}: {e}")
            
            elif pair["evaluation_type"] == "model_relatedness":
                # For model pairs, use metadata
                evaluation_pairs.append({
                    "id": pair["id"],
                    "model_a_id": pair["model_a_id"],
                    "model_b_id": pair["model_b_id"],
                    "model_a_metadata": pair["model_a_metadata"],
                    "model_b_metadata": pair["model_b_metadata"],
                    "evaluation_type": "model_relatedness"
                })
        
        return evaluation_pairs
    
    def _csv_to_markdown(self, csv_path: str, max_rows: int = 10) -> str:
        """Convert CSV to markdown format"""
        try:
            if not os.path.exists(csv_path):
                return f"(File not found: {csv_path})"
            
            df = pd.read_csv(csv_path, nrows=max_rows)
            return df.to_markdown(index=False)
        except Exception as e:
            return f"(Failed to read CSV: {e})"
    
    def save_pairs(self, pairs: List[Dict[str, Any]], output_path: str):
        """Save pairs to JSONL file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved {len(pairs)} pairs to {output_path}")
    
    def generate_sampling_report(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a report on the sampling distribution"""
        report = {
            "total_pairs": len(pairs),
            "evaluation_types": Counter(p["evaluation_type"] for p in pairs),
            "ground_truth_levels": Counter(p.get("ground_truth_level", "unknown") for p in pairs),
            "positive_negative": Counter(p.get("is_positive", False) for p in pairs),
            "table_sources": Counter(),
            "file_existence": Counter()
        }
        
        # Count table sources and file existence
        for pair in pairs:
            if "table_a_path" in pair:
                path_a = pair["table_a_path"]
                path_b = pair["table_b_path"]
                
                # Extract source from path
                for path in [path_a, path_b]:
                    if "hugging" in path:
                        report["table_sources"]["hugging"] += 1
                    elif "github" in path:
                        report["table_sources"]["github"] += 1
                    elif "html" in path or "tables_output" in path:
                        report["table_sources"]["html"] += 1
                    elif "llm" in path:
                        report["table_sources"]["llm"] += 1
                    else:
                        report["table_sources"]["unknown"] += 1
                    
                    # Check file existence
                    report["file_existence"][os.path.exists(path)] += 1
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Improved smart sampling for GPT evaluation")
    parser.add_argument("--output", default="output/improved_evaluation_pairs.jsonl", 
                       help="Output file path")
    parser.add_argument("--num-table-pairs", type=int, default=100,
                       help="Number of table pairs to sample")
    parser.add_argument("--num-model-pairs", type=int, default=50,
                       help="Number of model pairs to sample")
    parser.add_argument("--positive-ratio", type=float, default=0.5,
                       help="Ratio of positive pairs")
    parser.add_argument("--gt-dir", default=GT_DIR,
                       help="Ground truth directory")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR,
                       help="Processed data directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize sampler
    sampler = ImprovedSmartSampler(args.gt_dir, args.processed_dir)
    
    # Load data
    sampler.load_ground_truth()
    sampler.load_model_data()
    sampler.apply_ground_truth_filters()
    
    # Sample pairs
    all_pairs = []
    
    # Sample table pairs
    table_pairs = sampler.sample_table_pairs_with_ground_truth(
        num_pairs=args.num_table_pairs,
        positive_ratio=args.positive_ratio
    )
    all_pairs.extend(table_pairs)
    
    # Sample model pairs
    model_pairs = sampler.sample_model_pairs_with_metadata(num_pairs=args.num_model_pairs)
    all_pairs.extend(model_pairs)
    
    # Convert to evaluation format
    evaluation_pairs = sampler.convert_to_evaluation_format(all_pairs)
    
    # Save pairs
    sampler.save_pairs(evaluation_pairs, args.output)
    
    # Generate and print report
    report = sampler.generate_sampling_report(evaluation_pairs)
    print("\n" + "="*50)
    print("IMPROVED SAMPLING REPORT")
    print("="*50)
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Evaluation types: {dict(report['evaluation_types'])}")
    print(f"Ground truth levels: {dict(report['ground_truth_levels'])}")
    print(f"Positive/Negative: {dict(report['positive_negative'])}")
    print(f"Table sources: {dict(report['table_sources'])}")
    print(f"File existence: {dict(report['file_existence'])}")


if __name__ == "__main__":
    main()
