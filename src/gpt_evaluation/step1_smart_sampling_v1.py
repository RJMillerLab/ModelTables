#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-14
Description: Smart sampling strategy for GPT evaluation covering both model and table relatedness
             with balanced positive/negative examples across different levels and attributes.

This script implements a comprehensive sampling strategy that:
1. Covers both Model Relatedness and Table Relatedness
2. Includes different levels (direct, max_pr, model, dataset)
3. Balances positive and negative examples
4. Considers different table sources (hugging, github, html, llm)
5. Extracts attributes for better sampling
6. Ensures diverse and representative samples
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

# Different levels and their corresponding files
LEVELS = {
    "direct": "csv_pair_matrix_direct_label.npz",
    "direct_influential": "csv_pair_matrix_direct_label_influential.npz", 
    "direct_methodology_or_result": "csv_pair_matrix_direct_label_methodology_or_result.npz",
    "direct_methodology_or_result_influential": "csv_pair_matrix_direct_label_methodology_or_result_influential.npz",
    "max_pr": "csv_pair_matrix_max_pr.npz",
    "max_pr_influential": "csv_pair_matrix_max_pr_influential.npz",
    "max_pr_methodology_or_result": "csv_pair_matrix_max_pr_methodology_or_result.npz",
    "max_pr_methodology_or_result_influential": "csv_pair_matrix_max_pr_methodology_or_result_influential.npz",
    "model": "scilake_gt_modellink_model_adj_processed.npz",
    "dataset": "scilake_gt_modellink_dataset_adj_processed.npz"
}

# Table sources
TABLE_SOURCES = ["hugging", "github", "html", "llm"]

class SmartSampler:
    def __init__(self, gt_dir: str = GT_DIR, processed_dir: str = PROCESSED_DIR):
        self.gt_dir = gt_dir
        self.processed_dir = processed_dir
        self.ground_truth_matrices = {}
        self.csv_lists = {}
        self.model_data = None
        self.table_attributes = {}
        
    def load_ground_truth(self):
        """Load all ground truth matrices and CSV lists"""
        print("Loading ground truth matrices...")
        
        for level, filename in LEVELS.items():
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
        """Load model metadata and table mappings"""
        print("Loading model data...")
        
        model_file = os.path.join(self.processed_dir, "modelcard_step3_dedup.parquet")
        if os.path.exists(model_file):
            self.model_data = pd.read_parquet(model_file)
            print(f"  ✓ Loaded {len(self.model_data)} models")
        else:
            print(f"  ✗ Missing model file: {model_file}")
    
    def extract_table_attributes(self):
        """Extract attributes from table files for better sampling"""
        print("Extracting table attributes...")
        
        # Get all CSV files from different sources
        all_csvs = set()
        
        # From model data
        if self.model_data is not None:
            for source in TABLE_SOURCES:
                col_name = f"{source}_table_list_dedup"
                if col_name in self.model_data.columns:
                    for table_list in self.model_data[col_name]:
                        if isinstance(table_list, list):
                            all_csvs.update(table_list)
        
        # From ground truth CSV lists
        for csv_list in self.csv_lists.values():
            all_csvs.update(csv_list)
        
        print(f"  Found {len(all_csvs)} unique CSV files")
        
        # Extract attributes from CSV paths
        for csv_path in all_csvs:
            if not os.path.exists(csv_path):
                continue
                
            attributes = self._extract_csv_attributes(csv_path)
            self.table_attributes[csv_path] = attributes
        
        print(f"  ✓ Extracted attributes for {len(self.table_attributes)} tables")
    
    def _extract_csv_attributes(self, csv_path: str) -> Dict[str, Any]:
        """Extract attributes from CSV file path and content"""
        attributes = {
            "source": "unknown",
            "domain": "unknown", 
            "table_type": "unknown",
            "size": 0,
            "columns": 0,
            "rows": 0
        }
        
        # Extract source from path
        if "hugging" in csv_path:
            attributes["source"] = "hugging"
        elif "github" in csv_path:
            attributes["source"] = "github"
        elif "html" in csv_path or "tables_output" in csv_path:
            attributes["source"] = "html"
        elif "llm" in csv_path:
            attributes["source"] = "llm"
        
        # Extract domain/model info from filename
        filename = os.path.basename(csv_path)
        if "_table" in filename:
            base_name = filename.split("_table")[0]
            attributes["domain"] = base_name
        
        # Try to read CSV to get size info
        try:
            df = pd.read_csv(csv_path, nrows=5)  # Only read first 5 rows for efficiency
            attributes["columns"] = len(df.columns)
            attributes["rows"] = len(df)
            
            # Try to infer table type from column names
            col_names = [str(col).lower() for col in df.columns]
            if any("loss" in col for col in col_names):
                attributes["table_type"] = "training_metrics"
            elif any("accuracy" in col or "precision" in col or "recall" in col for col in col_names):
                attributes["table_type"] = "evaluation_metrics"
            elif any("benchmark" in col for col in col_names):
                attributes["table_type"] = "benchmark_results"
            elif any("model" in col for col in col_names):
                attributes["table_type"] = "model_comparison"
            else:
                attributes["table_type"] = "general"
                
        except Exception:
            pass
        
        return attributes
    
    def sample_table_pairs(self, 
                          num_pairs: int = 100,
                          positive_ratio: float = 0.5,
                          level_distribution: Dict[str, float] = None,
                          source_distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Sample table pairs with balanced positive/negative examples"""
        
        if level_distribution is None:
            level_distribution = {
                "direct": 0.3,
                "max_pr": 0.2, 
                "model": 0.25,
                "dataset": 0.25
            }
        
        if source_distribution is None:
            source_distribution = {
                "hugging": 0.3,
                "github": 0.2,
                "html": 0.3,
                "llm": 0.2
            }
        
        print(f"Sampling {num_pairs} table pairs...")
        print(f"  Positive ratio: {positive_ratio}")
        print(f"  Level distribution: {level_distribution}")
        print(f"  Source distribution: {source_distribution}")
        
        pairs = []
        num_positive = int(num_pairs * positive_ratio)
        num_negative = num_pairs - num_positive
        
        # Sample positive pairs from ground truth
        positive_pairs = self._sample_positive_pairs(num_positive, level_distribution)
        pairs.extend(positive_pairs)
        
        # Sample negative pairs (not in ground truth)
        negative_pairs = self._sample_negative_pairs(num_negative, source_distribution)
        pairs.extend(negative_pairs)
        
        # Shuffle and add metadata
        random.shuffle(pairs)
        
        for i, pair in enumerate(pairs):
            pair["id"] = f"smart-pair-{i+1}"
            pair["evaluation_type"] = "table_relatedness"
        
        print(f"  ✓ Generated {len(pairs)} pairs ({len(positive_pairs)} positive, {len(negative_pairs)} negative)")
        return pairs
    
    def _sample_positive_pairs(self, num_pairs: int, level_distribution: Dict[str, float]) -> List[Dict[str, Any]]:
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
            positive_indices = np.where(matrix.data)[0]
            
            if len(positive_indices) == 0:
                continue
            
            # Sample random positive pairs
            sampled_indices = random.sample(list(positive_indices), 
                                          min(num_level_pairs, len(positive_indices)))
            
            for idx in sampled_indices:
                row, col = matrix.indices[idx], matrix.indptr[idx]
                csv_a = csv_list[row]
                csv_b = csv_list[col]
                
                if csv_a != csv_b:  # Avoid self-pairs
                    pairs.append({
                        "table_a_path": csv_a,
                        "table_b_path": csv_b,
                        "ground_truth_level": level,
                        "is_positive": True,
                        "relation_strength": matrix.data[idx]
                    })
        
        return pairs
    
    def _sample_negative_pairs(self, num_pairs: int, source_distribution: Dict[str, float]) -> List[Dict[str, Any]]:
        """Sample negative pairs (not in ground truth)"""
        pairs = []
        
        # Get all available CSV files
        all_csvs = list(self.table_attributes.keys())
        
        if len(all_csvs) < 2:
            return pairs
        
        # Group CSVs by source for balanced sampling
        csvs_by_source = defaultdict(list)
        for csv_path in all_csvs:
            source = self.table_attributes[csv_path]["source"]
            csvs_by_source[source].append(csv_path)
        
        for source, ratio in source_distribution.items():
            if source not in csvs_by_source or len(csvs_by_source[source]) < 2:
                continue
            
            num_source_pairs = int(num_pairs * ratio)
            if num_source_pairs == 0:
                continue
            
            source_csvs = csvs_by_source[source]
            
            for _ in range(num_source_pairs):
                csv_a, csv_b = random.sample(source_csvs, 2)
                
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
    
    def sample_model_pairs(self, num_pairs: int = 50) -> List[Dict[str, Any]]:
        """Sample model pairs for model relatedness evaluation"""
        print(f"Sampling {num_pairs} model pairs...")
        
        if self.model_data is None:
            print("  ✗ No model data available")
            return []
        
        pairs = []
        models = self.model_data['modelId'].tolist()
        
        # Sample random model pairs
        for i in range(num_pairs):
            model_a, model_b = random.sample(models, 2)
            
            # Get tables for each model
            model_a_data = self.model_data[self.model_data['modelId'] == model_a].iloc[0]
            model_b_data = self.model_data[self.model_data['modelId'] == model_b].iloc[0]
            
            # Extract table lists
            tables_a = []
            tables_b = []
            
            for source in TABLE_SOURCES:
                col_name = f"{source}_table_list_dedup"
                if col_name in model_a_data:
                    tables_a.extend(model_a_data[col_name] if isinstance(model_a_data[col_name], list) else [])
                if col_name in model_b_data:
                    tables_b.extend(model_b_data[col_name] if isinstance(model_b_data[col_name], list) else [])
            
            if tables_a and tables_b:
                pairs.append({
                    "id": f"model-pair-{i+1}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_tables": tables_a[:5],  # Limit to first 5 tables
                    "model_b_tables": tables_b[:5],
                    "evaluation_type": "model_relatedness"
                })
        
        print(f"  ✓ Generated {len(pairs)} model pairs")
        return pairs
    
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
                        "table_a_attributes": self.table_attributes.get(pair["table_a_path"], {}),
                        "table_b_attributes": self.table_attributes.get(pair["table_b_path"], {})
                    })
                except Exception as e:
                    print(f"  ✗ Failed to load tables for {pair['id']}: {e}")
            
            elif pair["evaluation_type"] == "model_relatedness":
                # For model pairs, we'll use model metadata instead of table content
                evaluation_pairs.append({
                    "id": pair["id"],
                    "model_a": pair["model_a"],
                    "model_b": pair["model_b"],
                    "model_a_tables": pair["model_a_tables"],
                    "model_b_tables": pair["model_b_tables"],
                    "evaluation_type": "model_relatedness"
                })
        
        return evaluation_pairs
    
    def _csv_to_markdown(self, csv_path: str, max_rows: int = 10) -> str:
        """Convert CSV to markdown format"""
        try:
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
            "table_types": Counter()
        }
        
        # Count table sources and types
        for pair in pairs:
            if "table_a_attributes" in pair:
                attrs_a = pair["table_a_attributes"]
                attrs_b = pair["table_b_attributes"]
                
                report["table_sources"][attrs_a.get("source", "unknown")] += 1
                report["table_sources"][attrs_b.get("source", "unknown")] += 1
                
                report["table_types"][attrs_a.get("table_type", "unknown")] += 1
                report["table_types"][attrs_b.get("table_type", "unknown")] += 1
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Smart sampling for GPT evaluation")
    parser.add_argument("--output", default="output/smart_evaluation_pairs.jsonl", 
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
    sampler = SmartSampler(args.gt_dir, args.processed_dir)
    
    # Load data
    sampler.load_ground_truth()
    sampler.load_model_data()
    sampler.extract_table_attributes()
    
    # Sample pairs
    all_pairs = []
    
    # Sample table pairs
    table_pairs = sampler.sample_table_pairs(
        num_pairs=args.num_table_pairs,
        positive_ratio=args.positive_ratio
    )
    all_pairs.extend(table_pairs)
    
    # Sample model pairs
    model_pairs = sampler.sample_model_pairs(num_pairs=args.num_model_pairs)
    all_pairs.extend(model_pairs)
    
    # Convert to evaluation format
    evaluation_pairs = sampler.convert_to_evaluation_format(all_pairs)
    
    # Save pairs
    sampler.save_pairs(evaluation_pairs, args.output)
    
    # Generate and print report
    report = sampler.generate_sampling_report(evaluation_pairs)
    print("\n" + "="*50)
    print("SAMPLING REPORT")
    print("="*50)
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Evaluation types: {dict(report['evaluation_types'])}")
    print(f"Ground truth levels: {dict(report['ground_truth_levels'])}")
    print(f"Positive/Negative: {dict(report['positive_negative'])}")
    print(f"Table sources: {dict(report['table_sources'])}")
    print(f"Table types: {dict(report['table_types'])}")


if __name__ == "__main__":
    main()
