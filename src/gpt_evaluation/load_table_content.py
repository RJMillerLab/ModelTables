#!/usr/bin/env python3
"""
Load CSV content and create proper table pairs for GPT evaluation
"""

import os
import json
import pandas as pd
from typing import Dict, Any

def csv_to_markdown(csv_path: str, max_rows: int = 10) -> str:
    """Convert CSV to markdown table"""
    try:
        if not os.path.exists(csv_path):
            return f"(File not found: {csv_path})"
        
        df = pd.read_csv(csv_path, nrows=max_rows)
        if df.empty:
            return f"(Empty table: {os.path.basename(csv_path)})"
        
        return df.to_markdown(index=False)
    except Exception as e:
        return f"(Error reading {os.path.basename(csv_path)}: {str(e)})"

def load_csv_pairs_with_content(input_file: str, output_file: str):
    """Load CSV pairs and add actual table content"""
    print(f"Loading pairs from {input_file}...")
    
    pairs = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                pair = json.loads(line.strip())
                pairs.append(pair)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(pairs)} pairs")
    
    # Add table content
    print("Loading CSV content...")
    for i, pair in enumerate(pairs):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(pairs)}...")
        
        # Get full paths
        csv_a_path = pair.get('full_path_a', '')
        csv_b_path = pair.get('full_path_b', '')
        
        # Convert to markdown
        table_a_md = csv_to_markdown(csv_a_path)
        table_b_md = csv_to_markdown(csv_b_path)
        
        # Create new pair with table content
        new_pair = {
            "id": f"pair-{i+1}",
            "csv_a": pair['csv_a'],
            "csv_b": pair['csv_b'],
            "resource": pair['resource'],
            "is_related": pair['is_related'],
            "table_a_md": table_a_md,
            "table_b_md": table_b_md,
            "full_path_a": csv_a_path,
            "full_path_b": csv_b_path
        }
        
        pairs[i] = new_pair
    
    # Save with table content
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(pairs)} pairs with table content")

def test_single_pair(input_file: str, pair_index: int = 0):
    """Test a single pair to see the content"""
    with open(input_file, 'r') as f:
        pairs = [json.loads(line) for line in f]
    
    if pair_index >= len(pairs):
        print(f"Pair index {pair_index} out of range (max: {len(pairs)-1})")
        return
    
    pair = pairs[pair_index]
    print(f"=== PAIR {pair_index} ===")
    print(f"CSV A: {pair['csv_a']}")
    print(f"CSV B: {pair['csv_b']}")
    print(f"Resource: {pair['resource']}")
    print(f"GT Related: {pair['is_related']}")
    print(f"\n=== TABLE A ===")
    print(pair.get('table_a_md', 'No table_a_md field'))
    print(f"\n=== TABLE B ===")
    print(pair.get('table_b_md', 'No table_b_md field'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/quick_samples_100.jsonl")
    parser.add_argument("--output", default="output/table_pairs_with_content.jsonl")
    parser.add_argument("--test", type=int, default=-1, help="Test single pair (index)")
    args = parser.parse_args()
    
    if args.test >= 0:
        test_single_pair(args.input, args.test)
    else:
        load_csv_pairs_with_content(args.input, args.output)


