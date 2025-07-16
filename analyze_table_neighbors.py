#!/usr/bin/env python3
"""
Simple script to analyze key distribution in table neighbor JSON files
Usage: python analyze_table_neighbors.py --input <json_file>
"""

import json
import argparse
from collections import Counter
import os

def analyze_key_distribution(data):
    """Analyze key distribution patterns"""
    print("=" * 50)
    print("KEY DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Count different key types
    key_types = {
        'original': 0,      # No suffix
        'str_augmented': 0, # _s suffix
        'tr_augmented': 0,  # _t suffix
        'mixed_augmented': 0 # _s_t suffix
    }
    
    key_examples = {
        'original': [],
        'str_augmented': [],
        'tr_augmented': [],
        'mixed_augmented': []
    }
    
    for key in data.keys():
        base_key = key.replace('.csv', '')
        
        if base_key.endswith('_s_t'):
            key_types['mixed_augmented'] += 1
            if len(key_examples['mixed_augmented']) < 3:
                key_examples['mixed_augmented'].append(key)
        elif base_key.endswith('_s'):
            key_types['str_augmented'] += 1
            if len(key_examples['str_augmented']) < 3:
                key_examples['str_augmented'].append(key)
        elif base_key.endswith('_t'):
            key_types['tr_augmented'] += 1
            if len(key_examples['tr_augmented']) < 3:
                key_examples['tr_augmented'].append(key)
        else:
            key_types['original'] += 1
            if len(key_examples['original']) < 3:
                key_examples['original'].append(key)
    
    # Print statistics
    total = len(data)
    print(f"Total keys: {total}")
    print(f"Original tables: {key_types['original']} ({key_types['original']/total*100:.1f}%)")
    print(f"String augmented: {key_types['str_augmented']} ({key_types['str_augmented']/total*100:.1f}%)")
    print(f"Transpose augmented: {key_types['tr_augmented']} ({key_types['tr_augmented']/total*100:.1f}%)")
    print(f"Mixed augmented: {key_types['mixed_augmented']} ({key_types['mixed_augmented']/total*100:.1f}%)")
    
    # Print examples
    print("\nKEY EXAMPLES:")
    for key_type, examples in key_examples.items():
        if examples:
            print(f"{key_type}: {', '.join(examples)}")
    
    # Analyze neighbor count distribution
    neighbor_counts = [len(values) for values in data.values()]
    neighbor_dist = Counter(neighbor_counts)
    
    print(f"\nNEIGHBOR COUNT DISTRIBUTION:")
    print(f"Average neighbors: {sum(neighbor_counts)/len(neighbor_counts):.1f}")
    for count, freq in sorted(neighbor_dist.items()):
        print(f"  {count} neighbors: {freq} keys ({freq/total*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze key distribution in table neighbor JSON files")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found {args.input}")
        return
    
    print(f"Analyzing: {args.input}")
    
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    analyze_key_distribution(data)

if __name__ == "__main__":
    main() 