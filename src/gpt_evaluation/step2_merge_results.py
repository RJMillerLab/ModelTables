#!/usr/bin/env python3
"""
Merge results from different model runs into a single JSONL file
"""
import json
import argparse
from collections import defaultdict


def load_results(jsonl_path):
    """Load results from JSONL file"""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def merge_results(main_path, additional_paths, output_path):
    """
    Merge additional model responses into main results
    
    Args:
        main_path: Path to main results (with 4 models)
        additional_paths: List of paths to additional results (e.g., gemini only)
        output_path: Where to save merged results
    """
    print(f"Loading main results from {main_path}...")
    main_results = load_results(main_path)
    print(f"  ✓ Loaded {len(main_results)} pairs")
    
    # Create lookup by (csv_a, csv_b) as key (more reliable than pair_id)
    main_dict = {}
    for r in main_results:
        key = (r.get('csv_a', ''), r.get('csv_b', ''))
        if key[0] and key[1]:
            main_dict[key] = r
    
    # Merge additional results
    for add_path in additional_paths:
        print(f"\nLoading additional results from {add_path}...")
        add_results = load_results(add_path)
        print(f"  ✓ Loaded {len(add_results)} pairs")
        
        merged_count = 0
        for add_result in add_results:
            key = (add_result.get('csv_a', ''), add_result.get('csv_b', ''))
            if key in main_dict and key[0] and key[1]:
                # Merge model_responses
                main_models = main_dict[key].get('model_responses', {})
                add_models = add_result.get('model_responses', {})
                
                # Add new models
                for model, response in add_models.items():
                    if model not in main_models:
                        main_models[model] = response
                        merged_count += 1
                
                main_dict[key]['model_responses'] = main_models
            else:
                if key[0] and key[1]:
                    print(f"  Warning: Pair {key[0]}+{key[1]} not found in main results")
        
        print(f"  ✓ Merged {merged_count} new model responses")
    
    # Convert back to list and save
    merged_results = list(main_dict.values())
    
    print(f"\nSaving merged results to {output_path}...")
    with open(output_path, 'w') as f:
        for result in merged_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(merged_results)} merged pairs")
    
    # Print summary
    print("\n" + "="*60)
    print("MERGED RESULTS SUMMARY")
    print("="*60)
    
    models = set()
    for r in merged_results:
        models.update(r.get('model_responses', {}).keys())
    
    print(f"Total pairs: {len(merged_results)}")
    print(f"Total models: {len(models)}")
    print(f"Models: {', '.join(sorted(models))}")
    
    # Count per-model successes
    print("\nPer-model response counts:")
    for model in sorted(models):
        count = sum(1 for r in merged_results if model in r.get('model_responses', {}))
        print(f"  {model}: {count}/{len(merged_results)} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True, help="Main results JSONL (with existing models)")
    parser.add_argument("--additional", nargs='+', help="Additional results to merge")
    parser.add_argument("--output", required=True, help="Output path for merged results")
    args = parser.parse_args()
    
    merge_results(args.main, args.additional, args.output)

