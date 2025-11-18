#!/usr/bin/env python3
"""
Step 2: Batch evaluation with 5 different LLM models

Loads table pairs from Step 1, queries 5 models for each pair,
and saves results for analysis.

Usage:
    python src/gpt_evaluation/step2_batch_multi_model.py \
        --input output/gpt_evaluation/table_v2_all_levels_pairs.jsonl \
        --output output/gpt_evaluation/step2_results \
        --models gpt-4o-mini,gpt-3.5-turbo,llama3,mistral,gemma:2b \
        --limit 10
"""

import os
import json
import yaml
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.gpt_evaluation.multi_llm_handler import MultiLLMHandler


def csv_to_raw_text(csv_path: str, max_rows: int = 50) -> str:
    """Load CSV content as raw text (first N rows)"""
    try:
        if not os.path.exists(csv_path):
            return f"File not found: {csv_path}"
        
        df = pd.read_csv(csv_path, nrows=max_rows)
        if df.empty:
            return f"Empty table: {os.path.basename(csv_path)}"
        
        # Return as CSV string
        return df.to_csv(index=False)
    except Exception as e:
        return f"Error reading {os.path.basename(csv_path)}: {str(e)}"


def build_prompt(table_a: str, table_b: str) -> str:
    """Build the evaluation prompt with structural and level signals"""
    return f"""You are evaluating whether two data tables are semantically related.

Table A:
{table_a}

Table B:
{table_b}

Task: Determine if Tables A and B are related (YES/NO/UNSURE) and identify specific signals.

Consider STRUCTURAL signals (select ALL that apply):
- JOINABLE: share common column names that could be used to join the tables
- UNIONABLE: have similar schema/structure that could be combined vertically
- KEYWORD_OVERLAP: share common keywords or domain-specific terms
- SEMANTICALLY_SIMILAR: have related meaning or serve similar purposes

Consider LEVEL signals (select ALL that apply):
- PAPER_LEVEL: related because they are from the same research paper
- MODEL_LEVEL: related because they are about the same model(s) or training configuration
- DATASET_LEVEL: related because they use the same dataset(s) or evaluation data

Respond in YAML format (no markdown code fences):
related: [YES/NO/UNSURE]
structural_signals:
  joinable: [true/false]
  unionable: [true/false]
  keyword_overlap: [true/false]
  semantically_similar: [true/false]
level_signals:
  paper_level: [true/false]
  model_level: [true/false]
  dataset_level: [true/false]
rationale: "[1-2 sentences explaining your judgment with specific evidence]"
"""


def parse_yaml_response(response_text: str) -> Dict[str, Any]:
    """Parse YAML response from LLM with structural and level signals"""
    try:
        # Remove markdown code fences if present
        text = response_text.strip()
        if '```yaml' in text:
            text = text.split('```yaml')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        # Try parsing as YAML
        parsed = yaml.safe_load(text)
        if parsed is None:
            parsed = {}
        
        # Extract structural signals
        structural = parsed.get('structural_signals', {})
        level = parsed.get('level_signals', {})
        
        # Ensure required fields
        result = {
            'related': parsed.get('related', 'UNSURE'),
            'structural_signals': {
                'joinable': _parse_bool(structural.get('joinable', False)),
                'unionable': _parse_bool(structural.get('unionable', False)),
                'keyword_overlap': _parse_bool(structural.get('keyword_overlap', False)),
                'semantically_similar': _parse_bool(structural.get('semantically_similar', False))
            },
            'level_signals': {
                'paper_level': _parse_bool(level.get('paper_level', False)),
                'model_level': _parse_bool(level.get('model_level', False)),
                'dataset_level': _parse_bool(level.get('dataset_level', False))
            },
            'rationale': parsed.get('rationale', ''),
            'confidence': parsed.get('confidence', None),
            'raw_response': response_text,
            'parsed': True
        }
        
        return result
        
    except Exception as e:
        return {
            'related': 'UNSURE',
            'structural_signals': {},
            'level_signals': {},
            'rationale': f'Parse error: {str(e)}',
            'raw_response': response_text,
            'parsed': False,
            'error': str(e)
        }


def _parse_bool(value: Any) -> bool:
    """Parse boolean value from various formats"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ['true', 'yes', '1', 'y']
    return bool(value)


def load_table_pair(pair: Dict[str, Any], tables_base_dir: str = "data/scilake_tables") -> Dict[str, str]:
    """Load actual table content for a pair"""
    csv_a = pair.get('csv_a', '')
    csv_b = pair.get('csv_b', '')
    
    # Construct paths
    csv_a_path = os.path.join(tables_base_dir, csv_a)
    csv_b_path = os.path.join(tables_base_dir, csv_b)
    
    # Load content
    table_a = csv_to_raw_text(csv_a_path)
    table_b = csv_to_raw_text(csv_b_path)
    
    return {
        'table_a': table_a,
        'table_b': table_b,
        'table_a_path': csv_a_path,
        'table_b_path': csv_b_path
    }


def evaluate_pair_with_models(pair: Dict[str, Any],
                              model_names: List[str],
                              handler: MultiLLMHandler,
                              tables_base_dir: str = "data/scilake_tables",
                              verbose: bool = False) -> Dict[str, Any]:
    """Evaluate a single pair with multiple models"""
    
    # Load table content
    if verbose:
        print(f"Loading content for pair {pair.get('id', 'unknown')}...")
    
    table_content = load_table_pair(pair, tables_base_dir)
    
    # Build prompt
    prompt = build_prompt(table_content['table_a'], table_content['table_b'])
    
    # Query all models
    if verbose:
        print(f"Querying {len(model_names)} models...")
    
    model_responses = handler.query_batch(prompt, model_names, verbose=verbose)
    
    # Parse responses
    parsed_responses = {}
    for model_name, raw_response in model_responses.items():
        if raw_response['status'] == 'success':
            parsed = parse_yaml_response(raw_response['response'])
            parsed['model'] = model_name
            parsed['provider'] = raw_response['provider']
            parsed['elapsed_time'] = raw_response['elapsed_time']
        else:
            parsed = {
                'model': model_name,
                'provider': raw_response['provider'],
                'error': raw_response.get('error', 'Unknown error'),
                'status': 'error'
            }
        parsed_responses[model_name] = parsed
    
    # Aggregate results
    result = {
        'pair_id': pair.get('id', 'unknown'),
        'csv_a': pair.get('csv_a', ''),
        'csv_b': pair.get('csv_b', ''),
        'level': pair.get('level', ''),
        'gt_positive': pair.get('is_positive', None),
        'gt_labels': pair.get('labels', {}),
        'model_responses': parsed_responses,
        'table_a_path': table_content['table_a_path'],
        'table_b_path': table_content['table_b_path'],
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Compute majority vote
    votes = []
    for model_name, response in parsed_responses.items():
        if 'error' not in response and 'related' in response:
            votes.append(response['related'])
    
    if votes:
        from collections import Counter
        vote_counts = Counter(votes)
        majority = vote_counts.most_common(1)[0][0]
        result['majority_vote'] = majority
        result['vote_distribution'] = dict(vote_counts)
        result['agreement'] = max(vote_counts.values()) / len(votes)
    
    return result


def save_results(results: List[Dict[str, Any]], output_dir: str):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all results
    all_results_file = os.path.join(output_dir, 'all_model_responses.jsonl')
    with open(all_results_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(results)} results to {all_results_file}")
    
    # Save summary
    summary = {
        'total_pairs': len(results),
        'pairs_evaluated': len([r for r in results if r.get('model_responses')]),
        'models_queried': list(set([
            model_name 
            for r in results 
            for model_name in r.get('model_responses', {}).keys()
        ]))
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation with multiple LLM models")
    
    parser.add_argument("--input", required=True,
                       help="Input pairs JSONL from Step 1")
    parser.add_argument("--output", required=True,
                       help="Output directory for results")
    parser.add_argument("--models", 
                       default="gpt-4o-mini,gpt-3.5-turbo,llama3,mistral,gemma:2b",
                       help="Comma-separated list of model names")
    parser.add_argument("--tables-dir", default="data/scilake_tables",
                       help="Directory containing CSV tables")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of pairs (0=all)")
    parser.add_argument("--start-from", type=int, default=0,
                       help="Start from pair index")
    parser.add_argument("--verbose", action='store_true',
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Parse model names
    model_names = [m.strip() for m in args.models.split(',')]
    
    print(f"{'='*60}")
    print(f"Step 2: Batch Multi-Model Evaluation")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Models: {', '.join(model_names)}")
    if args.limit > 0:
        print(f"Limit: {args.limit} pairs")
    print(f"{'='*60}")
    
    # Load pairs
    print(f"\nLoading pairs from {args.input}...")
    pairs = []
    with open(args.input, 'r') as f:
        for line in f:
            try:
                pairs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    print(f"✓ Loaded {len(pairs)} pairs")
    
    # Apply limits
    if args.limit > 0:
        pairs = pairs[args.start_from:args.start_from + args.limit]
        print(f"Processing pairs {args.start_from} to {args.start_from + len(pairs)}")
    
    # Initialize handler
    handler = MultiLLMHandler()
    
    # Validate models
    available = handler.get_available_models()
    invalid = [m for m in model_names if m not in available]
    if invalid:
        print(f"⚠ Warning: Invalid models: {invalid}")
        print(f"Available: {available}")
        model_names = [m for m in model_names if m in available]
        print(f"Using: {model_names}")
    
    if not model_names:
        print("❌ No valid models to query")
        return
    
    # Evaluate pairs
    print(f"\nEvaluating {len(pairs)} pairs with {len(model_names)} models...")
    results = []
    
    for idx, pair in enumerate(pairs):
        if args.verbose or idx % 10 == 0:
            print(f"\n[{idx+1}/{len(pairs)}] Evaluating pair {pair.get('id', 'unknown')}")
        
        try:
            result = evaluate_pair_with_models(
                pair, model_names, handler, 
                tables_base_dir=args.tables_dir,
                verbose=args.verbose
            )
            results.append(result)
            
            if args.verbose:
                if 'majority_vote' in result:
                    print(f"  Majority: {result['majority_vote']}, Agreement: {result.get('agreement', 0):.2f}")
        
        except Exception as e:
            print(f"❌ Error evaluating pair {idx+1}: {e}")
            continue
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    save_results(results, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Pairs evaluated: {len(results)}")
    
    if results:
        votes = [r.get('majority_vote', 'N/A') for r in results]
        from collections import Counter
        vote_counts = Counter(votes)
        print(f"\nMajority Vote Distribution:")
        for vote, count in vote_counts.most_common():
            print(f"  {vote}: {count} ({100*count/len(results):.1f}%)")


if __name__ == "__main__":
    main()

