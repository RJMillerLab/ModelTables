#!/usr/bin/env python3
"""
Test new simplified prompt with structural and level signals
"""

import os
import json
import yaml
import argparse
from src.llm.model import LLM_response


def build_table_prompt(table_a_raw: str, table_b_raw: str) -> str:
    """Build the new simplified prompt with structural and level signals"""
    return f"""You are evaluating whether two data tables are semantically related.

Table A:
{table_a_raw}

Table B:
{table_b_raw}

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


def parse_yaml_response(response_text: str):
    """Parse YAML response with signal extraction"""
    try:
        # Remove markdown fences
        text = response_text.strip()
        if '```yaml' in text:
            text = text.split('```yaml')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        parsed = yaml.safe_load(text)
        if parsed is None:
            parsed = {}
        
        # Extract signals
        structural = parsed.get('structural_signals', {})
        level = parsed.get('level_signals', {})
        
        def parse_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ['true', 'yes', '1', 'y']
            return bool(v)
        
        result = {
            'related': parsed.get('related', 'UNSURE'),
            'structural_signals': {
                'joinable': parse_bool(structural.get('joinable', False)),
                'unionable': parse_bool(structural.get('unionable', False)),
                'keyword_overlap': parse_bool(structural.get('keyword_overlap', False)),
                'semantically_similar': parse_bool(structural.get('semantically_similar', False))
            },
            'level_signals': {
                'paper_level': parse_bool(level.get('paper_level', False)),
                'model_level': parse_bool(level.get('model_level', False)),
                'dataset_level': parse_bool(level.get('dataset_level', False))
            },
            'rationale': parsed.get('rationale', ''),
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


def test_with_demo_data(llm_model: str = "gpt-4o-mini"):
    """Test with sample table data"""
    print("="*80)
    print("TESTING NEW PROMPT WITH DEMO DATA")
    print("="*80)
    
    # Sample table data
    table_a = """model,accuracy,f1_score
bert-base,0.85,0.82
roberta-base,0.87,0.84
bert-large,0.89,0.86
"""
    
    table_b = """model,score
bert-base,0.83
roberta-base,0.86
bert-large,0.88
"""
    
    print("\nTable A:")
    print(table_a)
    print("\nTable B:")
    print(table_b)
    
    # Build prompt
    prompt = build_table_prompt(table_a, table_b)
    print("\n" + "="*80)
    print("PROMPT:")
    print("="*80)
    print(prompt)
    
    # Call LLM
    print("\n" + "="*80)
    print(f"CALLING {llm_model.upper()}...")
    print("="*80)
    
    try:
        response, _ = LLM_response(prompt, llm_model=llm_model, history=[], kwargs={}, max_tokens=1000)
        
        print("\n" + "="*80)
        print("RAW RESPONSE:")
        print("="*80)
        print(response)
        
        # Parse response
        parsed = parse_yaml_response(response)
        
        print("\n" + "="*80)
        print("PARSED RESULT:")
        print("="*80)
        print(yaml.dump(parsed, default_flow_style=False))
        
        # Validation
        print("\n" + "="*80)
        print("VALIDATION:")
        print("="*80)
        print(f"✓ Related: {parsed.get('related')}")
        print(f"✓ Parsed successfully: {parsed.get('parsed', False)}")
        
        if parsed.get('parsed'):
            struct = parsed.get('structural_signals', {})
            level = parsed.get('level_signals', {})
            
            print(f"\nStructural Signals:")
            for k, v in struct.items():
                print(f"  - {k}: {v}")
            
            print(f"\nLevel Signals:")
            for k, v in level.items():
                print(f"  - {k}: {v}")
            
            print(f"\nRationale: {parsed.get('rationale', 'N/A')[:100]}...")
        else:
            print(f"✗ Parse error: {parsed.get('error', 'Unknown')}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


def test_with_actual_pair(pairs_file: str, pair_index: int = 0, llm_model: str = "gpt-4o-mini"):
    """Test with actual pair from step1 output"""
    print("="*80)
    print("TESTING WITH ACTUAL PAIR")
    print("="*80)
    
    # Load pairs
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            try:
                pairs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if pair_index >= len(pairs):
        print(f"✗ Pair index {pair_index} out of range (max: {len(pairs)-1})")
        return
    
    pair = pairs[pair_index]
    print(f"\nPair ID: {pair.get('id', 'unknown')}")
    print(f"CSV A: {pair.get('csv_a', 'N/A')}")
    print(f"CSV B: {pair.get('csv_b', 'N/A')}")
    print(f"Level: {pair.get('level', 'N/A')}")
    print(f"GT Positive: {pair.get('is_positive', 'N/A')}")
    
    # Get table content (simplified - just show filenames for now)
    print("\n[Note: Would load actual CSV content here]")
    
    # Build prompt with placeholder
    table_a = "[Would load CSV A content here]"
    table_b = "[Would load CSV B content here]"
    
    prompt = build_table_prompt(table_a, table_b)
    
    print("\n" + "="*80)
    print(f"CALLING {llm_model.upper()}...")
    print("="*80)
    
    try:
        response, _ = LLM_response(prompt, llm_model=llm_model, history=[], kwargs={}, max_tokens=1000)
        
        print("\nRAW RESPONSE:")
        print("-"*80)
        print(response[:500] + "..." if len(response) > 500 else response)
        
        # Parse
        parsed = parse_yaml_response(response)
        
        print("\n" + "="*80)
        print("PARSED RESULT:")
        print("="*80)
        print(yaml.dump(parsed, default_flow_style=False))
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test new prompt with structural and level signals")
    parser.add_argument("--mode", choices=["demo", "pair"], default="demo",
                       help="Test mode: demo (built-in data) or pair (from step1)")
    parser.add_argument("--pairs", default="output/gpt_evaluation/table_v2_all_levels_pairs.jsonl",
                       help="Pairs file for --mode pair")
    parser.add_argument("--index", type=int, default=0,
                       help="Pair index (for --mode pair)")
    parser.add_argument("--llm", default="gpt-4o-mini",
                       help="LLM model to use")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        test_with_demo_data(llm_model=args.llm)
    else:
        test_with_actual_pair(args.pairs, args.index, llm_model=args.llm)


if __name__ == "__main__":
    main()

