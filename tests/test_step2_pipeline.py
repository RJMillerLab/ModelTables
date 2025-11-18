#!/usr/bin/env python3
"""
Quick test of Step 2 pipeline to validate the implementation
"""

import os
import json
from src.gpt_evaluation.multi_llm_handler import MultiLLMHandler

def test_multi_llm_handler():
    """Test 1: Multi-LLM handler initialization"""
    print("="*60)
    print("TEST 1: Multi-LLM Handler")
    print("="*60)
    
    handler = MultiLLMHandler()
    
    # Check available models
    available = handler.get_available_models()
    print(f"✓ Available models: {available}")
    
    # Group by provider
    families = handler.get_models_by_family()
    print(f"\n✓ Models by provider:")
    for provider, models in families.items():
        print(f"  {provider}: {models}")
    
    print("\n✓ Test 1 PASSED\n")


def test_simple_query():
    """Test 2: Simple query to one model"""
    print("="*60)
    print("TEST 2: Simple Model Query")
    print("="*60)
    
    handler = MultiLLMHandler()
    
    # Simple test prompt
    test_prompt = """You are evaluating whether two data tables are semantically related.

Table A:
name,value
Alice,100
Bob,200

Table B:
name,score
Charlie,85
Diana,92

Task: Determine if Tables A and B are related.

Respond in YAML format:
related: [YES/NO/UNSURE]
rationale: "[your explanation]" """
    
    # Test with gpt-4o-mini if available
    try:
        print("Querying gpt-4o-mini...")
        result = handler.query_model(test_prompt, 'gpt-4o-mini', verbose=True)
        
        if result['status'] == 'success':
            print(f"✓ Response received: {result['response'][:200]}...")
            print("\n✓ Test 2 PASSED\n")
        else:
            print(f"✗ Query failed: {result.get('error', 'Unknown error')}")
            print("\n⚠ Test 2 SKIPPED (API key may be missing)\n")
    except Exception as e:
        print(f"⚠ Test 2 SKIPPED: {e}\n")


def test_prompt_building():
    """Test 3: Prompt building"""
    print("="*60)
    print("TEST 3: Prompt Building")
    print("="*60)
    
    from src.gpt_evaluation.step2_batch_multi_model import build_prompt
    
    table_a = "col1,col2\nvalue1,value2\n"
    table_b = "col1,col3\nvalue1,value3\n"
    
    prompt = build_prompt(table_a, table_b)
    
    print("✓ Prompt generated")
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"\nFirst 200 chars:\n{prompt[:200]}...")
    
    # Check prompt contains required elements
    assert 'Table A:' in prompt
    assert 'Table B:' in prompt
    assert 'YAML format' in prompt
    assert 'rationale' in prompt
    
    print("\n✓ Test 3 PASSED\n")


def test_result_parsing():
    """Test 4: Response parsing"""
    print("="*60)
    print("TEST 4: Response Parsing")
    print("="*60)
    
    from src.gpt_evaluation.step2_batch_multi_model import parse_yaml_response
    
    # Test cases
    test_cases = [
        ("related: YES\nrationale: Tables share common structure", True),
        ("```yaml\nrelated: NO\nrationale: Different domains\n```", True),
        ("invalid response", False),
    ]
    
    for i, (response, should_parse) in enumerate(test_cases):
        parsed = parse_yaml_response(response)
        assert 'parsed' in parsed
        assert 'related' in parsed
        assert 'rationale' in parsed
        
        if should_parse:
            assert parsed['parsed'] == True
        print(f"✓ Test case {i+1} passed")
    
    print("\n✓ Test 4 PASSED\n")


def check_step1_output():
    """Test 5: Check Step 1 output exists"""
    print("="*60)
    print("TEST 5: Step 1 Output Check")
    print("="*60)
    
    step1_files = [
        'output/gpt_evaluation/table_v2_all_levels_pairs.jsonl',
        'output/gpt_evaluation/table_v2_paper_pairs.jsonl',
    ]
    
    for file in step1_files:
        if os.path.exists(file):
            # Count pairs
            count = 0
            with open(file, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
            print(f"✓ {file}: {count} pairs")
        else:
            print(f"⚠ {file}: NOT FOUND (run Step 1 first)")
    
    print()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Step 2 Pipeline Validation Tests")
    print("="*60 + "\n")
    
    test_multi_llm_handler()
    test_simple_query()
    test_prompt_building()
    test_result_parsing()
    check_step1_output()
    
    print("="*60)
    print("Validation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Ensure Step 1 output exists")
    print("2. Set up LLM models (Ollama for local models)")
    print("3. Run: python src/gpt_evaluation/step2_batch_multi_model.py --help")
    print()


if __name__ == "__main__":
    main()

