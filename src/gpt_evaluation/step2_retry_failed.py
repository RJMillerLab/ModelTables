#!/usr/bin/env python3
"""
Retry only failed requests, keep successful ones
"""
import os
import json
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.gpt_evaluation.step2_query_openrouter import (
    OPENROUTER_URL, MODELS, build_table_prompt, 
    load_table_content, query_model, parse_yaml_response
)

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

async def retry_failed(input_file, output_file, tables_dir="data/scilake_tables"):
    """Retry only failed requests"""
    
    print(f"Loading existing results from {input_file}...")
    
    # Load existing results
    results = []
    total_failed = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    print(f"✓ Loaded {len(results)} pairs")
    
    # Count how many need retry
    for result in results:
        for model, response in result['model_responses'].items():
            if 'error' in response:
                total_failed += 1
    
    print(f"✓ Found {total_failed} failed responses to retry\n")
    
    # Retry failed requests
    retry_count = 0
    async with aiohttp.ClientSession() as session:
        with tqdm(total=total_failed, desc="Retrying failed") as pbar:
            for result in results:
                pair_id = result.get('pair_id')
                
                # Check which models failed
                for model_name, response in result['model_responses'].items():
                    if 'error' in response:
                        # Retry this model
                        print(f"  Retrying {model_name} for pair {pair_id}...")
                        
                        # Load table content
                        content = load_table_content(result, tables_dir)
                        prompt = build_table_prompt(content['table_a'], content['table_b'])
                        
                        # Query model
                        model_response = await query_model(session, model_name, prompt)
                        
                        # Parse response
                        if 'error' in model_response:
                            result['model_responses'][model_name] = {
                                'model': model_name,
                                'error': model_response['error'],
                                'status': 'error'
                            }
                        else:
                            parsed = parse_yaml_response(model_response['response'], model_name)
                            result['model_responses'][model_name] = parsed
                        
                        retry_count += 1
                        pbar.update(1)
    
    # Save updated results
    print(f"\nSaving updated results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Retried {retry_count} failed responses")
    
    # Summary
    success_count = 0
    still_failed = 0
    
    for result in results:
        for model, response in result['model_responses'].items():
            if 'error' in response:
                still_failed += 1
            else:
                success_count += 1
    
    print(f"\nFinal status:")
    print(f"  Success: {success_count}")
    print(f"  Still failed: {still_failed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Sequire/gpt_evaluation/step2_openrouter_results_full.jsonl")
    parser.add_argument("--output", default="output/gpt_evaluation/step2_openrouter_results_retry.jsonl")
    parser.add_argument("--tables-dir", default="data/scilake_tables")
    
    args = parser.parse_args()
    asyncio.run(retry_failed(args.input, args.output, args.tables_dir))

