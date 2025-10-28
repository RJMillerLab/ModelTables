#!/usr/bin/env python3
"""
Step 2: Query multiple LLMs via OpenRouter for table relatedness evaluation

Input: pairs from Step 1 (JSONL)
Output: per-model responses for each pair
"""

import os
import json
import yaml
import asyncio
import aiohttp
import argparse
import pandas as pd
from typing import Dict, List, Any
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# OpenRouter configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3-70b-instruct",
    "openai/gpt-3.5-turbo",
]

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("Please set OPENROUTER_API_KEY in your .env file")


def build_table_prompt(table_a: str, table_b: str) -> str:
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
- MODEL_LEVEL: related because they are about the same model(s) or training+p configuration
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


def find_csv_file(csv_filename: str) -> str:
    """Find CSV file in multiple possible directories"""
    # Possible directories where tables are stored
    search_dirs = [
        "data/processed/deduped_hugging_csvs",
        "data/processed/deduped_github_csvs",
        "data/processed/tables_output",
        "data/processed/tables_output_v2",
        "data/processed/llm_tables",
    ]
    
    for dir_path in search_dirs:
        full_path = os.path.join(dir_path, csv_filename)
        if os.path.exists(full_path):
            return full_path
    
    return None


def csv_to_raw_text(csv_path: str, max_rows: int = 50) -> str:
    """Load CSV content as raw text"""
    try:
        # If csv_path is just a filename, try to find it
        if not os.path.dirname(csv_path):
            csv_path = find_csv_file(csv_path)
            if csv_path is None:
                return f"File not found: {csv_path}"
        
        if not os.path.exists(csv_path):
            return f"File not found: {csv_path}"
        
        df = pd.read_csv(csv_path, nrows=max_rows)
        if df.empty:
            return f"Empty table: {os.path.basename(csv_path)}"
        
        return df.to_csv(index=False)
    except Exception as e:
        return f"Error reading {os.path.basename(csv_path)}: {str(e)}"


def load_table_content(pair: Dict[str, Any], tables_base_dir: str = None) -> Dict[str, str]:
    """Load table content for a pair"""
    csv_a = pair.get('csv_a', '')
    csv_b = pair.get('csv_b', '')
    
    # csv_to_raw_text will handle finding the file
    table_a = csv_to_raw_text(csv_a)
    table_b = csv_to_raw_text(csv_b)
    
    return {
        'table_a': table_a,
        'table_b': table_b
    }


async def query_model(session, model, prompt, max_tokens=1000, temperature=0.1):
    """Query a single model via OpenRouter"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"model": model, "error": f"HTTP {resp.status}: {text}"}
            data = await resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"model": model, "response": content}
    except asyncio.TimeoutError:
        return {"model": model, "error": "Timeout"}
    except Exception as e:
        return {"model": model, "error": str(e)}


async def query_all_models(prompt, session):
    """Query all models in parallel"""
    tasks = [query_model(session, model, prompt) for model in MODELS]
    results = await asyncio.gather(*tasks)
    return results


def parse_yaml_response(response_text: str, model_name: str) -> Dict[str, Any]:
    """Parse YAML response from LLM"""
    try:
        text = response_text.strip()
        if '```yaml' in text:
            text = text.split('```yaml')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        parsed = yaml.safe_load(text)
        if parsed is None:
            parsed = {}
        
        structural = parsed.get('structural_signals', {})
        level = parsed.get('level_signals', {})
        
        def parse_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ['true', 'yes', '1', 'y']
            return bool(v)
        
        # Normalize related to YES/NO/UNSURE
        related_value = parsed.get('related', 'UNSURE')
        if isinstance(related_value, bool):
            related_value = 'YES' if related_value else 'NO'
        elif isinstance(related_value, str):
            related_value = related_value.upper()
            if related_value not in ['YES', 'NO', 'UNSURE']:
                related_value = 'UNSURE'
        
        result = {
            'related': related_value,
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
            'parsed': True,
            'model': model_name
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
            'error': str(e),
            'model': model_name
        }


async def process_pairs(input_file: str, output_file: str, tables_dir: str = "data/scilake_tables", limit: int = 0):
    """Process pairs and query all models"""
    
    # Load pairs
    print(f"Loading pairs from {input_file}...")
    pairs = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    pairs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    
    if limit > 0:
        pairs = pairs[:limit]
    
    print(f"✓ Loaded {len(pairs)} pairs")
    print(f"✓ Will query {len(MODELS)} models per pair")
    
    # Process pairs
    results = []
    
    async def process_single_pair(pair, session, pbar):
        """Process a single pair"""
        # Load table content
        content = load_table_content(pair, tables_dir)
        
        # Build prompt
        prompt = build_table_prompt(content['table_a'], content['table_b'])
        
        # Query all models
        model_responses = await query_all_models(prompt, session)
        
        # Parse responses
        parsed_responses = {}
        for response in model_responses:
            model_name = response['model']
            if 'error' in response:
                parsed_responses[model_name] = {
                    'model': model_name,
                    'error': response['error'],
                    'status': 'error'
                }
            else:
                parsed = parse_yaml_response(response['response'], model_name)
                parsed_responses[model_name] = parsed
        
        # Save result
        result = {
            'pair_id': pair.get('id', 'unknown'),
            'csv_a': pair.get('csv_a', ''),
            'csv_b': pair.get('csv_b', ''),
            'level': pair.get('level', ''),
            'gt_positive': pair.get('is_positive', None),
            'gt_labels': pair.get('labels', {}),
            'model_responses': parsed_responses
        }
        
        pbar.update(1)
        return result
    
    # Process all pairs with progress bar
    # Process sequentially to avoid overload
    print("\nProcessing pairs...")
    results = []
    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(pairs), desc="Evaluating") as pbar:
            for pair in pairs:
                result = await process_single_pair(pair, session, pbar)
                results.append(result)
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(results)} results")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Pairs processed: {len(results)}")
    print(f"Models queried: {', '.join(MODELS)}")
    
    for model in MODELS:
        success = sum(1 for r in results 
                     for m in r['model_responses'].values() 
                     if m.get('model') == model and 'error' not in m)
        print(f"  {model}: {success}/{len(results)} successful")


def main():
    parser = argparse.ArgumentParser(description="Query multiple LLMs via OpenRouter")
    
    parser.add_argument("--input", required=True,
                       help="Input pairs JSONL from Step 1")
    parser.add_argument("--output", required=True,
                       help="Output results JSONL")
    parser.add_argument("--tables-dir", default="data/scilake_tables",
                       help="Directory containing CSV tables")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of pairs (0=all)")
    
    args = parser.parse_args()
    
    asyncio.run(process_pairs(args.input, args.output, args.tables_dir, args.limit))


if __name__ == "__main__":
    main()

