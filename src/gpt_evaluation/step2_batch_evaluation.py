#!/usr/bin/env python3
"""
Batch GPT evaluation for table and model relatedness
Based on src/llm/batch.py and src/data_preprocess/step2_integration_order.py
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

def setup_client():
    """Setup OpenAI client"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
    client = OpenAI(api_key=api_key)
    return client

def create_batch_input_file(pairs_file: str, output_file: str, mode: str = "tables"):
    """Create batch input file from pairs"""
    print(f"Creating batch input from {pairs_file}...")
    
    with open(pairs_file, 'r') as f:
        pairs = [json.loads(line) for line in f]
    
    batch_requests = []
    for i, pair in enumerate(pairs):
        if mode == "tables":
            prompt = create_table_prompt(pair)
        else:
            prompt = create_model_prompt(pair)
        
        request = {
            "custom_id": f"pair-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.1
            }
        }
        batch_requests.append(request)
    
    with open(output_file, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"✅ Created batch input file with {len(batch_requests)} requests")

def create_table_prompt(pair: Dict[str, Any]) -> str:
    """Create table evaluation prompt"""
    return (
        "Human Evaluation: Semantic Table Search over Model Lake, A Benchmark\n"
        "You will be given two tables (A and B) in Markdown.\n"
        "Task: determine whether/how/why they are related.\n\n"
        "Return a strict JSON object with these fields:\n"
        "  related: one of ['YES','NO','UNSURE']\n"
        "  relation_types: array of any of ['JOINABLE','UNIONABLE','KEYWORDS','SEMANTIC','RELATED','OTHER']\n"
        "  closeness: integer 1-5 (Loosely Related=1 ... Very Closely Related=5)\n"
        "  rationale: short text\n"
        "  confidence: integer 1-5\n\n"
        f"Table A:\n{pair.get('table_a_md', '')}\n\n"
        f"Table B:\n{pair.get('table_b_md', '')}\n\n"
        "Respond with JSON only, no extra text."
    )

def create_model_prompt(pair: Dict[str, Any]) -> str:
    """Create model evaluation prompt"""
    return (
        "Human Evaluation: Semantic Model Card Relatedness\n"
        "You will be given two model cards (A and B).\n"
        "Task: determine whether/how/why these models are related.\n\n"
        "Return a strict JSON object with these fields:\n"
        "  related: one of ['YES','NO','UNSURE']\n"
        "  relation_types: array of any of ['KEYWORDS','SEMANTIC','RELATED','BENCHMARK','TASK','BASELINE','OTHER']\n"
        "  closeness: integer 1-5 (Loosely Related=1 ... Very Closely Related=5)\n"
        "  rationale: short text\n"
        "  confidence: integer 1-5\n\n"
        f"Model Card A:\n{pair.get('card_a', '')}\n\n"
        f"Model Card B:\n{pair.get('card_b', '')}\n\n"
        "Respond with JSON only, no extra text."
    )

def upload_batch_file(client, file_path: str) -> str:
    """Upload batch input file"""
    print(f"Uploading batch file: {file_path}")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="batch")
    print(f"✅ Uploaded file with ID: {response.id}")
    return response.id

def create_batch_job(client, file_id: str) -> str:
    """Create batch job"""
    print("Creating batch job...")
    response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"✅ Created batch job with ID: {response.id}")
    return response.id

def wait_for_completion(client, batch_id: str, poll_interval: int = 30) -> str:
    """Wait for batch completion"""
    print(f"Waiting for batch {batch_id} to complete...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"📡 Current status: {status}")
        
        if status in ["completed", "failed", "expired", "cancelled"]:
            if status != "completed":
                print(f"❌ Batch {status}")
                if hasattr(batch, 'errors') and batch.errors:
                    for i, err in enumerate(batch.errors):
                        print(f"  Error {i+1}: {err}")
            else:
                print("✅ Batch completed successfully!")
            break
        
        time.sleep(poll_interval)
    
    return status

def download_results(client, batch_id: str, output_file: str):
    """Download batch results"""
    print(f"Downloading results for batch {batch_id}...")
    
    try:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed" and batch.output_file_id:
            response = client.files.content(batch.output_file_id)
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded results to {output_file}")
        else:
            print(f"❌ Batch not completed or no output file")
    except Exception as e:
        print(f"❌ Error downloading results: {e}")

def process_batch_results(results_file: str, pairs_file: str, output_file: str):
    """Process batch results and merge with original pairs"""
    print(f"Processing batch results...")
    
    # Load original pairs
    with open(pairs_file, 'r') as f:
        pairs = [json.loads(line) for line in f]
    
    # Load batch results
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    # Merge results
    merged_results = []
    for i, pair in enumerate(pairs):
        # Find corresponding result
        result = None
        for r in results:
            if r.get('custom_id') == f"pair-{i+1}":
                result = r
                break
        
        if result:
            # Parse response
            try:
                response_text = result['response']['body']['choices'][0]['message']['content']
                # Try to parse JSON
                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    response_json = json.loads(response_text[start:end+1])
                else:
                    response_json = {"raw": response_text}
            except Exception as e:
                response_json = {"error": str(e), "raw": result.get('response', {})}
            
            merged_result = {
                "id": pair.get('id', f"pair-{i+1}"),
                "mode": "tables",
                "prompt": create_table_prompt(pair),
                "response": response_json,
                "original_pair": pair
            }
        else:
            merged_result = {
                "id": pair.get('id', f"pair-{i+1}"),
                "mode": "tables", 
                "prompt": create_table_prompt(pair),
                "response": {"error": "No result found"},
                "original_pair": pair
            }
        
        merged_results.append(merged_result)
    
    # Save merged results
    with open(output_file, 'w') as f:
        for result in merged_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Processed {len(merged_results)} results")

def main():
    parser = argparse.ArgumentParser(description="Batch GPT evaluation")
    parser.add_argument("--pairs", required=True, help="Input pairs JSONL file")
    parser.add_argument("--mode", choices=["tables", "models"], default="tables", help="Evaluation mode")
    parser.add_argument("--output", required=True, help="Output results JSONL file")
    parser.add_argument("--batch-input", default="output/batch_input.jsonl", help="Batch input file")
    parser.add_argument("--batch-results", default="output/batch_results.jsonl", help="Batch results file")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    # Create batch input
    create_batch_input_file(args.pairs, args.batch_input, args.mode)
    
    # Setup client
    client = setup_client()
    
    # Upload and create batch job
    file_id = upload_batch_file(client, args.batch_input)
    batch_id = create_batch_job(client, file_id)
    
    # Wait for completion
    status = wait_for_completion(client, batch_id, args.poll_interval)
    
    if status == "completed":
        # Download and process results
        download_results(client, batch_id, args.batch_results)
        process_batch_results(args.batch_results, args.pairs, args.output)
        print(f"✅ Batch evaluation complete! Results saved to {args.output}")
    else:
        print(f"❌ Batch evaluation failed with status: {status}")

if __name__ == "__main__":
    main()


