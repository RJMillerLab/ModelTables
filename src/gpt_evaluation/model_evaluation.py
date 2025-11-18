#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-14
Description: Model-relatedness evaluation using model cards and metadata
             This script evaluates how related two models are based on their
             model cards, tasks, domains, and other metadata.
"""

import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any
from src.llm.model import LLM_response

MODEL_CARD_PROMPT = (
    "Human Evaluation: Semantic Model Card Relatedness\n"
    "You will be given two model cards (A and B) with their metadata.\n"
    "Task: determine whether/how/why these models are related.\n\n"
    "Return a strict JSON object with these fields:\n"
    "  related: one of ['YES','NO','UNSURE']\n"
    "  relation_types: array of any of ['KEYWORDS','SEMANTIC','TASK','DOMAIN','ARCHITECTURE','DATASET','BENCHMARK','BASELINE','OTHER']\n"
    "  closeness: integer 1-5 (Loosely Related=1 ... Very Closely Related=5)\n"
    "  rationale: short text explaining the relationship\n"
    "  confidence: integer 1-5\n\n"
    "Model Card A:\n{model_a_info}\n\n"
    "Model Card B:\n{model_b_info}\n\n"
    "Respond with JSON only, no extra text."
)

def load_model_metadata(processed_dir: str = "data/processed") -> pd.DataFrame:
    """Load model metadata from processed files"""
    model_file = os.path.join(processed_dir, "modelcard_step1.parquet")
    if os.path.exists(model_file):
        return pd.read_parquet(model_file)
    else:
        print(f"Model metadata file not found: {model_file}")
        return None

def extract_model_info(model_row: pd.Series) -> str:
    """Extract relevant information from model metadata"""
    info_parts = []
    
    # Basic info
    if pd.notna(model_row.get('modelId')):
        info_parts.append(f"Model ID: {model_row['modelId']}")
    
    # Tags
    if pd.notna(model_row.get('tags')):
        tags = model_row['tags']
        if isinstance(tags, list):
            info_parts.append(f"Tags: {', '.join(tags[:10])}")  # Limit to first 10 tags
        else:
            info_parts.append(f"Tags: {tags}")
    
    # Pipeline tag
    if pd.notna(model_row.get('pipeline_tag')):
        info_parts.append(f"Pipeline Tag: {model_row['pipeline_tag']}")
    
    # Library name
    if pd.notna(model_row.get('library_name')):
        info_parts.append(f"Library: {model_row['library_name']}")
    
    # Downloads and likes
    if pd.notna(model_row.get('downloads')):
        info_parts.append(f"Downloads: {model_row['downloads']}")
    if pd.notna(model_row.get('likes')):
        info_parts.append(f"Likes: {model_row['likes']}")
    
    # Card content (first 500 chars)
    if pd.notna(model_row.get('card')):
        card_content = str(model_row['card'])[:500]
        info_parts.append(f"Card Content: {card_content}...")
    
    return "\n".join(info_parts)

def evaluate_model_pair(model_a_info: str, model_b_info: str, llm: str = "gpt-3.5-turbo-0125") -> Dict[str, Any]:
    """Evaluate a single model pair using LLM"""
    prompt = MODEL_CARD_PROMPT.format(
        model_a_info=model_a_info,
        model_b_info=model_b_info
    )
    
    try:
        response, _ = LLM_response(prompt, llm_model=llm, history=[], kwargs={}, max_tokens=800)
        
        # Try to parse JSON from response
        text = response if isinstance(response, str) else str(response)
        try:
            # Find first JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
            return json.loads(text)
        except Exception:
            return {"raw": text}
    except Exception as e:
        return {"error": str(e)}

def sample_model_pairs(df: pd.DataFrame, num_pairs: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample model pairs for evaluation"""
    import random
    random.seed(seed)
    
    # Filter models with valid metadata
    valid_models = df.dropna(subset=['modelId']).copy()
    
    if len(valid_models) < 2:
        print("Not enough valid models for sampling")
        return []
    
    pairs = []
    model_ids = valid_models['modelId'].tolist()
    
    for i in range(num_pairs):
        model_a_id, model_b_id = random.sample(model_ids, 2)
        
        model_a_row = valid_models[valid_models['modelId'] == model_a_id].iloc[0]
        model_b_row = valid_models[valid_models['modelId'] == model_b_id].iloc[0]
        
        model_a_info = extract_model_info(model_a_row)
        model_b_info = extract_model_info(model_b_row)
        
        pairs.append({
            "id": f"model-pair-{i+1}",
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "model_a_info": model_a_info,
            "model_b_info": model_b_info
        })
    
    return pairs

def evaluate_model_pairs(pairs: List[Dict[str, Any]], llm: str = "gpt-3.5-turbo-0125") -> List[Dict[str, Any]]:
    """Evaluate all model pairs"""
    results = []
    
    for pair in pairs:
        print(f"Evaluating {pair['id']}...")
        
        evaluation = evaluate_model_pair(
            pair['model_a_info'], 
            pair['model_b_info'], 
            llm
        )
        
        result = {
            "id": pair['id'],
            "model_a_id": pair['model_a_id'],
            "model_b_id": pair['model_b_id'],
            "model_a_info": pair['model_a_info'],
            "model_b_info": pair['model_b_info'],
            "evaluation": evaluation,
            "evaluation_type": "model_relatedness"
        }
        
        results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save evaluation results to JSONL"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(results)} model evaluations to {output_path}")

def generate_model_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a report on model evaluation results"""
    from collections import Counter
    
    report = {
        "total_pairs": len(results),
        "related_counts": Counter(),
        "relation_types": Counter(),
        "closeness_distribution": Counter(),
        "confidence_distribution": Counter()
    }
    
    for result in results:
        eval_data = result.get("evaluation", {})
        
        if "related" in eval_data:
            report["related_counts"][eval_data["related"]] += 1
        
        if "relation_types" in eval_data:
            for rel_type in eval_data["relation_types"]:
                report["relation_types"][rel_type] += 1
        
        if "closeness" in eval_data:
            report["closeness_distribution"][eval_data["closeness"]] += 1
        
        if "confidence" in eval_data:
            report["confidence_distribution"][eval_data["confidence"]] += 1
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Model-relatedness evaluation")
    parser.add_argument("--output", default="output/model_evaluation_results.jsonl",
                       help="Output file path")
    parser.add_argument("--num-pairs", type=int, default=50,
                       help="Number of model pairs to evaluate")
    parser.add_argument("--llm", default="gpt-3.5-turbo-0125",
                       help="LLM model to use")
    parser.add_argument("--processed-dir", default="data/processed",
                       help="Processed data directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("Loading model metadata...")
    df = load_model_metadata(args.processed_dir)
    
    if df is None:
        print("Failed to load model metadata")
        return
    
    print(f"Loaded {len(df)} models")
    
    print(f"Sampling {args.num_pairs} model pairs...")
    pairs = sample_model_pairs(df, args.num_pairs, args.seed)
    
    print(f"Evaluating {len(pairs)} model pairs...")
    results = evaluate_model_pairs(pairs, args.llm)
    
    print(f"Saving results...")
    save_results(results, args.output)
    
    print("Generating report...")
    report = generate_model_report(results)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Related counts: {dict(report['related_counts'])}")
    print(f"Relation types: {dict(report['relation_types'])}")
    print(f"Closeness distribution: {dict(report['closeness_distribution'])}")
    print(f"Confidence distribution: {dict(report['confidence_distribution'])}")

if __name__ == "__main__":
    main()
