#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-14
Description: Model-relatedness evaluation based on modelcard_matrix.py logic.
             This script evaluates model relationships using the same logic
             used to build ground truth: base model relationships, dataset relationships,
             and other metadata from model cards.
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
    "  relation_types: array of any of ['BASE_MODEL','DATASET','TASK','DOMAIN','ARCHITECTURE','KEYWORDS','SEMANTIC','OTHER']\n"
    "  closeness: integer 1-5 (Loosely Related=1 ... Very Closely Related=5)\n"
    "  rationale: short text explaining the relationship\n"
    "  confidence: integer 1-5\n\n"
    "Model Card A:\n{model_a_info}\n\n"
    "Model Card B:\n{model_b_info}\n\n"
    "Respond with JSON only, no extra text."
)

def load_model_metadata(processed_dir: str = "data/processed") -> pd.DataFrame:
    """Load model metadata from step1 (like modelcard_matrix.py)"""
    model_file = os.path.join(processed_dir, "modelcard_step1.parquet")
    if os.path.exists(model_file):
        return pd.read_parquet(model_file)
    else:
        print(f"Model metadata file not found: {model_file}")
        return None

def apply_modelcard_matrix_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same filtering logic as in modelcard_matrix.py"""
    print("Applying modelcard_matrix.py filtering logic...")
    
    initial_count = len(df)
    
    # Filter 1: Keep rows with at least one base/model link (line 304 in modelcard_matrix.py)
    df_model = df[df.apply(lambda r: bool(r.get('tag_base_model_list', []) or r.get('readme_modelid_list', [])), axis=1)]
    
    print(f"  Filtered from {initial_count} to {len(df_model)} models with base/model relationships")
    
    return df_model

def extract_model_info_for_evaluation(model_row: pd.Series) -> str:
    """Extract relevant information for model evaluation"""
    info_parts = []
    
    # Basic info
    if pd.notna(model_row.get('modelId')):
        info_parts.append(f"Model ID: {model_row['modelId']}")
    
    # Tags
    tags = model_row.get('tags', [])
    if isinstance(tags, list) and tags:
        info_parts.append(f"Tags: {', '.join(tags[:10])}")  # Limit to first 10 tags
    elif pd.notna(tags):
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
    
    # Base model relationships (key for model relatedness)
    tag_base_models = model_row.get('tag_base_model_list', [])
    readme_models = model_row.get('readme_modelid_list', [])
    
    if tag_base_models:
        info_parts.append(f"Tag Base Models: {', '.join(tag_base_models[:5])}")
    if readme_models:
        info_parts.append(f"README Model IDs: {', '.join(readme_models[:5])}")
    
    # Dataset relationships
    tag_datasets = model_row.get('tag_dataset_list', [])
    readme_datasets = model_row.get('readme_datasetid_list', [])
    
    if tag_datasets:
        info_parts.append(f"Tag Datasets: {', '.join(tag_datasets[:5])}")
    if readme_datasets:
        info_parts.append(f"README Dataset IDs: {', '.join(readme_datasets[:5])}")
    
    # Card content (first 500 chars)
    card_content = model_row.get('card', '')
    if isinstance(card_content, str) and card_content:
        truncated_content = card_content[:500] + "..." if len(card_content) > 500 else card_content
        info_parts.append(f"Card Content: {truncated_content}")
    
    return "\n".join(info_parts)

def sample_model_pairs_with_relationships(df: pd.DataFrame, num_pairs: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample model pairs based on actual relationships (like modelcard_matrix.py)"""
    import random
    random.seed(seed)
    
    print(f"Sampling {num_pairs} model pairs based on relationships...")
    
    # Build related model dictionary (like modelcard_matrix.py lines 305-317)
    related_model = {}
    
    # Process tag_base_model_list and readme_modelid_list
    for col in ["tag_base_model_list", "readme_modelid_list"]:
        if col not in df.columns:
            continue
            
        exploded = df[["modelId", col]].explode(col).dropna()
        for target, grp in exploded.groupby(col)["modelId"]:
            mem = grp.tolist()
            for model_id in mem:
                if model_id not in related_model:
                    related_model[model_id] = set()
                # Add other models that share the same base model
                related_model[model_id].update([m for m in mem if m != model_id])
                # Add the base model itself
                related_model[model_id].add(target)
    
    print(f"  Built relationship graph with {len(related_model)} models")
    
    # Sample pairs with different relationship types
    pairs = []
    model_ids = df['modelId'].tolist()
    
    # Sample 1: Related models (positive examples)
    related_pairs_count = int(num_pairs * 0.6)  # 60% related pairs
    related_pairs = []
    
    for model_id in model_ids:
        if model_id in related_model and related_model[model_id]:
            related_models = list(related_model[model_id])
            for related_id in related_models:
                if related_id in model_ids and len(related_pairs) < related_pairs_count:
                    related_pairs.append((model_id, related_id))
    
    # Sample random related pairs
    if related_pairs:
        sampled_related = random.sample(related_pairs, min(related_pairs_count, len(related_pairs)))
        for model_a_id, model_b_id in sampled_related:
            model_a_row = df[df['modelId'] == model_a_id].iloc[0]
            model_b_row = df[df['modelId'] == model_b_id].iloc[0]
            
            pairs.append({
                "id": f"related-model-pair-{len(pairs)+1}",
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "model_a_info": extract_model_info_for_evaluation(model_a_row),
                "model_b_info": extract_model_info_for_evaluation(model_b_row),
                "relationship_type": "related",
                "is_positive": True
            })
    
    # Sample 2: Random models (negative examples)
    random_pairs_count = num_pairs - len(pairs)
    for i in range(random_pairs_count):
        model_a_id, model_b_id = random.sample(model_ids, 2)
        
        # Check if they are actually unrelated
        is_unrelated = True
        if model_a_id in related_model and model_b_id in related_model[model_a_id]:
            is_unrelated = False
        
        model_a_row = df[df['modelId'] == model_a_id].iloc[0]
        model_b_row = df[df['modelId'] == model_b_id].iloc[0]
        
        pairs.append({
            "id": f"random-model-pair-{len(pairs)+1}",
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "model_a_info": extract_model_info_for_evaluation(model_a_row),
            "model_b_info": extract_model_info_for_evaluation(model_b_row),
            "relationship_type": "random",
            "is_positive": is_unrelated
        })
    
    print(f"  ✓ Generated {len(pairs)} model pairs ({len([p for p in pairs if p['relationship_type'] == 'related'])} related, {len([p for p in pairs if p['relationship_type'] == 'random'])} random)")
    return pairs

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
            "relationship_type": pair['relationship_type'],
            "is_positive": pair['is_positive'],
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
        "relationship_types": Counter(p["relationship_type"] for p in results),
        "related_counts": Counter(),
        "relation_types": Counter(),
        "closeness_distribution": Counter(),
        "confidence_distribution": Counter(),
        "accuracy_by_relationship": {}
    }
    
    for result in results:
        eval_data = result.get("evaluation", {})
        relationship_type = result["relationship_type"]
        is_positive = result["is_positive"]
        
        if "related" in eval_data:
            report["related_counts"][eval_data["related"]] += 1
        
        if "relation_types" in eval_data:
            for rel_type in eval_data["relation_types"]:
                report["relation_types"][rel_type] += 1
        
        if "closeness" in eval_data:
            report["closeness_distribution"][eval_data["closeness"]] += 1
        
        if "confidence" in eval_data:
            report["confidence_distribution"][eval_data["confidence"]] += 1
        
        # Calculate accuracy by relationship type
        if relationship_type not in report["accuracy_by_relationship"]:
            report["accuracy_by_relationship"][relationship_type] = {"correct": 0, "total": 0}
        
        # Check if evaluation matches ground truth
        predicted_related = eval_data.get("related", "").upper() == "YES"
        if predicted_related == is_positive:
            report["accuracy_by_relationship"][relationship_type]["correct"] += 1
        report["accuracy_by_relationship"][relationship_type]["total"] += 1
    
    # Calculate accuracy percentages
    for rel_type in report["accuracy_by_relationship"]:
        stats = report["accuracy_by_relationship"][rel_type]
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
    
    return report

def load_existing_pairs(pairs_file: str) -> List[Dict[str, Any]]:
    """Load existing pairs from JSONL file"""
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            try:
                pair = json.loads(line.strip())
                pairs.append(pair)
            except json.JSONDecodeError:
                continue
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Model-relatedness evaluation based on modelcard_matrix.py")
    parser.add_argument("--output", default="output/model_relatedness_evaluation.jsonl",
                       help="Output file path")
    parser.add_argument("--pairs", type=str, default="",
                       help="Path to existing pairs JSONL file (if not provided, will sample new pairs)")
    parser.add_argument("--num-pairs", type=int, default=50,
                       help="Number of model pairs to evaluate (only used if --pairs not provided)")
    parser.add_argument("--llm", default="gpt-3.5-turbo-0125",
                       help="LLM model to use")
    parser.add_argument("--processed-dir", default="data/processed",
                       help="Processed data directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if args.pairs and os.path.exists(args.pairs):
        print(f"Loading existing pairs from {args.pairs}...")
        pairs = load_existing_pairs(args.pairs)
        print(f"Loaded {len(pairs)} pairs")
    else:
        print("Loading model metadata...")
        df = load_model_metadata(args.processed_dir)
        
        if df is None:
            print("Failed to load model metadata")
            return
        
        print(f"Loaded {len(df)} models")
        
        print("Applying modelcard_matrix.py filtering...")
        df_filtered = apply_modelcard_matrix_filters(df)
        
        print(f"Sampling {args.num_pairs} model pairs...")
        pairs = sample_model_pairs_with_relationships(df_filtered, args.num_pairs, args.seed)
    
    print(f"Evaluating {len(pairs)} model pairs...")
    results = evaluate_model_pairs(pairs, args.llm)
    
    print(f"Saving results...")
    save_results(results, args.output)
    
    print("Generating report...")
    report = generate_model_report(results)
    
    print("\n" + "="*50)
    print("MODEL RELATEDNESS EVALUATION REPORT")
    print("="*50)
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Relationship types: {dict(report['relationship_types'])}")
    print(f"Related counts: {dict(report['related_counts'])}")
    print(f"Relation types: {dict(report['relation_types'])}")
    print(f"Closeness distribution: {dict(report['closeness_distribution'])}")
    print(f"Confidence distribution: {dict(report['confidence_distribution'])}")
    print(f"Accuracy by relationship type:")
    for rel_type, stats in report['accuracy_by_relationship'].items():
        print(f"  {rel_type}: {stats['correct']}/{stats['total']} = {stats.get('accuracy', 0):.2%}")

if __name__ == "__main__":
    main()
