#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-14
Description: Comprehensive GPT evaluation pipeline covering both table and model relatedness
             with balanced sampling across different levels and attributes.

This script provides a complete evaluation pipeline that:
1. Uses smart sampling to generate balanced datasets
2. Evaluates both table and model relatedness
3. Generates comprehensive reports
4. Supports different evaluation modes and configurations
"""

import os
import json
import argparse
import subprocess
from typing import List, Dict, Any
from pathlib import Path

def run_command(cmd: str, description: str = ""):
    """Run a command and handle errors"""
    print(f"Running: {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description or cmd} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description or cmd} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def run_smart_sampling(args):
    """Run smart sampling to generate evaluation pairs"""
    cmd = f"""python -m src.gpt_evaluation.smart_sampling \\
        --output {args.output_dir}/smart_evaluation_pairs.jsonl \\
        --num-table-pairs {args.num_table_pairs} \\
        --num-model-pairs {args.num_model_pairs} \\
        --positive-ratio {args.positive_ratio} \\
        --seed {args.seed}"""
    
    return run_command(cmd, "Smart sampling for evaluation pairs")

def run_table_evaluation(args):
    """Run table-relatedness evaluation"""
    cmd = f"""python -m src.gpt_evaluation.evaluate_pairs \\
        --mode tables \\
        --pairs {args.output_dir}/smart_evaluation_pairs.jsonl \\
        --output {args.output_dir}/table_evaluation_results.jsonl \\
        --llm {args.llm} \\
        --sample-n {args.num_table_pairs}"""
    
    return run_command(cmd, "Table-relatedness evaluation")

def run_model_evaluation(args):
    """Run model-relatedness evaluation"""
    cmd = f"""python -m src.gpt_evaluation.model_evaluation \\
        --output {args.output_dir}/model_evaluation_results.jsonl \\
        --num-pairs {args.num_model_pairs} \\
        --llm {args.llm} \\
        --seed {args.seed}"""
    
    return run_command(cmd, "Model-relatedness evaluation")

def generate_table_report(args):
    """Generate markdown report for table evaluation"""
    cmd = f"""python -m src.gpt_evaluation.jsonl_to_markdown \\
        --input {args.output_dir}/table_evaluation_results.jsonl \\
        --output {args.output_dir}/table_evaluation_report.md \\
        --show-prompt"""
    
    return run_command(cmd, "Table evaluation report generation")

def generate_model_report(args):
    """Generate markdown report for model evaluation"""
    # Create a simple model report generator
    model_report_script = f"""
import json
import os
from collections import Counter

def generate_model_report(input_file, output_file):
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    # Generate summary
    total_pairs = len(results)
    related_counts = Counter()
    relation_types = Counter()
    closeness_dist = Counter()
    confidence_dist = Counter()
    
    for result in results:
        eval_data = result.get('evaluation', {{}})
        
        if 'related' in eval_data:
            related_counts[eval_data['related']] += 1
        
        if 'relation_types' in eval_data:
            for rel_type in eval_data['relation_types']:
                relation_types[rel_type] += 1
        
        if 'closeness' in eval_data:
            closeness_dist[eval_data['closeness']] += 1
        
        if 'confidence' in eval_data:
            confidence_dist[eval_data['confidence']] += 1
    
    # Write markdown report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Model Evaluation Report\\n\\n")
        f.write(f"**Summary**: total={{total_pairs}}, YES={{related_counts['YES']}}, NO={{related_counts['NO']}}, UNSURE={{related_counts['UNSURE']}}\\n\\n")
        
        f.write("## Distribution Analysis\\n\\n")
        f.write(f"- **Relation Types**: {{dict(relation_types)}}\\n")
        f.write(f"- **Closeness Distribution**: {{dict(closeness_dist)}}\\n")
        f.write(f"- **Confidence Distribution**: {{dict(confidence_dist)}}\\n\\n")
        
        f.write("## Detailed Results\\n\\n")
        for i, result in enumerate(results, 1):
            f.write(f"### Model Pair {{i}}: {{result['id']}}\\n\\n")
            f.write(f"**Model A**: {{result['model_a_id']}}\\n")
            f.write(f"**Model B**: {{result['model_b_id']}}\\n\\n")
            
            eval_data = result.get('evaluation', {{}})
            f.write(f"**Related**: {{eval_data.get('related', 'UNKNOWN')}}\\n")
            f.write(f"**Relation Types**: {{eval_data.get('relation_types', [])}}\\n")
            f.write(f"**Closeness**: {{eval_data.get('closeness', 'N/A')}}\\n")
            f.write(f"**Confidence**: {{eval_data.get('confidence', 'N/A')}}\\n")
            f.write(f"**Rationale**: {{eval_data.get('rationale', 'N/A')}}\\n\\n")

if __name__ == "__main__":
    generate_model_report("{args.output_dir}/model_evaluation_results.jsonl", 
                         "{args.output_dir}/model_evaluation_report.md")
"""
    
    # Write and run the script
    script_path = f"{args.output_dir}/generate_model_report.py"
    with open(script_path, 'w') as f:
        f.write(model_report_script)
    
    cmd = f"python {script_path}"
    return run_command(cmd, "Model evaluation report generation")

def generate_comprehensive_report(args):
    """Generate a comprehensive report combining both evaluations"""
    report_content = f"""# Comprehensive GPT Evaluation Report

## Overview
This report presents the results of GPT-based evaluation for both table and model relatedness in the CitationLake dataset.

## Configuration
- **Table Pairs Evaluated**: {args.num_table_pairs}
- **Model Pairs Evaluated**: {args.num_model_pairs}
- **Positive Ratio**: {args.positive_ratio}
- **LLM Model**: {args.llm}
- **Random Seed**: {args.seed}

## Table-Relatedness Evaluation

### Summary
The table-relatedness evaluation assesses how semantically related two tables are based on their content, structure, and context.

### Key Findings
- Tables were sampled from different sources: Hugging Face, GitHub, HTML, and LLM-processed
- Ground truth levels included: direct, max_pr, model, and dataset relationships
- Evaluation covered both positive and negative examples

### Detailed Results
See `table_evaluation_report.md` for detailed table-by-table analysis.

## Model-Relatedness Evaluation

### Summary
The model-relatedness evaluation assesses how related two models are based on their metadata, tasks, domains, and characteristics.

### Key Findings
- Models were compared based on their model cards, tags, pipeline types, and descriptions
- Evaluation considered multiple relationship types: keywords, semantic, task, domain, architecture, dataset, benchmark, baseline

### Detailed Results
See `model_evaluation_report.md` for detailed model-by-model analysis.

## Methodology

### Sampling Strategy
1. **Smart Sampling**: Used ground truth matrices to ensure balanced positive/negative examples
2. **Level Coverage**: Sampled across different relationship levels (direct, max_pr, model, dataset)
3. **Source Diversity**: Included tables from multiple sources for comprehensive coverage
4. **Attribute-Based**: Considered table attributes (type, domain, size) for better sampling

### Evaluation Process
1. **Table Pairs**: Converted CSV tables to markdown format for LLM evaluation
2. **Model Pairs**: Extracted model metadata and descriptions for comparison
3. **LLM Assessment**: Used GPT-3.5-turbo for consistent evaluation
4. **Structured Output**: Collected structured JSON responses with multiple dimensions

### Evaluation Dimensions
- **Relatedness**: YES/NO/UNSURE
- **Relation Types**: Multiple categories (JOINABLE, UNIONABLE, KEYWORDS, SEMANTIC, etc.)
- **Closeness**: 1-5 scale (Loosely Related to Very Closely Related)
- **Confidence**: 1-5 scale for evaluation confidence
- **Rationale**: Textual explanation of the relationship

## Conclusions

This comprehensive evaluation provides insights into:
1. The quality of table relationships in the CitationLake dataset
2. The semantic relatedness of models based on their metadata
3. The effectiveness of different sampling strategies
4. The consistency of LLM-based evaluation across different data types

The results can be used to:
- Validate ground truth quality
- Improve sampling strategies
- Enhance evaluation methodologies
- Guide future dataset improvements

## Files Generated
- `smart_evaluation_pairs.jsonl`: Sampled pairs for evaluation
- `table_evaluation_results.jsonl`: Table evaluation results
- `model_evaluation_results.jsonl`: Model evaluation results
- `table_evaluation_report.md`: Detailed table analysis
- `model_evaluation_report.md`: Detailed model analysis
- `comprehensive_report.md`: This comprehensive report
"""
    
    report_path = f"{args.output_dir}/comprehensive_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ Generated comprehensive report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive GPT evaluation pipeline")
    
    # Output configuration
    parser.add_argument("--output-dir", default="output/gpt_evaluation",
                       help="Output directory for all results")
    
    # Sampling configuration
    parser.add_argument("--num-table-pairs", type=int, default=100,
                       help="Number of table pairs to evaluate")
    parser.add_argument("--num-model-pairs", type=int, default=50,
                       help="Number of model pairs to evaluate")
    parser.add_argument("--positive-ratio", type=float, default=0.5,
                       help="Ratio of positive pairs in table evaluation")
    
    # Evaluation configuration
    parser.add_argument("--llm", default="gpt-3.5-turbo-0125",
                       help="LLM model to use for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Pipeline control
    parser.add_argument("--skip-sampling", action="store_true",
                       help="Skip smart sampling step")
    parser.add_argument("--skip-table-eval", action="store_true",
                       help="Skip table evaluation step")
    parser.add_argument("--skip-model-eval", action="store_true",
                       help="Skip model evaluation step")
    parser.add_argument("--skip-reports", action="store_true",
                       help="Skip report generation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE GPT EVALUATION PIPELINE")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Table pairs: {args.num_table_pairs}")
    print(f"Model pairs: {args.num_model_pairs}")
    print(f"Positive ratio: {args.positive_ratio}")
    print(f"LLM model: {args.llm}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Step 1: Smart Sampling
    if not args.skip_sampling:
        print("\n[STEP 1] Smart Sampling")
        print("-" * 30)
        if not run_smart_sampling(args):
            print("Smart sampling failed, but continuing...")
    else:
        print("\n[STEP 1] Smart Sampling - SKIPPED")
    
    # Step 2: Table Evaluation
    if not args.skip_table_eval:
        print("\n[STEP 2] Table-Relatedness Evaluation")
        print("-" * 40)
        if not run_table_evaluation(args):
            print("Table evaluation failed, but continuing...")
    else:
        print("\n[STEP 2] Table-Relatedness Evaluation - SKIPPED")
    
    # Step 3: Model Evaluation
    if not args.skip_model_eval:
        print("\n[STEP 3] Model-Relatedness Evaluation")
        print("-" * 40)
        if not run_model_evaluation(args):
            print("Model evaluation failed, but continuing...")
    else:
        print("\n[STEP 3] Model-Relatedness Evaluation - SKIPPED")
    
    # Step 4: Report Generation
    if not args.skip_reports:
        print("\n[STEP 4] Report Generation")
        print("-" * 25)
        
        # Generate table report
        if os.path.exists(f"{args.output_dir}/table_evaluation_results.jsonl"):
            generate_table_report(args)
        
        # Generate model report
        if os.path.exists(f"{args.output_dir}/model_evaluation_results.jsonl"):
            generate_model_report(args)
        
        # Generate comprehensive report
        generate_comprehensive_report(args)
    else:
        print("\n[STEP 4] Report Generation - SKIPPED")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    print(f"Results saved in: {args.output_dir}")
    print("\nGenerated files:")
    
    output_files = [
        "smart_evaluation_pairs.jsonl",
        "table_evaluation_results.jsonl", 
        "model_evaluation_results.jsonl",
        "table_evaluation_report.md",
        "model_evaluation_report.md",
        "comprehensive_report.md"
    ]
    
    for file in output_files:
        file_path = f"{args.output_dir}/{file}"
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not generated)")

if __name__ == "__main__":
    main()
