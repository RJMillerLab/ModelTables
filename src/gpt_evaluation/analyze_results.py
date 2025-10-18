#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-17
Description: Analyze batch evaluation results and generate metrics/plots

Usage:
python -m src.gpt_evaluation.analyze_results --results output/fake_tables_results_5.jsonl --output output/analysis_tables
python -m src.gpt_evaluation.analyze_results --results output/fake_models_results_5.jsonl --output output/analysis_models
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from collections import Counter

def load_results(results_file: str) -> List[Dict[str, Any]]:
    """Load batch results from JSONL"""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def extract_predictions_and_gt(results: List[Dict[str, Any]]) -> Tuple[List[bool], List[bool], List[Dict]]:
    """Extract LLM predictions and ground truth"""
    predictions = []
    ground_truth = []
    metadata = []
    
    for result in results:
        # Extract LLM prediction
        response = result.get('response', {})
        if isinstance(response, dict):
            llm_related = response.get('related')
            if isinstance(llm_related, bool):
                predictions.append(llm_related)
            elif isinstance(llm_related, str):
                predictions.append(llm_related.upper() == 'YES')
            else:
                predictions.append(None)
        else:
            predictions.append(None)
        
        # Extract ground truth
        gt = result.get('gt', {})
        gt_related = gt.get('related')
        if isinstance(gt_related, bool):
            ground_truth.append(gt_related)
        else:
            ground_truth.append(None)
        
        # Extract metadata
        metadata.append({
            'id': result.get('id'),
            'mode': result.get('mode'),
            'gt_signals': gt.get('signals', {}),
            'identifiers': result.get('identifiers', {}),
            'response': response
        })
    
    return predictions, ground_truth, metadata

def compute_basic_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]:
    """Compute basic classification metrics"""
    # Filter out None values
    valid_indices = [i for i, (p, gt) in enumerate(zip(predictions, ground_truth)) 
                     if p is not None and gt is not None]
    
    if not valid_indices:
        return {"accuracy": 0.0, "total_samples": 0}
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_gt = [ground_truth[i] for i in valid_indices]
    
    # Calculate accuracy manually
    correct = sum(1 for p, gt in zip(valid_predictions, valid_gt) if p == gt)
    accuracy = correct / len(valid_indices)
    
    # Count predictions
    pred_counts = Counter(valid_predictions)
    gt_counts = Counter(valid_gt)
    
    return {
        "accuracy": accuracy,
        "total_samples": len(valid_indices),
        "llm_yes_count": pred_counts.get(True, 0),
        "llm_no_count": pred_counts.get(False, 0),
        "gt_yes_count": gt_counts.get(True, 0),
        "gt_no_count": gt_counts.get(False, 0),
    }

def plot_confusion_matrix(predictions: List[bool], ground_truth: List[bool], 
                         output_dir: str, title: str = "Confusion Matrix"):
    """Plot confusion matrix"""
    # Filter out None values
    valid_indices = [i for i, (p, gt) in enumerate(zip(predictions, ground_truth)) 
                     if p is not None and gt is not None]
    
    if not valid_indices:
        print("No valid predictions for confusion matrix")
        return
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_gt = [ground_truth[i] for i in valid_indices]
    
    # Calculate confusion matrix manually
    cm = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
    for pred, gt in zip(valid_predictions, valid_gt):
        if not pred and not gt:  # True Negative
            cm[0][0] += 1
        elif pred and not gt:  # False Positive
            cm[0][1] += 1
        elif not pred and gt:  # False Negative
            cm[1][0] += 1
        else:  # True Positive
            cm[1][1] += 1
    
    # Calculate accuracy
    correct = cm[0][0] + cm[1][1]
    total = sum(sum(row) for row in cm)
    accuracy = correct / total if total > 0 else 0
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'{title}\nAccuracy: {accuracy:.3f}')
    plt.colorbar()
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=16)
    
    plt.ylabel('Ground Truth')
    plt.xlabel('LLM Prediction')
    plt.xticks([0, 1], ['Predicted: NO', 'Predicted: YES'])
    plt.yticks([0, 1], ['Actual: NO', 'Actual: YES'])
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix to {output_path}")

def plot_prediction_distribution(predictions: List[bool], ground_truth: List[bool], 
                                output_dir: str, title: str = "Prediction Distribution"):
    """Plot prediction distribution comparison"""
    # Filter out None values
    valid_indices = [i for i, (p, gt) in enumerate(zip(predictions, ground_truth)) 
                     if p is not None and gt is not None]
    
    if not valid_indices:
        print("No valid predictions for distribution plot")
        return
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_gt = [ground_truth[i] for i in valid_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground Truth distribution
    gt_counts = Counter(valid_gt)
    ax1.bar(['NO', 'YES'], [gt_counts.get(False, 0), gt_counts.get(True, 0)], 
            color=['red', 'green'], alpha=0.7)
    ax1.set_title('Ground Truth Distribution')
    ax1.set_ylabel('Count')
    
    # LLM Prediction distribution
    pred_counts = Counter(valid_predictions)
    ax2.bar(['NO', 'YES'], [pred_counts.get(False, 0), pred_counts.get(True, 0)], 
            color=['red', 'green'], alpha=0.7)
    ax2.set_title('LLM Prediction Distribution')
    ax2.set_ylabel('Count')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'prediction_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved distribution plot to {output_path}")

def analyze_signal_types(metadata: List[Dict[str, Any]], predictions: List[bool], 
                        ground_truth: List[bool]) -> Dict[str, Any]:
    """Analyze performance by signal types"""
    signal_analysis = {}
    
    for i, (meta, pred, gt) in enumerate(zip(metadata, predictions, ground_truth)):
        if pred is None or gt is None:
            continue
            
        signals = meta.get('gt_signals', {})
        
        # Analyze by signal levels
        levels = signals.get('levels', [])
        for level in levels:
            if level not in signal_analysis:
                signal_analysis[level] = {'correct': 0, 'total': 0, 'predictions': [], 'gt': []}
            
            signal_analysis[level]['total'] += 1
            signal_analysis[level]['predictions'].append(pred)
            signal_analysis[level]['gt'].append(gt)
            
            if pred == gt:
                signal_analysis[level]['correct'] += 1
    
    # Calculate accuracy for each signal type
    for level, data in signal_analysis.items():
        if data['total'] > 0:
            data['accuracy'] = data['correct'] / data['total']
        else:
            data['accuracy'] = 0.0
    
    return signal_analysis

def plot_signal_analysis(signal_analysis: Dict[str, Any], output_dir: str):
    """Plot accuracy by signal types"""
    if not signal_analysis:
        print("No signal analysis data to plot")
        return
    
    levels = list(signal_analysis.keys())
    accuracies = [signal_analysis[level]['accuracy'] for level in levels]
    totals = [signal_analysis[level]['total'] for level in levels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy by signal type
    bars = ax1.bar(levels, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Accuracy by Signal Type')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Sample count by signal type
    ax2.bar(levels, totals, color='lightcoral', alpha=0.7)
    ax2.set_title('Sample Count by Signal Type')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'signal_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved signal analysis to {output_path}")

def generate_report(results: List[Dict[str, Any]], output_dir: str):
    """Generate a comprehensive analysis report"""
    predictions, ground_truth, metadata = extract_predictions_and_gt(results)
    
    # Basic metrics
    metrics = compute_basic_metrics(predictions, ground_truth)
    
    # Signal analysis
    signal_analysis = analyze_signal_types(metadata, predictions, ground_truth)
    
    # Create plots
    plot_confusion_matrix(predictions, ground_truth, output_dir)
    plot_prediction_distribution(predictions, ground_truth, output_dir)
    plot_signal_analysis(signal_analysis, output_dir)
    
    # Generate text report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== GPT Evaluation Analysis Report ===\n\n")
        f.write(f"Total samples: {len(results)}\n")
        f.write(f"Valid samples: {metrics['total_samples']}\n")
        f.write(f"Overall accuracy: {metrics['accuracy']:.3f}\n\n")
        
        f.write("Prediction distribution:\n")
        f.write(f"  LLM YES: {metrics['llm_yes_count']}\n")
        f.write(f"  LLM NO: {metrics['llm_no_count']}\n")
        f.write(f"  GT YES: {metrics['gt_yes_count']}\n")
        f.write(f"  GT NO: {metrics['gt_no_count']}\n\n")
        
        f.write("Signal type analysis:\n")
        for level, data in signal_analysis.items():
            f.write(f"  {level}: accuracy={data['accuracy']:.3f}, samples={data['total']}\n")
        
        f.write("\nDetailed results:\n")
        for i, (result, pred, gt) in enumerate(zip(results, predictions, ground_truth)):
            f.write(f"  {i+1}. ID: {result.get('id', 'unknown')}\n")
            f.write(f"     LLM: {pred}, GT: {gt}, Correct: {pred == gt if pred is not None and gt is not None else 'N/A'}\n")
    
    print(f"✅ Generated analysis report at {report_path}")
    return metrics, signal_analysis

def main():
    parser = argparse.ArgumentParser(description="Analyze batch evaluation results")
    parser.add_argument("--results", required=True, help="Results JSONL file")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    print(f"Loaded {len(results)} results")
    
    # Generate analysis
    metrics, signal_analysis = generate_report(results, args.output)
    
    print(f"\n=== Summary ===")
    print(f"Overall accuracy: {metrics['accuracy']:.3f}")
    print(f"Valid samples: {metrics['total_samples']}/{len(results)}")
    print(f"Analysis saved to: {args.output}")

if __name__ == "__main__":
    main()
