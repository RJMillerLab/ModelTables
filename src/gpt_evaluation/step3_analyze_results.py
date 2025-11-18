#!/usr/bin/env python3
"""
Step 3: Analyze multi-model evaluation results

Computes metrics to evaluate crowdsourcing validity:
- Inter-model agreement
- Agreement with GT labels
- Confidence distributions
- Disagreement patterns

Usage:
    python src/gpt_evaluation/step3_analyze_results.py \
        --input output/gpt_evaluation/step2_results/all_model_responses.jsonl \
        --output output/gpt_evaluation/step3_analysis
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from collections import Counter, defaultdict
from pathlib import Path


def load_results(input_file: str) -> List[Dict[str, Any]]:
    """Load evaluation results from JSONL"""
    results = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def compute_inter_model_agreement(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute inter-model agreement metrics"""
    metrics = {
        'consistency_scores': [],  # Agreement % for each pair
        'unanimity_count': 0,      # Pairs where all models agree
        'unanimity_rate': 0.0,
        'vote_distributions': Counter(),  # Distribution of majority votes
        'split_decisions': []       # Cases with 3-2 splits
    }
    
    total_pairs = len(results)
    
    for result in results:
        responses = result.get('model_responses', {})
        votes = []
        
        for model_name, response in responses.items():
            if 'error' not in response and 'related' in response:
                votes.append(response['related'])
        
        if votes:
            vote_counts = Counter(votes)
            majority = vote_counts.most_common(1)[0][0]
            agreement = max(vote_counts.values()) / len(votes)
            
            metrics['consistency_scores'].append(agreement)
            metrics['vote_distributions'][majority] += 1
            
            # Check for unanimity
            if agreement == 1.0:
                metrics['unanimity_count'] += 1
            
            # Check for split decisions (3-2)
            if len(votes) >= 5:
                counts = list(vote_counts.values())
                if sorted(counts) == [2, 3]:
                    metrics['split_decisions'].append({
                        'pair_id': result.get('pair_id'),
                        'majority': majority,
                        'distribution': dict(vote_counts)
                    })
    
    if metrics['consistency_scores']:
        metrics['unanimity_rate'] = metrics['unanimity_count'] / total_pairs
        metrics['avg_consistency'] = np.mean(metrics['consistency_scores'])
    
    return metrics


def compute_gt_agreement(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute agreement between model judgments and GT labels"""
    metrics = {
        'total_pairs': len(results),
        'pairs_with_gt': 0,
        'pairs_with_majority_vote': 0,
        'correct_predictions': 0,
        'accuracy': 0.0,
        'gt_positive_precision': 0.0,
        'gt_negative_precision': 0.0,
        'confusion_matrix': Counter(),
        'by_level': {}
    }
    
    # Group by level for per-level analysis
    by_level = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for result in results:
        gt_positive = result.get('gt_positive')
        majority_vote = result.get('majority_vote')
        level = result.get('level', 'unknown')
        
        if gt_positive is not None and majority_vote:
            metrics['pairs_with_gt'] += 1
            metrics['pairs_with_majority_vote'] += 1
            
            # Map judgment to binary
            if majority_vote == 'YES':
                predicted = True
            elif majority_vote == 'NO':
                predicted = False
            else:
                predicted = None  # UNSURE
            
            if predicted is not None:
                # Build confusion matrix
                gt_str = 'positive' if gt_positive else 'negative'
                pred_str = 'positive' if predicted else 'negative'
                metrics['confusion_matrix'][(gt_str, pred_str)] += 1
                
                # Check accuracy
                if predicted == gt_positive:
                    metrics['correct_predictions'] += 1
                    by_level[level]['correct'] += 1
                
                by_level[level]['total'] += 1
    
    if metrics['pairs_with_majority_vote'] > 0:
        metrics['accuracy'] = metrics['correct_predictions'] / metrics['pairs_with_majority_vote']
        
        # Precision for positive and negative
        tp = metrics['confusion_matrix'][('positive', 'positive')]
        fp = metrics['confusion_matrix'][('negative', 'positive')]
        fn = metrics['confusion_matrix'][('positive', 'negative')]
        tn = metrics['confusion_matrix'][('negative', 'negative')]
        
        if tp + fp > 0:
            metrics['gt_positive_precision'] = tp / (tp + fp)
        if tn + fn > 0:
            metrics['gt_negative_precision'] = tn / (tn + fn)
    
    # Per-level metrics
    for level, level_metrics in by_level.items():
        if level_metrics['total'] > 0:
            metrics['by_level'][level] = {
                'accuracy': level_metrics['correct'] / level_metrics['total'],
                'total': level_metrics['total']
            }
    
    return metrics


def compute_uncertainty_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze uncertainty patterns"""
    metrics = {
        'unsure_responses': 0,
        'unsure_rate': 0.0,
        'unsure_pairs': 0,  # Pairs where majority vote is UNSURE
        'by_model': defaultdict(lambda: {'unsure': 0, 'total': 0})
    }
    
    total_responses = 0
    
    for result in results:
        responses = result.get('model_responses', {})
        for model_name, response in responses.items():
            if 'related' in response:
                total_responses += 1
                metrics['by_model'][model_name]['total'] += 1
                
                if response['related'] == 'UNSURE':
                    metrics['unsure_responses'] += 1
                    metrics['by_model'][model_name]['unsure'] += 1
        
        # Check if majority vote is UNSURE
        majority_vote = result.get('majority_vote')
        if majority_vote == 'UNSURE':
            metrics['unsure_pairs'] += 1
    
    if total_responses > 0:
        metrics['unsure_rate'] = metrics['unsure_responses'] / total_responses
    
    # Compute per-model rates
    for model_name in metrics['by_model']:
        m = metrics['by_model'][model_name]
        if m['total'] > 0:
            m['unsure_rate'] = m['unsure'] / m['total']
    
    return metrics


def compute_signal_agreement(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute agreement on structural and level signals (multi-label)"""
    metrics = {
        'structural_signals': {},
        'level_signals': {}
    }
    
    structural_signal_names = ['joinable', 'unionable', 'keyword_overlap', 'semantically_similar']
    level_signal_names = ['paper_level', 'model_level', 'dataset_level']
    
    for signal in structural_signal_names:
        metrics['structural_signals'][signal] = {
            'total_pairs': 0,
            'agreement_scores': [],
            'majority_true': 0,
            'majority_false': 0
        }
    
    for signal in level_signal_names:
        metrics['level_signals'][signal] = {
            'total_pairs': 0,
            'agreement_scores': [],
            'majority_true': 0,
            'majority_false': 0
        }
    
    for result in results:
        model_responses = result.get('model_responses', {})
        
        # Structural signals
        for signal_name in structural_signal_names:
            votes = []
            for model_name, response in model_responses.items():
                if 'structural_signals' in response and signal_name in response['structural_signals']:
                    votes.append(response['structural_signals'][signal_name])
            
            if votes:
                metrics['structural_signals'][signal_name]['total_pairs'] += 1
                vote_counts = Counter(votes)
                agreement = max(vote_counts.values()) / len(votes)
                metrics['structural_signals'][signal_name]['agreement_scores'].append(agreement)
                
                majority = vote_counts.most_common(1)[0][0]
                if majority:
                    metrics['structural_signals'][signal_name]['majority_true'] += 1
                else:
                    metrics['structural_signals'][signal_name]['majority_false'] += 1
        
        # Level signals
        for signal_name in level_signal_names:
            votes = []
            for model_name, response in model_responses.items():
                if 'level_signals' in response and signal_name in response['level_signals']:
                    votes.append(response['level_signals'][signal_name])
            
            if votes:
                metrics['level_signals'][signal_name]['total_pairs'] += 1
                vote_counts = Counter(votes)
                agreement = max(vote_counts.values()) / len(votes)
                metrics['level_signals'][signal_name]['agreement_scores'].append(agreement)
                
                majority = vote_counts.most_common(1)[0][0]
                if majority:
                    metrics['level_signals'][signal_name]['majority_true'] += 1
                else:
                    metrics['level_signals'][signal_name]['majority_false'] += 1
    
    # Compute averages
    for signal, data in metrics['structural_signals'].items():
        if data['agreement_scores']:
            data['avg_agreement'] = np.mean(data['agreement_scores'])
    
    for signal, data in metrics['level_signals'].items():
        if data['agreement_scores']:
            data['avg_agreement'] = np.mean(data['agreement_scores'])
    
    return metrics


def compute_signal_gt_alignment(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute how level signals align with GT labels"""
    metrics = {
        'level_signal_vs_gt': {
            'paper': {'correct': 0, 'total': 0, 'false_pos': 0, 'false_neg': 0},
            'modelcard': {'correct': 0, 'total': 0, 'false_pos': 0, 'false_neg': 0},
            'dataset': {'correct': 0, 'total': 0, 'false_pos': 0, 'false_neg': 0}
        }
    }
    
    for result in results:
        gt_labels = result.get('gt_labels', {})
        model_responses = result.get('model_responses', {})
        
        # Get majority vote for each level signal
        for level_name in ['paper_level', 'model_level', 'dataset_level']:
            votes = []
            for model_name, response in model_responses.items():
                if 'level_signals' in response and level_name in response['level_signals']:
                    votes.append(response['level_signals'][level_name])
            
            if votes:
                majority = Counter(votes).most_common(1)[0][0]
                
                # Map to GT name
                gt_name = level_name.replace('_level', '').replace('model', 'modelcard')
                
                if gt_name in gt_labels:
                    gt_label = gt_labels[gt_name]
                    m = metrics['level_signal_vs_gt'].get(gt_name, {'correct': 0, 'total': 0, 'false_pos': 0, 'false_neg': 0})
                    
                    m['total'] += 1
                    if majority == (gt_label == 1):
                        m['correct'] += 1
                    elif majority and gt_label == 0:
                        m['false_pos'] += 1
                    elif not majority and gt_label == 1:
                        m['false_neg'] += 1
                    
                    metrics['level_signal_vs_gt'][gt_name] = m
    
    # Compute metrics
    for level, data in metrics['level_signal_vs_gt'].items():
        if data['total'] > 0:
            data['accuracy'] = data['correct'] / data['total']
            data['precision'] = data['correct'] / max(1, data['correct'] + data['false_pos'])
            data['recall'] = data['correct'] / max(1, data['correct'] + data['false_neg'])
    
    return metrics


def compute_model_specific_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-model statistics"""
    metrics = {
        'models': defaultdict(lambda: {
            'responses': 0,
            'errors': 0,
            'success_rate': 0.0,
            'vote_distribution': Counter(),
            'avg_elapsed_time': []
        })
    }
    
    for result in results:
        responses = result.get('model_responses', {})
        
        for model_name, response in responses.items():
            m = metrics['models'][model_name]
            
            if 'error' in response:
                m['errors'] += 1
            else:
                m['responses'] += 1
                
                if 'elapsed_time' in response:
                    m['avg_elapsed_time'].append(response['elapsed_time'])
                
                if 'related' in response:
                    m['vote_distribution'][response['related']] += 1
    
    # Compute rates and averages
    for model_name in metrics['models']:
        m = metrics['models'][model_name]
        total = m['responses'] + m['errors']
        if total > 0:
            m['success_rate'] = m['responses'] / total
        
        if m['avg_elapsed_time']:
            m['avg_elapsed_time'] = np.mean(m['avg_elapsed_time'])
        else:
            m['avg_elapsed_time'] = 0.0
    
    return metrics


def generate_report(agreement_metrics, gt_metrics, uncertainty_metrics, 
                   model_metrics, signal_agreement, signal_gt_alignment, output_dir: str):
    """Generate comprehensive report"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Multi-Model Crowdsourcing Evaluation Report")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Inter-model agreement
    report_lines.append("1. INTER-MODEL AGREEMENT")
    report_lines.append("-"*80)
    if agreement_metrics['consistency_scores']:
        report_lines.append(f"Average consistency: {agreement_metrics['avg_consistency']:.3f}")
        report_lines.append(f"Unanimity rate: {agreement_metrics['unanimity_rate']:.1%}")
        report_lines.append(f"Pairs with unanimous agreement: {agreement_metrics['unanimity_count']}")
    report_lines.append(f"\nMajority Vote Distribution:")
    for vote, count in agreement_metrics['vote_distributions'].most_common():
        report_lines.append(f"  {vote}: {count}")
    report_lines.append("")
    
    # GT agreement
    report_lines.append("2. AGREEMENT WITH GROUND TRUTH")
    report_lines.append("-"*80)
    report_lines.append(f"Pairs with GT labels: {gt_metrics['pairs_with_gt']}")
    report_lines.append(f"Pairs with majority vote: {gt_metrics['pairs_with_majority_vote']}")
    report_lines.append(f"Overall accuracy: {gt_metrics['accuracy']:.1%}")
    report_lines.append(f"\nConfusion Matrix (GT → Prediction):")
    for (gt, pred), count in sorted(gt_metrics['confusion_matrix'].items()):
        report_lines.append(f"  {gt} → {pred}: {count}")
    
    if gt_metrics['by_level']:
        report_lines.append(f"\nPer-Level Accuracy:")
        for level, metrics in gt_metrics['by_level'].items():
            report_lines.append(f"  {level}: {metrics['accuracy']:.1%} ({metrics['total']} pairs)")
    report_lines.append("")
    
    # Uncertainty
    report_lines.append("3. UNCERTAINTY ANALYSIS")
    report_lines.append("-"*80)
    report_lines.append(f"UNSURE responses: {uncertainty_metrics['unsure_responses']}")
    report_lines.append(f"Overall UNSURE rate: {uncertainty_metrics['unsure_rate']:.1%}")
    report_lines.append(f"Pairs with UNSURE majority: {uncertainty_metrics['unsure_pairs']}")
    report_lines.append("")
    
    # Signal agreement
    report_lines.append("4. SIGNAL AGREEMENT (MULTI-LABEL)")
    report_lines.append("-"*80)
    
    if signal_agreement.get('structural_signals'):
        report_lines.append("\nStructural Signals:")
        for signal_name, data in signal_agreement['structural_signals'].items():
            if 'avg_agreement' in data:
                report_lines.append(f"  {signal_name}: avg_agreement={data['avg_agreement']:.2f}, majority_true={data['majority_true']}/{data['total_pairs']}")
    
    if signal_agreement.get('level_signals'):
        report_lines.append("\nLevel Signals:")
        for signal_name, data in signal_agreement['level_signals'].items():
            if 'avg_agreement' in data:
                report_lines.append(f"  {signal_name}: avg_agreement={data['avg_agreement']:.2f}, majority_true={data['majority_true']}/{data['total_pairs']}")
    report_lines.append("")
    
    # Signal-GT alignment
    report_lines.append("5. SIGNAL vs GT ALIGNMENT")
    report_lines.append("-"*80)
    
    if signal_gt_alignment.get('level_signal_vs_gt'):
        for level_name, metrics in signal_gt_alignment['level_signal_vs_gt'].items():
            if metrics['total'] > 0:
                report_lines.append(f"\n{level_name.upper()} Level Signal:")
                report_lines.append(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
                report_lines.append(f"  Precision: {metrics['precision']:.1%}")
                report_lines.append(f"  Recall: {metrics['recall']:.1%}")
                report_lines.append(f"  False Positives: {metrics['false_pos']}")
                report_lines.append(f"  False Negatives: {metrics['false_neg']}")
    report_lines.append("")
    
    # Model-specific
    report_lines.append("6. MODEL-SPECIFIC METRICS")
    report_lines.append("-"*80)
    for model_name, metrics in model_metrics['models'].items():
        report_lines.append(f"\n{model_name}:")
        report_lines.append(f"  Responses: {metrics['responses']}")
        report_lines.append(f"  Errors: {metrics['errors']}")
        report_lines.append(f"  Success rate: {metrics['success_rate']:.1%}")
        report_lines.append(f"  Avg elapsed: {metrics['avg_elapsed_time']:.2f}s")
        report_lines.append(f"  Vote distribution:")
        for vote, count in metrics['vote_distribution'].most_common():
            report_lines.append(f"    {vote}: {count}")
    report_lines.append("")
    
    report_lines.append("="*80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-model evaluation results")
    
    parser.add_argument("--input", required=True,
                       help="Input JSONL file with model responses")
    parser.add_argument("--output", required=True,
                       help="Output directory for analysis")
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"Step 3: Analyze Results")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    print(f"\nLoading results...")
    results = load_results(args.input)
    print(f"✓ Loaded {len(results)} results")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    
    agreement_metrics = compute_inter_model_agreement(results)
    print("✓ Inter-model agreement")
    
    gt_metrics = compute_gt_agreement(results)
    print("✓ GT agreement")
    
    uncertainty_metrics = compute_uncertainty_analysis(results)
    print("✓ Uncertainty analysis")
    
    model_metrics = compute_model_specific_metrics(results)
    print("✓ Model-specific metrics")
    
    # NEW: Signal-specific metrics
    signal_agreement = compute_signal_agreement(results)
    print("✓ Signal agreement (multi-label)")
    
    signal_gt_alignment = compute_signal_gt_alignment(results)
    print("✓ Level signal vs GT alignment")
    
    # Save metrics
    all_metrics = {
        'agreement': agreement_metrics,
        'gt_agreement': gt_metrics,
        'uncertainty': uncertainty_metrics,
        'model_specific': model_metrics,
        'signal_agreement': signal_agreement,
        'signal_gt_alignment': signal_gt_alignment
    }
    
    metrics_file = os.path.join(args.output, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metrics to {metrics_file}")
    
    # Generate report
    print(f"\nGenerating report...")
    generate_report(agreement_metrics, gt_metrics, uncertainty_metrics, 
                   model_metrics, signal_agreement, signal_gt_alignment, args.output)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
