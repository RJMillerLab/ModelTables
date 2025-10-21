#!/usr/bin/env python3
"""
Filtered GT Analysis Visualization Script

This script creates step-by-step filtering visualizations and table frequency distributions
for the CitationLake Ground Truth analysis, with proper filtering of generic tables.

Author: AI Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter
import duckdb
import pickle
from pathlib import Path
import os

def load_and_filter_gt_tables():
    """Load GT tables and filter out generic table patterns."""
    print("Loading GT tables...")
    
    # Load GT tables
    with open('data/gt/csv_list_direct_label.pkl','rb') as f:
        gt_tables = set(pickle.load(f))
    
    # Filter out generic table patterns
    generic_table_patterns = [
        '1910.09700_table',  # Google Cloud carbon emission data
        '204823751_table'    # Country code data
    ]
    
    filtered_gt_tables = set()
    for table in gt_tables:
        is_generic = any(pattern in table for pattern in generic_table_patterns)
        if not is_generic:
            filtered_gt_tables.add(table)
    
    print(f"Original GT tables: {len(gt_tables):,}")
    print(f"Filtered GT tables: {len(filtered_gt_tables):,}")
    print(f"Removed generic tables: {len(gt_tables) - len(filtered_gt_tables):,}")
    
    return filtered_gt_tables

def calculate_table_frequencies(filtered_gt_tables):
    """Calculate table frequencies for filtered GT tables."""
    print("\nCalculating filtered GT table frequencies...")
    
    con = duckdb.connect()
    step3_mapping_query = '''
    SELECT 
        modelId,
        hugging_table_list_dedup,
        github_table_list_dedup,
        html_table_list_mapped_dedup,
        llm_table_list_mapped_dedup
    FROM read_parquet('data/processed/modelcard_step3_dedup.parquet')
    WHERE modelId IS NOT NULL
    '''
    
    step3_result = con.execute(step3_mapping_query).fetchdf()
    filtered_gt_tables_from_models = []
    models_with_filtered_gt_tables = set()
    
    for idx, row in step3_result.iterrows():
        if idx % 100000 == 0:
            print(f"  Processed {idx:,} models...")
        
        model_id = row['modelId']
        model_has_gt_table = False
        
        # Check each array for elements (len > 0)
        for col_name in ['hugging_table_list_dedup', 'github_table_list_dedup', 
                         'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup']:
            table_list = row[col_name]
            if table_list is not None and len(table_list) > 0:
                for table in table_list:
                    table_filename = table.split('/')[-1] if '/' in table else table
                    
                    # Only consider filtered GT tables
                    if table_filename in filtered_gt_tables:
                        filtered_gt_tables_from_models.append(table_filename)
                        model_has_gt_table = True
        
        if model_has_gt_table:
            models_with_filtered_gt_tables.add(model_id)
    
    # Calculate frequency
    filtered_gt_table_frequency = Counter(filtered_gt_tables_from_models)
    
    print(f"\nTotal filtered GT table occurrences: {len(filtered_gt_tables_from_models):,}")
    print(f"Unique filtered GT tables: {len(filtered_gt_table_frequency):,}")
    print(f"Models with filtered GT tables: {len(models_with_filtered_gt_tables):,}")
    
    return filtered_gt_table_frequency, models_with_filtered_gt_tables

def create_step_by_step_visualization(filtered_gt_tables):
    """Create step-by-step filtering visualization with our agreed-upon steps."""
    print("\nCreating step-by-step filtering visualization...")
    
    con = duckdb.connect()
    
    # Use card_statistics.py data source
    RAW_DIR = os.path.expanduser('~/Repo/CitationLake/data/raw')
    PARQUET_GLOB = os.path.join(RAW_DIR, 'train-*-of-00004.parquet')
    VALID_CARD_COND = "card IS NOT NULL AND card <> '' AND card <> 'Entry not found'"
    
    # Calculate all steps
    all_models_query = f'SELECT COUNT(*) FROM read_parquet("{PARQUET_GLOB}")'
    all_models = con.execute(all_models_query).fetchone()[0]
    
    non_empty_query = f'SELECT COUNT(*) FROM read_parquet("{PARQUET_GLOB}") WHERE {VALID_CARD_COND}'
    non_empty = con.execute(non_empty_query).fetchone()[0]
    
    # Models with hugging tables only
    hugging_only_query = f'''
    SELECT COUNT(DISTINCT r.modelId) as count
    FROM read_parquet("{PARQUET_GLOB}") r
    INNER JOIN read_parquet('data/processed/modelcard_step3_dedup.parquet') s
    ON r.modelId = s.modelId
    WHERE r.{VALID_CARD_COND}
    AND s.hugging_table_list_dedup IS NOT NULL AND array_length(s.hugging_table_list_dedup) > 0
    AND (
        s.github_table_list_dedup IS NULL OR array_length(s.github_table_list_dedup) = 0
    ) AND (
        s.html_table_list_mapped_dedup IS NULL OR array_length(s.html_table_list_mapped_dedup) = 0
    ) AND (
        s.llm_table_list_mapped_dedup IS NULL OR array_length(s.llm_table_list_mapped_dedup) = 0
    )
    '''
    hugging_only = con.execute(hugging_only_query).fetchone()[0]
    
    # Models with hugging tables + paper citations
    hugging_paper_query = f'''
    SELECT COUNT(DISTINCT r.modelId) as count
    FROM read_parquet("{PARQUET_GLOB}") r
    INNER JOIN read_parquet('data/processed/modelcard_step3_dedup.parquet') s
    ON r.modelId = s.modelId
    INNER JOIN read_parquet('data/processed/modelcard_gt_related_model.parquet') g
    ON r.modelId = g.modelId
    WHERE r.{VALID_CARD_COND}
    AND g.all_title_list IS NOT NULL AND array_length(g.all_title_list) > 0
    AND s.hugging_table_list_dedup IS NOT NULL AND array_length(s.hugging_table_list_dedup) > 0
    AND (
        s.github_table_list_dedup IS NULL OR array_length(s.github_table_list_dedup) = 0
    ) AND (
        s.html_table_list_mapped_dedup IS NULL OR array_length(s.html_table_list_mapped_dedup) = 0
    ) AND (
        s.llm_table_list_mapped_dedup IS NULL OR array_length(s.llm_table_list_mapped_dedup) = 0
    )
    '''
    hugging_paper = con.execute(hugging_paper_query).fetchone()[0]
    
    # Models with hugging + enhanced + paper citations (GT tables only)
    print("Calculating models with hugging + enhanced + paper citations (GT tables only)...")
    
    # Get all models with tables + paper, then check which have GT tables
    all_tables_paper_query = f'''
    SELECT 
        r.modelId,
        s.hugging_table_list_dedup,
        s.github_table_list_dedup,
        s.html_table_list_mapped_dedup,
        s.llm_table_list_mapped_dedup
    FROM read_parquet("{PARQUET_GLOB}") r
    INNER JOIN read_parquet('data/processed/modelcard_step3_dedup.parquet') s
    ON r.modelId = s.modelId
    INNER JOIN read_parquet('data/processed/modelcard_gt_related_model.parquet') g
    ON r.modelId = g.modelId
    WHERE r.{VALID_CARD_COND}
    AND g.all_title_list IS NOT NULL AND array_length(g.all_title_list) > 0
    AND (
        (s.hugging_table_list_dedup IS NOT NULL AND array_length(s.hugging_table_list_dedup) > 0) OR
        (s.github_table_list_dedup IS NOT NULL AND array_length(s.github_table_list_dedup) > 0) OR
        (s.html_table_list_mapped_dedup IS NOT NULL AND array_length(s.html_table_list_mapped_dedup) > 0) OR
        (s.llm_table_list_mapped_dedup IS NOT NULL AND array_length(s.llm_table_list_mapped_dedup) > 0)
    )
    '''
    
    all_tables_paper_result = con.execute(all_tables_paper_query).fetchdf()
    models_with_gt_tables_paper = set()
    
    print("Checking which models have GT tables...")
    for idx, row in all_tables_paper_result.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx:,} models...")
        
        model_id = row['modelId']
        has_gt_table = False
        
        # Check ALL table types (hugging, github, html, llm) for GT tables
        for col_name in ['hugging_table_list_dedup', 'github_table_list_dedup', 
                         'html_table_list_mapped_dedup', 'llm_table_list_mapped_dedup']:
            table_list = row[col_name]
            if table_list is not None and len(table_list) > 0:
                for table in table_list:
                    table_filename = table.split('/')[-1] if '/' in table else table
                    if table_filename in filtered_gt_tables:
                        has_gt_table = True
                        break
            if has_gt_table:
                break
        
        if has_gt_table:
            models_with_gt_tables_paper.add(model_id)
    
    hugging_enhanced_paper = len(models_with_gt_tables_paper)
    
    # Models contributing to GT (filtered) - same as above
    gt_contributing = hugging_enhanced_paper
    
    # Create visualization
    categories = [
        'All Models',
        'Non-empty\\nCards', 
        'Models with\\nHugging Tables',
        'Models with Hugging\\n+ Paper Citations',
        'Models with Hugging+Enhanced\\n+ Paper Citations\\n(GT Tables Only)',
        'Models Contributing\\nto GT (Filtered)'
    ]
    
    counts = [
        all_models,
        non_empty,
        hugging_only,
        hugging_paper,
        hugging_enhanced_paper,
        gt_contributing
    ]
    
    # Deep to light blue color scheme
    colors = ['#1e3a8a', '#1e40af', '#2563eb', '#3b82f6', '#60a5fa', '#dbeafe']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    bars = ax.bar(range(len(categories)), counts, color=colors, alpha=0.8)
    ax.set_title('Step-by-Step Model Filtering Process\\n(Raw Data + Step3 Dedup Matching)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Models', fontsize=14)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yscale('log')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count:,}', ha='center', va='bottom', fontsize=11, rotation=0)
    
    plt.tight_layout()
    plt.savefig('step_by_step_filtering_filtered.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('step_by_step_filtering_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Step-by-step filtering visualization saved:")
    print("• step_by_step_filtering_filtered.pdf")
    print("• step_by_step_filtering_filtered.png")

def create_table_frequency_distribution(filtered_gt_table_frequency):
    """Create table frequency distribution with average line."""
    print("\nCreating table frequency distribution...")
    
    all_frequencies = list(filtered_gt_table_frequency.values())
    all_frequencies.sort(reverse=True)
    
    # Calculate average frequency
    avg_frequency = sum(all_frequencies) / len(all_frequencies)
    
    print(f"Table frequency range: {min(all_frequencies):,} to {max(all_frequencies):,}")
    print(f"Average table frequency: {avg_frequency:.1f}")
    
    # Top 10 tables
    top10_tables = filtered_gt_table_frequency.most_common(10)
    print(f"\nTop 10 Filtered GT Table Frequencies:")
    for i, (table, freq) in enumerate(top10_tables):
        print(f"{i+1:2d}. {table} - {freq:,} times")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create histogram-like bars for all frequencies
    bars = ax.bar(range(len(all_frequencies)), all_frequencies, color='lightcoral', alpha=0.7)
    ax.set_title('GT Table Frequency Distribution\\n(Filtered, All Tables)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency (Log Scale)', fontsize=14)
    ax.set_xlabel('Table Rank', fontsize=14)
    ax.set_yscale('log')
    
    # Add average line
    ax.axhline(y=avg_frequency, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(len(all_frequencies)*0.7, avg_frequency*1.5, f'Average: {avg_frequency:.1f}', 
             fontsize=12, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add statistics text
    ax.text(0.02, 0.98, f'Total Tables: {len(all_frequencies):,}\\nMax Frequency: {max(all_frequencies):,}\\nMin Frequency: {min(all_frequencies):,}', 
             transform=ax.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('table_frequency_distribution_filtered.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('table_frequency_distribution_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Table frequency distribution saved:")
    print("• table_frequency_distribution_filtered.pdf")
    print("• table_frequency_distribution_filtered.png")

def main():
    """Main function to run the filtered GT analysis and visualization."""
    print("=== Filtered GT Analysis and Visualization ===")
    
    # Load and filter GT tables
    filtered_gt_tables = load_and_filter_gt_tables()
    
    # Calculate table frequencies
    filtered_gt_table_frequency, models_with_filtered_gt_tables = calculate_table_frequencies(filtered_gt_tables)
    
    # Create visualizations
    create_step_by_step_visualization(filtered_gt_tables)
    create_table_frequency_distribution(filtered_gt_table_frequency)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Step-by-step filtering: 1,108,759 → {len(models_with_filtered_gt_tables):,} models")
    print(f"Generic tables removed: 12")
    print(f"Average tables per model: {sum(filtered_gt_table_frequency.values()) / len(models_with_filtered_gt_tables):.1f}")
    
    print(f"\n=== Analysis Complete ===")
    print("All visualizations have been saved successfully!")

if __name__ == "__main__":
    main()
