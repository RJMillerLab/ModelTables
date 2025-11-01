#!/usr/bin/env python3
"""
Table Counts and Model Counts Over Time Visualization

This script creates a visualization showing:
1. Cumulative table counts over time
2. Cumulative model counts over time

Using efficient SQL queries with DuckDB for fast processing.

Author: Zhengyuan Dong
Date: 2025-01-XX
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_raw_parquet_files(config, pattern='train-*-of-00004.parquet'):
    """Get list of raw parquet files."""
    raw_base_path = os.path.join(config.get('base_path', 'data/'), 'raw')
    parquet_glob = os.path.join(raw_base_path, pattern)
    return parquet_glob

def build_table_count_query(step3_dedup_path, valid_title_path, raw_parquet_glob):
    """Build SQL query to count tables over time.
    
    Concatenates all table lists (hugging, github, html, llm) and counts unique tables per date.
    Tracks when each unique table first appears, then calculates cumulative counts.
    Filters by valid titles (models with all_title_list_valid) and excludes generic tables.
    """
    query = f"""
    WITH raw_data AS (
        SELECT 
            modelId,
            CAST(createdAt AS DATE) as created_date
        FROM read_parquet('{raw_parquet_glob}')
        WHERE modelId IS NOT NULL 
        AND createdAt IS NOT NULL
    ),
    step3_data AS (
        SELECT 
            modelId,
            hugging_table_list_dedup,
            github_table_list_dedup,
            html_table_list_mapped_dedup,
            llm_table_list_mapped_dedup
        FROM read_parquet('{step3_dedup_path}')
        WHERE modelId IS NOT NULL
    ),
    valid_title_data AS (
        SELECT 
            modelId,
            all_title_list_valid
        FROM read_parquet('{valid_title_path}')
        WHERE modelId IS NOT NULL
        AND all_title_list_valid IS NOT NULL
        AND array_length(all_title_list_valid) > 0
    ),
    merged_data AS (
        SELECT 
            r.modelId,
            r.created_date,
            s.hugging_table_list_dedup,
            s.github_table_list_dedup,
            s.html_table_list_mapped_dedup,
            s.llm_table_list_mapped_dedup
        FROM raw_data r
        INNER JOIN step3_data s
        ON r.modelId = s.modelId
        INNER JOIN valid_title_data v
        ON r.modelId = v.modelId
        WHERE (
            (s.hugging_table_list_dedup IS NOT NULL AND array_length(s.hugging_table_list_dedup) > 0) OR
            (s.github_table_list_dedup IS NOT NULL AND array_length(s.github_table_list_dedup) > 0) OR
            (s.html_table_list_mapped_dedup IS NOT NULL AND array_length(s.html_table_list_mapped_dedup) > 0) OR
            (s.llm_table_list_mapped_dedup IS NOT NULL AND array_length(s.llm_table_list_mapped_dedup) > 0)
        )
    ),
    -- Unnest all table lists and filter out generic tables
    table_unnested AS (
        SELECT 
            created_date,
            UNNEST(COALESCE(hugging_table_list_dedup, [])) as table_path
        FROM merged_data
        WHERE hugging_table_list_dedup IS NOT NULL 
        AND array_length(hugging_table_list_dedup) > 0
        
        UNION ALL
        
        SELECT 
            created_date,
            UNNEST(COALESCE(github_table_list_dedup, [])) as table_path
        FROM merged_data
        WHERE github_table_list_dedup IS NOT NULL 
        AND array_length(github_table_list_dedup) > 0
        
        UNION ALL
        
        SELECT 
            created_date,
            UNNEST(COALESCE(html_table_list_mapped_dedup, [])) as table_path
        FROM merged_data
        WHERE html_table_list_mapped_dedup IS NOT NULL 
        AND array_length(html_table_list_mapped_dedup) > 0
        
        UNION ALL
        
        SELECT 
            created_date,
            UNNEST(COALESCE(llm_table_list_mapped_dedup, [])) as table_path
        FROM merged_data
        WHERE llm_table_list_mapped_dedup IS NOT NULL 
        AND array_length(llm_table_list_mapped_dedup) > 0
    ),
    -- Filter out generic tables (1910.09700_table, 204823751_table)
    filtered_tables AS (
        SELECT 
            created_date,
            table_path,
            regexp_extract(table_path, '([^/\\\\]+)$', 1) as table_filename
        FROM table_unnested
        WHERE table_path IS NOT NULL 
        AND table_path != ''
        AND table_path NOT LIKE '%1910.09700_table%'
        AND table_path NOT LIKE '%204823751_table%'
    ),
    -- Get first appearance date for each unique table
    table_first_appearance AS (
        SELECT 
            table_path,
            MIN(created_date) as first_date
        FROM filtered_tables
        GROUP BY table_path
    ),
    -- Count new tables per date
    new_tables_per_date AS (
        SELECT 
            first_date as created_date,
            COUNT(*) as new_table_count
        FROM table_first_appearance
        GROUP BY first_date
    )
    -- Calculate cumulative table count
    SELECT 
        created_date,
        SUM(new_table_count) OVER (ORDER BY created_date ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumulative_table_count
    FROM new_tables_per_date
    ORDER BY created_date
    """
    return query

def build_model_count_query(step3_dedup_path, valid_title_path, raw_parquet_glob):
    """Build SQL query to count models over time.
    
    Only counts models with valid titles (all_title_list_valid) and tables.
    """
    query = f"""
    WITH raw_data AS (
        SELECT 
            modelId,
            CAST(createdAt AS DATE) as created_date
        FROM read_parquet('{raw_parquet_glob}')
        WHERE modelId IS NOT NULL 
        AND createdAt IS NOT NULL
    ),
    step3_data AS (
        SELECT 
            modelId,
            hugging_table_list_dedup,
            github_table_list_dedup,
            html_table_list_mapped_dedup,
            llm_table_list_mapped_dedup
        FROM read_parquet('{step3_dedup_path}')
        WHERE modelId IS NOT NULL
        AND (
            (hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0) OR
            (github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0) OR
            (html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0) OR
            (llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0)
        )
    ),
    valid_title_data AS (
        SELECT 
            modelId,
            all_title_list_valid
        FROM read_parquet('{valid_title_path}')
        WHERE modelId IS NOT NULL
        AND all_title_list_valid IS NOT NULL
        AND array_length(all_title_list_valid) > 0
    ),
    models_with_tables AS (
        SELECT DISTINCT
            r.modelId,
            r.created_date
        FROM raw_data r
        INNER JOIN step3_data s
        ON r.modelId = s.modelId
        INNER JOIN valid_title_data v
        ON r.modelId = v.modelId
    ),
    models_per_date AS (
        SELECT 
            created_date,
            COUNT(DISTINCT modelId) as model_count
        FROM models_with_tables
        GROUP BY created_date
    )
    SELECT 
        created_date,
        SUM(model_count) OVER (ORDER BY created_date ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumulative_model_count
    FROM models_per_date
    ORDER BY created_date
    """
    return query

def execute_query(con, query, description):
    """Execute SQL query and return DataFrame."""
    print(f"\n📊 {description}...")
    print("Executing SQL query...")
    try:
        df = con.execute(query).fetchdf()
        print(f"✅ Retrieved {len(df):,} rows")
        return df
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        raise

def create_visualization(table_df, model_df, output_dir='.', output_suffix=''):
    """Create visualization with two curves showing table counts and model counts over time."""
    print("\n🎨 Creating visualization...")
    
    # Convert date columns to datetime
    table_df['created_date'] = pd.to_datetime(table_df['created_date'])
    model_df['created_date'] = pd.to_datetime(model_df['created_date'])
    
    # Merge on date to align both series
    merged_df = pd.merge(
        table_df.rename(columns={'cumulative_table_count': 'table_count'}),
        model_df.rename(columns={'cumulative_model_count': 'model_count'}),
        on='created_date',
        how='outer'
    ).sort_values('created_date')
    
    # Forward fill missing values for cumulative counts
    merged_df['table_count'] = merged_df['table_count'].ffill().fillna(0)
    merged_df['model_count'] = merged_df['model_count'].ffill().fillna(0)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot table counts on left y-axis
    color1 = '#d96e44'  # Orange-red color (matching paper)
    ax1.set_xlabel('Date', fontsize=17, fontweight='bold')
    ax1.set_ylabel('Cumulative Table Count', fontsize=24, fontweight='bold', color=color1)
    line1 = ax1.plot(merged_df['created_date'], merged_df['table_count'], 
                     color=color1, linewidth=2.5, label='Cumulative Table Count', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=17)
    ax1.tick_params(axis='x', labelsize=17)
    ax1.grid(True, alpha=0.3)
    
    # Plot model counts on right y-axis
    ax2 = ax1.twinx()
    color2 = '#486f90'  # Deep blue color (matching paper)
    ax2.set_ylabel('Cumulative Model Count', fontsize=24, fontweight='bold', color=color2)
    line2 = ax2.plot(merged_df['created_date'], merged_df['model_count'], 
                     color=color2, linewidth=2.5, label='Cumulative Model Count', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=17)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=17)
    
    # Set title (caption is handled separately in LaTeX/Overleaf)
    plt.title('Table Counts and Model Counts Over Time', 
              fontsize=22, fontweight='bold', pad=20)
    
    # Add combined legend to avoid overlap with y-axis labels
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=17, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, f'table_model_counts_over_time{output_suffix}')
    plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved:")
    print(f"   • {output_base}.pdf")
    print(f"   • {output_base}.png")
    
    return merged_df

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Visualize table counts and model counts over time'
    )
    parser.add_argument('--step3-dedup', type=str, 
                       default='data/processed/modelcard_step3_dedup.parquet',
                       help='Path to step3_dedup parquet file')
    parser.add_argument('--raw-dir', type=str, default=None,
                       help='Directory containing raw parquet files (default: from config)')
    parser.add_argument('--raw-pattern', type=str, 
                       default='train-*-of-00004.parquet',
                       help='Pattern for raw parquet files')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config.yaml file')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save output files')
    parser.add_argument('--output-suffix', type=str, default='',
                       help='Suffix to append to output filenames')
    parser.add_argument('--use-v2', action='store_true',
                       help='Use v2 version of step3_dedup file')
    
    args = parser.parse_args()
    
    # Use v2 if requested
    if args.use_v2:
        if 'step3_dedup_v2' not in args.step3_dedup:
            args.step3_dedup = args.step3_dedup.replace('step3_dedup.parquet', 'step3_dedup_v2.parquet')
        if '_v2' not in args.output_suffix:
            args.output_suffix = f'_v2{args.output_suffix}' if args.output_suffix else '_v2'
        print(f"🔧 V2 mode enabled")
    
    # Load config
    config = load_config(args.config)
    
    # Get raw parquet files path
    if args.raw_dir:
        raw_parquet_glob = os.path.join(args.raw_dir, args.raw_pattern)
    else:
        raw_base_path = os.path.join(config.get('base_path', 'data/'), 'raw')
        raw_parquet_glob = os.path.join(raw_base_path, args.raw_pattern)
    
    # Expand user path
    raw_parquet_glob = os.path.expanduser(raw_parquet_glob)
    step3_dedup_path = os.path.expanduser(args.step3_dedup)
    
    print("="*60)
    print("📊 Table and Model Counts Over Time Visualization")
    print("="*60)
    print(f"Step3 dedup path: {step3_dedup_path}")
    print(f"Raw parquet glob: {raw_parquet_glob}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output suffix: {args.output_suffix}")
    
    # Initialize DuckDB connection
    con = duckdb.connect()
    
    # Build and execute queries
    # Determine valid_title_path based on v2 mode
    if args.use_v2:
        valid_title_path = os.path.expanduser('data/processed/all_title_list_valid.parquet')
    else:
        valid_title_path = os.path.expanduser('data/processed/all_title_list_valid.parquet')
    
    table_query = build_table_count_query(step3_dedup_path, valid_title_path, raw_parquet_glob)
    model_query = build_model_count_query(step3_dedup_path, valid_title_path, raw_parquet_glob)
    
    table_df = execute_query(con, table_query, "Calculating cumulative table counts")
    model_df = execute_query(con, model_query, "Calculating cumulative model counts")
    
    # Create visualization
    merged_df = create_visualization(table_df, model_df, args.output_dir, args.output_suffix)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📈 Summary Statistics")
    print("="*60)
    print(f"Date range: {merged_df['created_date'].min()} to {merged_df['created_date'].max()}")
    print(f"Final cumulative table count: {int(merged_df['table_count'].max()):,}")
    print(f"Final cumulative model count: {int(merged_df['model_count'].max()):,}")
    if merged_df['model_count'].max() > 0:
        print(f"Average tables per model: {merged_df['table_count'].max() / merged_df['model_count'].max():.2f}")
    else:
        print("Average tables per model: N/A (no models found)")
    print("="*60)
    
    # Close connection
    con.close()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

