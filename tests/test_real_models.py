"""
Test table extraction on real modelcards
"""
import pandas as pd
import os
from src.utils import load_config
from src.data_preprocess.step2_hugging_github_extract import detect_and_extract_markdown_tables

config = load_config('config.yaml')
processed_base_path = os.path.join(config.get('base_path'), 'processed')

# Load card data
step1_file = os.path.join(processed_base_path, "modelcard_step1.parquet")
df = pd.read_parquet(step1_file, columns=['modelId', 'card_readme'])

# Test different types of models
test_models = [
    # 1. Has HTML table
    "jartine/Meta-Llama-3-70B-Instruct-llamafile",
    
    # 2. Has markdown table
    "espnet/sluevoxceleb_whisper_finetune_sa",
    
    # 3. Has badge table (should filter)
    "rishitdagli/see-2-sound",
    
    # 4. Random popular model
    "bert-base-uncased",
]

print("="*80)
print("TESTING TABLE EXTRACTION ON REAL MODELS")
print("="*80)

for model_id in test_models:
    print(f"\n{'='*80}")
    print(f"MODEL: {model_id}")
    print(f"{'='*80}")
    
    model_row = df[df['modelId'] == model_id]
    
    if len(model_row) == 0:
        print(f"❌ Model not found in dataset")
        continue
    
    card_content = model_row.iloc[0]['card_readme']
    
    if not card_content or pd.isna(card_content):
        print(f"❌ Card content is empty")
        continue
    
    print(f"Card length: {len(card_content)} chars")
    
    # Count HTML tables
    html_count = card_content.count('<table')
    # Count markdown-like pipes
    pipe_lines = [line for line in card_content.split('\n') if '|' in line]
    
    print(f"HTML <table> tags: {html_count}")
    print(f"Lines with '|': {len(pipe_lines)}")
    
    # Extract tables
    found, tables = detect_and_extract_markdown_tables(card_content)
    
    print(f"\nExtraction result:")
    print(f"  Found: {found}")
    print(f"  Tables extracted: {len(tables)}")
    
    if tables:
        for i, table in enumerate(tables, 1):
            lines = table.split('\n')
            rows = [ln for ln in lines if ln.strip() and not ln.strip().startswith('|---')]
            
            # Determine type
            if '<table>' in table or '<tr>' in table:
                table_type = "HTML"
            else:
                table_type = "Markdown"
            
            # Count columns
            if rows:
                first_row = rows[0]
                cols = len([c for c in first_row.split('|') if c.strip()])
                
                # Preview
                preview_lines = lines[:3]
                preview = ' | '.join([ln[:40] for ln in preview_lines[:2]])
                
                print(f"\n  Table {i}: {table_type}, {len(rows)} rows, ~{cols} cols")
                print(f"    Preview: {preview[:80]}...")
                
                # Check if it looks like real data
                has_numbers = any(char.isdigit() for char in table)
                has_text = len([c for c in table if c.isalpha()]) > 20
                
                if has_numbers and has_text:
                    print(f"    ✅ Looks like real data table")
                elif '<|' in table and '|>' in table:
                    print(f"    ⚠️  Contains special tokens (might be code)")
                elif 'badge' in table.lower() or 'img.shields.io' in table:
                    print(f"    ⚠️  Contains badges (decoration)")
                else:
                    print(f"    ℹ️  Other content")
    else:
        print("  No tables extracted")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("All tests completed. Check if:")
print("  1. HTML tables are extracted ✅")
print("  2. Markdown tables are extracted ✅")
print("  3. No duplicates between HTML and markdown ✅")
print("  4. Badge tables are filtered ✅")

