"""Test General rowspan extraction"""
from src.data_preprocess.step2_gitcard_tab import detect_and_extract_markdown_tables
import pandas as pd
import re

# Load the original HTML content
df = pd.read_parquet('data/processed/modelcard_step1.parquet', columns=['modelId', 'card_readme'])
model_row = df[df['modelId'] == 'jartine/Meta-Llama-3-70B-Instruct-llamafile']
content = model_row.iloc[0]['card_readme']

# Extract HTML tables
tables = re.findall(r'<table.*?</table>', content, re.DOTALL)
print(f'Found {len(tables)} HTML tables')

# Extract table 3 (index 2)
if len(tables) >= 3:
    table3_html = tables[2]
    print(f'\nExtracting table 3 with NEW logic...')
    found, extracted_tables = detect_and_extract_markdown_tables(table3_html)
    
    if extracted_tables:
        table_md = extracted_tables[0]
        lines = table_md.split('\n')
        print(f'\nFirst 12 lines of extracted table:')
        for i, line in enumerate(lines[:12], 1):
            print(f'{i}: {line}')
        
        # Count "General" occurrences
        general_count = lines[0].count('General')
        print(f'\n\n❓ How many times does "General" appear in first row?')
        print(f'   Answer: {general_count} times')
        
        # Show Category column (first column of data rows)
        print(f'\n📊 Category column (first data column) for rows 2-9:')
        for i in range(2, min(10, len(lines))):
            cells = lines[i].split('|')
            if len(cells) > 1:
                print(f'   Row {i}: "{cells[1]}"' ) # First data column (after leading |)
else:
    print("Could not find table 3")

