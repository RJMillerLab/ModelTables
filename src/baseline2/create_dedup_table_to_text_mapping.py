import pandas as pd
import os
import json
import numpy as np
from pathlib import Path
import shutil
import re
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm

def create_tmp_folders():
    """
    Create temporary folders for processed text and embeddings.
    """
    tmp_dirs = {
        'readmes': os.path.join('data', 'tmp', 'processed_readmes'),
        'embeddings': {
            'single': os.path.join('data', 'tmp', 'embeddings', 'single'),
            'group': os.path.join('data', 'tmp', 'embeddings', 'group')
        }
    }
    
    # Create main directories
    for dir_path in [tmp_dirs['readmes'], tmp_dirs['embeddings']['single'], tmp_dirs['embeddings']['group']]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    return tmp_dirs

def remove_tables_from_text(text):
    """
    Remove all markdown tables from the text.
    Supports:
    1. Standard markdown tables
    2. HTML tables
    3. LaTeX tables
    """
    if not isinstance(text, str):
        return text
    
    # Pattern to match markdown tables (same as in step2_gitcard_tab.py)
    markdown_table_pattern = r"(?:\|[^\n]*?\|[\s]*\n)+\|[-:| ]*\|[\s]*\n(?:\|[^\n]*?\|(?:\n|$))+"
    
    # Pattern to match HTML tables
    html_table_pattern = r"<table[^>]*>.*?</table>"
    
    # Pattern to match LaTeX tables
    latex_table_pattern = r"\\begin{table}.*?\\end{table}"
    
    # Remove markdown tables
    text = re.sub(markdown_table_pattern, '', text, flags=re.DOTALL)
    
    # Remove HTML tables
    text = re.sub(html_table_pattern, '', text, flags=re.DOTALL)
    
    # Remove LaTeX tables
    text = re.sub(latex_table_pattern, '', text, flags=re.DOTALL)
    
    return text

def extract_table_caption(html_content, table_name):
    """
    Extract all table captions from HTML content.
    For arXiv: extract all table captions regardless of table number to avoid formula interference.
    """
    if not isinstance(html_content, str):
        return ""
    
    # Patterns to match any table captions (not specific to table number)
    caption_patterns = [
        r'<caption[^>]*>.*?Table\s*\d+[.:]\s*([^<]*)</caption>',
        r'Table\s*\d+[.:]\s*([^\n\r]*)',
        r'TABLE\s*\d+[.:]\s*([^\n\r]*)',
        r'<p[^>]*>\s*Table\s*\d+[.:]\s*([^<]*)</p>',
    ]
    
    captions = []
    for pattern in caption_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            caption = re.sub(r'<[^>]+>', '', match).strip()  # Remove HTML tags
            if caption and len(caption) > 10:  # Filter out very short captions
                captions.append(caption)
    
    # Return unique captions
    unique_captions = list(set(captions))
    return '\n'.join(unique_captions) if unique_captions else ""

def extract_table_context_paragraphs(html_content, table_name):
    """
    Extract all paragraphs that mention any table.
    For arXiv: extract all table-related paragraphs to avoid issues with formula interference.
    """
    if not isinstance(html_content, str):
        return ""
    
    # Remove HTML tables first to avoid matching within table content
    content_no_tables = re.sub(r'<table[^>]*>.*?</table>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Split into paragraphs (by <p> tags or double newlines)
    paragraphs = []
    
    # Extract paragraphs from <p> tags
    p_paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', content_no_tables, re.DOTALL | re.IGNORECASE)
    paragraphs.extend(p_paragraphs)
    
    # Also try splitting by double newlines for plain text sections
    text_content = re.sub(r'<[^>]+>', ' ', content_no_tables)  # Remove remaining HTML tags
    text_paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
    paragraphs.extend(text_paragraphs)
    
    # Patterns to match references to any table (not specific table number)
    table_reference_patterns = [
        r'\btable\s*\d+\b',
        r'\bTable\s*\d+\b',
        r'\bTABLE\s*\d+\b',
        r'\btab\.\s*\d+\b',
        r'\bTab\.\s*\d+\b',
        r'\btables?\b',  # Also capture general mentions of "table" or "tables"
        r'\bTables?\b',
        r'\bTABLES?\b',
    ]
    
    relevant_paragraphs = []
    for paragraph in paragraphs:
        # Clean HTML tags from paragraph
        clean_paragraph = re.sub(r'<[^>]+>', ' ', paragraph).strip()
        if not clean_paragraph or len(clean_paragraph) < 20:  # Skip very short paragraphs
            continue
            
        # Check if paragraph mentions any table
        for pattern in table_reference_patterns:
            if re.search(pattern, clean_paragraph, re.IGNORECASE):
                relevant_paragraphs.append(clean_paragraph)
                break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paragraphs = []
    for para in relevant_paragraphs:
        if para not in seen:
            unique_paragraphs.append(para)
            seen.add(para)
    
    return '\n\n'.join(unique_paragraphs)

def extract_all_table_content(html_content, table_name=None):
    """
    Extract table-related content from HTML (NO table data, only text content):
    1. All table captions
    2. All paragraphs mentioning tables  
    3. Any other table-related text mentions
    
    For arXiv: comprehensive text extraction without actual table data
    """
    if not isinstance(html_content, str):
        return ""
    
    all_content = []
    
    # 1. Extract all table captions (including those outside <table> tags)
    captions = extract_table_caption(html_content, table_name or "")
    if captions:
        all_content.append("=== TABLE CAPTIONS ===\n" + captions)
    
    # 2. Extract all table context paragraphs
    context = extract_table_context_paragraphs(html_content, table_name or "")
    if context:
        all_content.append("=== TABLE CONTEXT ===\n" + context)
    
    # 3. Extract any other table-related text patterns
    other_patterns = [
        r'(?:see|refer to|shown in|presented in|listed in)\s+table\s*\d*',
        r'table\s*\d*\s+(?:shows|presents|lists|demonstrates|indicates)',
        r'(?:above|below|following|preceding)\s+table',
        r'tabular\s+(?:data|results|format)',
        r'as\s+(?:shown|presented|indicated)\s+in\s+table',
        r'table\s+\d+\s+(?:shows|demonstrates|presents)',
        r'according\s+to\s+table\s*\d*',
        r'table\s*\d*\s+(?:reveals|confirms|illustrates)',
    ]
    
    other_mentions = []
    content_no_tables = re.sub(r'<table[^>]*>.*?</table>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    clean_content = re.sub(r'<[^>]+>', ' ', content_no_tables)
    
    for pattern in other_patterns:
        matches = re.finditer(pattern, clean_content, re.IGNORECASE)
        for match in matches:
            # Get surrounding context (100 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(clean_content), match.end() + 100)
            context_snippet = clean_content[start:end].strip()
            if context_snippet and len(context_snippet) > 20:
                other_mentions.append(context_snippet)
    
    if other_mentions:
        unique_mentions = list(dict.fromkeys(other_mentions))  # Remove duplicates while preserving order
        all_content.append("=== OTHER TABLE MENTIONS ===\n" + '\n\n'.join(unique_mentions))
    
    # Join all content sections
    return '\n\n'.join(all_content) if all_content else ""

def process_readme_content(row, tmp_dir):
    """
    Process readme content based on source type.
    For arXiv: extract ALL table-related content (captions, data, context, mentions) - comprehensive approach
    For Huggingface: directly process the content and save as modelId.txt
    For GitHub: read from file and process, keeping original filename
    """
    source = row['source']
    readme_path = row['readme_path']
    csv_path = row['csv_path']  # Original CSV path for filename
    dedup_csv_path = row['dedup_csv_path']  # Use this for table number extraction
    
    if source == 'huggingface':
        # Huggingface already has the content
        content = readme_path
        # Use modelId as filename for huggingface content
        model_id = row.get('modelId', '')
        output_filename = f"{model_id}.txt"
        # Remove tables from content (original logic for huggingface)
        cleaned_content = remove_tables_from_text(content)
        
    elif source == 'arxiv':
        # For arXiv, extract ALL table-related content (comprehensive approach)
        if not os.path.exists(readme_path):
            print(f"Warning: File not found: {readme_path}")
            return None
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use comprehensive extraction to get ALL table-related content
            all_table_content = extract_all_table_content(content, csv_path)
            
            if all_table_content.strip():
                cleaned_content = all_table_content
                # Count sections extracted for logging
                sections = all_table_content.count('===')
                #print(f"✅ Extracted {sections//2} sections of table content for {csv_path}")
            else:
                # If no table content found, fall back to removing tables from full content
                print(f"⚠️  No table content found in {csv_path}, using fallback method")
                cleaned_content = remove_tables_from_text(content)
            
            # Keep original filename
            output_filename = os.path.basename(readme_path)
            if not output_filename.endswith('.txt'):
                output_filename = f"{os.path.splitext(output_filename)[0]}.txt"
                
        except Exception as e:
            print(f"❌ Error reading {readme_path}: {e}")
            return None
            
    else:
        # For GitHub and other sources, use original logic
        if not os.path.exists(readme_path):
            print(f"Warning: File not found: {readme_path}")
            return None
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Keep original filename
            output_filename = os.path.basename(readme_path)
            if not output_filename.endswith('.txt'):
                output_filename = f"{os.path.splitext(output_filename)[0]}.txt"
        except Exception as e:
            print(f"Error reading {readme_path}: {e}")
            return None
        
        # Remove tables from content (original logic for github)
        cleaned_content = remove_tables_from_text(content)
    
    # Save to processed_readmes directory
    output_path = os.path.join(tmp_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    return output_path

def process_group_texts(group_row):
    """
    Process a group of texts and return the combined content.
    """
    combined_text = ""
    for text_path in group_row['processed_text_path']:
        try:
            with open(text_path, 'r', encoding='utf-8') as tf:
                combined_text += tf.read() + "\n\n"
        except Exception as e:
            print(f"Error reading {text_path}: {e}")
    
    table_name = os.path.splitext(os.path.basename(group_row['dedup_csv_path']))[0]
    return {
        "id": table_name,
        "contents": combined_text.strip()
    }

def create_enhanced_mapping():
    """
    Create enhanced mapping with dedup paths and processed text paths.
    For each deduped table (e.g., table1), combine all surrounding texts from its variants
    (e.g., table2, table3 that map to table1) into one document.
    """
    # First load valid titles
    print("Step1: Loading valid titles list...")
    valid_titles = set()
    with open('data/analysis/all_valid_title_valid.txt', 'r') as f:
        for line in f:
            valid_titles.add(os.path.basename(line.strip()))
    print(f"Loaded {len(valid_titles)} valid titles")
    
    df_raw = pd.read_parquet(os.path.join('data', 'processed', 'raw_csv_to_text_mapping.parquet'))
    df_raw['csv_path'] = df_raw['csv_path'].apply(lambda x: os.path.basename(x))

    print(f"Step1: Loaded raw mapping with {len(df_raw)} rows")
    
    # Check for duplicates in csv_path
    print("\nChecking for duplicates in csv_path:")
    duplicate_csv = df_raw[df_raw.duplicated(['csv_path'], keep=False)]
    if len(duplicate_csv) > 0:
        print(f"Found {len(duplicate_csv)} rows with duplicate csv_path")
        print(duplicate_csv[['csv_path', 'source', 'readme_path']].head())
    
    mapping_path = os.path.join('data', 'deduped', 'duplicate_mapping.json')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Duplicate mapping not found at {mapping_path}")
    with open(mapping_path, 'r') as f:
        duplicate_mapping = json.load(f)
    df_raw['dedup_csv_path'] = df_raw['csv_path'].map(lambda x: duplicate_mapping.get(x, x))
    
    # Check mapping results
    print("\nChecking mapping results:")
    print(f"Total unique csv_path: {df_raw['csv_path'].nunique()}")
    print(f"Total unique dedup_csv_path: {df_raw['dedup_csv_path'].nunique()}")
    
    # Check for cases where mapping didn't change the path
    unchanged = df_raw[df_raw['csv_path'] == df_raw['dedup_csv_path']]
    print(f"Paths unchanged by mapping: {len(unchanged)}")
    
    # filter by valid titles
    print('\nbefore filtering: ', len(df_raw))
    df_raw = df_raw[df_raw['dedup_csv_path'].isin(valid_titles)]
    print('after filtering: ', len(df_raw))
    print(f"Step2: Loaded duplicate mapping with {len(duplicate_mapping)} entries")
    
    # Check results after filtering
    print("\nResults after filtering:")
    print(f"Unique dedup_csv_path after filtering: {df_raw['dedup_csv_path'].nunique()}")
    print("\nSample of filtered data:")
    print(df_raw[['csv_path', 'dedup_csv_path', 'source']].head())
    
    print("Creating temporary directories...")
    tmp_dirs = create_tmp_folders()
    
    print("Step3: Processing readme files, remove tables from the text...")
    # Process each readme file in parallel
    processed_paths = Parallel(n_jobs=-1)(
        delayed(process_readme_content)(row, tmp_dirs['readmes'])
        for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Processing readme files")
    )
    df_raw['processed_text_path'] = processed_paths
    
    # Get embedding for all related readme paths for unique dedup_csv_path
    print("Step4: Grouping by dedup paths...")
    # Sort by source priority: huggingface > arxiv > html
    source_priority = {'huggingface': 0, 'arxiv': 1, 'html': 2}
    df_raw['source_priority'] = df_raw['source'].map(source_priority)
    
    # Save the processed paths DataFrame for later use
    print("Step5: Saving processed paths DataFrame...")
    processed_paths_path = os.path.join('data', 'processed', 'processed_paths.parquet')
    df_raw.to_parquet(processed_paths_path)
    print(f"Saved processed paths to {processed_paths_path}")
    
    # Create a mapping of dedup_csv_path to its group of processed text paths
    dedup_groups = df_raw.sort_values('source_priority').groupby('dedup_csv_path').agg({
        'processed_text_path': lambda x: [p for p in x if p is not None],
        'source': 'first'  # Keep the source of the canonical file
    }).reset_index()
    
    # Check group results
    print("\nGroup results:")
    print(f"Total groups: {len(dedup_groups)}")
    print("Sample of groups:")
    for _, row in dedup_groups.head().iterrows():
        print(f"\nDedup path: {row['dedup_csv_path']}")
        print(f"Number of texts: {len(row['processed_text_path'])}")
        print(f"Source: {row['source']}")
    
    # Save the group mapping
    print("Step6: Saving group mapping...")
    group_mapping_path = os.path.join('data', 'processed', 'group_mapping.parquet')
    dedup_groups.to_parquet(group_mapping_path)
    print(f"Saved group mapping to {group_mapping_path}")
    
    # Build corpus in jsonl format
    print("Step7: Building corpus in jsonl format...")
    corpus_dir = os.path.join('data', 'tmp', 'corpus')
    os.makedirs(corpus_dir, exist_ok=True)
    corpus_file = os.path.join(corpus_dir, 'collection.jsonl')
    
    # Process each group and write to jsonl
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for _, row in tqdm(dedup_groups.iterrows(), total=len(dedup_groups), desc="Writing corpus"):
            # Read and combine texts for this group
            combined_text = []
            for text_path in row['processed_text_path']:
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as tf:
                        text = tf.read().strip()
                        if text:
                            combined_text.append(text)
            
            if combined_text:
                entry = {
                    'id': os.path.splitext(os.path.basename(row['dedup_csv_path']))[0],
                    'contents': '\n\n'.join(combined_text)
                }
                f.write(json.dumps(entry) + '\n')
    
    print(f"Step8: Corpus built at {corpus_file}")
    print("Next steps:")
    print("1. Build index using pyserini:")
    print("   python -m pyserini.index.lucene \\")
    print("     --collection JsonCollection \\")
    print("     --input data/tmp/corpus \\")
    print("     --index data/tmp/index \\")
    print("     --generator DefaultLuceneDocumentGenerator \\")
    print("     --threads 1 \\")
    print("     --storePositions --storeDocvectors --storeRaw")
    print("2. Use batch_search.py for retrieval")
    
    return corpus_file

def main():
    try:
        corpus_file = create_enhanced_mapping()
        print(f"\nCorpus built at {corpus_file}")
        print("\nNext steps:")
        print("1. Build index using pyserini:")
        print("   python -m pyserini.index.lucene \\")
        print("     --collection JsonCollection \\")
        print("     --input data/tmp/corpus \\")
        print("     --index data/tmp/index \\")
        print("     --generator DefaultLuceneDocumentGenerator \\")
        print("     --threads 1 \\")
        print("     --storePositions --storeDocvectors --storeRaw")
        print("2. Use batch_search.py for retrieval")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 