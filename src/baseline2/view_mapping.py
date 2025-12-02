import pandas as pd
import os

def view_mapping():
    """
    View the contents of csv_to_readme_mapping.parquet
    """
    # Support TAG environment variable for versioning
    tag = os.environ.get('TAG', '')
    suffix = f"_{tag}" if tag else ""
    
    # Load the mapping file
    mapping_path = os.path.join('data', 'processed', f'raw_csv_to_text_mapping{suffix}.parquet')
    if not os.path.exists(mapping_path):
        print(f"Error: Mapping file not found at {mapping_path}")
        return
    
    # Read the parquet file
    df = pd.read_parquet(mapping_path)
    # get basename
    df['csv_path'] = df['csv_path'].apply(lambda x: os.path.basename(x))
    print(df[df['source'] != 'huggingface'])

if __name__ == "__main__":
    view_mapping()