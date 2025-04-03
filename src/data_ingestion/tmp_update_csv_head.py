"""
Warning: never run second time, only for case that the header are nonsense and remove once
"""

import pandas as pd
from pathlib import Path

def fix_table_header(input_csv, output_csv=None):
    if output_csv is None:
        output_csv = input_csv
    try:
        original_df = pd.read_csv(input_csv, header=None)
        print("="*50)
        print(original_df.head(3).to_string(index=False, header=False))
        
        df = pd.read_csv(input_csv, header=None, skiprows=1)
        
        if len(df) > 1:
            new_header = [str(item).strip() for item in df.iloc[0].values.tolist()]
            fixed_df = df[1:].copy()
            fixed_df.columns = new_header
            fixed_df.reset_index(drop=True, inplace=True)
            fixed_df.to_csv(output_csv, index=False)
        else:
            print("\nWarning: insufficient data, can not remove")
            
    except Exception as e:
        print(f"Process failure: {str(e)}")

def process_directory(directory_path):
    csv_files = Path(directory_path).glob("*.csv")
    for csv_file in csv_files:
        print(f"\nStart processing: {csv_file}")
        fix_table_header(str(csv_file), output_csv=str(csv_file))

if __name__ == "__main__":
    target_directory = "data/processed/tables_output"
    process_directory(target_directory)
