#!/usr/bin/env python3
"""
Smart CSV Handler - General solution for handling different table types
"""

import pandas as pd
import os

class SmartCSVHandler:
    """General solution for handling different CSV table types."""
    
    def __init__(self, label_separator='|', csv_separator=','):
        """
        Initialize the handler.
        
        Args:
            label_separator: Separator for labels within cells (default: '|')
            csv_separator: Separator for CSV columns (default: ',')
        """
        self.label_separator = label_separator
        self.csv_separator = csv_separator
    
    def detect_table_type(self, df):
        """Detect if this is a label scheme table or performance table."""
        component_col = None
        labels_col = None
        
        # Look for Component and Labels columns (handle spaces)
        for col in df.columns:
            if 'Component' in col.strip():
                component_col = col
            if 'Labels' in col.strip():
                labels_col = col
        
        return component_col, labels_col
    
    def process_label_scheme_table(self, df, component_col, labels_col):
        """Process label scheme table by converting comma-separated labels."""
        result = []
        for i, row in df.iterrows():
            component = row[component_col]
            labels = row[labels_col]
            if pd.notna(labels):
                # Convert comma-separated to custom separator
                label_list = [label.strip() for label in str(labels).split(',')]
                result.append([component, self.label_separator.join(label_list)])
        return result
    
    def process_performance_table(self, df):
        """Process performance table as-is."""
        return df.values.tolist()
    
    def smart_process_csv(self, csv_path, output_path=None):
        """
        Smartly process CSV file based on its content.
        
        Args:
            csv_path: Path to input CSV
            output_path: Path to output CSV (if None, overwrites input)
        
        Returns:
            Processed data and table type
        """
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Detect table type
        component_col, labels_col = self.detect_table_type(df)
        is_label_scheme = component_col and labels_col
        
        # Process based on type
        if is_label_scheme:
            print(f"🎯 Processing Label Scheme table: {os.path.basename(csv_path)}")
            processed_data = self.process_label_scheme_table(df, component_col, labels_col)
            table_type = "label_scheme"
        else:
            print(f"📈 Processing Performance table: {os.path.basename(csv_path)}")
            processed_data = self.process_performance_table(df)
            table_type = "performance"
        
        # Save processed data
        if output_path is None:
            output_path = csv_path
        
        # Create DataFrame and save
        if is_label_scheme:
            result_df = pd.DataFrame(processed_data, columns=['Component', 'Labels'])
        else:
            result_df = pd.DataFrame(processed_data, columns=df.columns)
        
        result_df.to_csv(output_path, index=False, sep=self.csv_separator)
        
        return processed_data, table_type

def test_general_solution():
    """Test the general solution."""
    print("🧪 Testing General Smart CSV Handler")
    print("="*60)
    
    handler = SmartCSVHandler(label_separator='|', csv_separator=',')
    
    # Test files
    test_files = [
        "data/processed/deduped_hugging_csvs/ec8b87737d_table1.csv",
        "data/processed/deduped_hugging_csvs/b82734632e_table2.csv", 
        "data/processed/deduped_hugging_csvs/c8ea08177c_table2.csv"
    ]
    
    for csv_file in test_files:
        if os.path.exists(csv_file):
            print(f"\n--- Processing {os.path.basename(csv_file)} ---")
            try:
                processed_data, table_type = handler.smart_process_csv(csv_file)
                print(f"✅ Processed as {table_type} table")
                print(f"📊 Data shape: {len(processed_data)} rows")
                
                # Show first few rows
                for i, row in enumerate(processed_data[:2]):
                    if table_type == "label_scheme":
                        print(f"  Row {i}: {row[0]} -> {row[1][:50]}...")
                    else:
                        print(f"  Row {i}: {row[:3]}...")
                        
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_general_solution()
