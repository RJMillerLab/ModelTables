"""
Author: Zhengyuan Dong
Created: 2025-08-30
Description: Load raw parquet files into SQLite database for step1 processing
"""
import sqlite3
import os
import time
import pandas as pd
from src.utils import load_config

def create_sqlite_connection(db_path):
    """Create a SQLite connection to the specified database file."""
    print(f"Connecting to SQLite: {db_path}")
    start_time = time.time()
    
    # Create connection
    con = sqlite3.connect(db_path)
    
    # Test connection
    con.execute("SELECT 1")
    
    print(f"✅ Connection established. Time cost: {time.time() - start_time:.2f} seconds.")
    return con

def load_raw_data_to_sqlite(con, data_type, config):
    """Load raw parquet data into SQLite database using pandas."""
    print(f"⚠️ Loading raw {data_type} data into SQLite...")
    start_time = time.time()
    
    # Define file paths - use the same logic as load_combined_data
    raw_base_path = os.path.join(config.get('base_path'), 'raw')
    
    # Define file names based on data_type (same as load_combined_data)
    if data_type == "modelcard":
        file_names = [f"train-0000{i}-of-00004.parquet" for i in range(4)]
    elif data_type == "datasetcard":
        file_names = [f"train-0000{i}-of-00002.parquet" for i in range(2)]
    else:
        raise ValueError(f"data_type must be 'modelcard' or 'datasetcard', got {data_type}")
    
    # Check if all files exist
    missing_files = []
    for file_name in file_names:
        file_path = os.path.join(raw_base_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing raw data files: {missing_files}")
    
    # Table name
    table_name = f"raw_{data_type}"
    
    # Drop existing table if it exists
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Load data from multiple parquet files to SQLite
    print(f"📥 Loading {len(file_names)} parquet files into table '{table_name}'...")
    
    all_dataframes = []
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(raw_base_path, file_name)
        print(f"   📁 Loading file {i+1}/{len(file_names)}: {file_name}")
        
        # Read parquet file using pandas
        df = pd.read_parquet(file_path)
        all_dataframes.append(df)
        print(f"     ✅ Loaded {len(df):,} rows from {file_name}")
    
    # Combine all dataframes
    print(f"🔄 Combining {len(all_dataframes)} dataframes...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"   📊 Combined total: {len(combined_df):,} rows")
    
    # Save to SQLite
    print(f"💾 Saving to SQLite table '{table_name}'...")
    combined_df.to_sql(table_name, con, if_exists='replace', index=False)
    
    # Get table info
    cursor = con.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    print(f"📊 Table '{table_name}' created with {row_count:,} rows")
    
    # Show table schema
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    print(f"📋 Table schema ({len(schema)} columns):")
    for col in schema:
        print(f"   - {col[1]}: {col[2]}")
    
    load_time = time.time() - start_time
    print(f"✅ Raw data loading completed. Time cost: {load_time:.2f} seconds.")
    
    return table_name, row_count

def verify_raw_table(con, table_name, expected_columns):
    """Verify that the raw table has the expected structure."""
    print(f"🔍 Verifying table '{table_name}' structure...")
    
    # Check if table exists
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        raise ValueError(f"❌ Table '{table_name}' not found!")
    
    # Get actual columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    actual_columns = [col[1] for col in cursor.fetchall()]
    
    # Check required columns
    missing_columns = [col for col in expected_columns if col not in actual_columns]
    if missing_columns:
        print(f"⚠️ Missing columns: {missing_columns}")
        return False
    
    print(f"✅ Table structure verified. All required columns present.")
    return True

def main():
    """Main function to load raw data into SQLite."""
    print("🚀 LOAD RAW DATA TO SQLITE")
    print("=" * 50)
    print("This script loads raw parquet files into modellake_all_sqlite.db for unified data access")
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Create SQLite connection
    db_path = "modellake_all_sqlite.db"
    con = create_sqlite_connection(db_path)
    
    try:
        # Define data types to process
        data_types = ["modelcard", "datasetcard"]
        
        for data_type in data_types:
            print(f"\n{'='*20} Processing {data_type.upper()} {'='*20}")
            
            try:
                # Load raw data
                table_name, row_count = load_raw_data_to_sqlite(con, data_type, config)
                
                # Verify table structure
                expected_columns = ["modelId", "card"] if data_type == "modelcard" else ["datasetId", "card"]
                verify_raw_table(con, table_name, expected_columns)
                
                print(f"✅ {data_type.upper()} data loaded successfully!")
                print(f"   - Table: {table_name}")
                print(f"   - Rows: {row_count:,}")
                
            except Exception as e:
                print(f"❌ Failed to load {data_type}: {e}")
                continue
        
        print(f"\n🎉 Raw data loading completed!")
        print("Database structure:")
        for data_type in data_types:
            table_name = f"raw_{data_type}"
            try:
                cursor = con.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                print(f"  - '{table_name}' table: {row_count:,} rows")
            except:
                print(f"  - '{table_name}' table: Not created")
        
        # Show database file size
        if os.path.exists(db_path):
            db_size = os.path.getsize(db_path)
            print(f"\n📊 SQLite database size: {db_size / (1024*1024):.2f} MB")
        
        print("\nYou can now use SQL queries directly on modellake_all_sqlite.db!")
        print("Example queries:")
        print("  - SELECT COUNT(*) FROM raw_modelcard;")
        print("  - SELECT COUNT(*) FROM raw_datasetcard;")
        print("  - SELECT modelId, downloads FROM raw_modelcard LIMIT 10;")
        
    finally:
        # Close connection
        con.close()
        print("🔌 Database connection closed.")

if __name__ == "__main__":
    main()
