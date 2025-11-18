"""
Author: Zhengyuan Dong
Created: 2025-08-30
Description: Load raw parquet files into DuckDB or SQLite database for step1 processing
"""
import sqlite3
import duckdb
import os
import time
import pandas as pd
from src.utils import load_config

def create_database_connection(db_path, mode="duckdb"):
    """Create a database connection (DuckDB or SQLite)."""
    if mode.lower() == "duckdb":
        print(f"Connecting to DuckDB: {db_path}")
        start_time = time.time()
        
        # Create DuckDB connection
        con = duckdb.connect(db_path)
        
        # Test connection
        con.execute("SELECT 1")
        
        print(f"✅ DuckDB connection established. Time cost: {time.time() - start_time:.2f} seconds.")
        return con
        
    elif mode.lower() == "sqlite":
        print(f"Connecting to SQLite: {db_path}")
        start_time = time.time()
        
        # Create SQLite connection
        con = sqlite3.connect(db_path)
        
        # Test connection
        con.execute("SELECT 1")
        
        print(f"✅ SQLite connection established. Time cost: {time.time() - start_time:.2f} seconds.")
        return con
        
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'duckdb' or 'sqlite'")

def load_raw_data_to_db(con, data_type, config, mode="duckdb"):
    """Load raw parquet data into database (DuckDB or SQLite)."""
    print(f"⚠️ Loading raw {data_type} data into {mode.upper()}...")
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
    
    # Load data from multiple parquet files
    print(f"📥 Loading {len(file_names)} parquet files into table '{table_name}'...")
    
    if mode.lower() == "duckdb":
        # DuckDB mode: use direct SQL loading
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(raw_base_path, file_name)
            print(f"   📁 Loading file {i+1}/{len(file_names)}: {file_name}")
            
            if i == 0:
                # First file: create table
                try:
                    con.execute(f"""
                        CREATE TABLE {table_name} AS 
                        SELECT * FROM read_parquet('{file_path}')
                    """)
                    print(f"     ✅ Created table from {file_name}")
                except Exception as e:
                    print(f"     ⚠️ Direct loading failed: {e}")
                    print(f"     🔄 Falling back to pandas loading method...")
                    df = pd.read_parquet(file_path)
                    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                    print(f"     ✅ Created table using pandas from {file_name}")
            else:
                # Subsequent files: append to table
                try:
                    con.execute(f"""
                        INSERT INTO {table_name} 
                        SELECT * FROM read_parquet('{file_path}')
                    """)
                    print(f"     ✅ Appended {file_name}")
                except Exception as e:
                    print(f"     ⚠️ Direct loading failed: {e}")
                    print(f"     🔄 Falling back to pandas loading method...")
                    df = pd.read_parquet(file_path)
                    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                    print(f"     ✅ Appended using pandas from {file_name}")
    
    elif mode.lower() == "sqlite":
        # SQLite mode: use pandas
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
    if mode.lower() == "duckdb":
        table_info = con.execute(f"SELECT COUNT(*) as total_rows FROM {table_name}").fetchone()
        row_count = table_info[0]
        print(f"📊 Table '{table_name}' created with {row_count:,} rows")
        
        # Show table schema
        schema = con.execute(f"DESCRIBE {table_name}").df()
        print(f"📋 Table schema ({len(schema)} columns):")
        for _, col in schema.iterrows():
            print(f"   - {col['column_name']}: {col['column_type']}")
            
    elif mode.lower() == "sqlite":
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

def verify_raw_table(con, table_name, expected_columns, mode="duckdb"):
    """Verify that the raw table has the expected structure."""
    print(f"🔍 Verifying table '{table_name}' structure...")
    
    if mode.lower() == "duckdb":
        # Check if table exists
        table_exists = con.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'").fetchone()[0]
        
        if table_exists == 0:
            raise ValueError(f"❌ Table '{table_name}' not found!")
        
        # Get actual columns
        actual_columns = con.execute(f"DESCRIBE {table_name}").df()['column_name'].tolist()
        
    elif mode.lower() == "sqlite":
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

def main(mode="duckdb"):
    """Main function to load raw data into database (DuckDB or SQLite)."""
    print(f"🚀 LOAD RAW DATA TO {mode.upper()}")
    print("=" * 50)
    print(f"This script loads raw parquet files into database for unified data access")
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Create database connection
    if mode.lower() == "duckdb":
        db_path = "modellake_all.db"
    elif mode.lower() == "sqlite":
        db_path = "modellake_all.sqlite"
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'duckdb' or 'sqlite'")
    
    con = create_database_connection(db_path, mode)
    
    try:
        # Define data types to process
        data_types = ["modelcard", "datasetcard"]
        
        for data_type in data_types:
            print(f"\n{'='*20} Processing {data_type.upper()} {'='*20}")
            
            try:
                # Load raw data
                table_name, row_count = load_raw_data_to_db(con, data_type, config, mode)
                
                # Verify table structure
                expected_columns = ["modelId", "card"] if data_type == "modelcard" else ["datasetId", "card"]
                verify_raw_table(con, table_name, expected_columns, mode)
                
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
                if mode.lower() == "duckdb":
                    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                elif mode.lower() == "sqlite":
                    cursor = con.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                print(f"  - '{table_name}' table: {row_count:,} rows")
            except:
                print(f"  - '{table_name}' table: Not created")
        
        # Show database file size
        if os.path.exists(db_path):
            db_size = os.path.getsize(db_path)
            print(f"\n📊 {mode.upper()} database size: {db_size / (1024*1024):.2f} MB")
        
        print(f"\nYou can now use SQL queries directly on {db_path}!")
        print("Example queries:")
        print("  - SELECT COUNT(*) FROM raw_modelcard;")
        print("  - SELECT COUNT(*) FROM raw_datasetcard;")
        print("  - SELECT modelId, downloads FROM raw_modelcard LIMIT 10;")
        
    finally:
        # Close connection
        con.close()
        print("🔌 Database connection closed.")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["duckdb", "sqlite"]:
            print("❌ Invalid mode. Use 'duckdb' or 'sqlite'")
            print("Usage: python load_raw_to_db.py [duckdb|sqlite]")
            sys.exit(1)
    else:
        mode = "duckdb"  # Default mode
    
    print(f"🔧 Running in {mode.upper()} mode")
    main(mode)
