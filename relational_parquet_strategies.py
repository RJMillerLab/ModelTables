#!/usr/bin/env python3
"""
Relational Parquet Strategies for CitationLake
Store multiple related tables with shared primary key (modelId) for efficient querying
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import duckdb

class RelationalParquetStrategies:
    """Strategies for storing and querying related parquet tables with shared primary key"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / "relational_dataset"
        self.dataset_dir.mkdir(exist_ok=True)
    
    def strategy_1_parquet_dataset_with_partitioning(self, table_configs: Dict[str, Dict]):
        """
        Strategy 1: Parquet Dataset with Partitioning
        Store each table as separate parquet files, but create a dataset that can be queried efficiently
        """
        print("🔄 Strategy 1: Creating Parquet Dataset with Partitioning...")
        
        # Create separate parquet files for each table type
        for table_name, config in table_configs.items():
            print(f"  Processing table: {table_name}")
            
            # Read source files and combine
            combined_data = []
            for file_path in config['source_files']:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if 'modelId' in df.columns:
                        # Add table type identifier
                        df['_table_type'] = table_name
                        combined_data.append(df)
            
            if not combined_data:
                continue
                
            # Combine and optimize
            table_df = pd.concat(combined_data, ignore_index=True, sort=False)
            table_df = self._optimize_for_modelid_queries(table_df, table_name)
            
            # Save as parquet
            output_path = self.dataset_dir / f"{table_name}.parquet"
            table_df.to_parquet(
                output_path,
                compression='zstd',
                engine='pyarrow',
                index=False
            )
            
            print(f"    ✅ Saved: {output_path} ({len(table_df):,} rows)")
        
        # Create dataset metadata
        self._create_dataset_metadata(table_configs)
    
    def strategy_2_wide_table_with_structured_columns(self, table_configs: Dict[str, Dict]):
        """
        Strategy 2: Wide Table with Structured Columns
        Store all related data in one table with structured columns for different data types
        """
        print("🔄 Strategy 2: Creating Wide Table with Structured Columns...")
        
        # First, get all unique modelIds
        all_modelids = set()
        for config in table_configs.values():
            for file_path in config['source_files']:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if 'modelId' in df.columns:
                        all_modelids.update(df['modelId'].unique())
        
        print(f"  Found {len(all_modelids):,} unique modelIds")
        
        # Create base dataframe with all modelIds
        wide_df = pd.DataFrame({'modelId': list(all_modelids)})
        
        # Add structured columns for each table type
        for table_name, config in table_configs.items():
            print(f"  Adding structured column for: {table_name}")
            
            # Collect data for this table type
            table_data = {}
            for file_path in config['source_files']:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if 'modelId' in df.columns:
                        for _, row in df.iterrows():
                            modelid = row['modelId']
                            if modelid not in table_data:
                                table_data[modelid] = {}
                            
                            # Store relevant columns (exclude modelId)
                            for col in df.columns:
                                if col != 'modelId' and col not in ['_table_type']:
                                    table_data[modelid][col] = row[col]
            
            # Add as structured column
            wide_df[f'{table_name}_data'] = wide_df['modelId'].apply(
                lambda x: table_data.get(x, {})
            )
        
        # Save wide table
        output_path = self.dataset_dir / "wide_structured_table.parquet"
        wide_df.to_parquet(
            output_path,
            compression='zstd',
            engine='pyarrow',
            index=False
        )
        
        print(f"    ✅ Wide table saved: {output_path} ({len(wide_df):,} rows)")
    
    def strategy_3_normalized_tables_with_foreign_keys(self, table_configs: Dict[str, Dict]):
        """
        Strategy 3: Normalized Tables with Foreign Keys
        Create separate tables with integer foreign keys, plus a master lookup table
        """
        print("🔄 Strategy 3: Creating Normalized Tables with Foreign Keys...")
        
        # Create master modelId lookup table
        all_modelids = set()
        for config in table_configs.values():
            for file_path in config['source_files']:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if 'modelId' in df.columns:
                        all_modelids.update(df['modelId'].unique())
        
        # Create master table with integer IDs
        master_df = pd.DataFrame({
            'modelId_int': range(len(all_modelids)),
            'modelId': list(all_modelids)
        })
        
        master_path = self.dataset_dir / "master_modelids.parquet"
        master_df.to_parquet(master_path, compression='zstd', engine='pyarrow')
        
        # Create lookup dictionary
        modelid_to_int = dict(zip(master_df['modelId'], master_df['modelId_int']))
        
        print(f"    ✅ Master table: {master_path} ({len(master_df):,} modelIds)")
        
        # Create normalized tables
        for table_name, config in table_configs.items():
            print(f"  Creating normalized table: {table_name}")
            
            table_data = []
            for file_path in config['source_files']:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if 'modelId' in df.columns:
                        for _, row in df.iterrows():
                            modelid = row['modelId']
                            if modelid in modelid_to_int:
                                # Convert modelId to integer foreign key
                                row_dict = row.to_dict()
                                row_dict['modelId_int'] = modelid_to_int[modelid]
                                del row_dict['modelId']  # Remove original modelId
                                table_data.append(row_dict)
            
            if table_data:
                table_df = pd.DataFrame(table_data)
                table_df = self._optimize_dtypes(table_df)
                
                output_path = self.dataset_dir / f"{table_name}_normalized.parquet"
                table_df.to_parquet(
                    output_path,
                    compression='zstd',
                    engine='pyarrow',
                    index=False
                )
                
                print(f"    ✅ Normalized table: {output_path} ({len(table_df):,} rows)")
    
    def strategy_4_parquet_dataset_with_indexing(self, table_configs: Dict[str, Dict]):
        """
        Strategy 4: Parquet Dataset with Indexing
        Use PyArrow Dataset with built-in indexing for fast modelId-based queries
        """
        print("🔄 Strategy 4: Creating Parquet Dataset with Indexing...")
        
        # Create dataset directory structure
        dataset_path = self.dataset_dir / "indexed_dataset"
        dataset_path.mkdir(exist_ok=True)
        
        # Process each table type
        for table_name, config in table_configs.items():
            print(f"  Creating indexed table: {table_name}")
            
            # Combine source files
            combined_data = []
            for file_path in config['source_files']:
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    if 'modelId' in df.columns:
                        df['_table_type'] = table_name
                        combined_data.append(df)
            
            if not combined_data:
                continue
                
            table_df = pd.concat(combined_data, ignore_index=True, sort=False)
            table_df = self._optimize_for_modelid_queries(table_df, table_name)
            
            # Save with partitioning by first character of modelId for better query performance
            table_df['_partition_key'] = table_df['modelId'].str[0]
            
            # Group by partition and save
            for partition_key, group_df in table_df.groupby('_partition_key'):
                partition_dir = dataset_path / f"{table_name}" / f"partition={partition_key}"
                partition_dir.mkdir(parents=True, exist_ok=True)
                
                group_df = group_df.drop('_partition_key', axis=1)
                output_path = partition_dir / "data.parquet"
                group_df.to_parquet(
                    output_path,
                    compression='zstd',
                    engine='pyarrow',
                    index=False
                )
        
        # Create dataset metadata
        self._create_indexed_dataset_metadata(dataset_path, table_configs)
    
    def _optimize_for_modelid_queries(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Optimize dataframe for efficient modelId-based queries"""
        # Sort by modelId for better query performance
        if 'modelId' in df.columns:
            df = df.sort_values('modelId').reset_index(drop=True)
        
        # Optimize data types
        df = self._optimize_dtypes(df)
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for better compression and performance"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert to category if low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # Downcast integers
                if df[col].min() >= 0 and df[col].max() <= 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].min() >= 0 and df[col].max() <= 65535:
                    df[col] = df[col].astype('uint16')
        return df
    
    def _create_dataset_metadata(self, table_configs: Dict[str, Dict]):
        """Create metadata file for the dataset"""
        metadata = {
            'dataset_type': 'relational_parquet',
            'tables': {},
            'query_examples': {
                'get_model_data': "SELECT * FROM modelcard_core WHERE modelId = 'your_model_id'",
                'join_tables': "SELECT * FROM modelcard_core c JOIN citations cit ON c.modelId = cit.modelId WHERE c.modelId = 'your_model_id'"
            }
        }
        
        for table_name, config in table_configs.items():
            metadata['tables'][table_name] = {
                'description': config.get('description', f'Data for {table_name}'),
                'source_files': config['source_files'],
                'parquet_file': f"{table_name}.parquet"
            }
        
        import json
        with open(self.dataset_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_indexed_dataset_metadata(self, dataset_path: Path, table_configs: Dict[str, Dict]):
        """Create metadata for indexed dataset"""
        metadata = {
            'dataset_type': 'indexed_parquet_dataset',
            'dataset_path': str(dataset_path),
            'tables': list(table_configs.keys()),
            'query_examples': {
                'load_dataset': "import pyarrow.dataset as ds; dataset = ds.dataset('path/to/dataset')",
                'query_by_modelid': "dataset.to_table(filter=ds.field('modelId') == 'your_model_id')"
            }
        }
        
        import json
        with open(dataset_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

class ParquetQueryHelper:
    """Helper class for querying the relational parquet files efficiently"""
    
    def __init__(self, dataset_dir: str = "data/processed/relational_dataset"):
        self.dataset_dir = Path(dataset_dir)
    
    def get_model_data(self, model_id: str, table_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific modelId across all or specified table types
        
        Args:
            model_id: The modelId to query for
            table_types: List of table types to query (None = all tables)
        
        Returns:
            Dictionary mapping table_type to DataFrame
        """
        results = {}
        
        if table_types is None:
            # Find all parquet files in dataset directory
            table_files = list(self.dataset_dir.glob("*.parquet"))
        else:
            table_files = [self.dataset_dir / f"{table_type}.parquet" for table_type in table_types]
        
        for table_file in table_files:
            if table_file.exists():
                table_name = table_file.stem
                
                # Load and filter by modelId
                df = pd.read_parquet(table_file)
                if 'modelId' in df.columns:
                    filtered_df = df[df['modelId'] == model_id]
                    if not filtered_df.empty:
                        results[table_name] = filtered_df
        
        return results
    
    def get_model_data_wide(self, model_id: str) -> pd.DataFrame:
        """
        Get all data for a specific modelId as a single wide DataFrame
        (for Strategy 2 - wide table with structured columns)
        """
        wide_file = self.dataset_dir / "wide_structured_table.parquet"
        if not wide_file.exists():
            raise FileNotFoundError("Wide structured table not found. Run Strategy 2 first.")
        
        df = pd.read_parquet(wide_file)
        return df[df['modelId'] == model_id]
    
    def get_model_data_normalized(self, model_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific modelId from normalized tables
        (for Strategy 3 - normalized tables with foreign keys)
        """
        # First, get the integer ID for this modelId
        master_file = self.dataset_dir / "master_modelids.parquet"
        if not master_file.exists():
            raise FileNotFoundError("Master modelIds table not found. Run Strategy 3 first.")
        
        master_df = pd.read_parquet(master_file)
        model_row = master_df[master_df['modelId'] == model_id]
        
        if model_row.empty:
            return {}
        
        model_id_int = model_row['modelId_int'].iloc[0]
        
        # Query all normalized tables
        results = {}
        for table_file in self.dataset_dir.glob("*_normalized.parquet"):
            table_name = table_file.stem.replace('_normalized', '')
            df = pd.read_parquet(table_file)
            
            if 'modelId_int' in df.columns:
                filtered_df = df[df['modelId_int'] == model_id_int]
                if not filtered_df.empty:
                    results[table_name] = filtered_df
        
        return results
    
    def query_with_duckdb(self, model_id: str, sql_query: str = None) -> pd.DataFrame:
        """
        Use DuckDB to query across multiple parquet files efficiently
        
        Args:
            model_id: The modelId to query for
            sql_query: Custom SQL query (if None, uses default)
        """
        if sql_query is None:
            sql_query = f"""
            SELECT * FROM read_parquet('{self.dataset_dir}/*.parquet')
            WHERE modelId = '{model_id}'
            """
        
        import duckdb
        return duckdb.execute(sql_query).df()

def main():
    """Example usage of relational parquet strategies"""
    
    # Define table configurations based on your data
    table_configs = {
        'modelcard_core': {
            'description': 'Core model card information',
            'source_files': [
                'data/processed/modelcard_step1.parquet',
                'data/processed/modelcard_step2.parquet',
                'data/processed/modelcard_step3_merged.parquet',
                'data/processed/modelcard_step4.parquet'
            ]
        },
        'citations': {
            'description': 'Citation and reference data',
            'source_files': [
                'data/processed/modelcard_citation_enriched.parquet',
                'data/processed/modelcard_citation_API.parquet',
                'data/processed/extracted_annotations.parquet',
                'data/processed/s2orc_citations_cache.parquet',
                'data/processed/s2orc_references_cache.parquet'
            ]
        },
        'tables': {
            'description': 'Table and CSV data',
            'source_files': [
                'data/processed/llm_markdown_table_results.parquet',
                'data/processed/html_table.parquet',
                'data/processed/step_pdf_table.parquet',
                'data/processed/step_tex_table.parquet'
            ]
        },
        'metadata': {
            'description': 'Model metadata and titles',
            'source_files': [
                'data/processed/modelcard_all_title_list.parquet',
                'data/processed/all_title_list_valid.parquet',
                'data/processed/giturl_info.parquet',
                'data/processed/github_readmes_info.parquet'
            ]
        }
    }
    
    # Initialize strategies
    strategies = RelationalParquetStrategies()
    
    print("🚀 Creating relational parquet datasets...")
    
    # Strategy 1: Parquet Dataset with Partitioning
    strategies.strategy_1_parquet_dataset_with_partitioning(table_configs)
    
    # Strategy 2: Wide Table with Structured Columns
    strategies.strategy_2_wide_table_with_structured_columns(table_configs)
    
    # Strategy 3: Normalized Tables with Foreign Keys
    strategies.strategy_3_normalized_tables_with_foreign_keys(table_configs)
    
    # Strategy 4: Parquet Dataset with Indexing
    strategies.strategy_4_parquet_dataset_with_indexing(table_configs)
    
    print("\n✅ All strategies completed!")
    print("\n📖 Usage Examples:")
    print("""
    # Query helper examples:
    from relational_parquet_strategies import ParquetQueryHelper
    
    helper = ParquetQueryHelper()
    
    # Get all data for a specific model
    model_data = helper.get_model_data('your_model_id')
    
    # Get data from specific table types
    core_data = helper.get_model_data('your_model_id', ['modelcard_core', 'citations'])
    
    # Get wide table data (Strategy 2)
    wide_data = helper.get_model_data_wide('your_model_id')
    
    # Get normalized data (Strategy 3)
    normalized_data = helper.get_model_data_normalized('your_model_id')
    
    # Use DuckDB for complex queries
    result = helper.query_with_duckdb('your_model_id')
    """)

if __name__ == "__main__":
    main()
