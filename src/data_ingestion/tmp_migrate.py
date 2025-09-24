import os
import shutil
import pandas as pd
from src.utils import to_parquet

def migrate_files_and_update_paths():
    parquet_path = "data/processed/html_table.parquet"
    df = pd.read_parquet(parquet_path)
    
    output_dir = 'data/processed/tables_output'
    migrated_count = 0
    
    for dir_name in os.listdir(output_dir):
        dir_path = os.path.join(output_dir, dir_name)
        
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.csv'):
                    src_path = os.path.join(dir_path, file_name)
                    dest_path = os.path.join(output_dir, file_name)
                    
                    if os.path.exists(dest_path):
                        base_name = os.path.splitext(file_name)[0]
                        counter = 1
                        while os.path.exists(dest_path):
                            new_name = f"{base_name}_dup{counter}.csv"
                            dest_path = os.path.join(output_dir, new_name)
                            counter += 1
                    
                    shutil.move(src_path, dest_path)
                    migrated_count += 1
            
            try:
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")
            except OSError:
                print(f"Directory not empty, keeping: {dir_path}")

    def update_paths(table_list):
        updated = []
        for path in table_list:
            # old_dir/paper_id/paper_id_tableX.csv → tables_output/paper_id_tableX.csv
            parts = path.split(os.sep)
            if len(parts) > 2 and parts[-2] == parts[-1].split('_')[0]:
                new_path = os.path.join(output_dir, parts[-1])
                updated.append(new_path)
            else:
                updated.append(path)
        return updated

    df['table_list'] = df['table_list'].apply(update_paths)
    
    backup_path = "data/processed/html_table_backup.parquet"
    shutil.copy(parquet_path, backup_path)
    to_parquet(df, parquet_path)
    
    print(f"Migration complete. {migrated_count} files moved.")
    print(f"Original data backed up at: {backup_path}")

if __name__ == "__main__":
    migrate_files_and_update_paths()
