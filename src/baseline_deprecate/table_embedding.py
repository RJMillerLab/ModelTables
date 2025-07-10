import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Any
import json
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import gc
import time

class FilteredTableEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """Initialize encoder optimized for filtered processing"""
        print(f"Loading SentenceTransformer model: {model_name}")
        
        # Force CPU if CUDA has issues
        if device == 'cpu' or not torch.cuda.is_available():
            self.model = SentenceTransformer(model_name, device='cpu')
            print("Using CPU for inference")
        else:
            try:
                self.model = SentenceTransformer(model_name, device='cuda')
                print("Using CUDA for acceleration")
            except Exception as e:
                print(f"CUDA failed, falling back to CPU: {e}")
                self.model = SentenceTransformer(model_name, device='cpu')
                print("Using CPU for inference")
        
        # Enable evaluation mode and optimize
        self.model.eval()
        
        # Optimize batch size based on device
        if self.model.device.type == 'cuda':
            self.batch_size = 64  # Larger batches for GPU
        else:
            self.batch_size = 32  # Smaller batches for CPU
        
        print(f"Using device: {self.model.device}")
        print(f"Batch size: {self.batch_size}")

    def preprocess_table_text(self, df: pd.DataFrame) -> str:
        """Efficiently convert DataFrame to text with size limits"""
        # Limit table size to prevent memory issues and speed up processing
        max_rows = 500  # Reduced from 1000 for faster processing
        max_cols = 30   # Reduced from 50 for faster processing
        
        if len(df) > max_rows:
            df = df.head(max_rows)
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        
        # Convert to string efficiently
        df_str = df.astype(str)
        
        # Use more efficient string concatenation
        row_texts = []
        for _, row in df_str.iterrows():
            # Filter out empty strings and join
            non_empty = [str(val).strip() for val in row.values if str(val).strip()]
            if non_empty:
                row_texts.append(' '.join(non_empty))
        
        return ' '.join(row_texts)

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts with error handling"""
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return np.array([])
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    valid_texts, 
                    show_progress_bar=False, 
                    batch_size=self.batch_size,
                    convert_to_numpy=True
                )
            
            return embeddings
            
        except Exception as e:
            print(f"Error in batch encoding: {e}")
            return np.array([])

def load_filtered_files(mask_file: str) -> Dict[str, List[str]]:
    """Load and organize files by folder from mask file"""
    print(f"Loading mask file: {mask_file}")
    
    with open(mask_file, "r") as f:
        mask_list = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(mask_list)} entries from mask list")
    
    # Organize files by folder
    files_by_folder = {}
    for full_path in mask_list:
        folder = os.path.dirname(full_path).split('/')[-1]  # Get folder name
        filename = os.path.basename(full_path)
        
        if folder not in files_by_folder:
            files_by_folder[folder] = []
        files_by_folder[folder].append(filename)
    
    # Print summary
    for folder, files in files_by_folder.items():
        print(f"  {folder}: {len(files)} files")
    
    return files_by_folder

def safe_load_csv(file_path: str) -> Tuple[str, pd.DataFrame]:
    filename = os.path.basename(file_path)
    encodings = ['utf-8', 'latin1', 'cp1252']
    for encoding in encodings:
        # 先用默认engine
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            if df is not None and not df.empty:
                return filename, df
        except Exception:
            pass
        # 再用python engine和on_bad_lines
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False, engine='python', on_bad_lines='skip')
            if df is not None and not df.empty:
                return filename, df
        except Exception:
            pass
    return filename, None

def process_batch_filtered(batch_data: List[Tuple[str, pd.DataFrame]], encoder: FilteredTableEncoder) -> Dict[str, Dict[str, Any]]:
    """Process a batch of tables with optimized memory usage"""
    batch_embeddings = {}
    
    # Prepare texts for batch encoding
    texts = []
    valid_tables = []
    
    for filename, df in batch_data:
        if df is None or df.empty:
            continue
        
        try:
            table_text = encoder.preprocess_table_text(df)
            if table_text.strip():
                texts.append(table_text)
                valid_tables.append((filename, df))
        except Exception as e:
            continue
    
    if not texts:
        return batch_embeddings
    
    # Get embeddings in batch
    try:
        embeddings = encoder.get_embeddings_batch(texts)
        
        if len(embeddings) == len(valid_tables):
            # Process results
            for i, (filename, df) in enumerate(valid_tables):
                try:
                    structure_info = {
                        "num_rows": len(df),
                        "num_cols": len(df.columns),
                        "column_names": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "table_text": df.to_string(index=False, max_rows=50)  # Reduced for speed
                    }
                    
                    batch_embeddings[filename] = {
                        "table_embedding": embeddings[i],
                        "structure_info": structure_info
                    }
                except Exception as e:
                    continue
    except Exception as e:
        print(f"Error in batch processing: {e}")
    
    # Clear memory
    del texts, valid_tables
    gc.collect()
    
    return batch_embeddings

def process_folder_filtered(folder_path: str, files_to_process: List[str], encoder: FilteredTableEncoder, 
                          batch_size: int = 32, max_workers: int = 4) -> Dict[str, Dict[str, Any]]:
    """Process specific files in a folder with maximum efficiency"""
    embeddings = {}
    
    if not files_to_process:
        return embeddings
    
    print(f"Processing {len(files_to_process)} files in {os.path.basename(folder_path)}")
    
    # Load files in parallel
    print("Loading CSV files...")
    file_paths = [os.path.join(folder_path, f) for f in files_to_process]
    
    loaded_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(safe_load_csv, path): path for path in file_paths}
        
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Loading CSVs"):
            try:
                filename, df = future.result()
                if df is not None:
                    loaded_files.append((filename, df))
            except Exception as e:
                continue
    
    print(f"Successfully loaded {len(loaded_files)} files")
    
    # Process in batches
    for i in tqdm(range(0, len(loaded_files), batch_size), desc="Processing embeddings"):
        batch = loaded_files[i:i + batch_size]
        batch_embeddings = process_batch_filtered(batch, encoder)
        embeddings.update(batch_embeddings)
        
        # Clear memory periodically
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    return embeddings

def main():
    # Initialize encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = FilteredTableEncoder(model_name="all-MiniLM-L6-v2", device=device)
    
    base_path = "data/processed"
    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist.")
        return
    
    # Load filtered files
    mask_file = "data/analysis/all_valid_title_valid.txt"
    if not os.path.exists(mask_file):
        print(f"Mask file {mask_file} not found!")
        return
    
    files_by_folder = load_filtered_files(mask_file)
    
    # Process folders with optimized settings
    all_embeddings = {}
    batch_size = 32
    max_workers = 4
    
    start_time = time.time()
    
    for folder, files_to_process in files_by_folder.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue
        
        print(f"\nProcessing folder: {folder_path}...")
        folder_start = time.time()
        
        folder_embeddings = process_folder_filtered(
            folder_path, files_to_process, encoder, 
            batch_size=batch_size, max_workers=max_workers
        )
        
        all_embeddings.update(folder_embeddings)
        folder_time = time.time() - folder_start
        
        print(f"Processed {len(folder_embeddings)} tables from {folder} in {folder_time:.2f}s")
        print(f"Total embeddings so far: {len(all_embeddings)}")
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    
    if not all_embeddings:
        print("No embeddings generated. Exiting.")
        return
    
    # Save results
    output_dir = os.path.join(base_path, "embeddings_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    output_path_pkl = os.path.join(output_dir, "table_embeddings.pkl")
    try:
        with open(output_path_pkl, "wb") as f:
            pickle.dump(all_embeddings, f)
        print(f"Saved {len(all_embeddings)} embeddings to {output_path_pkl}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
    
    # Save structure info
    structure_path_json = os.path.join(output_dir, "table_structure.json")
    structure_info = {tid: emb_data["structure_info"] for tid, emb_data in all_embeddings.items() 
                     if emb_data.get("structure_info")}
    try:
        with open(structure_path_json, "w") as f:
            json.dump(structure_info, f, indent=2)
        print(f"Saved structure info for {len(structure_info)} tables")
    except Exception as e:
        print(f"Error saving structure info: {e}")

if __name__ == "__main__":
    main() 