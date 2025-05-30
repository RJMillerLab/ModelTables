import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import TapasTokenizer, TapasModel
from PIL import Image
import io
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Any
import json
import matplotlib.pyplot as plt

class TableEncoder:
    def __init__(self, model_name: str = "google/tapas-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load TAPAS model and tokenizer
        self.model = TapasModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        
    def _table_to_image(self, df: pd.DataFrame) -> Image.Image:
        """Convert table to image for structure recognition"""
        # Create a figure with the table
        fig = plt.figure(figsize=(10, len(df) * 0.4))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )
        
        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
        buf.seek(0)
        
        # Open image and convert to RGB mode
        image = Image.open(buf)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def get_embedding(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get table embedding using TAPAS model"""
        # Convert all data to strings
        df = df.astype(str)
        
        # Convert table to string format
        table_text = df.to_string(index=False)
        
        # Tokenize table with optimized settings
        inputs = self.tokenizer(
            table=df,
            queries="",  # Empty query since we only want table embeddings
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model outputs with memory optimization
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get table representation (using pooled output)
        table_embedding = outputs.pooler_output.cpu().numpy()
        
        # Extract structure information
        structure_info = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "table_text": table_text
        }
        
        return {
            "table_embedding": table_embedding,
            "structure_info": structure_info
        }

class TableRetriever:
    def __init__(self, embeddings: Dict[str, Dict[str, Any]]):
        self.embeddings = embeddings
        self.table_ids = list(embeddings.keys())
        
        # Pre-compute normalized table embeddings
        self.table_matrix = np.stack([emb["table_embedding"] for emb in embeddings.values()])
        self.table_matrix = self.table_matrix / np.linalg.norm(self.table_matrix, axis=1, keepdims=True)
        
    def retrieve(self, query_embedding: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k most similar tables"""
        # Normalize query embedding
        query_vector = query_embedding["table_embedding"]
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Calculate similarities
        similarities = np.dot(self.table_matrix, query_vector)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results
        return [(self.table_ids[idx], similarities[idx]) for idx in top_indices]

def process_csv_folder(folder_path: str, encoder: TableEncoder, batch_size: int = 32, mask_list: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Process all CSV files in a folder and return their embeddings"""
    embeddings = {}
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Convert mask list to basenames if provided
    if mask_list is not None:
        mask_list = [os.path.basename(f) for f in mask_list]
        print(f"\nMask list contains {len(mask_list)} tables")
    
    # Filter files based on mask list
    if mask_list is not None:
        original_count = len(files)
        files = [f for f in files if f in mask_list]
        filtered_count = len(files)
        print(f"Filtered {original_count - filtered_count} tables (from {original_count} to {filtered_count})")
    
    # Process files in batches
    for i in tqdm(range(0, len(files), batch_size)):
        batch_files = files[i:i + batch_size]
        batch_dfs = []
        
        # Load batch of files
        for filename in batch_files:
            try:
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                batch_dfs.append((filename, df))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        # Process batch
        for filename, df in batch_dfs:
            try:
                embedding = encoder.get_embedding(df)
                embeddings[filename] = embedding
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return embeddings

def main():
    # Initialize encoder
    encoder = TableEncoder()
    
    # Process all CSV folders
    base_path = "data/processed"
    folders = ["deduped_hugging_csvs", "deduped_github_csvs", "tables_output", "llm_tables"]
    
    # Load mask list if provided
    mask_list = None
    mask_file = os.path.join(base_path, "valid_tables.txt")
    if os.path.exists(mask_file):
        with open(mask_file, "r") as f:
            mask_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded mask list from {mask_file}")
    
    all_embeddings = {}
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            print(f"\nProcessing {folder}...")
            folder_embeddings = process_csv_folder(folder_path, encoder, mask_list=mask_list)
            all_embeddings.update(folder_embeddings)
    
    # Save embeddings
    output_path = os.path.join(base_path, "table_embeddings.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_embeddings, f)
    print(f"\nSaved {len(all_embeddings)} embeddings to {output_path}")
    
    # Save structure information
    structure_path = os.path.join(base_path, "table_structure.json")
    structure_info = {tid: emb["structure_info"] for tid, emb in all_embeddings.items()}
    with open(structure_path, "w") as f:
        json.dump(structure_info, f, indent=2)
    print(f"Saved structure information to {structure_path}")
    
    # Example retrieval
    retriever = TableRetriever(all_embeddings)
    query_file = list(all_embeddings.keys())[0]
    query_embedding = all_embeddings[query_file]
    results = retriever.retrieve(query_embedding, top_k=5)
    
    print("\nExample retrieval results:")
    for table_id, score in results:
        print(f"{table_id}: {score:.4f}")

if __name__ == "__main__":
    main() 