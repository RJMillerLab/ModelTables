import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer # Changed import
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Any
import json

# Removed PIL, io, matplotlib.pyplot, torch, nn, transformers as they are not needed for the new embedding

class TableEncoder:
    def __init__(self, model_name: str = "Mozilla/smart-tab-embedding"): # Default to your preferred model
        # Device handling is usually managed by SentenceTransformer automatically
        # or can be specified via model.to(device) if needed.
        # For 'Mozilla/smart-tab-embedding', it typically runs efficiently on CPU.
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Using device: {self.model.device}") # SentenceTransformer has its own device property

    # _table_to_image method removed as it was TAPAS-specific

    def get_embedding(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get table embedding using SentenceTransformer model"""
        # Convert all data to strings, similar to your snippet
        df_str = df.astype(str)

        # Create a list of strings, where each string is a row's content joined together
        row_texts = df_str.apply(lambda row: ' '.join(row.values), axis=1).to_list()

        # Join all row texts into a single string representing the whole table for embedding
        # This matches your provided snippet: model.encode(' '.join(texts), ...)
        full_table_text_for_embedding = ' '.join(row_texts)

        # Get model outputs (embedding for the entire table string)
        # SentenceTransformer's encode method typically doesn't require torch.no_grad() context
        table_embedding = self.model.encode(full_table_text_for_embedding, show_progress_bar=False) # Progress bar handled by tqdm in process_csv_folder

        # Extract structure information (remains useful)
        # For table_text in structure_info, we can store the more standard string representation
        table_text_for_storage = df.to_string(index=False)
        structure_info = {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "table_text": table_text_for_storage
        }

        return {
            "table_embedding": table_embedding, # This will be a 1D numpy array
            "structure_info": structure_info
        }

class TableRetriever:
    def __init__(self, embeddings: Dict[str, Dict[str, Any]]):
        self.embeddings = embeddings
        self.table_ids = list(embeddings.keys())

        # Pre-compute normalized table embeddings
        # table_embedding from SentenceTransformer for a single string is already a 1D array.
        # np.stack will create a 2D array where each row is an embedding.
        self.table_matrix = np.stack([emb_data["table_embedding"] for emb_data in embeddings.values() if emb_data["table_embedding"] is not None])

        if self.table_matrix.ndim == 1: # Handle case where only one embedding was processed
             if self.table_matrix.size > 0 : # If it's not empty
                self.table_matrix = np.expand_dims(self.table_matrix, axis=0)
             else: # if it is empty (e.g. no valid tables found)
                self.table_matrix = np.array([]).reshape(0,0) # or some other appropriate shape like (0, embedding_dim)

        if self.table_matrix.size > 0:
            self.table_matrix_norms = np.linalg.norm(self.table_matrix, axis=1, keepdims=True)
            # Avoid division by zero if any norm is zero
            self.table_matrix_norms[self.table_matrix_norms == 0] = 1e-9
            self.normalized_table_matrix = self.table_matrix / self.table_matrix_norms
        else:
            self.normalized_table_matrix = np.array([]).reshape(0,0)


    def retrieve(self, query_embedding_data: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k most similar tables"""
        if self.normalized_table_matrix.size == 0 or not self.table_ids:
            print("Warning: No table embeddings available for retrieval.")
            return []

        query_vector = query_embedding_data["table_embedding"]
        if query_vector is None:
            print("Warning: Query embedding is None.")
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            print("Warning: Query vector norm is zero. Cannot compute similarity.")
            return []
        normalized_query_vector = query_vector / query_norm
        
        # Ensure query_vector is 1D for dot product with 2D matrix
        if normalized_query_vector.ndim > 1:
            normalized_query_vector = normalized_query_vector.squeeze()
            if normalized_query_vector.ndim == 0 : # Squeezed to scalar
                 print("Warning: Query vector became scalar after squeeze.")
                 return []


        # Calculate similarities (cosine similarity)
        # Similarities will be 1D array of shape (num_tables,)
        try:
            similarities = np.dot(self.normalized_table_matrix, normalized_query_vector)
        except ValueError as e:
            print(f"Error during similarity calculation: {e}")
            print(f"Shape of normalized_table_matrix: {self.normalized_table_matrix.shape}")
            print(f"Shape of normalized_query_vector: {normalized_query_vector.shape}")
            return []


        # Get top-k indices
        # Ensure we don't request more items than available
        actual_top_k = min(top_k, len(similarities))
        if actual_top_k == 0:
            return []
            
        top_indices = np.argsort(similarities)[-actual_top_k:][::-1]

        # Return results
        return [(self.table_ids[idx], float(similarities[idx])) for idx in top_indices]

def process_csv_folder(folder_path: str, encoder: TableEncoder, batch_size: int = 32, mask_list: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Process all CSV files in a folder and return their embeddings"""
    embeddings = {}
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Convert mask list to basenames if provided
    if mask_list is not None:
        mask_basenames = {os.path.basename(f) for f in mask_list} # Use a set for faster lookups
        print(f"\nMask list contains {len(mask_basenames)} unique table basenames")

    # Filter files based on mask list
    if mask_list is not None:
        original_count = len(all_files)
        files_to_process = [f for f in all_files if f in mask_basenames]
        filtered_count = len(files_to_process)
        print(f"Filtered {original_count - filtered_count} tables (from {original_count} to {filtered_count} based on mask list)")
    else:
        files_to_process = all_files
        print(f"Processing all {len(files_to_process)} tables in the folder (no mask list).")

    if not files_to_process:
        print(f"No CSV files to process in {folder_path} after filtering.")
        return embeddings

    # Process files in batches (though SentenceTransformer encodes one by one in get_embedding)
    # Batching here mainly helps manage file loading and memory for DataFrames.
    for i in tqdm(range(0, len(files_to_process), batch_size), desc=f"Processing CSVs in {os.path.basename(folder_path)}"):
        batch_files = files_to_process[i:i + batch_size]
        # batch_dfs = [] # Not needed to store all DFs of a batch if processing one by one

        for filename in batch_files:
            try:
                file_path = os.path.join(folder_path, filename)
                # Try different encodings if utf-8 fails
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
                    except Exception as e_latin1:
                        print(f"Error loading {filename} with utf-8 and latin1: {e_latin1}")
                        continue
                except pd.errors.EmptyDataError:
                    print(f"Warning: {filename} is empty. Skipping.")
                    continue
                except Exception as e_load:
                    print(f"Error loading {filename}: {e_load}")
                    continue
                
                if df.empty:
                    print(f"Warning: {filename} loaded as an empty DataFrame. Skipping.")
                    continue

                embedding_data = encoder.get_embedding(df)
                if embedding_data["table_embedding"] is not None:
                    embeddings[filename] = embedding_data
                else:
                    print(f"Warning: Could not generate embedding for {filename}.")

            except Exception as e_proc:
                print(f"Error processing {filename}: {e_proc}")

    return embeddings

def main():
    # Initialize encoder with the SentenceTransformer model
    encoder = TableEncoder(model_name="Mozilla/smart-tab-embedding")

    base_path = "data/processed"
    # Ensure base_path exists or handle appropriately
    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist. Please create it or check the path.")
        # You might want to create it: os.makedirs(base_path, exist_ok=True)
        # For now, let's assume it exists or the script will fail gracefully later.

    folders = ["deduped_hugging_csvs", "deduped_github_csvs", "tables_output", "llm_tables"]
    # Example for testing:
    # folders = ["llm_tables"] # Process only one folder for quicker testing

    mask_list = None
    mask_file = os.path.join("../starmie_internal/val_file/all_valid_title_valid.txt")
    if os.path.exists(mask_file):
        try:
            with open(mask_file, "r") as f:
                mask_list = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(mask_list)} entries from mask list: {mask_file}")
        except Exception as e:
            print(f"Error loading mask list from {mask_file}: {e}")
            mask_list = None # Proceed without mask list if loading fails
    else:
        print(f"Mask file {mask_file} not found. Proceeding without mask list.")


    all_embeddings = {}
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder_path}...")
            folder_embeddings = process_csv_folder(folder_path, encoder, batch_size=16, mask_list=mask_list)
            all_embeddings.update(folder_embeddings)
            print(f"Processed {len(folder_embeddings)} tables from {folder}. Total embeddings: {len(all_embeddings)}")
        else:
            print(f"\nFolder {folder_path} does not exist or is not a directory. Skipping.")

    if not all_embeddings:
        print("\nNo embeddings were generated. Exiting.")
        return

    # Save embeddings
    output_dir = os.path.join(base_path, "embeddings_output") # Save to a subfolder
    os.makedirs(output_dir, exist_ok=True)
    output_path_pkl = os.path.join(output_dir, "table_embeddings_st.pkl")
    try:
        with open(output_path_pkl, "wb") as f:
            pickle.dump(all_embeddings, f)
        print(f"\nSaved {len(all_embeddings)} embeddings to {output_path_pkl}")
    except Exception as e:
        print(f"Error saving embeddings to {output_path_pkl}: {e}")

    # Save structure information
    structure_path_json = os.path.join(output_dir, "table_structure_st.json")
    structure_info = {tid: emb_data["structure_info"] for tid, emb_data in all_embeddings.items() if "structure_info" in emb_data}
    try:
        with open(structure_path_json, "w") as f:
            json.dump(structure_info, f, indent=2)
        print(f"Saved structure information for {len(structure_info)} tables to {structure_path_json}")
    except Exception as e:
        print(f"Error saving structure information to {structure_path_json}: {e}")

if __name__ == "__main__":
    main()