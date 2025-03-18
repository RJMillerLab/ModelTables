import os
import html2text
from joblib import Parallel, delayed
from tqdm import tqdm

def process_md_file(filename, input_dir, output_dir):
    input_path = os.path.join(input_dir, filename)
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if content seems to be raw HTML (case-insensitive check)
    if "<!DOCTYPE" in content.upper():
        print(f"Reprocessing {filename}: detected raw HTML content.")
        new_content = html2text.html2text(content)
    else:
        print(f"Skipping {filename}: no raw HTML detected.")
        new_content = content

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Saved reprocessed file to: {output_path}")
    return filename

def reprocess_md_files(input_dir, output_dir, n_jobs=-1):
    """
    Scan all .md files in the input directory. For each file, check if the content contains HTML markers
    (e.g., <!DOCTYPE). If so, reprocess the content using html2text and save it with the same filename
    in the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Gather all markdown files in input_dir
    md_files = [filename for filename in os.listdir(input_dir) if filename.lower().endswith(".md")]
    
    # Process files in parallel with a tqdm progress bar
    Parallel(n_jobs=n_jobs)(
        delayed(process_md_file)(filename, input_dir, output_dir) for filename in tqdm(md_files, desc="Processing MD files")
    )

    print(f"Reprocessing completed. All processed files are saved in '{output_dir}'.")

if __name__ == "__main__":
    # Example usage:
    input_directory = "data/downloaded_github_readmes/"       # replace with your source folder path
    output_directory = "data/downloaded_github_readmes_processed"   # replace with your desired output folder
    reprocess_md_files(input_directory, output_directory, n_jobs=-1)
