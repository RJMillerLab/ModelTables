import os
import argparse
import traceback
import html2text
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import pandas as pd
from src.utils import to_parquet, load_config

def process_md_file(filepath, input_root, output_dir, convert_only_when_html=True):
    rel_path = os.path.relpath(filepath, input_root)
    result = {
        'input_path': filepath,
        'relative_path': rel_path,
        'output_path': None,
        'converted_from_html': False,
        'error': None,
    }
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        is_html_like = "<!DOCTYPE" in content.upper() or content.strip().lower().startswith("<html")

        if is_html_like:
            new_content = html2text.html2text(content)
            result['converted_from_html'] = True
        else:
            if convert_only_when_html:
                new_content = content
            else:
                # Normalize markdown lightly (no-op placeholder; extend if needed)
                new_content = content

        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        result['output_path'] = out_path
        return result
    except Exception as e:
        result['error'] = f"{e}\n{traceback.format_exc()}"
        return result

def reprocess_md_files(input_dir, output_dir, n_jobs=4, convert_only_when_html=True, results_parquet="data/processed/md_parsing_results_v2.parquet"):
    """
    Recursively scan .md files in input_dir. If content contains HTML, convert to markdown via html2text.
    Save results preserving relative paths in output_dir. Produce a parquet summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    md_files = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".md"):
                md_files.append(os.path.join(root, name))

    print(f"Found {len(md_files)} markdown files under {input_dir}")

    with tqdm_joblib(tqdm(total=len(md_files), desc="Processing MD files")):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_md_file)(path, input_dir, output_dir, convert_only_when_html) for path in md_files
        )

    df = pd.DataFrame(results)
    try:
        to_parquet(df, results_parquet)
        print(f"Saved results parquet to {results_parquet}")
    except Exception as e:
        print(f"Warning: failed to save results parquet: {e}")

    num_errors = sum(1 for r in results if r.get('error'))
    print(f"Completed. {len(results) - num_errors} succeeded, {num_errors} failed.")
    return results

def main():
    parser = argparse.ArgumentParser(description='Parse/normalize markdown files (GitHub/HuggingFace READMEs)')
    parser.add_argument('--tag', dest='tag', default=None,
                        help='Tag suffix for versioning (e.g., 251117). Enables versioning mode.')
    parser.add_argument('--input_dir', default=None, help='Root directory containing .md files (default: auto-detect from tag)')
    parser.add_argument('--output_dir', default=None, help='Directory to write processed .md files (default: auto-detect from tag)')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')
    parser.add_argument('--convert_only_when_html', action='store_true', default=True, help='Only convert when content is HTML-like')
    parser.add_argument('--results_parquet', default=None, help='Where to save summary parquet (default: auto-detect from tag)')
    args = parser.parse_args()
    
    # Load config to get base_path
    config = load_config('config.yaml')
    base_path = config.get('base_path', 'data')
    processed_base_path = os.path.join(base_path, 'processed')
    
    # Determine paths based on tag
    tag = args.tag
    if args.input_dir:
        input_dir = args.input_dir
    else:
        if tag:
            input_dir = os.path.join(base_path, f"downloaded_github_readmes_{tag}")
        else:
            input_dir = os.path.join(base_path, "downloaded_github_readmes")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if tag:
            output_dir = os.path.join(base_path, f"downloaded_github_readmes_{tag}_processed")
        else:
            output_dir = os.path.join(base_path, "downloaded_github_readmes_processed")
    
    if args.results_parquet:
        results_parquet = args.results_parquet
    else:
        results_suffix = f"_{tag}" if tag else ""
        results_parquet = os.path.join(processed_base_path, f"md_parsing_results_v2{results_suffix}.parquet")
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Results parquet: {results_parquet}")

    reprocess_md_files(
        input_dir=input_dir,
        output_dir=output_dir,
        n_jobs=args.n_jobs,
        convert_only_when_html=args.convert_only_when_html,
        results_parquet=results_parquet,
    )

if __name__ == "__main__":
    main()
