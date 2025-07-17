#!/usr/bin/env python3
import json
import argparse
import re
from pathlib import Path
from typing import Any

def standardize_csv_filename(filename: str) -> str:
    filename = re.sub(r'_s\.csv$', '.csv', filename)
    filename = re.sub(r'_t\.csv$', '.csv', filename)
    return filename

def process_json_content(data: Any) -> Any:
    if isinstance(data, dict):
        return {standardize_csv_filename(k): process_json_content(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [process_json_content(item) for item in data]
    elif isinstance(data, str):
        if data.endswith('.csv'):
            return standardize_csv_filename(data)
        return data
    else:
        return data

def process_json_file(input_file: Path, output_file: Path = None) -> Path:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processed_data = process_json_content(data)
    if output_file is None:
        output_file = input_file.with_name(input_file.stem + '_processed.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"Processed: {input_file} -> {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Standardize CSV filenames in JSON files")
    parser.add_argument('--input', required=True, help='Input JSON file or directory')
    parser.add_argument('--recursive', action='store_true', help='Process all JSON files in directory recursively')
    args = parser.parse_args()
    input_path = Path(args.input)
    if input_path.is_file():
        process_json_file(input_path)
    elif input_path.is_dir() and args.recursive:
        for json_file in input_path.rglob('*.json'):
            process_json_file(json_file)
    else:
        print('Error: --input must be a file or a directory with --recursive flag')
        exit(1)

if __name__ == '__main__':
    main() 