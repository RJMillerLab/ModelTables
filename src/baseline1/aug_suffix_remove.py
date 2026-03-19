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
        # First handle dictionaries where keys are CSV filenames mapping to lists/objects.
        # 1. Standardise the key name.
        # 2. Recursively process the value to standardise any filenames it contains.
        # 3. If the value is a list, remove any element that equals the (standardised) key itself
        #    and also remove duplicates while preserving order.
        # 4. If the value is a single string equal to the key, set it to None (will become null in JSON).
        new_dict = {}
        for k, v in data.items():
            std_k = standardize_csv_filename(k)
            processed_v = process_json_content(v)

            # Remove self-references from list values
            if isinstance(processed_v, list):
                seen = set()
                filtered_list = []
                for item in processed_v:
                    if item == std_k:
                        continue  # skip self
                    if item in seen:
                        continue  # skip duplicates
                    seen.add(item)
                    filtered_list.append(item)
                processed_v = filtered_list

            # Remove self-reference from string value
            elif isinstance(processed_v, str) and processed_v == std_k:
                processed_v = None

            new_dict[std_k] = processed_v
        return new_dict
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

    # Basic statistics before processing
    keys_before = len(data) if isinstance(data, dict) else None

    processed_data = process_json_content(data)

    # Statistics after processing
    if isinstance(processed_data, dict):
        keys_after = len(processed_data)
        non_empty_value_list_keys = sum(
            1 for v in processed_data.values() if isinstance(v, list) and len(v) > 0
        )
    else:
        keys_after = None
        non_empty_value_list_keys = None

    if output_file is None:
        output_file = input_file.with_name(input_file.stem + '_processed.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # Print statistics for user visibility
    stats_msg = (
        f"Processed: {input_file} -> {output_file}\n"
        f"  Keys before processing: {keys_before}\n"
        f"  Keys after processing: {keys_after}\n"
        f"  Keys with non-empty value list after processing: {non_empty_value_list_keys}"
    )
    print(stats_msg)
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