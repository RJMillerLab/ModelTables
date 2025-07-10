#!/usr/bin/env python3
import json
import sys

def append_csv_suffix(input_path, output_path=None):
    # Load the original JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Append .csv to each key and neighbor string
    new_data = {
        key + '.csv': [neighbor + '.csv' for neighbor in neighbors]
        for key, neighbors in data.items()
    }

    # Determine output file path
    out_path = output_path or input_path

    # Write the updated JSON back
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_json> [output_json]", file=sys.stderr)
        sys.exit(1)

    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else None
    append_csv_suffix(inp, outp)
