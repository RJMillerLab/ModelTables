import json
import sys
import argparse
import re
from typing import List

def get_table_base(name):
    # e.g. table1, table1_s, table1_t, table1_s_t
    return re.sub(r'(_s|_t|_s_t)?$', '', name)

def ensure_csv_suffix(name):
    return name if name.endswith('.csv') else name + '.csv'

def filter_keys(data, key_types: List[str]):
    # key_types: list of '', '_s', '_t', etc.
    filtered = {}
    for k, v in data.items():
        k_csv = ensure_csv_suffix(k)
        
        # Special handling for empty string key_types
        if "" in key_types:
            # Empty string should match keys that don't end with _s, _t, or _s_t
            if not (k_csv.endswith('_s.csv') or k_csv.endswith('_t.csv') or k_csv.endswith('_s_t.csv')):
                filtered[k_csv] = v
        else:
            # Normal handling for other key_types
            if any(k_csv.endswith(kt + '.csv') for kt in key_types):
                filtered[k_csv] = v
    return filtered

def filter_values(vlist, value_types: List[str]):
    # value_types: list of '', '_s', '_t', etc.
    result = []
    for v in vlist:
        v_csv = ensure_csv_suffix(v)
        if any(v_csv.endswith(vt + '.csv') for vt in value_types):
            result.append(v_csv)
    return result

def remove_self_variants(neighbors_dict, key_types, value_types):
    result = {}
    for k, vlist in neighbors_dict.items():
        base = get_table_base(k.replace('.csv', ''))
        filtered = [n for n in vlist if get_table_base(n.replace('.csv', '')) != base]
        result[k] = filtered
    return result

def process(input_json, key_types, value_types, output):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    filtered_keys = filter_keys(data, key_types)
    filtered = {k: filter_values(v, value_types) for k, v in filtered_keys.items()}
    cleaned = remove_self_variants(filtered, key_types, value_types)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output}, keys: {len(cleaned)}")

def parse_types(types_str):
    if isinstance(types_str, list):
        return types_str
    return [t.strip() for t in types_str.split(',') if t.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General postprocess for table retrieval experiments.")
    parser.add_argument('--input', required=True, help='input neighbor json')
    parser.add_argument('--key_types', nargs='+', required=True, help='key types, e.g. "" _s _t')
    parser.add_argument('--value_types', nargs='+', required=True, help='value types, e.g. "" _s _t')
    parser.add_argument('--output', required=True, help='output json')
    args = parser.parse_args()
    key_types = parse_types(args.key_types)
    value_types = parse_types(args.value_types)
    process(args.input, key_types, value_types, args.output) 