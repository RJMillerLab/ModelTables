"""
Test rowspan/colspan parsing
"""
import tempfile
import os
from src.data_preprocess.step2_gitcard_tab import detect_and_extract_markdown_tables

# Sample HTML with rowspan (like the Llama table)
test_html = """
<table>
  <tr>
    <th rowspan="6">General</th>
    <th>MMLU (5-shot)</th>
    <td>66.6</td>
    <td>45.7</td>
    <td>53.8</td>
    <td>79.5</td>
  </tr>
  <tr>
    <th>AGIEval English (3-5 shot)</th>
    <td>45.9</td>
    <td>28.8</td>
    <td>38.7</td>
    <td>63.0</td>
  </tr>
  <tr>
    <th>CommonSenseQA (7-shot)</th>
    <td>72.6</td>
    <td>57.6</td>
    <td>67.6</td>
    <td>83.8</td>
  </tr>
  <tr>
    <th>Winogrande (5-shot)</th>
    <td>76.1</td>
    <td>73.3</td>
    <td>75.4</td>
    <td>83.1</td>
  </tr>
  <tr>
    <th>BIG-Bench Hard (3-shot, CoT)</th>
    <td>61.1</td>
    <td>38.1</td>
    <td>47.0</td>
    <td>81.3</td>
  </tr>
  <tr>
    <th>ARC-Challenge (25-shot)</th>
    <td>78.6</td>
    <td>53.7</td>
    <td>67.6</td>
    <td>93.0</td>
  </tr>
  <tr>
    <th rowspan="1">Knowledge reasoning</th>
    <th>TriviaQA-Wiki (5-shot)</th>
    <td>78.5</td>
    <td>72.1</td>
    <td>79.6</td>
    <td>89.7</td>
  </tr>
  <tr>
    <th rowspan="4">Reading comprehension</th>
    <th>SQuAD (1-shot)</th>
    <td>76.4</td>
    <td>72.2</td>
    <td>72.1</td>
    <td>85.6</td>
  </tr>
  <tr>
    <th>QuAC (1-shot, F1)</th>
    <td>44.4</td>
    <td>39.6</td>
    <td>44.9</td>
    <td>51.1</td>
  </tr>
  <tr>
    <th>BoolQ (0-shot)</th>
    <td>75.7</td>
    <td>65.5</td>
    <td>66.9</td>
    <td>79.0</td>
  </tr>
  <tr>
    <th>DROP (3-shot, F1)</th>
    <td>58.4</td>
    <td>37.9</td>
    <td>49.8</td>
    <td>79.7</td>
  </tr>
</table>
"""

print("Testing rowspan/colspan parsing...")
found, tables = detect_and_extract_markdown_tables(test_html)

print(f"\nExtracted {len(tables)} table(s)")
for i, table in enumerate(tables, 1):
    print(f"\n{'='*80}")
    print(f"Table {i}:")
    print(f"{'='*80}")
    print(table)
    print()

# Check first few rows
if tables:
    lines = tables[0].split('\n')
    print("\nFirst 10 rows:")
    for i, line in enumerate(lines[:10], 1):
        print(f"{i}: {line}")

