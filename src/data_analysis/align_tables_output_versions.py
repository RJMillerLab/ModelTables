import argparse
import collections
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def list_csv_files(directory: Path) -> List[Path]:
    """Recursively list all CSV files under a directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return [p for p in directory.rglob("*.csv") if p.is_file()]


def get_base_id(filename: str) -> str:
    """
    Extract a base ID from a CSV filename.

    Assumes filenames look like:
        <some_prefix>_<index>.csv

    We treat everything before the last '_' as the base ID.
    Example:
        arxiv_2502.11111_1.csv -> arxiv_2502.11111
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    if "_" not in name:
        return name
    # split on last underscore
    base, _ = name.rsplit("_", 1)
    return base


def build_counts(csv_paths: List[Path]) -> Tuple[Dict[str, int], Dict[str, List[Path]]]:
    """Build counts and file lists per base ID."""
    counts: Dict[str, int] = collections.Counter()
    files_by_base: Dict[str, List[Path]] = collections.defaultdict(list)
    for p in csv_paths:
        base = get_base_id(p.name)
        counts[base] += 1
        files_by_base[base].append(p)
    return counts, files_by_base


def rank_key_from_base(base: str) -> int:
    """
    Ranking key: use the first 4 consecutive digits in the base ID.

    Examples:
        '2409.19581_table'      -> 2409
        'arxiv_2502.11111'      -> 2502

    If no 4-digit sequence is found, return -1 so it goes to the end.
    """
    m = re.search(r"(\d{4,})", base)
    if not m:
        return -1
    # Only use the first 4 digits for ranking.
    return int(m.group(1)[:4])


def compare_directories(dir_a: Path, dir_b: Path) -> None:
    print(f"Comparing:\n  A: {dir_a}\n  B: {dir_b}\n")

    csv_a = list_csv_files(dir_a)
    csv_b = list_csv_files(dir_b)

    counts_a, files_a = build_counts(csv_a)
    counts_b, files_b = build_counts(csv_b)

    # Only keep base IDs that appear in BOTH dirs, but with different counts.
    common_bases = set(counts_a) & set(counts_b)

    mismatches = []
    for base in common_bases:
        ca = counts_a.get(base, 0)
        cb = counts_b.get(base, 0)
        if ca != cb:
            mismatches.append((base, ca, cb))

    if not mismatches:
        print("✅ All base IDs have matching counts between the two directories.")
        return

    # Rank by "newest first": use first 4 digits in base ID, descending.
    mismatches.sort(key=lambda x: rank_key_from_base(x[0]), reverse=True)

    print("Found base IDs with different CSV counts (ranked by first 4 digits, newest first):\n")
    print(f"{'base_id':60}  {'count_A':>8}  {'count_B':>8}  {'rank_key':>8}")
    print("-" * 96)
    for base, ca, cb in mismatches:
        rk = rank_key_from_base(base)
        print(f"{base:60}  {ca:8d}  {cb:8d}  {rk:8d}")

    # For a few examples, print the actual file paths so it's easy to inspect.
    print("\nExamples (first 10 mismatched base IDs with file paths):\n")
    for base, ca, cb in mismatches[:10]:
        print(f"Base ID: {base}")
        print(f"  A ({ca} files):")
        for p in files_a.get(base, []):
            print(f"    - {p}")
        print(f"  B ({cb} files):")
        for p in files_b.get(base, []):
            print(f"    - {p}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CSV counts per base ID between two tables_output directories.\n\n"
            "Base ID is everything before the last '_' in the filename "
            "(e.g., arxiv_2502.11111_1.csv -> arxiv_2502.11111)."
        )
    )
    parser.add_argument(
        "--dir-a",
        type=Path,
        default=Path("data/processed/tables_output"),
        help="First directory to compare (default: data/processed/tables_output)",
    )
    parser.add_argument(
        "--dir-b",
        type=Path,
        default=Path("data/processed/tables_output_v2_251117"),
        help="Second directory to compare (default: data/processed/tables_output_v2_251117)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_directories(args.dir_a, args.dir_b)


if __name__ == "__main__":
    main()

