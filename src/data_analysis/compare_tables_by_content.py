import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple


def get_base_id(filename: str) -> str:
    """
    Extract a base ID from a CSV filename.

    Assumes filenames look like:
        <some_prefix>_<index>.csv

    We treat everything before the last '_' as the base ID.
    Example:
        arxiv_2502.11111_1.csv -> arxiv_2502.11111
        2409.19581_table7.csv  -> 2409.19581_table
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    if "_" not in name:
        return name
    base, _ = name.rsplit("_", 1)
    return base


def list_csvs_for_base(directory: Path, base_id: str) -> List[Path]:
    """List all CSVs in a directory whose base_id matches the given one."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    result: List[Path] = []
    for p in directory.rglob("*.csv"):
        if get_base_id(p.name) == base_id:
            result.append(p)
    return sorted(result)


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_hash_index(paths: List[Path]) -> Tuple[Dict[Path, str], Dict[str, List[Path]]]:
    """
    For a list of paths, compute:
      - file_to_hash: Path -> hash
      - hash_to_files: hash -> [Paths]
    """
    file_to_hash: Dict[Path, str] = {}
    hash_to_files: Dict[str, List[Path]] = {}

    for p in paths:
        h = sha256_file(p)
        file_to_hash[p] = h
        hash_to_files.setdefault(h, []).append(p)

    return file_to_hash, hash_to_files


def _sanitize_for_filename(text: str) -> str:
    """Make a safe filename fragment from an arbitrary string."""
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def csv_to_markdown_table(path: Path) -> str:
    """
    Convert a CSV file to a Markdown table string.

    This is for quick manual inspection; we don't try to be clever about types,
    just render all cells as strings.
    """
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([cell.replace("|", "\\|") for cell in row])

    if not rows:
        return "_Empty table_\n"

    header = rows[0]
    body = rows[1:]

    lines = []
    # Header
    lines.append("| " + " | ".join(header) + " |")
    # Separator
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    # Rows
    for r in body:
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines) + "\n"


def compare_base_id(base_id: str, dir_a: Path, dir_b: Path) -> None:
    print(f"Base ID: {base_id}")
    print(f"Dir A: {dir_a}")
    print(f"Dir B: {dir_b}")
    print()

    files_a = list_csvs_for_base(dir_a, base_id)
    files_b = list_csvs_for_base(dir_b, base_id)

    print(f"Found {len(files_a)} CSV(s) in A, {len(files_b)} CSV(s) in B for base_id = {base_id}\n")

    if not files_a and not files_b:
        print("No CSVs found for this base_id in either directory.")
        return

    file_to_hash_a, hash_to_files_a = build_hash_index(files_a)
    file_to_hash_b, hash_to_files_b = build_hash_index(files_b)

    hashes_a = set(hash_to_files_a.keys())
    hashes_b = set(hash_to_files_b.keys())

    common_hashes = hashes_a & hashes_b

    # 1) Print pairings/groups with the same content (by hash)
    if common_hashes:
        print("=== Matched by content (SHA256) ===\n")
        for h in sorted(common_hashes):
            print(f"Hash: {h}")
            print("  A:")
            for p in hash_to_files_a[h]:
                print(f"    - {p}")
            print("  B:")
            for p in hash_to_files_b[h]:
                print(f"    - {p}")
            print()
    else:
        print("No content matches (no shared SHA256 hashes) between A and B for this base_id.\n")

    # 2) Remaining files that have no mapping (hash only on one side)
    unmatched_a = [p for p in files_a if file_to_hash_a[p] not in common_hashes]
    unmatched_b = [p for p in files_b if file_to_hash_b[p] not in common_hashes]

    print("=== Unmatched files (no content match on the other side) ===\n")
    print("In A only:")
    if unmatched_a:
        for p in unmatched_a:
            print(f"  - {p}  (hash={file_to_hash_a[p]})")
    else:
        print("  (none)")

    print("\nIn B only:")
    if unmatched_b:
        for p in unmatched_b:
            print(f"  - {p}  (hash={file_to_hash_b[p]})")
    else:
        print("  (none)")

    # Also write all unmatched tables into a single Markdown file for manual inspection.
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_base = _sanitize_for_filename(base_id)
    md_path = logs_dir / f"compare_tables_unmatched_{safe_base}.md"

    md_lines = []
    md_lines.append(f"# Unmatched tables for base_id `{base_id}`\n")
    md_lines.append("## Summary (from script output)\n")
    md_lines.append("### In A only\n")
    if unmatched_a:
        for p in unmatched_a:
            md_lines.append(f"- `{p}`  (hash={file_to_hash_a[p]})")
    else:
        md_lines.append("- (none)")
    md_lines.append("\n### In B only\n")
    if unmatched_b:
        for p in unmatched_b:
            md_lines.append(f"- `{p}`  (hash={file_to_hash_b[p]})")
    else:
        md_lines.append("- (none)")

    md_lines.append("\n---\n")
    md_lines.append("## Unmatched tables rendered as Markdown\n")

    # Render all unmatched tables as Markdown tables.
    if unmatched_a:
        md_lines.append("\n### Tables from A only\n")
        for p in unmatched_a:
            md_lines.append(f"\n#### `{p}`\n")
            md_lines.append(f"`hash = {file_to_hash_a[p]}`\n")
            md_lines.append("\n" + csv_to_markdown_table(p))

    if unmatched_b:
        md_lines.append("\n### Tables from B only\n")
        for p in unmatched_b:
            md_lines.append(f"\n#### `{p}`\n")
            md_lines.append(f"`hash = {file_to_hash_b[p]}`\n")
            md_lines.append("\n" + csv_to_markdown_table(p))

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nMarkdown with unmatched tables written to: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Given a base_id, compare CSVs under two tables_output directories by SHA256 content.\n"
                                    "We do NOT assume same filenames imply same content. Instead we group by hash,\n"
                                    "print all content-equal groups (A/B), and then list remaining unmatched files.")
    parser.add_argument("base_id", nargs="?", default=None, type=str, help="The base ID to inspect (e.g., '2409.19581_table' or 'arxiv_2502.11111').",)
    parser.add_argument("--dir-a", type=Path, default=Path("data/processed/tables_output"), help="First directory (default: data/processed/tables_output)",)
    parser.add_argument("--dir-b", type=Path, default=Path("data/processed/tables_output_v2_251117"), help="Second directory (default: data/processed/tables_output_v2_251117)",)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_base_id(args.base_id, args.dir_a, args.dir_b)


if __name__ == "__main__":
    main()

