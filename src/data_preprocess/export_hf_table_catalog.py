"""Prepare a Hugging Face-ready table catalog for ModelTables / ModelSearch.

This command is deliberately local-only: it creates Parquet files and a
manifest, but never creates or uploads a Hugging Face repository.

For the ModelSearch table-search release, use the default ``--source hugging``.
The resulting files contain no queries, evaluation outputs, model-card text,
or API artifacts. All deduplicated tables are retained.

Example (small smoke test)::

    python -m src.data_preprocess.export_hf_table_catalog \
      --tag 251117 --source hugging --limit 100 \
      --output-dir data/hf_export/modelsearch_tables_251117_sample

Full local preparation::

    python -m src.data_preprocess.export_hf_table_catalog \
      --tag 251117 --source hugging \
      --output-dir data/hf_export/modelsearch_tables_251117
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_SPECS = {
    "hugging": {
        "directory": "deduped_hugging_csvs_v2_{tag}",
        "relationship_column": "hugging_table_list_dedup",
    },
    "github": {
        "directory": "deduped_github_csvs_v2_{tag}",
        "relationship_column": "github_table_list_dedup",
    },
    "html": {
        "directory": "tables_output_v2_{tag}",
        "relationship_column": "html_table_list_mapped_dedup",
    },
    "llm": {
        "directory": "llm_tables_{tag}",
        "relationship_column": "llm_table_list_mapped_dedup",
    },
}

TABLE_SCHEMA = pa.schema(
    [
        pa.field("table_csv", pa.string()),
        pa.field("source", pa.string()),
        pa.field("model_ids", pa.list_(pa.string())),
        pa.field("num_rows", pa.int32()),
        pa.field("num_columns", pa.int32()),
    ]
)


def iter_table_paths(directory: Path) -> Iterable[Path]:
    """Yield CSVs in deterministic order, including nested source folders."""
    yield from sorted(path for path in directory.rglob("*.csv") if path.is_file())


def normalize_path(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def build_model_map(relationship_path: Path, column: str) -> dict[str, set[str]]:
    """Build filename -> model ids from the pipeline's deduplicated mapping."""
    if not relationship_path.is_file():
        raise FileNotFoundError(f"Relationship parquet not found: {relationship_path}")

    mapping: dict[str, set[str]] = defaultdict(set)
    frame = pd.read_parquet(relationship_path, columns=["modelId", column])
    for model_id, paths in frame.itertuples(index=False, name=None):
        if paths is None:
            continue
        for path in paths:
            # Mapping paths can be absolute or project-relative.  The source
            # folders use unique file basenames, so this is the stable join key.
            name = Path(normalize_path(path)).name
            if name:
                mapping[name].add(str(model_id))
    return mapping


def csv_stats(text: str) -> tuple[int, int, str | None]:
    """Return data-row/maximum-column counts without altering CSV text."""
    try:
        rows = 0
        max_columns = 0
        # StringIO preserves legal newlines inside quoted CSV cells.
        for row in csv.reader(io.StringIO(text, newline="")):
            rows += 1
            max_columns = max(max_columns, len(row))
        # ModelTables QC counts data rows, excluding the CSV header.
        return max(rows - 1, 0), max_columns, None
    except csv.Error as exc:
        return 0, 0, str(exc)


def write_batch(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=TABLE_SCHEMA))
        rows.clear()


def export_catalog(
    *,
    processed_dir: Path,
    output_dir: Path,
    tag: str,
    source: str,
    batch_size: int,
    limit: int | None,
    overwrite: bool,
) -> dict:
    spec = SOURCE_SPECS[source]
    source_dir = processed_dir / spec["directory"].format(tag=tag)
    relationship_path = processed_dir / f"modelcard_step3_dedup_v2_{tag}.parquet"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source table directory not found: {source_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    model_map = build_model_map(relationship_path, spec["relationship_column"])
    table_paths = list(iter_table_paths(source_dir))
    existing_names = {path.name for path in table_paths}
    mapped_names = set(model_map)
    missing_csvs = mapped_names - existing_names
    unexpected_csvs = existing_names - mapped_names
    if missing_csvs or unexpected_csvs:
        details = (
            f"{len(missing_csvs)} parquet-referenced CSVs missing from disk; "
            f"{len(unexpected_csvs)} on-disk CSVs absent from the parquet mapping"
        )
        raise ValueError(
            "Table folder and model-table mapping are not the same corpus: " + details
        )
    table_path = output_dir / "tables.parquet"
    rows: list[dict] = []
    parse_errors = 0
    unmapped = 0
    table_count = 0
    model_table_pairs = 0

    with pq.ParquetWriter(table_path, TABLE_SCHEMA, compression="zstd") as table_writer:
        for csv_path in table_paths:
            if limit is not None and table_count >= limit:
                break
            text = csv_path.read_text(encoding="utf-8", errors="replace")
            num_rows, num_columns, error = csv_stats(text)
            model_ids = sorted(model_map.get(csv_path.name, set()))
            rows.append(
                {
                    "table_csv": text,
                    "source": source,
                    "model_ids": model_ids,
                    "num_rows": num_rows,
                    "num_columns": num_columns,
                }
            )
            model_table_pairs += len(model_ids)
            table_count += 1
            parse_errors += error is not None
            unmapped += not model_ids

            if len(rows) >= batch_size:
                write_batch(table_writer, rows)
        write_batch(table_writer, rows)

    manifest = {
        "format_version": 2,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "source": source,
        "input": {
            "table_directory": str(source_dir),
            "relationship_parquet": str(relationship_path),
            "relationship_column": spec["relationship_column"],
        },
        "outputs": {
            "tables": "tables.parquet",
        },
        "counts": {
            "tables": table_count,
            "candidate_tables": len(existing_names),
            "parquet_referenced_tables": len(mapped_names),
            "on_disk_tables": len(existing_names),
            "model_table_pairs": model_table_pairs,
            "unmapped_tables": unmapped,
            "csv_parse_errors": parse_errors,
        },
        "scope": "tables and model-table mapping only; no queries, evaluation results, or model-card text",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local HF-ready serialized table catalog; never uploads data.")
    parser.add_argument("--tag", default="251117", help="Snapshot tag used in existing ModelTables filenames.")
    parser.add_argument("--source", choices=sorted(SOURCE_SPECS), default="hugging", help="Table source to export.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="ModelTables processed data directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New local directory for Parquet outputs.")
    parser.add_argument("--batch-size", type=int, default=1_000, help="Rows written per Parquet batch.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of tables; useful for a smoke test.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--batch-size and --limit must be positive")

    manifest = export_catalog(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        tag=args.tag,
        source=args.source,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
