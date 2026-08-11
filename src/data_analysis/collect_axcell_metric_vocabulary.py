"""
Collect metric/context names from the AxCell/Papers-with-Code released taxonomy files.

The AxCell release archive contains metrics.json and taxonomy.json. Metric names
are used as direct performance-table evidence; task and dataset names are
written separately as weak context evidence.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path


AXCELL_MODELS_URL = "https://github.com/paperswithcode/axcell/releases/download/v1.0/models.tar.xz"
AMBIGUOUS_METRIC_NAMES = {
    "all",
    "action",
    "average",
    "best",
    "books",
    "cc",
    "cnn",
    "count",
    "cs",
    "daily mail",
    "dev",
    "euclidean",
    "exact",
    "h",
    "is",
    "loss",
    "mean",
    "medium",
    "median",
    "mrpc",
    "overall",
    "parameters",
    "params",
    "performance",
    "pos",
    "price",
    "q1",
    "q2",
    "q3",
    "quality",
    "r1",
    "lr",
    "mad",
    "score",
    "sentiment",
    "sim",
    "sts",
    "test",
    "val",
    "value",
    "vs",
}
AMBIGUOUS_CONTEXT_NAMES = {
    "coin",
    "company",
    "drive",
    "general",
    "quantization",
    "seed",
    "spect",
}


def normalize_metric(value: str) -> str:
    return " ".join(str(value).strip().split())


def keep_metric(value: str, include_ambiguous: bool) -> bool:
    normalized = normalize_metric(value)
    lowered = normalized.lower()
    compact = "".join(ch for ch in lowered if ch.isalnum())
    if not normalized or lowered == "none":
        return False
    if re_fullmatch(r"[\d\s.%+-]+", normalized):
        return False
    if len(compact) <= 1:
        return False
    if "param" in lowered:
        return False
    if lowered.startswith("# of "):
        return False
    if not include_ambiguous and lowered in AMBIGUOUS_METRIC_NAMES:
        return False
    return True


def re_fullmatch(pattern: str, value: str) -> bool:
    import re

    return re.fullmatch(pattern, value) is not None


def download_models_archive(archive_path: Path, url: str) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, archive_path)


def extract_required_jsons(archive_path: Path, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    required = {"metrics.json", "taxonomy.json"}
    with tarfile.open(archive_path, "r:xz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in required:
                continue
            target = models_dir / name
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target.write_bytes(extracted.read())


def read_metrics(path: Path, include_ambiguous: bool) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        metric = normalize_metric(str(row.get("metric", "")))
        if keep_metric(metric, include_ambiguous):
            metrics.append(metric)
    return metrics


def keep_context(value: str) -> bool:
    normalized = normalize_metric(value)
    lowered = normalized.lower()
    compact = "".join(ch for ch in lowered if ch.isalnum())
    if not normalized or lowered == "none":
        return False
    if len(compact) < 3:
        return False
    if re_fullmatch(r"[\d\s.%+-]+", normalized):
        return False
    if "*" in normalized:
        return False
    if lowered in AMBIGUOUS_CONTEXT_NAMES:
        return False
    return True


def read_context_terms(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    terms: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        for field in ("task", "dataset"):
            value = normalize_metric(str(row.get(field, "")))
            if keep_context(value):
                terms.append(value)
    return terms


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    models_dir = Path(args.models_dir)
    archive_path = Path(args.archive_path)
    if args.download and not archive_path.exists():
        download_models_archive(archive_path, args.url)
    if args.extract and (not (models_dir / "metrics.json").exists() or not (models_dir / "taxonomy.json").exists()):
        extract_required_jsons(archive_path, models_dir)

    source_files = [models_dir / "metrics.json", models_dir / "taxonomy.json"]
    all_metrics: list[str] = []
    all_context_terms: list[str] = []
    source_counts: dict[str, int] = {}
    context_source_counts: dict[str, int] = {}
    for path in source_files:
        metrics = read_metrics(path, args.include_ambiguous)
        source_counts[path.name] = len(set(metrics))
        all_metrics.extend(metrics)
        context_terms = read_context_terms(path)
        context_source_counts[path.name] = len(set(context_terms))
        all_context_terms.extend(context_terms)

    counts = Counter(all_metrics)
    metrics = sorted(counts, key=lambda item: (item.lower(), item))
    context_counts = Counter(all_context_terms)
    context_terms = sorted(context_counts, key=lambda item: (item.lower(), item))
    out_dir = Path(args.out_dir)
    write_lines(out_dir / "axcell_metric_vocabulary.txt", metrics)
    write_lines(out_dir / "axcell_task_dataset_vocabulary.txt", context_terms)

    meta_lines = [
        f"source_url: {args.url}",
        f"archive_path: {archive_path}",
        f"models_dir: {models_dir}",
        "metric_fields_used: metric",
        "context_fields_used: task,dataset",
        f"ambiguous_metric_names_excluded: {not args.include_ambiguous}",
        f"unique_metrics: {len(metrics)}",
        f"unique_context_terms: {len(context_terms)}",
    ]
    meta_lines.extend(f"{name}_unique_metrics: {count}" for name, count in source_counts.items())
    meta_lines.extend(f"{name}_unique_context_terms: {count}" for name, count in context_source_counts.items())
    write_lines(out_dir / "axcell_metric_vocabulary_source.txt", meta_lines)

    print(f"unique_metrics={len(metrics)}")
    print(f"unique_context_terms={len(context_terms)}")
    for name, count in source_counts.items():
        print(f"{name}: {count}")
    print(f"saved: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AxCell/PWC metric vocabulary.")
    parser.add_argument("--models-dir", default="data/table_type/external_axcell/models_download/models")
    parser.add_argument("--archive-path", default="data/table_type/external_axcell/models_download/models.tar.xz")
    parser.add_argument("--url", default=AXCELL_MODELS_URL)
    parser.add_argument("--out-dir", default="data/table_type")
    parser.add_argument("--download", action="store_true", help="Download models.tar.xz if it is missing.")
    parser.add_argument("--extract", action="store_true", help="Extract metrics.json/taxonomy.json if missing.")
    parser.add_argument("--include-ambiguous", action="store_true", help="Keep very generic metric names such as loss/score/test.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
