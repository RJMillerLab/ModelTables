"""
Lightweight keyword-based model-lake table type classification.

The goal is not to build a final classifier. This script tests whether a
simple keyword taxonomy gives enough signal to support a
three-way type-aware retrieval analysis:

- performance: result, metric, leaderboard, and benchmark comparison tables.
- configuration: model, training, dataset, and hyperparameter specification tables.
- other: everything else, including label schemas and ambiguous/noisy tables.

Example:
python -m src.data_analysis.table_type_keyword \
  --data-root /Users/z6dong/Repo/ModelTables \
  --valid-list data/analysis/all_valid_title_valid_v2_251117.txt \
  --sample-size 3000 \
  --out-dir data/table_type
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PERFORMANCE_KEYWORDS = ["accuracy", "acc", "acc@", "f1", "f1-score", "precision", "recall", "auc", "auroc", "bleu", "sacrebleu", "rouge", "meteor", "bertscore", "perplexity", "ppl", "wer", "cer", "mrr", "map", "ndcg", "hit@", "hits@", "score", "metric", "result", "performance", "benchmark", "leaderboard", "rank", "win rate", "pass@", "exact match", "em", "avg", "average", "overall", "top 1", "top 5", "top-1", "top-5", "top1", "top5", "error", "rate", "success ratio", "mae", "rmse", "absrel", "silog", "sqrel", "iou", "miou", "ap", "ap50", "ap75", "fid", "gfid", "cider", "spice", "elo", "micro", "macro", "naturalness", "similarity", "diversity", "mmlu", "hellaswag", "gsm8k", "arc", "truthfulqa", "humaneval", "mt-bench", "winogrande", "bbh", "gpqa", "math", "drop", "squad", "cifar-10", "cifar-100", "imagenet", "objectnet", "mnli", "hans", "glue", "superglue", "ifeval", "alignbench", "wildbench", "arenahard", "vsi-bench", "cmu-mosi", "tnews", "iflytek", "ocnli", "afqmc", "csl", "wsc", "flores", "mkqa", "lambada", "nl4opt", "optmath", "optibench", "screenspot", "ceval", "cmmlu", "coco", "xcopa", "sighan", "openclip", "siglip", "domainnet", "fixmatch"]

CONFIGURATION_KEYWORDS = ["model", "method", "architecture", "backbone", "encoder", "decoder", "embedding", "checkpoint", "version", "config", "parameter", "parameters", "param", "params", "#params", "hyperparameter", "hyperparameters", "search value", "search values", "layer", "layers", "hidden", "head", "heads", "dimension", "dim", "width", "depth", "flops", "macs", "context", "context length", "sequence length", "max length", "dataset", "corpus", "split", "train", "training", "test", "validation", "dev", "samples", "instances", "examples", "tokens", "classes", "domain", "language", "task", "learning rate", "lr", "batch", "batch size", "epoch", "epochs", "step", "steps", "optimizer", "scheduler", "warmup", "dropout", "loss", "weight decay", "temperature", "top-p", "top p", "beam", "seed", "augmentation", "pretrain", "fine-tune", "finetune", "quant", "quantization", "bits", "group size", "act order"]

OTHER_KEYWORDS = ["label", "labels", "class", "classes", "category", "component", "entity", "tag", "type", "description", "license", "url", "link", "reference", "citation"]

TAXONOMY: dict[str, list[str]] = {
    "performance": PERFORMANCE_KEYWORDS,
    "configuration": CONFIGURATION_KEYWORDS,
    "other": OTHER_KEYWORDS,
}

LEGACY_TAXONOMY: dict[str, list[str]] = {
    "performance_leaderboard": [
        "accuracy",
        "acc",
        "f1",
        "precision",
        "recall",
        "auc",
        "bleu",
        "rouge",
        "perplexity",
        "ppl",
        "score",
        "metric",
        "result",
        "benchmark",
        "dataset",
        "task",
        "method",
        "model",
    ],
    "dataset_statistics": [
        "dataset",
        "split",
        "train",
        "test",
        "validation",
        "dev",
        "samples",
        "instances",
        "examples",
        "size",
        "tokens",
        "classes",
        "labels",
        "domain",
        "language",
    ],
    "model_configuration": [
        "model",
        "parameter",
        "parameters",
        "params",
        "layer",
        "layers",
        "hidden",
        "head",
        "heads",
        "embedding",
        "architecture",
        "backbone",
        "checkpoint",
        "version",
        "config",
    ],
    "training_hyperparameter": [
        "learning rate",
        "lr",
        "batch",
        "epoch",
        "optimizer",
        "scheduler",
        "warmup",
        "dropout",
        "loss",
        "weight decay",
        "training",
    ],
    "ablation_comparison": [
        "ablation",
        "variant",
        "setting",
        "component",
        "w/o",
        "without",
        "with",
        "baseline",
        "ours",
        "module",
    ],
    "label_schema": [
        "label",
        "labels",
        "class",
        "classes",
        "category",
        "component",
        "entity",
        "tag",
        "type",
        "description",
    ],
}

PRIORITY = ["performance", "configuration", "other"]

WEAK_PERFORMANCE_KEYWORDS = {"avg", "average", "overall", "error", "rate", "rank", "result", "score"}

LEGACY_PRIORITY = [
    "training_hyperparameter",
    "performance_leaderboard",
    "dataset_statistics",
    "model_configuration",
    "ablation_comparison",
    "label_schema",
]


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9@/._+-]+", " ", value.lower()).strip()


def compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_metric_vocabulary(path: Path) -> list[str]:
    metrics: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        metric = " ".join(line.strip().split())
        key = metric.lower()
        if not metric or key == "none" or key in seen:
            continue
        metrics.append(metric)
        seen.add(key)
    return metrics


def load_vocabulary(path: Path) -> list[str]:
    return load_metric_vocabulary(path)


def has_numeric_evidence(cells: list[str]) -> bool:
    text = " ".join(cells)
    return re.search(r"(?<![a-zA-Z])[-+]?\d+(?:\.\d+)?%?(?![a-zA-Z])", text) is not None


def has_result_structure(cells: list[str]) -> bool:
    text = " ".join(normalize_text(cell) for cell in cells)
    return re.search(
        r"(?<![a-z0-9])(model|models|method|methods|baseline|benchmark|benchmarks|ours|sota)(?![a-z0-9])",
        text,
    ) is not None


def has_metadata_shape(cells: list[str]) -> bool:
    text = " ".join(normalize_text(cell) for cell in cells[:12])
    metadata_terms = {"description", "license", "link", "model detail", "model authors", "paper or other resources"}
    pair_terms = {"anchor", "positive", "negative", "pos attr name", "neg attr name"}
    if any(term in text for term in metadata_terms):
        return True
    if any(len(cell) > 240 for cell in cells[:6]) and re.search(
        r"(?<![a-z0-9])(arxiv|pre print|webpage|github|honors|awards|scholarship|commendation)(?![a-z0-9])",
        text,
    ):
        return True
    return sum(term in text for term in pair_terms) >= 2


def tokenize_header(value: str) -> list[str]:
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "the",
        "this",
        "that",
        "with",
        "unnamed",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9@_+-]{1,}", value.lower())
        if token not in stopwords
    ]


COMPACT_PERFORMANCE_KEYWORDS = {
    compact_text(keyword)
    for keyword in PERFORMANCE_KEYWORDS
    if "@" not in keyword and len(compact_text(keyword)) >= 5
}


def classify_source(path: str) -> str:
    name = Path(path).name
    if "_github_" in path or re.fullmatch(r"[0-9a-f]{32}_table_\d+\.csv", name):
        return "github"
    if "_hugging_" in path or re.fullmatch(r"[0-9a-f]{10}_table\d+\.csv", name):
        return "huggingface"
    if "tables_output" in path or re.fullmatch(r"\d+\.\d+(?:v\d+)?_table\d+\.csv", name):
        return "paper_html"
    if "llm" in path:
        return "llm"
    return "unknown"


def read_table_preview(path: Path, content_rows: int, first_column_rows: int) -> tuple[list[str], list[list[str]], list[str]]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if any(cell.strip() for cell in row):
                header = [cell.strip() for cell in row]
                rows: list[list[str]] = []
                first_column: list[str] = []
                for content_row in csv.reader(f):
                    has_content = any(cell.strip() for cell in content_row)
                    if not has_content:
                        continue
                    if len(rows) < content_rows:
                        rows.append([cell.strip() for cell in content_row])
                    if len(first_column) < first_column_rows and content_row and content_row[0].strip():
                        first_column.append(content_row[0].strip())
                    if len(rows) >= content_rows and len(first_column) >= first_column_rows:
                        break
                return header, rows, first_column[:first_column_rows]
    return [], [], []


def keyword_matches(text: str, compact: str, keyword: str) -> bool:
    keyword = normalize_text(keyword)
    if not keyword:
        return False
    if "@" in keyword:
        return keyword in text
    if " " in keyword or "@" in keyword or "-" in keyword or "/" in keyword:
        return keyword in text or compact_text(keyword) in compact
    if re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None:
        return True
    if compact_text(keyword) in COMPACT_PERFORMANCE_KEYWORDS:
        return compact_text(keyword) in compact
    return False


def match_taxonomy(text: str, taxonomy: dict[str, list[str]]) -> dict[str, list[str]]:
    compact = compact_text(text)
    hits: dict[str, list[str]] = {}
    for table_type, keywords in taxonomy.items():
        matched = [keyword for keyword in keywords if keyword_matches(text, compact, keyword)]
        if matched:
            hits[table_type] = matched
    return hits


def match_context_terms(cells: list[str], context_terms: list[str], max_hits: int = 20) -> list[str]:
    text = " ".join(normalize_text(cell) for cell in cells)
    compact = compact_text(text)
    matched: list[str] = []
    for term in context_terms:
        normalized = normalize_text(term)
        if not normalized:
            continue
        term_compact = compact_text(term)
        if len(term_compact) < 4:
            continue
        if normalized in text or term_compact in compact:
            matched.append(term)
            if len(matched) >= max_hits:
                break
    return matched


def classify_text(
    cells: list[str],
    require_numeric_evidence: bool,
) -> tuple[str, dict[str, list[str]], dict[str, list[str]]]:
    text = " ".join(normalize_text(cell) for cell in cells)
    hits = match_taxonomy(text, TAXONOMY)
    legacy_hits = match_taxonomy(text, LEGACY_TAXONOMY)
    if not hits:
        return "other", hits, legacy_hits
    performance_hits = hits.get("performance", [])
    strong_performance_hits = [keyword for keyword in performance_hits if keyword not in WEAK_PERFORMANCE_KEYWORDS]
    has_required_numeric = not require_numeric_evidence or has_numeric_evidence(cells)
    if has_required_numeric and (strong_performance_hits or len(performance_hits) >= 2):
        return "performance", hits, legacy_hits

    def rank(table_type: str) -> tuple[int, int]:
        return len(hits.get(table_type, [])), -PRIORITY.index(table_type)

    return max(PRIORITY, key=rank), hits, legacy_hits


def classify_table(
    header: list[str],
    content_rows: list[list[str]],
    first_column: list[str],
    evidence_scope: str,
    max_evidence_cell_chars: int,
    require_numeric_evidence: bool,
) -> tuple[str, dict[str, list[str]], dict[str, list[str]]]:
    cells = list(header)
    if evidence_scope == "header_rows":
        for row in content_rows:
            cells.extend(cell for cell in row if len(cell) <= max_evidence_cell_chars)
    cells.extend(cell for cell in first_column if len(cell) <= max_evidence_cell_chars)
    return classify_text(cells, require_numeric_evidence)


def is_hf_training_log(source: str, header: list[str]) -> bool:
    if source != "huggingface":
        return False
    normalized = {normalize_text(cell) for cell in header}
    return {"training loss", "epoch", "step"}.issubset(normalized)


def is_metric_value_performance(header: list[str], cells: list[str]) -> bool:
    normalized_header = [normalize_text(cell) for cell in header if cell.strip()]
    if normalized_header != ["metric", "value"]:
        return False
    text = " ".join(normalize_text(cell) for cell in cells)
    if not has_numeric_evidence(cells):
        return False
    metadata_terms = {
        "architecture",
        "base model",
        "context length",
        "license",
        "model author",
        "model type",
        "parameter",
        "parameters",
        "precision",
        "tokenizer",
        "version",
    }
    if any(term in text for term in metadata_terms):
        return False
    metric_terms = (
        "accuracy",
        "arc",
        "bbh",
        "boolq",
        "cosine",
        "gpqa",
        "gsm8",
        "hellaswag",
        "humaneval",
        "ifeval",
        "math",
        "mmlu",
        "musr",
        "pearson",
        "silhouette",
        "spearman",
        "truthfulqa",
        "winogrande",
    )
    return any(term in text for term in metric_terms)


def is_context_performance(
    cells: list[str],
    context_terms: list[str],
    require_numeric_evidence: bool,
) -> tuple[bool, dict[str, list[str]]]:
    if not context_terms:
        return False, {}
    if require_numeric_evidence and not has_numeric_evidence(cells):
        return False, {}
    if not has_result_structure(cells):
        return False, {}
    if has_metadata_shape(cells):
        return False, {}
    matched = match_context_terms(cells, context_terms)
    if not matched:
        return False, {}
    if len(matched) < 2 and not re.search(r"(?<![a-z0-9])(benchmark|benchmarks|avg|average|performance)(?![a-z0-9])", " ".join(normalize_text(cell) for cell in cells)):
        return False, {"performance_context": matched}
    return True, {"performance_context": matched}


def resolve_path(data_root: Path, listed_path: str) -> Path:
    path = Path(listed_path)
    if path.is_absolute():
        return path
    return data_root / path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_review_txt(path: Path, data_root: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            full_path = resolve_path(data_root, str(row["table_path"])).resolve()
            f.write(
                "\t".join(
                    [
                        str(full_path),
                        str(row["table_type"]),
                        str(row["source"]),
                        str(row["matched_keywords"]),
                        str(row["matched_context_keywords"]),
                        str(row["header"]),
                    ]
                )
                + "\n"
            )


def write_by_source_txt(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                "\t".join(
                    [
                        str(row["source"]),
                        str(row["table_type"]),
                        str(row["count"]),
                        str(row["share_within_source"]),
                    ]
                )
                + "\n"
            )


def clean_tsv_field(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().replace("\t", " ")


def write_label_tsv(path: Path, data_root: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["table_path", "label", "resource", "header"])
        for row in rows:
            full_path = resolve_path(data_root, str(row["table_path"])).resolve()
            writer.writerow(
                [
                    clean_tsv_field(full_path),
                    clean_tsv_field(row["table_type"]),
                    clean_tsv_field(row["source"]),
                    clean_tsv_field(row["header"]),
                ]
            )


def run_cluster_audit(records: list[dict[str, object]], out_dir: Path, args: argparse.Namespace) -> None:
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = [record for record in records if str(record["header"]).strip()]
    if args.cluster_scope != "all":
        rows = [record for record in rows if record["table_type"] == args.cluster_scope]
    if len(rows) < 2:
        return
    if args.cluster_dedupe:
        deduped: dict[str, dict[str, object]] = {}
        for record in rows:
            deduped.setdefault(compact_text(str(record["header"])), record)
        clustering_rows = list(deduped.values())
    else:
        clustering_rows = rows
    texts = [str(record["header"]) for record in clustering_rows]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_df=0.9,
        max_features=5000,
        min_df=1 if args.cluster_dedupe else 2,
        ngram_range=(1, 2),
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9@_+-]{1,}\b",
    )
    matrix = vectorizer.fit_transform(texts)
    cluster_count = min(args.clusters, matrix.shape[0])
    terms = vectorizer.get_feature_names_out()
    if args.cluster_method == "kmeans":
        model = KMeans(n_clusters=cluster_count, random_state=args.seed, n_init=args.cluster_n_init)
        clustering_labels = model.fit_predict(matrix)
        components = model.cluster_centers_
    else:
        model = NMF(n_components=cluster_count, random_state=args.seed, init="nndsvda", max_iter=args.nmf_max_iter)
        weights = model.fit_transform(matrix)
        clustering_labels = np.asarray(weights.argmax(axis=1)).ravel()
        components = model.components_
    if args.cluster_dedupe:
        header_labels = {
            compact_text(str(record["header"])): int(label)
            for record, label in zip(clustering_rows, clustering_labels)
        }
        labels = [header_labels[compact_text(str(record["header"]))] for record in rows]
    else:
        labels = [int(label) for label in clustering_labels]

    cluster_records: dict[int, list[dict[str, object]]] = defaultdict(list)
    for label, record in zip(labels, rows):
        cluster_records[label].append(record)

    summary_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    wordfreq_rows: list[dict[str, object]] = []
    for cluster_id in range(cluster_count):
        members = cluster_records[cluster_id]
        label_counts = Counter(str(record["table_type"]) for record in members)
        top_label, top_count = label_counts.most_common(1)[0]
        component = components[cluster_id]
        top_indices = component.argsort()[-args.top_terms :][::-1]
        token_counts: Counter[str] = Counter()
        for record in members:
            token_counts.update(tokenize_header(str(record["header"])))
        summary_rows.append(
            {
                "cluster": cluster_id,
                "scope": args.cluster_scope,
                "method": args.cluster_method,
                "deduped_for_fit": args.cluster_dedupe,
                "count": len(members),
                "unique_headers": len({compact_text(str(record["header"])) for record in members}),
                "majority_seed_label": top_label,
                "majority_share": f"{top_count / len(members):.6f}",
                "seed_label_counts": ";".join(f"{label}:{count}" for label, count in label_counts.most_common()),
                "top_terms": ";".join(terms[index] for index in top_indices if component[index] > 0),
                "top_words": ";".join(token for token, _ in token_counts.most_common(args.top_words)),
            }
        )
        for record in members[: args.examples_per_cluster]:
            example_rows.append(
                {
                    "cluster": cluster_id,
                    "seed_label": record["table_type"],
                    "table_path": record["table_path"],
                    "header": record["header"],
                    "matched_keywords": record["matched_keywords"],
                }
            )
        for rank, (token, count) in enumerate(token_counts.most_common(args.top_words), start=1):
            wordfreq_rows.append(
                {
                    "cluster": cluster_id,
                    "rank": rank,
                    "token": token,
                    "count": count,
                    "share": f"{count / sum(token_counts.values()):.6f}" if token_counts else "0",
                }
            )

    write_csv(
        out_dir / "table_type_cluster_summary.csv",
        [
            "cluster",
            "scope",
            "method",
            "deduped_for_fit",
            "count",
            "unique_headers",
            "majority_seed_label",
            "majority_share",
            "seed_label_counts",
            "top_terms",
            "top_words",
        ],
        summary_rows,
    )
    write_csv(
        out_dir / "table_type_cluster_examples.csv",
        ["cluster", "seed_label", "table_path", "header", "matched_keywords"],
        example_rows,
    )
    write_csv(
        out_dir / "table_type_cluster_wordfreq.csv",
        ["cluster", "rank", "token", "count", "share"],
        wordfreq_rows,
    )


def run(args: argparse.Namespace) -> None:
    global PERFORMANCE_KEYWORDS, TAXONOMY, WEAK_PERFORMANCE_KEYWORDS, COMPACT_PERFORMANCE_KEYWORDS

    context_terms: list[str] = []
    if args.metric_vocab:
        metric_keywords = load_metric_vocabulary(Path(args.metric_vocab))
        if args.metric_vocab_only:
            PERFORMANCE_KEYWORDS = metric_keywords
        else:
            PERFORMANCE_KEYWORDS = sorted({*PERFORMANCE_KEYWORDS, *metric_keywords}, key=lambda item: item.lower())
        TAXONOMY = {
            "performance": PERFORMANCE_KEYWORDS,
            "configuration": CONFIGURATION_KEYWORDS,
            "other": OTHER_KEYWORDS,
        }
        WEAK_PERFORMANCE_KEYWORDS = set() if args.metric_vocab_only else WEAK_PERFORMANCE_KEYWORDS
        COMPACT_PERFORMANCE_KEYWORDS = {
            compact_text(keyword)
            for keyword in PERFORMANCE_KEYWORDS
            if "@" not in keyword and len(compact_text(keyword)) >= 5
        }
    if args.context_vocab:
        context_terms = load_vocabulary(Path(args.context_vocab))

    data_root = Path(args.data_root)
    valid_list = data_root / args.valid_list
    paths = [line.strip() for line in valid_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.sample_size and args.sample_size < len(paths):
        random.seed(args.seed)
        paths = random.sample(paths, args.sample_size)

    records: list[dict[str, object]] = []
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    missing_header = 0
    multi_hit = 0

    for listed_path in paths:
        path = resolve_path(data_root, listed_path)
        source = classify_source(listed_path)
        try:
            header, content_rows, first_column = read_table_preview(path, args.content_rows, args.first_column_rows)
        except OSError:
            header, content_rows, first_column = [], [], []
        if not header:
            missing_header += 1
            table_type, hits, legacy_hits, context_hits = "other", {}, {}, {}
        else:
            table_type, hits, legacy_hits = classify_table(
                header,
                content_rows,
                first_column,
                args.evidence_scope,
                args.max_evidence_cell_chars,
                args.require_numeric_evidence,
            )
            evidence_cells = list(header)
            if args.evidence_scope == "header_rows":
                for row in content_rows:
                    evidence_cells.extend(cell for cell in row if len(cell) <= args.max_evidence_cell_chars)
            evidence_cells.extend(cell for cell in first_column if len(cell) <= args.max_evidence_cell_chars)
            context_is_performance, context_hits = is_context_performance(
                evidence_cells,
                context_terms,
                args.require_numeric_evidence,
            )
            metric_value_is_performance = is_metric_value_performance(header, evidence_cells)
            if args.diagnostic_labels:
                if table_type == "performance" or metric_value_is_performance:
                    table_type = "performance_metric"
                elif context_is_performance:
                    table_type = "performance_context"
                elif args.separate_training_logs and is_hf_training_log(source, header):
                    table_type = "hf_training_log"
                else:
                    table_type = "non_performance"
        if args.binary_performance_config and table_type != "performance":
            table_type = "configuration"
        if len(hits) > 1:
            multi_hit += 1
        record = {
            "table_path": listed_path,
            "source": source,
            "table_type": table_type,
            "header": " | ".join(header[:12]),
            "content_preview": " || ".join(" | ".join(row[:12]) for row in content_rows[: args.content_rows]),
            "first_column_preview": " | ".join(first_column[: args.first_column_rows]),
            "matched_types": ";".join(hits),
            "matched_keywords": ";".join(f"{k}:{','.join(v)}" for k, v in hits.items()),
            "matched_context_keywords": ";".join(f"{k}:{','.join(v)}" for k, v in context_hits.items()),
            "legacy_matched_types": ";".join(legacy_hits),
            "legacy_matched_keywords": ";".join(f"{k}:{','.join(v)}" for k, v in legacy_hits.items()),
        }
        records.append(record)
        if len(examples[table_type]) < args.examples_per_type:
            examples[table_type].append(record)

    counts = Counter(record["table_type"] for record in records)
    source_counts = Counter((record["source"], record["table_type"]) for record in records)
    source_totals = Counter(record["source"] for record in records)
    total = len(records)
    summary_rows = [
        {
            "table_type": table_type,
            "count": count,
            "share": f"{count / total:.6f}" if total else "0",
        }
        for table_type, count in counts.most_common()
    ]
    by_source_rows = [
        {
            "source": source,
            "table_type": table_type,
            "count": count,
            "share_within_source": f"{count / source_totals[source]:.6f}" if source_totals[source] else "0",
        }
        for (source, table_type), count in sorted(source_counts.items())
    ]
    example_rows = [row for table_type in sorted(examples) for row in examples[table_type]]

    out_dir = Path(args.out_dir)
    if args.label_tsv:
        write_label_tsv(Path(args.label_tsv), data_root, records)
    if not args.compact_output:
        write_csv(out_dir / "table_type_keyword_summary.csv", ["table_type", "count", "share"], summary_rows)
        write_csv(
            out_dir / "table_type_keyword_by_source.csv",
            ["source", "table_type", "count", "share_within_source"],
            by_source_rows,
        )
        write_by_source_txt(out_dir / "table_type_keyword_by_source.txt", by_source_rows)
        write_csv(
            out_dir / "table_type_keyword_examples.csv",
            [
                "table_path",
                "source",
                "table_type",
                "header",
                "content_preview",
                "first_column_preview",
                "matched_types",
                "matched_keywords",
                "matched_context_keywords",
                "legacy_matched_types",
                "legacy_matched_keywords",
            ],
            example_rows,
        )
        if args.write_review_txt:
            write_review_txt(out_dir / "table_type_keyword_review.txt", data_root, records)
        if args.write_all:
            write_csv(
                out_dir / "table_type_keyword_predictions.csv",
                [
                    "table_path",
                    "source",
                    "table_type",
                    "header",
                    "content_preview",
                    "first_column_preview",
                    "matched_types",
                    "matched_keywords",
                    "matched_context_keywords",
                    "legacy_matched_types",
                    "legacy_matched_keywords",
                ],
                records,
            )
    if args.cluster_audit:
        run_cluster_audit(records, out_dir, args)

    print(f"sampled_tables={total}")
    print(f"missing_header={missing_header}")
    print(f"multi_hit={multi_hit}")
    for row in summary_rows:
        print(f"{row['table_type']}: {row['count']} ({float(row['share']):.1%})")
    if args.cluster_audit:
        print("cluster_audit=enabled")
    print(f"saved: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run keyword-based table type classification.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--valid-list", default="data/analysis/all_valid_title_valid_v2_251117.txt")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="data/table_type")
    parser.add_argument("--examples-per-type", type=int, default=10)
    parser.add_argument("--evidence-scope", choices=["header", "header_rows"], default="header")
    parser.add_argument("--content-rows", type=int, default=3)
    parser.add_argument("--first-column-rows", type=int, default=0)
    parser.add_argument("--max-evidence-cell-chars", type=int, default=80)
    parser.add_argument("--metric-vocab", default="", help="One metric name per line, e.g. AxCell/PWC metric vocabulary.")
    parser.add_argument("--context-vocab", default="", help="One weak context term per line, e.g. AxCell/PWC task/dataset names.")
    parser.add_argument("--metric-vocab-only", action="store_true")
    parser.add_argument("--require-numeric-evidence", action="store_true")
    parser.add_argument("--diagnostic-labels", action="store_true")
    parser.add_argument("--separate-training-logs", action="store_true")
    parser.add_argument("--cluster-audit", action="store_true")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--cluster-method", choices=["kmeans", "nmf"], default="kmeans")
    parser.add_argument("--cluster-scope", choices=["all", "performance", "configuration", "other"], default="all")
    parser.add_argument("--cluster-dedupe", action="store_true")
    parser.add_argument("--cluster-n-init", type=int, default=20)
    parser.add_argument("--nmf-max-iter", type=int, default=400)
    parser.add_argument("--binary-performance-config", action="store_true")
    parser.add_argument("--write-review-txt", action="store_true")
    parser.add_argument("--label-tsv", default="")
    parser.add_argument("--compact-output", action="store_true")
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--top-words", type=int, default=30)
    parser.add_argument("--examples-per-cluster", type=int, default=8)
    parser.add_argument("--write-all", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
