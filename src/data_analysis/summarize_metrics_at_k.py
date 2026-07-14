"""
Summarize retrieval metrics JSON files at selected k values.

The starmie evaluation JSON already stores full precision/recall/map/f1
curves under system_metrics. This script extracts paper-table friendly
P/R/MAP/F1@k columns from one or more JSON files.

Example:
python -m src.data_analysis.summarize_metrics_at_k \
  --inputs "metrics_1030/metrics_scilake_final_none_tfidf_entity_*.json" \
  --out-csv metrics_1030/metrics_scilake_final_none_at_1_3_5_10.csv \
  --fig-dir experiments/metrics_v1_final
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Iterable


DEFAULT_KS = (1, 3, 5, 10)
METRIC_KEYS = ("precision", "recall", "map", "f1")
GT_ORDER = ("direct", "model", "dataset", "union")
GT_LABELS = {
    "direct": "Paper GT",
    "model": "Model GT",
    "dataset": "Dataset GT",
    "union": "Union GT",
}
METHOD_LABELS = {
    "starmie_none": "Starmie",
    "starmie_drop_cell": "Starmie + Drop Cell",
    "starmie_shuffle_row": "Starmie + Shuffle Row",
    "starmie_shuffle_col": "Starmie + Shuffle Col",
    "baseline": "Dense",
    "baseline2": "Sparse/BM25",
    "baseline3": "Hybrid",
    "baseline3_0712": "Hybrid",
    "baseline5": "Keyword Search",
    "baseline6": "Joinable Search",
}
COLORS = ("#e5f2ff", "#b8dcff", "#78b8ee", "#3f8fd2", "#0b5fae", "#083b7a")


def expand_inputs(patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def infer_labels(path: str) -> dict[str, str]:
    stem = Path(path).stem
    labels = {"source_group": Path(path).parent.name, "source_file": path}

    match = re.match(r"metrics_scilake_final_(.+?)_tfidf_entity_(.+)$", stem)
    if match:
        labels["method"] = f"starmie_{match.group(1)}"
        labels["gt_level"] = match.group(2)
        return labels

    match = re.match(r"metrics_baseline(\d*)_(.+)$", stem)
    if match:
        index = match.group(1)
        labels["method"] = f"baseline{index}" if index else "baseline"
        labels["gt_level"] = match.group(2)
        return labels

    labels["method"] = stem
    labels["gt_level"] = ""
    return labels


def value_at(values: list[float], k: int) -> float | None:
    if k <= 0 or len(values) < k:
        return None
    return float(values[k - 1])


def row_from_metrics(path: str, ks: list[int]) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    system = data.get("system_metrics", data)
    row: dict[str, object] = infer_labels(path)

    for metric in METRIC_KEYS:
        values = system.get(metric, [])
        for k in ks:
            row[f"{metric}@{k}"] = value_at(values, k)

    metrics_at_k = system.get("metrics_at_k", {})
    for k in ks:
        by_k = metrics_at_k.get(str(k)) or metrics_at_k.get(k) or {}
        for metric in METRIC_KEYS:
            key = f"{metric}@{k}"
            if row.get(key) is None and metric in by_k:
                row[key] = by_k[metric]

    return row


def format_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: str, rows: list[dict[str, object]], ks: list[int]) -> None:
    fieldnames = ["source_group", "method", "gt_level"]
    for metric in METRIC_KEYS:
        fieldnames.extend(f"{metric}@{k}" for k in ks)
    fieldnames.append("source_file")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_number(row.get(key)) for key in fieldnames})


def write_markdown(path: str, rows: list[dict[str, object]], ks: list[int]) -> None:
    fieldnames = ["source_group", "method", "gt_level"]
    for metric in ("precision", "recall"):
        fieldnames.extend(f"{metric}@{k}" for k in ks)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join("---" for _ in fieldnames) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_number(row.get(key)) for key in fieldnames) + " |\n")


def read_summary_csv(path: str) -> list[dict[str, object]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def metric_value(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def method_label(method: str, label_overrides: dict[str, str] | None = None) -> str:
    if label_overrides and method in label_overrides:
        return label_overrides[method]
    return METHOD_LABELS.get(method, method.replace("_", " "))


def display_method(row: dict[str, object], use_source_suffix: bool = False) -> str:
    method = str(row.get("method", ""))
    if use_source_suffix and method == "baseline3":
        group = str(row.get("source_group", ""))
        if group.endswith("0706"):
            return "baseline3_0706"
        if group.endswith("0712"):
            return "baseline3_0712"
    return method


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def text_width(draw, value: str, font) -> float:
    bbox = draw.textbbox((0, 0), value, font=font)
    return bbox[2] - bbox[0]


def save_png_as_pdf(png_path: Path, pdf_path: Path) -> None:
    from PIL import Image

    img = Image.open(png_path).convert("RGB")
    img.save(pdf_path, "PDF", resolution=150)


def select_rows(
    rows: list[dict[str, object]],
    methods: list[str],
    use_source_suffix: bool = False,
) -> list[dict[str, object]]:
    wanted = set(methods)
    selected = []
    seen = set()
    for row in rows:
        gt = str(row.get("gt_level", ""))
        shown_method = display_method(row, use_source_suffix)
        if gt not in GT_ORDER or shown_method not in wanted:
            continue
        key = (shown_method, gt)
        if key in seen:
            continue
        copied = dict(row)
        copied["display_method"] = shown_method
        selected.append(copied)
        seen.add(key)
    return selected


def draw_centered_legend(
    draw,
    methods: list[str],
    width: int,
    y: int,
    font,
    scale: int,
    label_overrides: dict[str, str] | None = None,
) -> None:
    def s(value: float) -> int:
        return int(round(value * scale))

    swatch_w = 24
    label_gap = 10
    item_gap = 52
    per_row = 3 if len(methods) > 4 else len(methods)
    for row_idx in range(0, len(methods), per_row):
        row_methods = methods[row_idx : row_idx + per_row]
        item_widths = [
            swatch_w + label_gap + text_width(draw, method_label(method, label_overrides), font)
            for method in row_methods
        ]
        legend_w = sum(item_widths) + item_gap * (len(row_methods) - 1)
        legend_x = max(24, (width - legend_w) / 2)
        row_y = y + (row_idx // per_row) * 26
        for m_idx, method in enumerate(row_methods):
            x = legend_x + sum(item_widths[:m_idx]) + item_gap * m_idx
            color = hex_to_rgb(COLORS[(row_idx + m_idx) % len(COLORS)])
            draw.rectangle([s(x), s(row_y - 11), s(x + swatch_w), s(row_y + 3)], fill=color)
            draw.text(
                (s(x + swatch_w + label_gap), s(row_y - 3)),
                method_label(method, label_overrides),
                fill=(0, 0, 0),
                anchor="lm",
                font=font,
            )


def render_metric_pair_grid(
    rows: list[dict[str, object]],
    methods: list[str],
    title: str,
    out_path: Path,
    ks: list[int],
    label_overrides: dict[str, str] | None = None,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to generate figures.") from exc

    scale = 2
    width, height = 1380, 520
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)

    def s(value: float) -> int:
        return int(round(value * scale))

    try:
        font = ImageFont.truetype("Arial.ttf", s(13))
        font_sm = ImageFont.truetype("Arial.ttf", s(10))
        font_xs = ImageFont.truetype("Arial.ttf", s(8))
        font_title = ImageFont.truetype("Arial Bold.ttf", s(22))
        font_panel = ImageFont.truetype("Arial Bold.ttf", s(16))
    except Exception:
        font = ImageFont.load_default()
        font_sm = ImageFont.load_default()
        font_xs = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_panel = ImageFont.load_default()

    def rotated_text(x: int, y: int, value: str) -> None:
        bbox = draw.textbbox((0, 0), value, font=font_panel)
        text_img = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((4, 4), value, fill=(0, 0, 0), font=font_panel)
        rotated = text_img.rotate(90, expand=True)
        img.paste(rotated.convert("RGB"), (s(x) - rotated.width // 2, s(y) - rotated.height // 2), rotated)

    panel_w, panel_h = 306, 150
    lefts = (74, 405, 736, 1067)
    tops = {"precision": 70, "recall": 250}
    pad_l, pad_r, pad_t, pad_b = 39, 8, 15, 28
    y_max = {}
    for metric in ("precision", "recall"):
        values = [metric_value(row, f"{metric}@{k}") for row in rows for k in ks]
        y_max[metric] = max(max(values or [1.0]) * 1.15, 0.01)

    def sy(metric: str, top: int, value: float) -> float:
        y0 = top + panel_h - pad_b
        y1 = top + pad_t
        return y0 - value / y_max[metric] * (y0 - y1)

    def draw_value_label(x: float, y: float, value: float) -> None:
        label = f"{value:.2f}" if value >= 0.01 else f"{value:.3f}"
        bbox = draw.textbbox((0, 0), label, font=font_xs)
        text_img = Image.new("RGBA", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((2, 2), label, fill=(25, 25, 25), font=font_xs)
        rotated = text_img.rotate(90, expand=True)
        img.paste(rotated.convert("RGB"), (s(x) - rotated.width // 2, s(y) - rotated.height - s(2)), rotated)

    draw.text((s(width / 2), s(24)), title, fill=(0, 0, 0), anchor="mm", font=font_title)
    for gt, left in zip(GT_ORDER, lefts):
        draw.text((s(left + panel_w / 2), s(49)), GT_LABELS[gt], fill=(0, 0, 0), anchor="mm", font=font_panel)

    for metric in ("precision", "recall"):
        top = tops[metric]
        row_label = "Precision@k" if metric == "precision" else "Recall@k"
        rotated_text(28, int(top + panel_h / 2), row_label)
        for gt, left in zip(GT_ORDER, lefts):
            gt_rows = {str(row["display_method"]): row for row in rows if row.get("gt_level") == gt}
            x0, x1 = left + pad_l, left + panel_w - pad_r
            y0, y1 = top + panel_h - pad_b, top + pad_t
            draw.rectangle([s(left), s(top), s(left + panel_w), s(top + panel_h)], fill=(253, 253, 253))
            draw.line([s(x0), s(y0), s(x1), s(y0)], fill=(51, 51, 51), width=s(1))
            draw.line([s(x0), s(y0), s(x0), s(y1)], fill=(51, 51, 51), width=s(1))
            for tick in (0, y_max[metric] / 2, y_max[metric]):
                y = sy(metric, top, tick)
                draw.line([s(x0), s(y), s(x1), s(y)], fill=(229, 229, 229), width=s(1))
                draw.text((s(x0 - 8), s(y)), f"{tick:.3f}", fill=(0, 0, 0), anchor="rm", font=font_sm)

            plot_w = x1 - x0
            cluster_w = plot_w / len(ks)
            inner_gap = cluster_w * 0.16
            usable_cluster_w = cluster_w - inner_gap
            bar_gap = 3
            bar_w = max(7, (usable_cluster_w - bar_gap * (len(methods) - 1)) / len(methods))
            for k_idx, k in enumerate(ks):
                cluster_left = x0 + k_idx * cluster_w + inner_gap / 2
                cluster_center = cluster_left + usable_cluster_w / 2
                if metric == "recall":
                    draw.text((s(cluster_center), s(y0 + 18)), f"@{k}", fill=(0, 0, 0), anchor="mm", font=font_sm)
                draw.line([s(cluster_center), s(y0), s(cluster_center), s(y0 + 5)], fill=(51, 51, 51), width=s(1))
                for m_idx, method in enumerate(methods):
                    row = gt_rows.get(method)
                    if not row:
                        continue
                    value = metric_value(row, f"{metric}@{k}")
                    bar_h = y0 - sy(metric, top, value)
                    x = cluster_left + m_idx * (bar_w + bar_gap)
                    y = y0 - bar_h
                    color = hex_to_rgb(COLORS[m_idx % len(COLORS)])
                    draw.rectangle([s(x), s(y), s(x + bar_w), s(y0)], fill=color)
                    draw_value_label(x + bar_w / 2, y, value)

    draw_centered_legend(draw, methods, width, 460, font, scale, label_overrides=label_overrides)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = img.resize((width, height))
    img.save(out_path)
    save_png_as_pdf(out_path, out_path.with_suffix(".pdf"))


def write_figures(rows: list[dict[str, object]], fig_dir: str, ks: list[int]) -> None:
    out_dir = Path(fig_dir)
    main_methods = ["baseline5", "baseline6", "starmie_shuffle_col", "baseline", "baseline2", "baseline3_0712"]
    starmie_methods = ["starmie_none", "starmie_drop_cell", "starmie_shuffle_row", "starmie_shuffle_col"]
    specs = [
        (
            "main_results_precision_recall_at_k",
            main_methods,
            "Main Results Precision/Recall@k",
            True,
            {"starmie_shuffle_col": "Union Search"},
        ),
        (
            "ablation_starmie_precision_recall_at_k",
            starmie_methods,
            "Starmie Structural Ablation Precision/Recall@k",
            False,
            None,
        ),
    ]
    for stem, methods, title, use_source_suffix, label_overrides in specs:
        selected = select_rows(rows, methods, use_source_suffix=use_source_suffix)
        if selected:
            path = out_dir / f"fig_{stem}.png"
            render_metric_pair_grid(selected, methods, title, path, ks, label_overrides=label_overrides)
            print(f"saved figure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract metrics at k and optionally render PNG/PDF figures.")
    parser.add_argument("--inputs", nargs="+", default=None, help="Metrics JSON paths or glob patterns.")
    parser.add_argument("--from-csv", default=None, help="Use an existing summary CSV instead of reading JSON.")
    parser.add_argument("--ks", default="1,3,5,10", help="Comma-separated k values.")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--fig-dir", default=None, help="Directory for PNG/PDF figures.")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if args.from_csv:
        rows = read_summary_csv(args.from_csv)
    else:
        if not args.inputs:
            raise ValueError("Pass --inputs or --from-csv.")
        paths = expand_inputs(args.inputs)
        if not paths:
            raise FileNotFoundError(f"No metrics JSON files matched: {args.inputs}")
        rows = [row_from_metrics(path, ks) for path in paths]

    rows.sort(
        key=lambda r: (
            str(r.get("method", "")),
            str(r.get("source_group", "")),
            str(r.get("gt_level", "")),
            str(r.get("source_file", "")),
        )
    )

    if args.out_csv:
        write_csv(args.out_csv, rows, ks)
        print(f"saved csv: {args.out_csv}")

    if args.out_md:
        write_markdown(args.out_md, rows, ks)
        print(f"saved md: {args.out_md}")

    if args.fig_dir:
        write_figures(rows, args.fig_dir, ks)


if __name__ == "__main__":
    main()
