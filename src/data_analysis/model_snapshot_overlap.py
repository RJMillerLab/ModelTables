#!/usr/bin/env python
"""
Compare modelId overlap between two fixed snapshots: V1 (no tag, no v2) vs V2 (tag 251117 + v2).
Visualization: two-circle Venn diagram per state (V1 circle, V2 circle, overlap); each region shows count.

Based on hf_models_analysis: same four conditions, modelId lists (not counts).
Output: data/analysis/model_snapshot_overlap.png (.pdf).
"""

import math
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch
from scipy.optimize import brentq



VALID_CARD_COND = "card IS NOT NULL AND card <> '' AND card <> 'Entry not found'"

# Same "any table" condition as hf_models_analysis
ANY_TABLE_COND = """
    (hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0)
    OR (html_table_list_mapped_dedup IS NOT NULL AND array_length(html_table_list_mapped_dedup) > 0)
    OR (llm_table_list_mapped_dedup IS NOT NULL AND array_length(llm_table_list_mapped_dedup) > 0)
    OR (github_table_list_dedup IS NOT NULL AND array_length(github_table_list_dedup) > 0)
"""


def build_raw_glob(tag: str | None) -> str:
    if tag:
        return os.path.join("data", f"raw_{tag}", "train-*-of-00006.parquet")
    return os.path.join("data", "raw", "train-*-of-00004.parquet")


def build_step3_dedup_path(tag: str | None, v2_mode: bool) -> str:
    suffix = f"_{tag}" if tag else ""
    v2_suffix = "_v2" if v2_mode else ""
    return os.path.join("data", "processed", f"modelcard_step3_dedup{v2_suffix}{suffix}.parquet")


def get_all_model_ids(con: duckdb.DuckDBPyConnection, raw_glob: str) -> set[str]:
    q = f"SELECT DISTINCT modelId FROM read_parquet('{raw_glob}') WHERE modelId IS NOT NULL"
    df = con.execute(q).fetchdf()
    return {str(mid) for mid in df["modelId"].tolist()}


def get_cards_model_ids(con: duckdb.DuckDBPyConnection, raw_glob: str) -> set[str]:
    q = f"""
        SELECT DISTINCT modelId FROM read_parquet('{raw_glob}')
        WHERE {VALID_CARD_COND} AND modelId IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    return {str(mid) for mid in df["modelId"].tolist()}


def get_any_table_model_ids(con: duckdb.DuckDBPyConnection, step3_path: str) -> set[str]:
    q = f"""
        SELECT DISTINCT modelId FROM read_parquet('{step3_path}')
        WHERE {ANY_TABLE_COND}
    """
    df = con.execute(q).fetchdf()
    return {str(mid) for mid in df["modelId"].tolist()}


def get_hugging_table_model_ids(con: duckdb.DuckDBPyConnection, step3_path: str) -> set[str]:
    q = f"""
        SELECT DISTINCT modelId FROM read_parquet('{step3_path}')
        WHERE hugging_table_list_dedup IS NOT NULL AND array_length(hugging_table_list_dedup) > 0
    """
    df = con.execute(q).fetchdf()
    return {str(mid) for mid in df["modelId"].tolist()}


def overlap_stats(name: str, a: set[str], b: set[str]) -> None:
    inter = a & b
    only_a = a - b
    only_b = b - a
    print(f"\n--- {name} (modelId list overlap) ---")
    print(f"  |A| (V1)           = {len(a):,}")
    print(f"  |B| (V2)           = {len(b):,}")
    print(f"  |A ∩ B|            = {len(inter):,}")
    print(f"  |A \\ B| (only V1)  = {len(only_a):,}")
    print(f"  |B \\ A| (only V2)  = {len(only_b):,}")
    if len(a) > 0:
        print(f"  Overlap rate V1→V2 = {len(inter)/len(a)*100:.2f}%  (fraction of V1 modelIds also in V2)")
    if len(b) > 0:
        print(f"  Overlap rate V2→V1 = {len(inter)/len(b)*100:.2f}%  (fraction of V2 modelIds also in V1)")


# Hardcoded: V1 = no tag, no v2; V2 = tag 251117 + v2
TAG_V2 = "251117"
V2_MODE = True

STATE_NAMES = [
    "All Models",
    "Models w/ Cards",
    "Models w/ Any Table",
    "Models w/ Hugging Tables",
]


def _overlap_numbers(a: set[str], b: set[str]) -> dict:
    inter = a & b
    only_a = a - b
    only_b = b - a
    n_a, n_b = len(a), len(b)
    rate_v1_v2 = (len(inter) / n_a * 100) if n_a else 0.0
    rate_v2_v1 = (len(inter) / n_b * 100) if n_b else 0.0
    return {
        "|A|": n_a,
        "|B|": n_b,
        "|A∩B|": len(inter),
        "|A\\B|": len(only_a),
        "|B\\A|": len(only_b),
        "rate_V1→V2": rate_v1_v2,
        "rate_V2→V1": rate_v2_v1,
    }


def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    """Area of intersection of two circles with radii r1, r2 and center distance d."""
    if d <= 0 or d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    d2 = d * d
    r12, r22 = r1 * r1, r2 * r2
    term = (d2 + r12 - r22) / (2 * d * r1)
    term = max(-1, min(1, term))
    a1 = r12 * math.acos(term)
    term2 = (d2 + r22 - r12) / (2 * d * r2)
    term2 = max(-1, min(1, term2))
    a2 = r22 * math.acos(term2)
    s = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    if s <= 0:
        return a1 + a2
    return a1 + a2 - 0.5 * math.sqrt(s)


# Base count: circle area = (π/base)*count => radius = sqrt(count/base). So same count => same radius across panels.
BASE = 400_000


def _solve_d_for_intersection(r1: float, r2: float, target_area: float) -> float:
    """Find center distance d so that intersection area of two circles (r1, r2) = target_area."""
    d_lo = max(abs(r1 - r2), 1e-6)
    d_hi = r1 + r2 - 1e-6
    if d_lo >= d_hi:
        return (r1 + r2) * 0.5
    if target_area <= 0:
        return r1 + r2  # no overlap
    if target_area >= _circle_intersection_area(r1, r2, d_lo):
        return d_lo
    if target_area <= _circle_intersection_area(r1, r2, d_hi):
        return d_hi

    def objective(d_val: float) -> float:
        return _circle_intersection_area(r1, r2, d_val) - target_area

    try:
        if objective(d_lo) <= 0 <= objective(d_hi) or objective(d_hi) <= 0 <= objective(d_lo):
            return brentq(objective, d_lo, d_hi)
    except Exception:
        pass
    return (d_lo + d_hi) / 2


def draw_venn2_overlap(
    ax,
    only_v1: int,
    only_v2: int,
    both: int,
    title: str,
    base: float | int = BASE,
    xlim_ylim: tuple[float, float] | None = None,
) -> None:
    """
    Draw one two-circle Venn from counts. All lengths in same scale as base.

    Logic:
      - |V1| = only_v1 + both, |V2| = only_v2 + both.
      - Circle area ∝ count => radius = sqrt(count/base). So r1 = sqrt(|V1|/base), r2 = sqrt(|V2|/base).
        E.g. if base=110w and one diagram's red circle count is 1/4 of another's, its red radius is 1/2.
      - Overlap area ∝ both => target intersection = (π/base)*both; solve for center distance d.
      - If xlim_ylim is set (e.g. from plot_overlap_venn), use it so all panels share the same data range
        and circle sizes are comparable across panels (smaller counts => smaller circles on the same axes).
    """
    base = max(float(base), 1.0)
    n_a = only_v1 + both
    n_b = only_v2 + both

    # Radii from area ∝ count: π r^2 = (π/base)*count => r = sqrt(count/base)
    r1 = math.sqrt(n_a / base) if n_a > 0 else 0.0
    r2 = math.sqrt(n_b / base) if n_b > 0 else 0.0
    if r1 <= 0 and r2 <= 0:
        r1 = r2 = 0.5
    elif r1 <= 0:
        r1 = r2 * 0.3
    elif r2 <= 0:
        r2 = r1 * 0.3

    # Overlap area = (π/base)*both => solve for d
    target_I = (math.pi / base) * both
    d = _solve_d_for_intersection(r1, r2, target_I)

    cx1, cx2 = -d / 2, d / 2
    c1 = Circle((cx1, 0), r1, facecolor="#4a90d9", alpha=0.4, edgecolor="black", linewidth=1)
    c2 = Circle((cx2, 0), r2, facecolor="#e74c3c", alpha=0.4, edgecolor="black", linewidth=1)
    ax.add_patch(c1)
    ax.add_patch(c2)

    if xlim_ylim is not None:
        lim, _ = xlim_ylim
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    else:
        margin = 0.3
        lim = max(r1, r2) + abs(d) / 2 + margin
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    ax.set_aspect("equal")
    ax.axis("off")

    cur_lim = (xlim_ylim[0] if xlim_ylim is not None else max(r1, r2) + abs(d) / 2 + 0.3)
    # Numbers below the diagram: equal spacing, minimal gap to circles
    num_y = -cur_lim - 0.02
    x_left, x_mid, x_right = -cur_lim * 0.6, 0, cur_lim * 0.6
    ax.text(x_left, num_y, f"Only V1:\n{only_v1:,}", ha="center", va="top", fontsize=11)
    ax.text(x_mid, num_y, f"V1∩V2:\n{both:,}", ha="center", va="top", fontsize=11)
    ax.text(x_right, num_y, f"Only V2:\n{only_v2:,}", ha="center", va="top", fontsize=11)

    # Widen ylim so bottom numbers are visible (minimal padding)
    if xlim_ylim is not None:
        ax.set_ylim(-cur_lim - 0.3, cur_lim)
    ax.set_title(title, fontsize=13, pad=2)


def plot_overlap_venn(rows: list[dict], out_path: str, base: float | int = BASE) -> None:
    """1×4 row of Venn diagrams. Same axis range for all panels so circle size is comparable (radius = sqrt(count/base))."""
    base = max(float(base), 1.0)
    margin = 0.35
    global_lim = 0.0
    for r in rows:
        only_v1, only_v2, both = r["|A\\B|"], r["|B\\A|"], r["|A∩B|"]
        n_a = only_v1 + both
        n_b = only_v2 + both
        r1 = math.sqrt(n_a / base) if n_a > 0 else 0.0
        r2 = math.sqrt(n_b / base) if n_b > 0 else 0.0
        target_I = (math.pi / base) * both
        d = _solve_d_for_intersection(r1, r2, target_I) if (r1 > 0 and r2 > 0) else (r1 + r2)
        ext = max(r1, r2) + abs(d) / 2 + margin
        global_lim = max(global_lim, ext)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, r, name in zip(axes, rows, STATE_NAMES):
        draw_venn2_overlap(
            ax,
            r["|A\\B|"],
            r["|B\\A|"],
            r["|A∩B|"],
            title=name,
            base=base,
            xlim_ylim=(global_lim, global_lim),
        )

    # Legend: blue = V1, red = V2
    fig.legend(
        [Patch(facecolor="#4a90d9", alpha=0.6, edgecolor="black"), Patch(facecolor="#e74c3c", alpha=0.6, edgecolor="black")],
        ["250118 version", "250925 version"],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.985, 0.985),
        fontsize=12,
        frameon=True,
    )
    fig.suptitle("ModelId overlap", fontsize=14, y=0.98)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.08, wspace=0.12)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved overlap figure to {out_path} and .pdf")


def main() -> None:
    raw_v1 = build_raw_glob(None)
    step3_v1 = build_step3_dedup_path(None, False)
    raw_v2 = build_raw_glob(TAG_V2)
    step3_v2 = build_step3_dedup_path(TAG_V2, V2_MODE)

    print("ModelId list overlap: V1 (no tag, no v2) vs V2 (tag 251117 + v2)")
    print("Paths (same logic as hf_models_analysis, modelId lists not counts):")
    print(f"  V1 raw:   {raw_v1}")
    print(f"  V2 raw:   {raw_v2}")
    print(f"  V1 step3: {step3_v1}")
    print(f"  V2 step3: {step3_v2}")

    con = duckdb.connect()

    rows: list[dict] = []

    print("\nLoading state 1 (All Models)...")
    a1 = get_all_model_ids(con, raw_v1)
    b1 = get_all_model_ids(con, raw_v2)
    overlap_stats("1. All Models", a1, b1)
    rows.append(_overlap_numbers(a1, b1))

    print("\nLoading state 2 (Models w/ Cards)...")
    a2 = get_cards_model_ids(con, raw_v1)
    b2 = get_cards_model_ids(con, raw_v2)
    overlap_stats("2. Models w/ Cards", a2, b2)
    rows.append(_overlap_numbers(a2, b2))

    print("\nLoading state 3 (Models w/ Any Table)...")
    a3 = get_any_table_model_ids(con, step3_v1)
    b3 = get_any_table_model_ids(con, step3_v2)
    overlap_stats("3. Models w/ Any Table", a3, b3)
    rows.append(_overlap_numbers(a3, b3))

    print("\nLoading state 4 (Models w/ Hugging Tables)...")
    a4 = get_hugging_table_model_ids(con, step3_v1)
    b4 = get_hugging_table_model_ids(con, step3_v2)
    overlap_stats("4. Models w/ Hugging Tables", a4, b4)
    rows.append(_overlap_numbers(a4, b4))

    con.close()

    os.makedirs("data/analysis", exist_ok=True)
    plot_overlap_venn(rows, "data/analysis/model_snapshot_overlap.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

