# -*- coding: utf-8 -*-
"""
One-time script: Load modelcard_all_title_list_251117.parquet, deduplicate all_title_list
within each row (normalize: remove -, space, .). Keep the title with fewest such symbols.
Save parquet back + group mapping JSON for later s2orc_titles2ids processing.
"""
import os
import json
import pandas as pd
from src.utils import load_config, to_parquet, to_list_safe
from src.data_preprocess.title_dedup_utils import dedup_row_titles


def main():
    config = load_config("config.yaml")
    base_path = config.get("base_path", "data")
    processed = os.path.join(base_path, "processed")
    parquet_path = os.path.join(processed, "modelcard_all_title_list_251117.parquet")
    groups_path = os.path.join(processed, "all_title_list_intra_row_dedup_groups_251117.json")

    print("Loading modelcard_all_title_list_251117.parquet...")
    df = pd.read_parquet(parquet_path)

    total_before = 0
    total_after = 0
    all_groups = []  # [{kept, duplicates}, ...]

    new_lists = []
    for idx, row in df.iterrows():
        titles = row.get("all_title_list")
        n_before = 0
        if titles is not None:
            lst = to_list_safe(titles)
            lst = [x for x in lst if x is not None and str(x).strip()]
            n_before = len(lst)
        total_before += n_before

        deduped, row_groups = dedup_row_titles(titles)
        total_after += len(deduped)
        all_groups.extend(row_groups)
        new_lists.append(deduped if deduped else [])

    df["all_title_list"] = new_lists

    removed = total_before - total_after
    print(f"\n=== Intra-row dedup stats ===")
    print(f"  Before dedup: {total_before} title items across {len(df)} rows")
    print(f"  After dedup:  {total_after} title items")
    print(f"  Removed:      {removed} repeated items")
    print(f"  Groups (kept->duplicates): {len(all_groups)}")

    # Build mapping for s2orc: duplicate -> kept
    dup_to_kept = {}
    for g in all_groups:
        k = g["kept"]
        for d in g["duplicates"]:
            dup_to_kept[d] = k

    groups_output = {
        "groups": all_groups,
        "duplicate_to_kept": dup_to_kept,
        "stats": {"before": total_before, "after": total_after, "removed": removed, "num_groups": len(all_groups)},
    }
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(groups_output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved group mapping to {groups_path}")

    to_parquet(df, parquet_path)
    print(f"Overwritten: {parquet_path}")


if __name__ == "__main__":
    main()
