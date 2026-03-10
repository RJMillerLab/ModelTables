#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download arXiv HTML (via ar5iv) for titles that already have arxiv_id in `title2arxiv_cache_<tag>.parquet`.
"""

import os
import re
import argparse
import time
import requests
import pandas as pd

from src.utils import load_config


def fetch_ar5iv_html(arxiv_id: str, html_folder: str) -> str | None:
    """
    Fetch HTML from ar5iv (ar5iv.labs.arxiv.org) given an arXiv ID.
    Save the HTML to a local file in folder html_folder with filename '{arxiv_id}.html'
    and return the file path, or None on failure.
    """
    file_path = os.path.join(html_folder, f"{arxiv_id}.html")
    if os.path.exists(file_path):
        return file_path

    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            html_text = resp.text
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_text)
            return file_path
        elif resp.status_code == 404:
            print(f"[WARN] HTML not exist: {arxiv_id}")
            return None
        else:
            print(f"[WARN] ar5iv HTML not found for {arxiv_id}, status={resp.status_code}")
            # Fallback: try base arXiv ID without version suffix (e.g. 2101.12345v3 -> 2101.12345)
            base_arxiv_id = re.sub(r"v\\d+$", "", arxiv_id)
            if base_arxiv_id != arxiv_id:
                print(f"[INFO] Fallback: trying base arXiv ID '{base_arxiv_id}' for {arxiv_id}.")
                base_file_path = os.path.join(html_folder, f"{base_arxiv_id}.html")
                if os.path.exists(base_file_path):
                    return base_file_path
                base_url = f"https://ar5iv.labs.arxiv.org/html/{base_arxiv_id}"
                try:
                    base_resp = requests.get(base_url, timeout=15)
                    if base_resp.status_code == 200:
                        html_text = base_resp.text
                        with open(base_file_path, "w", encoding="utf-8") as f:
                            f.write(html_text)
                        return base_file_path
                    else:
                        print(
                            f"[WARN] Fallback ar5iv HTML not found for {base_arxiv_id}, "
                            f"status={base_resp.status_code}"
                        )
                        return None
                except Exception as ex:
                    print(f"[ERROR] Fallback ar5iv HTML fetch error for {base_arxiv_id}: {ex}")
                    return None
            return None
    except Exception as e:
        print(f"[ERROR] ar5iv HTML fetch error for {arxiv_id}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv HTML via ar5iv for titles that already have arxiv_id in title2arxiv_cache_<tag>.parquet")
    parser.add_argument("--tag", dest="tag", default=None, help="Tag suffix for versioning (e.g., 251117).")
    args = parser.parse_args()

    config = load_config("config.yaml")
    base_path = config.get("base_path", "data")
    processed_base_path = os.path.join(base_path, "processed")
    tag = args.tag
    suffix = f"_{tag}" if tag else ""

    cache_parquet_path = os.path.join(processed_base_path, f"title2arxiv_cache{suffix}.parquet")
    html_folder = os.path.join(base_path, f"arxiv_fulltext_html{suffix}")
    os.makedirs(html_folder, exist_ok=True)

    print(f"[INFO] Cache parquet: {cache_parquet_path}")
    print(f"[INFO] HTML folder: {html_folder}")

    if not os.path.isfile(cache_parquet_path):
        raise FileNotFoundError(f"title2arxiv_cache parquet not found at {cache_parquet_path}. Run arxiv_title2ids_oai.py first.")

    df_cache = pd.read_parquet(cache_parquet_path)
    print(f"[INFO] Loaded cache with {len(df_cache)} rows from {cache_parquet_path}")

    # Ensure expected columns exist
    for col in ["title", "norm_title", "arxiv_id", "html_path", "query_status"]:
        if col not in df_cache.columns:
            if col == "query_status":
                df_cache[col] = "unknown"
            else:
                df_cache[col] = ""

    # Build in‑memory html_cache (arxiv_id -> html_path)
    html_cache: dict[str, str] = {}
    for _, row in df_cache.iterrows():
        aid = str(row.get("arxiv_id") or "").strip()
        path = row.get("html_path") or ""
        if aid:
            html_cache[aid] = path

    all_ids = [str(a).strip() for a in df_cache["arxiv_id"].dropna() if str(a).strip()]
    unique_ids = sorted(set(all_ids))
    print(f"[INFO] Found {len(unique_ids)} unique arxiv_id values in cache.")

    n_already = 0
    n_downloaded = 0
    n_failed = 0

    for idx, aid in enumerate(unique_ids, start=1):
        cached_path = html_cache.get(aid, "")
        if cached_path and os.path.isfile(cached_path):
            n_already += 1
            continue

        html_file_path = fetch_ar5iv_html(aid, html_folder)
        if html_file_path:
            html_cache[aid] = html_file_path
            n_downloaded += 1
        else:
            html_cache[aid] = ""
            n_failed += 1

        if idx % 100 == 0:
            print(
                f"[STATS] Processed {idx}/{len(unique_ids)} IDs "
                f"(existing={n_already}, downloaded={n_downloaded}, failed={n_failed})"
            )
            # Be a bit polite to ar5iv, short sleep
            time.sleep(1.0)

    print(
        f"[INFO] Finished HTML pass over {len(unique_ids)} IDs: "
        f"{n_already} already had files, {n_downloaded} downloaded, {n_failed} failed."
    )

    # Update html_path column in cache from html_cache
    df_cache["html_path"] = df_cache["arxiv_id"].map(
        lambda aid: html_cache.get(str(aid).strip(), "") if str(aid).strip() else ""
    )

    df_cache.to_parquet(cache_parquet_path, index=False)
    print(f"[INFO] Updated cache written back to {cache_parquet_path}")


if __name__ == "__main__":
    main()

