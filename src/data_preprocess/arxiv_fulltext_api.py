#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download arXiv HTML (via ar5iv) for arxiv_ids in title2arxiv_cache_<tag>.parquet.
Run arxiv_title2ids_oai first (which merges s2orc + arxiv_titles_cache + bibtex + OAI).
"""

import os
import re
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

from tqdm import tqdm

from src.utils import load_config

def fetch_ar5iv_html(arxiv_id: str, html_folder: str) -> str:
    """
    Fetch HTML from ar5iv. Save to html_folder/{arxiv_id}.html if 200 and real fulltext.
    Ar5iv redirects to arxiv.org/abs/ for papers it hasn't converted; we reject those.
    Writes .redirect marker to skip retries. Returns status in ("success", "exists", "404", "429", "500", "redirect", "error").
    """
    file_path = os.path.join(html_folder, f"{arxiv_id}.html")
    if os.path.exists(file_path):
        return "exists"
    redirect_marker = os.path.join(html_folder, f"{arxiv_id}.redirect")
    if os.path.exists(redirect_marker):
        return "redirect"

    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            # Ar5iv redirects to arxiv.org/abs/ for unconverted papers - reject, don't save abstract page
            if "ar5iv" not in resp.url:
                open(redirect_marker, "w").close()
                return "redirect"
            tmp_path = os.path.join(html_folder, f"{arxiv_id}.html.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            os.rename(tmp_path, file_path)
            return "success"
        elif resp.status_code == 404:
            return "404"
        elif resp.status_code == 429:
            return "429"
        elif resp.status_code >= 500:
            return "500"
        else:
            # try fallback: base id without version
            base_id = re.sub(r"v\d+$", "", arxiv_id)
            if base_id != arxiv_id:
                base_path = os.path.join(html_folder, f"{base_id}.html")
                if os.path.exists(base_path):
                    return "exists"
                base_redirect = os.path.join(html_folder, f"{base_id}.redirect")
                if os.path.exists(base_redirect):
                    return "redirect"
                base_url = f"https://ar5iv.labs.arxiv.org/html/{base_id}"
                try:
                    br = requests.get(base_url, timeout=15)
                    if br.status_code == 200:
                        if "ar5iv" not in br.url:
                            open(base_redirect, "w").close()
                            return "redirect"
                        tmp_path = os.path.join(html_folder, f"{base_id}.html.tmp")
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            f.write(br.text)
                        os.rename(tmp_path, base_path)
                        return "success"
                    elif br.status_code == 404:
                        return "404"
                    elif br.status_code == 429:
                        return "429"
                    elif br.status_code >= 500:
                        return "500"
                except Exception:
                    return "error"
            return "error"
    except requests.exceptions.Timeout:
        return "error"
    except Exception as e:
        print(f"[ERROR] ar5iv fetch {arxiv_id}: {e}", flush=True)
        return "error"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv HTML via ar5iv for titles that already have arxiv_id in title2arxiv_cache_<tag>.parquet")
    parser.add_argument("--tag", dest="tag", default=None, help="Tag suffix for versioning (e.g., 251117).")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2).")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests per worker (default: 1.5).")
    args = parser.parse_args()

    config = load_config("config.yaml")
    base_path = config.get("base_path", "data")
    processed_base_path = os.path.join(base_path, "processed")
    tag = args.tag
    suffix = f"_{tag}" if tag else ""

    cache_parquet_path = os.path.join(processed_base_path, f"title2arxiv_cache{suffix}.parquet")
    html_folder = os.path.join(base_path, f"arxiv_fulltext_html{suffix}")
    os.makedirs(html_folder, exist_ok=True)
    for f in os.listdir(html_folder):
        if f.endswith(".html.tmp"):
            try:
                os.remove(os.path.join(html_folder, f))
            except OSError:
                pass

    print(f"[INFO] Cache: {cache_parquet_path}", flush=True)
    print(f"[INFO] HTML folder: {html_folder}", flush=True)

    if not os.path.isfile(cache_parquet_path):
        raise FileNotFoundError(
            f"title2arxiv_cache not found at {cache_parquet_path}. Run arxiv_title2ids_oai.py first."
        )
    # Only download for rows where html_path is empty (no existing HTML): this only work when each time we update cache right after running this script
    df = pd.read_parquet(cache_parquet_path, columns=["arxiv_id", "html_path"])
    has_id = df["arxiv_id"].notna() & (df["arxiv_id"].astype(str).str.strip() != "")
    no_html = df["html_path"].isna() | (df["html_path"].astype(str).str.strip() == "")
    all_ids = df.loc[has_id & no_html, "arxiv_id"].astype(str).str.strip().unique()
    all_ids = sorted([x for x in all_ids if x and x.lower() not in ("nan", "none")])
    workers = max(1, min(args.workers, 8))
    delay = max(0.0, args.delay)
    print(f"[INFO] arxiv_id without html_path to download: {len(all_ids)} (workers={workers}, delay={delay}s)", flush=True)

    stats = {"exists": 0, "success": 0, "404": 0, "429": 0, "500": 0, "redirect": 0, "error": 0}

    def _fetch(aid: str) -> str:
        time.sleep(delay)
        return fetch_ar5iv_html(aid, html_folder)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, aid): aid for aid in all_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="ar5iv HTML", unit="id"):
            status = future.result()
            stats[status] = stats.get(status, 0) + 1

    print(
        f"[DONE] exists={stats['exists']}, success={stats['success']}, 404={stats['404']}, "
        f"429={stats['429']}, 5xx={stats['500']}, redirect={stats['redirect']}, error={stats['error']}",
        flush=True,
    )
    print("[INFO] html_path: run arxiv_title2ids_oai (sync) or scan folder when needed.", flush=True)
    print("Run `python -m src.data_preprocess.arxiv_title2ids_oai --tag <tag>` to sync html_path from folder to cache.", flush=True)

if __name__ == "__main__":
    main()
