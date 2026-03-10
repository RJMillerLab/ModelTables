"""
TEMP script: extract (title, arxiv_id) from BibTeX columns to avoid redundant arXiv API
queries in step2_arxiv_get_html. Uses extract_titles (same as all_bibtex_titles) for title.

Pipeline:
  1) Extract cell strings from both BibTeX columns via SQL.
  2) Filter nonempty; quick filter by "arxiv" in string.
  3) Parse to entries; extract title with extract_titles([entry]), arxiv_id with extract_arxiv_id_from_bib_entry.
  4) Output: list of {title, arxiv_id}. No modelId.
"""

import os, re, json, argparse, ast, pandas as pd
from typing import Dict, List, Tuple, Set, Any
from src.utils import load_config, extract_non_empty_column_list_sql, to_parquet
from src.data_preprocess.step2_arxiv_github_title import extract_arxiv_id, extract_titles
from src.data_preprocess.step2_arxiv_get_html import normalize_title


def _parse_cell_to_entries(s: str) -> List[Any]:
    """Parse a cell string (TEXT from parquet) to a list of entries (dicts or fallback to raw str)."""
    s = (s or "").strip()
    if not s:
        return []
    # Try Python literal (single-quoted dicts)
    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return out
        if isinstance(out, dict):
            return [out]
        return []
    except (ValueError, SyntaxError):
        pass
    # Try JSON (double-quoted)
    try:
        out = json.loads(s)
        if isinstance(out, list):
            return out
        if isinstance(out, dict):
            return [out]
        return []
    except (json.JSONDecodeError, TypeError):
        pass
    return [s]


def _arxiv_id_from_string(s: str) -> str | None:
    """Extract arXiv ID from a raw string (e.g. 2403.12345)."""
    if not s:
        return None
    m = re.search(r"\b(\d{4}\.\d{5})(?:v\d+)?\b", s)
    return m.group(1) if m else None


def _title_from_string(s: str) -> str:
    """When entry is still a string (parse failed), try to extract title with regex."""
    if not s or not isinstance(s, str):
        return ""
    for pattern in (
        r'["\']title["\']\s*:\s*["\']([^"\']+)["\']',
        r'["\']title["\']\s*:\s*\{([^}]+)\}',
    ):
        m = re.search(pattern, s, re.IGNORECASE)
        if m:
            t = m.group(1).replace("{", "").replace("}", "").strip()
            if t:
                return t
    return ""


def extract_arxiv_id_from_bib_entry(entry: Any) -> str | None:
    """Extract arXiv ID from a single BibTeX entry (dict or fallback from string)."""
    if isinstance(entry, dict):
        for key in ("eprint", "arxivid", "arxiv_id", "url", "howpublished", "note"):
            val = entry.get(key)
            if isinstance(val, str) and val.strip():
                aid = extract_arxiv_id(val)
                if aid:
                    return aid
        s = str(entry)
    else:
        s = str(entry)
    return _arxiv_id_from_string(s)


def _nonempty_cells(cell_strings: List[str]) -> List[str]:
    """Filter to nonempty cells: drop empty, '[]', 'nan', whitespace-only."""
    out = []
    for s in cell_strings:
        t = (s or "").strip()
        if not t or t == "[]" or t.lower() == "nan":
            continue
        out.append(s)
    return out


def _cells_containing_arxiv(cell_strings: List[str]) -> List[str]:
    """Keep only cells whose string contains 'arxiv' (case-insensitive)."""
    return [s for s in cell_strings if "arxiv" in (s or "").lower()]


def run_pipeline(parquet_path: str, tag: str | None) -> List[Dict[str, str]]:
    """
    Run full pipeline: SQL extract → nonempty filter → arxiv filter → parse & extract.
    Title is extracted with extract_titles([entry]) (same as all_bibtex_titles). Returns title, arxiv_id.
    """
    # Step 1: Extract all non-empty cell strings from both BibTeX columns
    all_cells: List[str] = []
    for col in ("parsed_bibtex_tuple_list", "parsed_bibtex_tuple_list_github"):
        try:
            values = extract_non_empty_column_list_sql(parquet_path, col)
            all_cells.extend(values)
        except Exception as e:
            print(f"[WARN] Failed to extract column '{col}': {e}")

    n_raw = len(all_cells)
    # Step 2: Nonempty filter (drop [], nan, empty)
    nonempty = _nonempty_cells(all_cells)
    n_nonempty = len(nonempty)
    # Step 3: Quick filter by "arxiv" in string
    arxiv_candidates = _cells_containing_arxiv(nonempty)
    n_arxiv_candidates = len(arxiv_candidates)

    print(f"[STATS] Step 1 – cells from SQL (both cols): {n_raw}")
    print(f"[STATS] Step 2 – after nonempty filter (drop [], nan, empty): {n_nonempty}")
    print(f"[STATS] Step 3 – after 'arxiv' in string filter: {n_arxiv_candidates}")

    # Step 4: Parse each candidate; extract title via extract_titles([entry]), arxiv_id via extract_arxiv_id_from_bib_entry
    seen: Set[Tuple[str, str]] = set()
    rows: List[Dict[str, str]] = []

    for cell_str in arxiv_candidates:
        entries = _parse_cell_to_entries(cell_str)
        for entry in entries:
            arxiv_id = extract_arxiv_id_from_bib_entry(entry)
            if not arxiv_id:
                continue
            # Title: extract_titles(entry) when entry is dict; when entry is str (parse failed), regex fallback
            if isinstance(entry, dict):
                titles = extract_titles(entry)
                raw_title = (titles[0] if titles else "").strip()
            else:
                raw_title = _title_from_string(str(entry)).strip()
            retrieved_title = normalize_title(raw_title) if raw_title else ""
            key = (retrieved_title or "", arxiv_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "title": retrieved_title,
                "arxiv_id": arxiv_id,
            })

    print(f"[STATS] Step 4 – unique (title, arxiv_id) pairs extracted: {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TEMP: Extract (title, arxiv_id) from BibTeX using extract_titles (same as all_bibtex_titles).",
    )
    parser.add_argument("--tag", default=None, help="Tag suffix (e.g. 251117).")
    args = parser.parse_args()
    tag = args.tag
    suffix = f"_{tag}" if tag else ""

    config = load_config("config.yaml")
    base_path = config.get("base_path", "data")
    processed_base_path = os.path.join(base_path, "processed")
    parquet_path = os.path.join(processed_base_path, f"modelcard_all_title_list{suffix}.parquet")

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    print(f"📁 Reading BibTeX columns via SQL: {parquet_path}")

    rows = run_pipeline(parquet_path, tag)

    if not rows:
        print("[INFO] No (title, arxiv_id) pairs extracted; nothing more to analyze.")
        return

    # 1) Filter: keep only rows where both title and arxiv_id are non-empty
    df_bib = pd.DataFrame(rows)[["title", "arxiv_id"]]
    df_bib = df_bib[df_bib["title"].astype(str).str.strip().ne("") & df_bib["arxiv_id"].astype(str).str.strip().ne("")].drop_duplicates()
    print(f"[STATS] After filter (title and arxiv_id non-empty): {len(df_bib)} rows")

    # 2) Save BibTeX title->arxiv_id as parquet
    bibtex_parquet_path = os.path.join(processed_base_path, f"bibtex_title_arxiv{suffix}.parquet")
    to_parquet(df_bib, bibtex_parquet_path)
    print(f"[INFO] Saved BibTeX (title, arxiv_id) to {bibtex_parquet_path}")

    # 3) Load s2orc_titles2ids: only query_title and retrieved_title; join on title -> final bibtex parquet
    s2orc_path = os.path.join(processed_base_path, f"s2orc_titles2ids{suffix}.parquet")
    if not os.path.isfile(s2orc_path):
        print(f"[WARN] s2orc_titles2ids not found at {s2orc_path}; skipping final merge.")
        return

    df_s2orc = pd.read_parquet(s2orc_path, columns=["query_title", "retrieved_title"])
    df_s2orc["norm_title"] = df_s2orc["retrieved_title"].astype(str).apply(normalize_title)

    final = df_bib.merge(df_s2orc, left_on="title", right_on="norm_title", how="inner").drop(columns=["norm_title"])
    final = final[["title", "arxiv_id", "query_title", "retrieved_title"]].drop_duplicates()
    final_parquet_path = os.path.join(processed_base_path, f"bibtex_title_arxiv_s2orc{suffix}.parquet")
    to_parquet(final, final_parquet_path)
    print(f"[INFO] Final (title, arxiv_id, query_title, retrieved_title) saved to {final_parquet_path} (deduplicated)")

if __name__ == "__main__":
    main()
