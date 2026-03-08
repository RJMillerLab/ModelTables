# -*- coding: utf-8 -*-
"""Shared title dedup utils (normalize: remove -, space, .). Minimal deps, no PyPDF2."""
from src.utils import to_list_safe


def normalize(s):
    """Remove: - (hyphen), space, . (小数点) for comparison."""
    if not isinstance(s, str):
        return ""
    return str(s).replace("-", "").replace(" ", "").replace(".", "").lower().strip()


def count_symbols(s):
    return sum(1 for c in str(s) if c in "- .")


def pick_kept(titles):
    """Among titles that normalize to same form, keep the one with fewest '-', ' ', '.'."""
    if not titles:
        return None
    return min(titles, key=lambda t: (count_symbols(t), len(t), t))


def dedup_row_titles(titles):
    """Dedup titles within a row. Returns (deduped_list, groups)."""
    if titles is None or (hasattr(titles, "__len__") and len(titles) == 0):
        return [], []
    try:
        items = to_list_safe(titles)
        items = [str(x).strip() for x in items if x is not None and str(x).strip()]
    except (TypeError, ValueError):
        return [], []
    if not items:
        return [], []
    seen = {}
    for s in items:
        norm = normalize(s)
        if norm:
            seen.setdefault(norm, []).append(s)
    deduped = []
    groups = []
    for norm, originals in seen.items():
        uniq = list(dict.fromkeys(originals))
        kept = pick_kept(uniq)
        deduped.append(kept)
        if len(uniq) > 1:
            groups.append({"kept": kept, "duplicates": [x for x in uniq if x != kept]})
    return deduped, groups
