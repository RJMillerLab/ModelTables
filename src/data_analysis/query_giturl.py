r"""
Author: Zhengyuan Dong
Created: 2025-02-24
Last Modified: 2025-03-11
Description: Look up original GitHub URL by local README path (from github_readme_cache).
Usage:
    python -m src.data_analysis.query_giturl "data/downloaded_github_readmes/xxx.md"
    python -m src.data_analysis.query_giturl --cache data/processed/github_readme_cache_251117.parquet "data/downloaded_github_readmes_251117/xxx.md"
"""

import sys
import duckdb

def query_direct(query_value: str, cache_path: str = "data/processed/github_readme_cache.parquet") -> None:
    """Look up raw_url by downloaded_path. Use cache_path for tagged runs (e.g. github_readme_cache_251117.parquet)."""
    import os
    cache_path = os.path.abspath(os.path.expanduser(cache_path))
    if not os.path.isfile(cache_path):
        print(f"Cache not found: {cache_path}")
        return
    result = duckdb.execute(
        "SELECT raw_url FROM read_parquet(?) WHERE downloaded_path = ? LIMIT 1",
        [cache_path, query_value],
    ).fetchone()
    if result:
        print(result[0])
    else:
        print("Not found")


if __name__ == "__main__":
    path = None
    cache_path = "data/processed/github_readme_cache.parquet"
    argv = sys.argv[1:]
    if "--cache" in argv:
        i = argv.index("--cache")
        if i + 1 < len(argv):
            cache_path = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    if "--query" in argv:
        i = argv.index("--query")
        if i + 1 < len(argv):
            path = argv[i + 1]
    elif len(argv) == 1:
        path = argv[0]
    if not path:
        print("Usage: python -m src.data_analysis.query_giturl [--cache <cache.parquet>] [load --query] <path>")
        sys.exit(1)
    query_direct(path, cache_path)
