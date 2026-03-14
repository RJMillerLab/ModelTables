"""
Symlink all .md files from source_dir into target_dir.
If target is a symlink: force overwrite (replace). If target is a real file: skip.

Usage:
  python -m src.data_preprocess.ln_giturl --source-dir data/downloaded_github_readmes --target-dir data/downloaded_github_readmes_251117
"""
from __future__ import annotations

import argparse
import os


def main(source_dir: str, target_dir: str) -> None:
    source_dir = os.path.abspath(os.path.expanduser(source_dir))
    target_dir = os.path.abspath(os.path.expanduser(target_dir))

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    os.makedirs(target_dir, exist_ok=True)

    linked = 0
    skipped = 0
    for name in os.listdir(source_dir):
        if not name.endswith(".md"):
            continue
        src_path = os.path.join(source_dir, name)
        if not os.path.isfile(src_path) and not os.path.islink(src_path):
            continue
        tgt_path = os.path.join(target_dir, name)
        if os.path.lexists(tgt_path) and not os.path.islink(tgt_path):
            skipped += 1
            continue
        try:
            if os.path.islink(tgt_path):
                os.remove(tgt_path)
            os.symlink(src_path, tgt_path)
            linked += 1
        except OSError as e:
            print(f"Error linking {name}: {e}")

    print(f"Symlinks created/updated: {linked:,}")
    print(f"Skipped (real file in target): {skipped:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Symlink .md from source_dir into target_dir; overwrite symlinks only, skip real files."
    )
    parser.add_argument("--source-dir", required=True, help="Directory to link from (e.g. data/downloaded_github_readmes)")
    parser.add_argument("--target-dir", required=True, help="Directory to link into (e.g. data/downloaded_github_readmes_251117)")
    args = parser.parse_args()
    main(args.source_dir, args.target_dir)
