#!/usr/bin/env python3
import os
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
MODE_CONFIG = {
    "": ("", ".csv"),
    "_str": ("_str", "_s.csv"),
    "_tr": ("_tr", "_t.csv"),
}


def build_src(base_line, repo_root, dir_suffix, file_suffix, suffix, v2_suffix):
    """
    base_line: a line from the mask file, containing the *base* CSV path
               (e.g. data/processed/deduped_hugging_csvs_v2_251117/xxx.csv)
    For each mode, we:
      - replace 'deduped_hugging_csvs_v2_251117' -> 'deduped_hugging_csvs_v2_251117{dir_suffix}'
      - replace '.csv' -> '{file_suffix}' (e.g. '_s.csv' / '_t.csv')
    We DO NOT check existence; we assume the corresponding file exists.
    """
    # Build mode-specific source path purely by string replace
    src_rel = base_line.replace(
        f"{v2_suffix}{suffix}",
        f"{v2_suffix}{suffix}{dir_suffix}",
    ).replace(".csv", file_suffix)
    src = os.path.join(repo_root, "ModelTables", src_rel)
    return src


def link_one(src, target_dir, base_basename):
    """
    Always remove then re-link, without checking whether src exists.
    Target filename is exactly the basename from the mask file (base_basename).
    """
    target = os.path.join(target_dir, base_basename)
    try:
        try:
            os.remove(target)
        except FileNotFoundError:
            pass
        os.symlink(src, target)
        return 1
    except Exception:
        return 0
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo_root",default="/u1/z6dong/Repo")
    parser.add_argument("--tag",default="251117")
    parser.add_argument("--v2_mode",action="store_true")
    parser.add_argument("--n_jobs",type=int,default=32)
    args=parser.parse_args()
    v2_suffix="_v2" if args.v2_mode else ""
    suffix=f"_{args.tag}" if args.tag else ""
    mask_file = os.path.join(args.repo_root, "ModelTables", "data", "analysis", f"all_valid_title_valid{v2_suffix}{suffix}.txt")
    base_lines = []
    with open(mask_file) as f:
        for line in f:
            line = line.strip()
            if not line.endswith(".csv"):
                continue
            base_lines.append(line)
    print("mask entries:", len(base_lines))
    for mode, (dir_suffix, file_suffix) in MODE_CONFIG.items():
        target_dir = os.path.join(args.repo_root, "starmie_internal", "data", f"scilake_final{v2_suffix}{suffix}{mode}", "datalake")
        os.makedirs(target_dir, exist_ok=True)
        print(f"\n=== mode {mode or 'base'} ===")
        src_paths = [build_src(line, args.repo_root, dir_suffix, file_suffix, suffix, v2_suffix) for line in base_lines]
        base_basenames = [os.path.basename(line).replace(".csv", file_suffix) for line in base_lines]
        jobs = (delayed(link_one)(src, target_dir, base_basename) for src, base_basename in zip(src_paths, base_basenames))
        results = Parallel(n_jobs=args.n_jobs, backend="threading")(tqdm(jobs, total=len(src_paths), desc=f"linking ({mode or 'base'})", unit="file"))
        linked = sum(results)
        print(f"linked={linked}, failed={len(results)-linked}, total={len(results)}\n")
if __name__=="__main__":
    main()
