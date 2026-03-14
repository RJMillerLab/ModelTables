#!/usr/bin/env python3
import os
import argparse
from joblib import Parallel, delayed
MODE_CONFIG={"":("",".csv"),"_str":("_str","_s.csv"),"_tr":("_tr","_t.csv")}
def build_src(base_line,repo_root,dir_suffix,file_suffix):
    basename=os.path.basename(base_line)[:-4]
    src_dir=base_line.replace("deduped_hugging_csvs_v2_251117",f"deduped_hugging_csvs_v2_251117{dir_suffix}")
    src_dir=os.path.dirname(src_dir)
    src=os.path.join(repo_root,"ModelTables",src_dir,basename+file_suffix)
    return src
def link_one(src,target_dir):
    basename=os.path.basename(src)
    if basename.endswith("_s.csv"):basename=basename[:-6]+".csv"
    if basename.endswith("_t.csv"):basename=basename[:-6]+".csv"
    target=os.path.join(target_dir,basename)
    try:
        os.symlink(src,target)
        return 1
    except FileExistsError:
        try:
            os.remove(target)
            os.symlink(src,target)
            return 1
        except:return 0
    except:return 0
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo_root",default="/u1/z6dong/Repo")
    parser.add_argument("--tag",default="251117")
    parser.add_argument("--v2_mode",action="store_true")
    parser.add_argument("--n_jobs",type=int,default=32)
    args=parser.parse_args()
    v2_suffix="_v2" if args.v2_mode else ""
    suffix=f"_{args.tag}" if args.tag else ""
    mask_file=os.path.join(args.repo_root,"ModelTables","data","analysis",f"all_valid_title_valid{v2_suffix}{suffix}.txt")
    base_lines=[]
    with open(mask_file) as f:
        for line in f:
            line=line.strip()
            if line.endswith(".csv"):base_lines.append(line)
    print("mask entries:",len(base_lines))
    for mode,(dir_suffix,file_suffix) in MODE_CONFIG.items():
        target_dir=os.path.join(args.repo_root,"starmie_internal","data",f"scilake_final{v2_suffix}{suffix}{mode}","datalake")
        os.makedirs(target_dir,exist_ok=True)
        print(f"\n=== mode {mode or 'base'} ===")
        src_paths=[build_src(line,args.repo_root,dir_suffix,file_suffix) for line in base_lines]
        results=Parallel(n_jobs=args.n_jobs,backend="threading")(delayed(link_one)(src,target_dir) for src in src_paths)
        linked=sum(results)
        print(f"linked={linked}, failed={len(results)-linked}, total={len(results)}")
if __name__=="__main__":
    main()
