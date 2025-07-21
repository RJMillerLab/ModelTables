import subprocess
import os

def run_shell(cmd):
    print(f"[modellake] {cmd}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return out.stdout

def download(resource, mode='scratch', dest='./data/'):
    if resource == 'modelcard':
        cmd = "python -m src.data_preprocess.step1"
    elif resource == 'github':
        cmd = "python -m src.data_preprocess.step1_down_giturl"
    elif resource == 'arxiv':
        cmd = "python -m src.data_preprocess.step2_get_html"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

def extract_table(resource, mode='scratch', dest='./data/'):
    if resource in ['modelcard', 'github']:
        cmd = "python -m src.data_preprocess.step2_gitcard_tab"
    elif resource == 'arxiv':
        cmd = "python -m src.data_preprocess.step2_html_parsing"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

def quality_control(mode='intra', dest='./data/'):
    if mode == 'intra':
        cmd = "python -m src.data_analysis.qc_dedup"
    elif mode == 'inter':
        cmd = "python -m src.data_gt.step3_pre_merge"
    else:
        raise ValueError("mode must be 'intra' or 'inter'")
    return run_shell(cmd)

def extract_relatedness(resource='paper'):
    if resource == 'paper':
        cmd = "python -m src.data_gt.overlap_rate"
    elif resource in ['model', 'dataset', 'all']:
        cmd = "python -m src.data_gt.modelcard_matrix"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

def table_search(input_table, method='dense', directory='./data/'):
    # method: dense/sparse/hybrid 可扩展
    if method == 'dense':
        cmd = f"bash scripts/step3_search_hnsw.sh"  # 实际你可根据具体实现调整输入参数
    elif method == 'sparse':
        cmd = "bash src/baseline2/sparse_search.sh"
    elif method == 'hybrid':
        cmd = "bash src/baseline2/hybrid_search.sh"
    else:
        raise ValueError("method not supported")
    return run_shell(cmd)

def plot_analysis():
    cmds = [
        "python -m src.data_analysis.qc_stats_fig",
        "python -m src.data_analysis.gt_distri"
    ]
    for cmd in cmds:
        run_shell(cmd)
    return "Analysis plots generated."

def repeat_experiments(method='unionable', resource='modelcard', relatedness='paper'):
    # method/relatedness/resource 控制脚本
    if method == 'unionable':
        cmd = "bash src/baseline1/table_retrieval_pipeline.sh"
    elif method == 'dense':
        cmd = "bash scripts/step3_search_hnsw.sh"
    elif method == 'sparse':
        cmd = "bash src/baseline2/sparse_search.sh"
    elif method == 'hybrid':
        cmd = "bash src/baseline2/hybrid_search.sh"
    else:
        cmd = f"bash scripts/step3_processmetrics.sh"
    return run_shell(cmd)

