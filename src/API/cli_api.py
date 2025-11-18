import subprocess
import os

def run_shell(cmd):
    print(f"[modellake] {cmd}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return out.stdout

# ===== Corresponds to download call in pipeline.py =====
def download(resource, mode='scratch', dest='./data/'):
    """Corresponds to: modellake.download('modelcard/github/arxiv') in pipeline.py"""
    if resource == 'modelcard':
        cmd = "python -m src.data_preprocess.step1_parse"
    elif resource == 'github':
        cmd = "python -m src.data_preprocess.step1_down_giturl"
    elif resource == 'arxiv':
        cmd = "python -m src.data_preprocess.step2_arxiv_get_html"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

# ===== Corresponds to extract_table call in pipeline.py =====
def extract_table(resource, mode='scratch', dest='./data/'):
    """Corresponds to: modellake.extract_table('modelcard/github/arxiv') in pipeline.py"""
    if resource in ['modelcard', 'github']:
        cmd = "python -m src.data_preprocess.step2_hugging_github_extract"
    elif resource == 'arxiv':
        cmd = "python -m src.data_preprocess.step2_arxiv_parse"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

# ===== Corresponds to quality_control call in pipeline.py =====
def quality_control(mode='intra', dest='./data/'):
    """Corresponds to: modellake.quality_control('intra/inter') in pipeline.py"""
    if mode == 'intra':
        cmd = "python -m src.data_preprocess.step2_dedup_tables"
    elif mode == 'inter':
        cmd = "python -m src.data_preprocess.step2_merge_tables"
    else:
        raise ValueError("mode must be 'intra' or 'inter'")
    return run_shell(cmd)

# ===== Corresponds to extract_relatedness call in pipeline.py =====
def extract_relatedness(resource='paper'):
    """Corresponds to: modellake.extract_relatedness('paper') in pipeline.py"""
    if resource == 'paper':
        cmd = "python -m src.data_gt.paper_citation_overlap"
    elif resource in ['model', 'dataset', 'all']:
        cmd = "python -m src.data_gt.modelcard_matrix"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

# ===== Corresponds to table_search call in pipeline.py =====
def table_search(input_table, method='dense', directory='./data/'):
    """Corresponds to: modellake.table_search('tables/example.csv', method='dense') in pipeline.py"""
    if method == 'dense':
        cmd = f"bash scripts/step3_search_hnsw.sh"  # TODO: verify path
    elif method == 'sparse':
        cmd = "bash src/baseline2/sparse_search.sh"
    elif method == 'hybrid':
        cmd = "bash src/baseline2/hybrid_search.sh"
    else:
        raise ValueError("method not supported")
    return run_shell(cmd)

# ===== Corresponds to plot_analysis call in pipeline.py =====
def plot_analysis():
    """Corresponds to: modellake.plot_analysis() in pipeline.py"""
    cmds = [
        "python -m src.data_analysis.qc_stats_fig",
        "python -m src.data_analysis.gt_distri"
    ]
    for cmd in cmds:
        run_shell(cmd)
    return "Analysis plots generated."

# ===== Corresponds to repeat_experiments call in pipeline.py =====
def repeat_experiments(method='unionable', resource='modelcard', relatedness='paper'):
    """Corresponds to: modellake.repeat_experiments(method='dense', resource='modelcard', relatedness='paper') in pipeline.py"""
    if method == 'unionable':
        cmd = "bash src/baseline1/table_retrieval_pipeline.sh"
    elif method == 'dense':
        cmd = "bash scripts/step3_search_hnsw.sh"  # TODO: verify path
    elif method == 'sparse':
        cmd = "bash src/baseline2/sparse_search.sh"
    elif method == 'hybrid':
        cmd = "bash src/baseline2/hybrid_search.sh"
    else:
        cmd = f"bash scripts/step3_processmetrics.sh"  # TODO: verify path
    return run_shell(cmd)

