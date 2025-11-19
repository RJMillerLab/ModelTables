import subprocess
import os

def run_shell(cmd):
    print(f"[modellake] {cmd}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return out.stdout

# ===== Corresponds to download call in pipeline.py =====
def download(resource, mode='scratch', dest='./data/', tag=None):
    """Corresponds to: modellake.download('modelcard/github/arxiv') in pipeline.py
    
    Args:
        resource: 'modelcard', 'github', or 'arxiv'
        mode: 'scratch' (unused, kept for compatibility)
        dest: destination directory (unused, kept for compatibility)
        tag: Tag suffix for versioning (e.g., '251117'). Required for most scripts.
    """
    tag_arg = f" --tag {tag}" if tag else ""
    if resource == 'modelcard':
        # step1_parse uses --raw-date instead of --tag
        if tag:
            cmd = f"python -m src.data_preprocess.step1_parse --raw-date {tag}"
        else:
            cmd = "python -m src.data_preprocess.step1_parse"
    elif resource == 'github':
        cmd = f"python -m src.data_preprocess.step1_down_giturl{tag_arg}"
    elif resource == 'arxiv':
        cmd = f"python -m src.data_preprocess.step2_arxiv_get_html{tag_arg}"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

# ===== Corresponds to extract_table call in pipeline.py =====
def extract_table(resource, mode='scratch', dest='./data/', tag=None):
    """Corresponds to: modellake.extract_table('modelcard/github/arxiv') in pipeline.py
    
    Args:
        resource: 'modelcard', 'github', or 'arxiv'
        mode: 'scratch' (unused, kept for compatibility)
        dest: destination directory (unused, kept for compatibility)
        tag: Tag suffix for versioning (e.g., '251117'). Required for most scripts.
    """
    tag_arg = f" --tag {tag}" if tag else ""
    if resource in ['modelcard', 'github']:
        cmd = f"python -m src.data_preprocess.step2_hugging_github_extract{tag_arg}"
    elif resource == 'arxiv':
        cmd = f"python -m src.data_preprocess.step2_arxiv_parse{tag_arg}"
    else:
        raise ValueError("resource not supported")
    return run_shell(cmd)

# ===== Corresponds to quality_control call in pipeline.py =====
def quality_control(mode='intra', dest='./data/', tag=None):
    """Corresponds to: modellake.quality_control('intra/inter') in pipeline.py
    
    Args:
        mode: 'intra' (dedup) or 'inter' (merge)
        dest: destination directory (unused, kept for compatibility)
        tag: Tag suffix for versioning (e.g., '251117'). Required for most scripts.
    """
    tag_arg = f" --tag {tag}" if tag else ""
    if mode == 'intra':
        cmd = f"python -m src.data_preprocess.step2_dedup_tables{tag_arg}"
    elif mode == 'inter':
        cmd = f"python -m src.data_preprocess.step2_merge_tables{tag_arg}"
    else:
        raise ValueError("mode must be 'intra' or 'inter'")
    return run_shell(cmd)

# ===== Corresponds to extract_relatedness call in pipeline.py =====
def extract_relatedness(resource='paper', tag=None):
    """Corresponds to: modellake.extract_relatedness('paper') in pipeline.py
    
    Args:
        resource: 'paper', 'model', 'dataset', or 'all'
        tag: Tag suffix for versioning (e.g., '251117'). Required for most scripts.
    """
    tag_arg = f" --tag {tag}" if tag else ""
    if resource == 'paper':
        cmd = f"python -m src.data_gt.paper_citation_overlap{tag_arg}"
    elif resource in ['model', 'dataset', 'all']:
        cmd = f"python -m src.data_gt.modelcard_matrix{tag_arg}"
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
def plot_analysis(tag=None):
    """Corresponds to: modellake.plot_analysis() in pipeline.py
    
    Args:
        tag: Tag suffix for versioning (e.g., '251117'). Required for most scripts.
    """
    tag_arg = f" --tag {tag}" if tag else ""
    cmds = [
        f"python -m src.data_analysis.qc_stats_fig{tag_arg}",
        f"python -m src.data_analysis.gt_distri{tag_arg}"
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

