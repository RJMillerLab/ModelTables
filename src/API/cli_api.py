import subprocess
import os

def run_shell(cmd):
    print(f"[modellake] {cmd}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return out.stdout

# ===== Corresponds to download call in pipeline.py =====
def download(tag):
    tag_arg = f" --tag {tag}" if tag else ""
    cmds = [
        f"python -m src.data_preprocess.download_hf_dataset --date {tag} --type modelcard",
        f"python -m src.data_preprocess.step1_parse --raw-date {tag}",
        f"python -m src.data_preprocess.step1_down_giturl{tag_arg} --versioning --baseline-cache data/processed/github_readme_cache.parquet",
        f"python -m src.data_preprocess.ln_giturl --source-dir data/downloaded_github_readmes --target-dir data/downloaded_github_readmes_{tag}",
        f"python -m src.data_preprocess.step2_git_md2text{tag_arg}"
    ]
    for cmd in cmds:
        run_shell(cmd)

# ===== Corresponds to extract_table call in pipeline.py =====
def extract_table(tag):
    tag_arg = f" --tag {tag}" if tag else ""
    cmds = [
        f"python -m src.data_preprocess.step2_hugging_github_extract{tag_arg}",
        f"python -m src.data_preprocess.step2_arxiv_github_title{tag_arg}",
        f"python -m src.data_preprocess.step2_s2orc_save{tag_arg}",
        f"python -m src.data_preprocess.s2orc_title2ids_API{tag_arg}",
        f"python -m src.data_preprocess.s2orc_refcit_API{tag_arg}",
        f"python -m src.data_preprocess.s2orc_merge{tag_arg}",
        f"python -m src.data_preprocess.arxiv_title2ids_oai{tag_arg}",
        f"bash scripts/ln_arxiv_html.sh {tag}",
        f"python -m src.data_preprocess.arxiv_fulltext_api{tag_arg}",
        f"python -m src.data_preprocess.step2_arxiv_parse_v2{tag_arg} --n_jobs 16 --save_mode csv",
        f"python -m src.data_preprocess.step2_merge_tables_simplify{tag_arg} --v2_mode"
    ]
    for cmd in cmds:
        run_shell(cmd)

# ===== Corresponds to quality_control call in pipeline.py =====
def quality_control(tag):
    tag_arg = f" --tag {tag}" if tag else ""
    cmds = [
        f"python -m src.data_preprocess.step2_dedup_tables{tag_arg} --v2_mode",
        f"python -m src.data_analysis.qc_dedup_fig{tag_arg} --v2_mode",
        f"python -m src.data_analysis.qc_stats{tag_arg} --v2_mode",
        f"python -m src.data_analysis.qc_stats_fig{tag_arg} --v2_mode --exclude_resources llm"
    ]
    for cmd in cmds:
        run_shell(cmd)

# ===== Corresponds to extract_relatedness call in pipeline.py =====
def extract_relatedness(tag):
    tag_arg = f" --tag {tag}" if tag else ""
    cmds = [
        f"python -m src.data_gt.paper_citation_overlap{tag_arg}",
    ]
    for cmd in cmds:
        run_shell(cmd)
    # TODO: add gt scripts

# ===== Corresponds to plot_analysis call in pipeline.py =====
def plot_analysis(tag):
    tag_arg = f" --tag {tag}" if tag else ""
    cmds = [
        f"python -m src.data_analysis.table_model_counts_over_time{tag_arg} --v2_mode",
        f"python -m src.data_analysis.card_statistics{tag_arg}",
        f"python -m src.data_analysis.hf_models_analysis{tag_arg} --v2_mode",
        f"python -m src.data_analysis.model_snapshot_overlap{tag_arg}",
        f"python -m src.data_analysis.align_tables_output_versions --dir-a data/processed/tables_output --dir-b data/processed/tables_output_v2_251117",
        f"python -m src.data_analysis.compare_tables_by_content 2503.03556v1",
        f"python -m src.data_analysis.valid_table_shapes{tag_arg} --v2_mode",
        f"python -m src.data_analysis.table_usage_stats{tag_arg} --v2_mode",
    ]
    for cmd in cmds:
        run_shell(cmd)



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
    # TODO: check the scripts

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
    # TODO: check the scripts

