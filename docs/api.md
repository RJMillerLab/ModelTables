
# Modellake: Unified Data Lake Table Benchmark Pipeline

This module provides a unified Python API for all core data lake table benchmark workflows. Each function directly maps to one or more underlying shell scripts or Python modules, offering simple, robust orchestration.

## API Overview & Command Mapping

| API Command                       | Supported Resource / Mode                                | Underlying Script(s) / Command(s)                                                                                                                                                    | Key Input/Output                        | Notes                                          |
| --------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- | ---------------------------------------------- |
| `download(resource, mode)`        | `modelcard` / `github` / `arxiv` <br> `scratch`/`update` | - `python -m src.data_preprocess.step1`<br>- `python -m src.data_preprocess.step1_down_giturl`<br>- `python -m src.data_preprocess.step2_get_html`                                   | Local cache and folders                 | Downloads modelcard/github/arxiv artifacts     |
| `extract_table(resource, mode)`   | Same as above                                            | - `python -m src.data_preprocess.step2_gitcard_tab`<br>- `python -m src.data_preprocess.step2_html_parsing`                                                                          | Table CSV folders, mapping parquet/json | Extracts/cleans tables to local CSVs           |
| `quality_control(mode)`           | `intra` / `inter`                                        | - `python -m src.data_analysis.qc_dedup`<br>- `python -m src.data_gt.step3_pre_merge`                                                                                                | Cleaned table folders, stats/figures    | Intra: within source; Inter: cross-source      |
| `extract_relatedness(resource)`   | `paper`/`model`/`dataset`/`all`                          | - `python -m src.data_gt.overlap_rate`<br>- `python -m src.data_gt.modelcard_matrix`                                                                                                 | Binary matrix (npz/pkl), CSV lists      | Extracts unionability/relatedness ground truth |
| `table_search(input, method)`     | Table path; `dense`/`sparse`/`hybrid`                    | - `bash scripts/step3_search_hnsw.sh`<br>- `bash src/baseline2/sparse_search.sh`<br>- `bash src/baseline2/hybrid_search.sh`                                                          | Retrieved table paths                   | Runs table retrieval experiments               |
| `plot_analysis()`                 | --                                                       | - `python -m src.data_analysis.qc_stats_fig`<br>- `python -m src.data_analysis.gt_distri`                                                                                            | Benchmark figures, stats                | Plots statistics/benchmark figures             |
| `repeat_experiments(method, ...)` | See below; method: `unionable`/`joinable`/`keyword`/...  | - `bash src/baseline1/table_retrieval_pipeline.sh`<br>- `bash src/baseline2/sparse_search.sh`<br>- `bash src/baseline2/hybrid_search.sh`<br>- `bash scripts/step3_processmetrics.sh` | Performance tables, metrics             | Repeat experiments across methods/resources    |

**Legend:**

* "resource" is one of: `modelcard`, `github`, `arxiv`
* "mode" is `scratch` (fresh/full) or `update` (incremental)
* "method" in `repeat_experiments` can be `unionable`, `joinable`, `keyword`, `dense`, `sparse`, `hybrid`

---

## Python API Usage

### 1. Install requirements and import

```python
import modellake.cli_api as modellake
```

### 2. Typical Workflow Example

```python
# Download artifacts from all sources
modellake.download('modelcard')
modellake.download('github')
modellake.download('arxiv')

# Extract and standardize tables
modellake.extract_table('modelcard')
modellake.extract_table('github')
modellake.extract_table('arxiv')

# Quality control (deduplication/integration)
modellake.quality_control('intra')
modellake.quality_control('inter')

# Compute relatedness/ground truth
modellake.extract_relatedness('paper')

# Table retrieval/search experiments
modellake.table_search('tables/example.csv', method='dense')

# Analyze and plot benchmarks
modellake.plot_analysis()

# Repeat main experiments (customizable)
modellake.repeat_experiments(method='dense', resource='modelcard', relatedness='paper')
```

### 3. API Reference

```python
def download(resource: str, mode: str = 'scratch', dest: str = './data/') -> str:
    """
    Download raw metadata or content for a specified resource.
    Args:
        resource: 'modelcard', 'github', or 'arxiv'
        mode: 'scratch' (full), 'update' (incremental)
        dest: output directory
    Returns: Shell stdout log as string
    """

def extract_table(resource: str, mode: str = 'scratch', dest: str = './data/') -> str:
    """
    Extract and clean tables from the downloaded resource.
    Args:
        resource: as above
        mode: as above
        dest: output directory
    Returns: Shell stdout log as string
    """

def quality_control(mode: str = 'intra', dest: str = './data/') -> str:
    """
    Run quality control (deduplication/merging).
    Args:
        mode: 'intra' (within-source), 'inter' (cross-source)
        dest: output directory
    Returns: Shell stdout log as string
    """

def extract_relatedness(resource: str = 'paper') -> str:
    """
    Compute relatedness/unionability ground truth.
    Args:
        resource: 'paper', 'model', 'dataset', or 'all'
    Returns: Shell stdout log as string
    """

def table_search(input_table: str, method: str = 'dense', directory: str = './data/') -> str:
    """
    Search for related tables using a specified retrieval method.
    Args:
        input_table: path to input table file
        method: 'dense', 'sparse', or 'hybrid'
        directory: data directory
    Returns: Shell stdout log as string
    """

def plot_analysis() -> str:
    """
    Generate and save figures/plots for statistics and benchmarks.
    Returns: Shell stdout log as string
    """

def repeat_experiments(method: str = 'unionable', resource: str = 'modelcard', relatedness: str = 'paper') -> str:
    """
    Run main experiments for retrieval/evaluation using specified method.
    Args:
        method: retrieval approach
        resource: 'modelcard', 'github', 'arxiv'
        relatedness: 'paper', 'model', or 'dataset'
    Returns: Shell stdout log as string
    """
```

---

## Full Mapping Table: API ⟷ Scripts

| API Command           | Mapped Scripts                                                                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `download`            | `src.data_preprocess.step1`, `src.data_preprocess.step1_down_giturl`, `src.data_preprocess.step2_get_html`                                         |
| `extract_table`       | `src.data_preprocess.step2_gitcard_tab`, `src.data_preprocess.step2_html_parsing`                                                                  |
| `quality_control`     | `src.data_analysis.qc_dedup`, `src.data_gt.step3_pre_merge`                                                                                        |
| `extract_relatedness` | `src.data_gt.overlap_rate`, `src.data_gt.modelcard_matrix`                                                                                         |
| `table_search`        | `scripts/step3_search_hnsw.sh`, `src/baseline2/sparse_search.sh`, `src/baseline2/hybrid_search.sh`                                                 |
| `plot_analysis`       | `src.data_analysis.qc_stats_fig`, `src.data_analysis.gt_distri`                                                                                    |
| `repeat_experiments`  | `src/baseline1/table_retrieval_pipeline.sh`, `src/baseline2/sparse_search.sh`, `src/baseline2/hybrid_search.sh`, `scripts/step3_processmetrics.sh` |

---

## Remaining / Not (Yet) Covered Utility Scripts

These scripts are not wrapped in the main API, as they are primarily used for one-off debugging, data fixing, augmentation, or low-frequency manual analysis:

* Incremental recovery, retry, merge, or “fix” scripts (e.g., s2orc\_merge, s2orc\_retry\_missing, step1\_query\_giturl)
* Debugging/validation scripts (e.g., `check_empty.sh`, `python -m src.data_gt.check_pair_in_gt`)
* Symlink/augmentation utilities (`src.data_symlink.*`, e.g., `trick_aug`, `ln_scilake`, `prepare_sample`, etc.)
* Legacy PDF/TeX-based extraction scripts (deprecated)
* One-off stats or metrics scripts (`count_files.sh`, `standardize_filenames.sh`, etc.)
* LLM/OpenAI batch job status scripts

These scripts are **not required** for the main benchmarking pipeline but may be used for advanced customization, debugging, or offline data augmentation.

---

## Summary

* **Your API design covers >90% of the full benchmark pipeline**
* **Scripts not covered are for advanced use, debugging, or legacy workflows**
* **You can easily extend the API to cover more commands as needed (e.g., symlink/augmentation, one-off fixes, batch job management)**

---

## Example: Minimal End-to-End Pipeline

```python
import modellake.cli_api as modellake

modellake.download('modelcard')
modellake.download('github')
modellake.download('arxiv')
modellake.extract_table('modelcard')
modellake.extract_table('github')
modellake.extract_table('arxiv')
modellake.quality_control('intra')
modellake.quality_control('inter')
modellake.extract_relatedness('paper')
modellake.table_search('tables/example.csv', method='dense')
modellake.plot_analysis()
modellake.repeat_experiments(method='dense', resource='modelcard', relatedness='paper')
```


Now the scripts are stored in 
```bash
src/API/cli_api.py
src/API/pipeline.py
```
---

**For further extension (symlink/augmentation/one-off stats), see the “Not (Yet) Covered” section above.**
