# Step 2: Multi-Model Crowdsourcing Evaluation

## Overview

Step 2 implements a crowdsourcing approach to evaluate table pair relatedness using 5 different LLM models. This creates redundancy and allows for agreement-based quality assessment.

## Design Document

See `step2_design.md` for complete design rationale and metric definitions.

## Quick Start

### 1. Run Step 1 (if not done)

```bash
python src/gpt_evaluation/step1_table_sampling.py \
    --n-samples-pool 100000 \
    --target-positive 150 \
    --target-negative 150 \
    --total-target 300 \
    --seed 42
```

This generates `output/gpt_evaluation/table_v2_all_levels_pairs.jsonl`

### 2. Run Step 2: Multi-Model Evaluation

```bash
# Test with a few pairs first
python src/gpt_evaluation/step2_batch_multi_model.py \
    --input output/gpt_evaluation/table_v2_all_levels_pairs.jsonl \
    --output output/gpt_evaluation/step2_results \
    --models gpt-4o-mini,gpt-3.5-turbo,llama3,mistral,gemma:2b \
    --limit 10 \
    --verbose
```

### 3. Analyze Results

```bash
python src/gpt_evaluation/step3_analyze_results.py \
    --input output/gpt_evaluation/step2_results/all_model_responses.jsonl \
    --output output/gpt_evaluation/step3_analysis
```

## Expected Response Format

Each LLM returns a YAML response:

```yaml
related: YES | NO | UNSURE
rationale: "1-2 sentences explaining your judgment"
confidence: 1-5  # Optional
```

## Evaluation Metrics

### 1. Inter-Model Agreement
- **Consistency Score**: % of models that agree on the same judgment
- **Unanimity Rate**: % of pairs where all 5 models agree
- **Majority Vote**: Most common judgment across 5 models

### 2. Agreement with GT Labels
- **Accuracy**: % where majority vote matches GT label
- **Confusion Matrix**: GT label vs Majority Vote
- **Per-Level Analysis**: Metrics broken down by paper/modelcard/dataset

### 3. Uncertainty Analysis
- **UNSURE Rate**: % of UNSURE responses
- **Uncertainty Distribution**: Per-model uncertainty patterns

### 4. Model-Specific Metrics
- Per-model accuracy vs GT
- Success rates and error rates
- Vote distributions per model
- Average response time

## Available LLM Models

### OpenAI (requires API key)
- `gpt-4o-mini` - Fast, cost-effective
- `gpt-3.5-turbo` - Classic baseline
- `gpt-4-turbo` - Most capable

### Ollama (requires local installation)
Run: `ollama pull llama3` before using
- `llama3` - Meta's latest open model
- `mistral` - Mistral's 7B model
- `gemma:2b` - Google's small efficient model

**Note**: To use Ollama models, you need to:
1. Install Ollama: https://ollama.ai
2. Download models: `ollama pull llama3`, `ollama pull mistral`, etc.
3. Ensure Ollama is running (default: http://localhost:11434)

## Command-Line Options

### Step 2: Batch Evaluation

```bash
python src/gpt_evaluation/step2_batch_multi_model.py \
    --input <pairs.jsonl>          # Required: Step 1 output
    --output <output_dir>           # Required: Where to save results
    --models <model1,model2,...>   # Comma-separated model names
    --tables-dir <path>            # Where CSV tables are stored
    --limit <n>                    # Limit to first N pairs (0=all)
    --start-from <n>               # Start from pair index (for resuming)
    --verbose                      # Show detailed progress
```

### Step 3: Analysis

```bash
python src/gpt_evaluation/step3_analyze_results.py \
    --input <responses.jsonl>      # Required: Step 2 output
    --output <analysis_dir>        # Required: Where to save analysis
```

## Output Structure

```
output/gpt_evaluation/
├── step2_results/
│   ├── all_model_responses.jsonl  # All model responses (one per pair)
│   └── summary.json                # Basic summary stats
│
└── step3_analysis/
    ├── metrics.json                # All computed metrics (JSON)
    └── evaluation_report.txt       # Human-readable report
```

## Example: Full Pipeline

```bash
# 1. Sample table pairs (Step 1)
python src/gpt_evaluation/step1_table_sampling.py \
    --n-samples-pool 100000 \
    --target-positive 150 \
    --target-negative 150 \
    --total-target 300

# 2. Evaluate with 5 models (Step 2) - test with 10 pairs first
python src/gpt_evaluation/step2_batch_multi_model.py \
    --input output/gpt_evaluation/table_v2_all_levels_pairs.jsonl \
    --output output/gpt_evaluation/step2_results \
    --models gpt-4o-mini,gpt-3.5-turbo,llama3,mistral,gemma:2b \
    --limit 10

# 3. Analyze results (Step 3)
python src/gpt_evaluation/step3_analyze_results.py \
    --input output/gpt_evaluation/step2_results/all_model_responses.jsonl \
    --output output/gpt_evaluation/step3_analysis

# 4. If successful, run on full dataset (remove --limit)
python src/gpt_evaluation/step2_batch_multi_model.py \
    --input output/gpt_evaluation/table_v2_all_levels_pairs.jsonl \
    --output output/gpt_evaluation/step2_results \
    --models gpt-4o-mini,gpt-3.5-turbo,llama3,mistral,gemma:2b
```

## Crowdsourcing Validity Indicators

The evaluation computes several signals that help assess reliability:

1. **High Inter-Model Agreement**: When models agree, they likely found strong signal
2. **Rationale Quality**: Good rationales cite concrete evidence (column names, data types, keywords)
3. **Confidence Calibration**: Higher confidence on clear cases
4. **GT Agreement**: Check if models' judgments align with known ground truth

## Tips

1. **Start Small**: Use `--limit 10` to test before running on full dataset
2. **Check Models**: Run `python src/gpt_evaluation/multi_llm_handler.py` to see available models
3. **Parallel Processing**: The script queries models sequentially. For true parallelism, consider modifying to use ThreadPoolExecutor
4. **Resume**: Use `--start-from N` to resume from a specific pair index
5. **Error Handling**: Failed queries are logged; run again to retry

## Troubleshooting

### Ollama models not working
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull models
ollama pull llama3
ollama pull mistral
```

### OpenAI API errors
- Set `OPENAI_API_KEY` in `.env` file
- Check API rate limits

### Missing CSV files
- Ensure `--tables-dir` points to correct directory
- Check that CSV paths in pairs match actual files

## Next Steps

After Step 2-3, you can:
- Visualize agreement patterns
- Compare model performance
- Identify edge cases requiring human review
- Build a high-quality dataset from unanimous judgments

