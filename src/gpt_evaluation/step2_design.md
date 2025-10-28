# Step 2: Multi-LLM Crowdsourcing Evaluation Design

## 1. Objective

Evaluate table pair relatedness using crowdsourcing approach with 5 different LLM models.

## 2. Task Definition

**Input**: Two tables (A and B) in CSV format  
**Output**: Binary judgment + evidence

### 2.1 Expected Response Format (YAML)

```yaml
related: YES | NO | UNSURE
structural_signals: 
  - joinable: boolean          # Share common column names
  - unionable: boolean         # Similar schema/structure
  - keyword_overlap: boolean   # Share common keywords/terms
  - semantically_similar: boolean  # Related meaning/purpose
level_signals:
  - paper_level: boolean       # Related via research papers
  - model_level: boolean       # Related via same model(s)
  - dataset_level: boolean     # Related via same dataset(s)
rationale: "1-2 sentence evidence-based explanation"
confidence: 1-5  # Optional
```

### 2.2 Prompt Design

Detailed prompt that captures structural and level signals:

```
You are evaluating whether two data tables are semantically related.

Table A:
[raw CSV content]

Table B:
[raw CSV content]

Task: Determine if Tables A and B are related (YES/NO/UNSURE) and identify specific signals.

Consider STRUCTURAL signals (select all that apply):
- JOINABLE: share common column names that could be used to join
- UNIONABLE: have similar schema/structure that could be combined
- KEYWORD_OVERLAP: share common keywords or domain terms
- SEMANTICALLY_SIMILAR: have related meaning or purpose

Consider LEVEL signals (select all that apply):
- PAPER_LEVEL: related because they are from the same research paper
- MODEL_LEVEL: related because they are about the same model(s)
- DATASET_LEVEL: related because they use the same dataset(s)

Respond in YAML format (no markdown code fences):
related: [YES/NO/UNSURE]
structural_signals:
  joinable: [true/false]
  unionable: [true/false]
  keyword_overlap: [true/false]
  semantically_similar: [true/false]
level_signals:
  paper_level: [true/false]
  model_level: [true/false]
  dataset_level: [true/false]
rationale: "[1-2 sentences explaining your judgment with specific evidence]"
```

## 3. Evaluation Metrics Design

Since we only have table content and LLM judgments, we design metrics to evaluate the **crowdsourcing validity**:

### 3.1 Inter-Model Agreement (Binary)
- **Consistency Score**: % of models agreeing on YES/NO/UNSURE
- **Majority Vote**: Most common judgment across 5 models
- **Unanimity**: % of pairs where all 5 models agree

### 3.2 Signal Agreement (Multi-label)
- **Structural Signal Agreement**: For each signal (joinable, unionable, etc.), % of models that agree
- **Level Signal Agreement**: For each level (paper/model/dataset), % of models that agree
- **Signal Co-occurrence**: Which signals appear together (e.g., joinable + model_level)

### 3.3 Agreement with GT Labels (Multi-level)
We have MULTIPLE GTs per pair:
- 1 dataset GT
- 1 model GT  
- 8 paper GTs

For each GT level:
- **Accuracy**: % where majority vote matches that GT's label
- **Signal-based Precision**: For pairs predicted as related, what % of structural/level signals match GT level
- **Per-GT Agreement**:
  - Dataset GT agreement
  - Model GT agreement
  - Paper GT agreement (8 separate GT matrices)

Example: If LLM returns `model_level=true` and model GT says related, that's a correct level signal.

### 3.4 Signal-Specific Analysis
- **Joinable Detection Accuracy**: When LLM says joinable, what % have actually matching columns?
- **Level Signal Validation**: When LLM says paper_level, check against paper GT
- **Signal Combination Patterns**: Which signal combinations are most common for different types of relatedness?

### 3.5 Model-Specific Metrics
- Per-model accuracy vs each GT (dataset, model, paper)
- Model correlation matrix for each signal
- Model disagreement patterns per signal type

## 4. LLM Models Selection

Target 5 diverse models from different families:

1. **GPT-4o-mini** (OpenAI) - Fast, cost-effective
2. **GPT-3.5-turbo** (OpenAI) - Classic baseline
3. **Llama3** (Meta, via Ollama) - Open-source foundation model
4. **Claude Sonnet 3.5** (Anthropic) - Different reasoning style
5. **Mistral 7B** (Mistral, via Ollama) - Small efficient model

**Alternative**: If Claude/AWS not available:
- GPT-4, GPT-4o-mini, GPT-3.5-turbo
- Llama3, Mistral

## 5. Implementation Plan

### 5.1 Core Components

1. **Multi-LLM Query Handler** (`multi_llm_handler.py`)
   - Wrapper for querying different LLM providers
   - Unified interface for OpenAI, Ollama, Anthropic
   - Error handling and retry logic

2. **Batch Evaluator** (`step2_batch_evaluation.py`)
   - Load pairs from step1 output
   - Query 5 models for each pair
   - Save results with metadata

3. **Metrics Analyzer** (`step3_analyze_results.py`)
   - Compute all designed metrics
   - Generate visualization and reports

### 5.2 Data Flow

```
Step1 Output (pairs.jsonl)
    ↓
Load CSV Content for each table pair
    ↓
Build Prompt (table content → prompt)
    ↓
Query 5 LLMs (batch processing)
    ↓
Parse & Save Results (per-model responses)
    ↓
Compute Metrics (agreement, accuracy, etc.)
    ↓
Generate Report
```

## 6. Expected Output Structure

```
output/gpt_evaluation/
├── step2_results/
│   ├── llm_responses_all_models.jsonl  # All model responses
│   ├── per_model_responses/             # Split by model
│   │   ├── gpt-4o-mini_responses.jsonl
│   │   ├── gpt-3.5-turbo_responses.jsonl
│   │   └── ...
│   └── aggregated_results.jsonl        # Per-pair aggregated judgments
└── step3_analysis/
    ├── metrics_report.json
    ├── agreement_analysis.json
    ├── gt_comparison.json
    └── visualizations/
```

## 7. Key Design Decisions

1. **Simplified Prompt**: Focus on binary judgment + rationale (remove complex multi-choice)
2. **YAML Output**: Easier to parse than JSON for LLMs
3. **Batch Processing**: Query 5 models in parallel where possible
4. **Metadata Preservation**: Keep all intermediate data for analysis
5. **GT-Agnostic Metrics**: Design metrics that work even if GT is noisy/incomplete

## 8. Crowdsourcing Validity Indicators

Since tables are our only signal, we evaluate:

1. **Structural Signals**: Analyze if tables have matching column names, similar schemas
2. **Semantic Signals**: Keyword overlap, topic similarity
3. **Model Consensus**: When models agree, they likely found strong signal
4. **Rationale Quality**: Good rationales should cite concrete evidence
5. **Confidence Calibration**: Higher confidence on clear cases, lower on ambiguous

These metrics help assess whether the crowdsourcing annotation is reliable and useful.

