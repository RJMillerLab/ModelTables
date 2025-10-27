#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-17
Description: Test single prompt for table relatedness evaluation
Test single prompt for table relatedness evaluation

This script tests a single prompt for table relatedness evaluation.
It uses the LLM_response function to call the LLM and get the response.
It then parses the response and prints the results.

Usage:
python -m src.gpt_evaluation.test_single_prompt --mode tables --pairs output/table_pairs_with_content.jsonl --index 0
python -m src.gpt_evaluation.test_single_prompt --mode models --pairs PATH_TO_MODEL_PAIRS.jsonl --index 0
"""

import os
import json
import argparse
from src.llm.model import LLM_response
from src.utils import load_combined_data


def csv_to_raw_text(csv_path: str) -> str:
	"""Convert CSV to raw text format"""
	try:
		if not os.path.exists(csv_path):
			return f"File not found: {csv_path}"
		
		with open(csv_path, 'r', encoding='utf-8') as f:
			content = f.read()
		return content
	except Exception as e:
		return f"Error reading {os.path.basename(csv_path)}: {str(e)}"

def build_table_prompt(table_a_raw: str, table_b_raw: str) -> str:
	"""Build the single-sample prompt for table relatedness."""
	return f"""
Your task is to determine whether two tables are related, together with picking relatedness types and reasons from multiple choices.

You will be given two tables (A and B) in raw CSV format.
Please evaluate their relationship from both structural and semantic perspectives and answer the following questions:

**Question 1 (Single Selection):** Are these two tables related? Choose ONE:
- YES, they are related
- NO, they are not related  
- UNSURE, I cannot tell

Based on your answer above, please provide additional details:

**If you answered YES (they are related):**

**Question 2 (Multiple Selection):** What is the nature of the relationship? Select ALL that apply:
- JOINABLE: share a common column
- UNIONABLE: have a similar structure
- KEYWORDS: share common terms or topics
- SEMANTIC: strong overlap in meaning
- RELATED: clearly related for other reasons
- Other: (specify reason)

**Question 3 (Single Selection):** How closely are they related? Choose ONE:
- **Loosely Related** (1)
- **Somewhat Related** (2)
- **Moderately Related** (3)
- **Closely Related** (4)
- **Very Closely Related** (5)

**Question 4 (Multiple Selection):** Why do you believe this relationship exists? Select ALL that apply:
- BENCHMARK: part of a standard dataset
- TASK: used for the same task/context
- BASELINE: one serves as reference for the other
- Other: (specify reason)

**If you answered NO (they are not related):**

**Question 2 (Multiple Selection):** Why do you believe they are NOT related? Select ALL that apply:
- SEMANTICS: different table meaning (e.g., performance vs. config)
- TOPICS: completely different topics or tasks
- BENCHMARK: not part of the same dataset/benchmark
- Other: (specify reason)

**If you answered UNSURE (I cannot tell):**

**Question 2 (Multiple Selection):** Why are you unsure? Select ALL that apply:
- NOT CLEAR if STRUCTURALLY SIMILAR (JOIN/UNION)
- NOT CLEAR if same TOPIC/KEYWORDS
- NOT CLEAR if SEMANTICALLY related
- NOT SURE if part of same BENCHMARK/DATASET
- NOT SURE if used for same TASK/CONTEXT
- NOT SURE if one is BASELINE/REFERENCE for the other
- Other: (specify reason)

**For all answers, also provide:**

**Question 5 (Single Selection):** How confident are you in your answer? Choose ONE:
- **Not Confident** (1)
- **Somewhat Confident** (2)
- **Moderately Confident** (3)
- **Confident** (4)
- **Very Confident** (5)

**Table A:**
{table_a_raw}

**Table B:**
{table_b_raw}

**Instructions for response:**
- Use standard YAML format: keyword: value
- For lists, use: keyword: [item1, item2] or keyword: [] for empty
- For single values, use: keyword: value (no quotes unless needed)
- Return EXACT KEYWORDS ONLY for all selections
- For closeness and confidence, return the exact keyword without **bold** markers
- If you select "Other", provide the specific reason in other_reasons field
- Always include ALL fields in YAML
- YAML only; no extra text or markdown formatting

**CRITICAL: Use clean YAML format with keyword: value pairs, no **bold** or special formatting.**

**Please respond with a YAML format containing your answers:**
related: [YES/NO/UNSURE]
relation_types: [list of selected types - only if related=YES]
closeness: [bolded keyword - only if related=YES]
reasons: [list of selected reasons - only if related=YES]
not_related_reasons: [list of selected reasons - only if related=NO]
unsure_reasons: [list of selected reasons - only if related=UNSURE]
confidence: [bolded keyword]
rationale: [brief explanation]
other_reasons: [any reasons for "Other" selections]
"""

def build_model_prompt(card_a_raw: str, card_b_raw: str) -> str:
	"""Build the single-sample prompt for model relatedness (using raw card text).
	Captures three-level signals: paper, model, dataset.
	"""
	return f"""
Your task is to determine whether two model cards (A and B) describe related models. Make a decision and then select relation types at three levels (Model / Paper / Dataset). Use exact tokens and cite concrete evidence.

You will be given two model cards in raw text format.
Please evaluate their relationship from both structural and semantic perspectives and answer the following questions:

**Question 1 (Single Selection):** Are these two models related? Choose ONE:
- YES, they are related
- NO, they are not related  
- UNSURE, I cannot tell

Based on your answer above, please provide additional details:

**If you answered YES (they are related):**

**Question 2A (Multiple Selection, Model level):** Select ALL that apply (use exact tokens):
- EXPLICIT_REFERENCE: one card explicitly references/links the other model
- BASE_INHERITANCE: one model is based on/derived from the other (base -> finetune, distilled, adapter, LoRA, etc.)
- SAME_ARCH_FAMILY: same architecture family (e.g., T5 family)
- VERSION_VARIANT: same model with size/version/ckpt variant
- SAME_ORG: same organization/project lineage
- CROSS_REFERENCE: cards cite each other or a shared canonical id
- RELATED: clearly related for other reasons
- Other: (specify reason)

**Question 2B (Multiple Selection, Paper level):** Select ALL that apply (use exact tokens):
- DIRECT_CITATION: papers directly cite each other
- REFERENCE_OVERLAP: significant overlap in reference lists
- SAME_TASK_PAPERS: papers are for the same task/context
- OTHER: other paper-level relationships

**Question 2C (Multiple Selection, Dataset level):** Select ALL that apply (use exact tokens):
- DATASET_COUSAGE: trained/evaluated on the same dataset(s)
- SAME_BENCHMARK: part of the same benchmark suite
- SHARED_PRETRAIN_DATA: share pretraining corpora
- OTHER: other dataset-level relationships

**Question 3 (Single Selection):** How closely are they related? Choose ONE:
- **Loosely Related** (1)
- **Somewhat Related** (2)
- **Moderately Related** (3)
- **Closely Related** (4)
- **Very Closely Related** (5)

**For all answers, also provide:**

**Question 4 (Single Selection):** How confident are you in your answer? Choose ONE:
- **Not Confident** (1)
- **Somewhat Confident** (2)
- **Moderately Confident** (3)
- **Confident** (4)
- **Very Confident** (5)

**Evidence requirements:**
- Cite concrete evidence in the rationale: list matching phrases, ids, dataset names, architecture names, version strings, or direct links.
- Do not invent entities not present in the text.

**Model Card A:**
{card_a_raw}

**Model Card B:**
{card_b_raw}

**Instructions for response:**
- Use standard YAML format: keyword: value
- For lists, use: keyword: [item1, item2] or keyword: [] for empty
- For single values, use: keyword: value (no quotes unless needed)
- Return EXACT KEYWORDS ONLY for all selections
- For closeness and confidence, return the exact keyword without **bold** markers
- If you select "OTHER" in any list, provide the specific reason in other_reasons
- Always include ALL fields in YAML
- YAML only; no extra text or markdown formatting

**CRITICAL: Use clean YAML format with keyword: value pairs, no **bold** or special formatting.**

**Please respond with a YAML format containing your answers:**
related: [YES/NO/UNSURE]
model_relation_types: [list - only if related=YES]
paper_relation_types: [list - only if related=YES]
dataset_relation_types: [list - only if related=YES]
closeness: [bolded keyword - only if related=YES]
confidence: [bolded keyword]
rationale: [brief explanation citing evidence]
other_reasons: [any reasons for "OTHER" selections]
unsure_reasons: [list - only if related=UNSURE]
not_related_reasons: [list - only if related=NO]
"""

def parse_llm_yaml(response_text: str):
	"""Strip code fences if any and parse YAML into a dict."""
	try:
		import yaml
	except Exception:
		return {"raw": response_text}
	yaml_content = response_text
	if '```yaml' in yaml_content:
		try:
			yaml_content = yaml_content.split('```yaml')[1].split('```')[0].strip()
		except Exception:
			pass
	elif '```' in yaml_content:
		try:
			yaml_content = yaml_content.split('```')[1].split('```')[0].strip()
		except Exception:
			pass
	try:
		return yaml.safe_load(yaml_content)
	except Exception:
		return {"raw": response_text}

def test_single_table_pair(pairs_file: str, pair_index: int = 0, llm_model: str = "gpt-4o-mini"):
	"""Test a single table pair"""
	with open(pairs_file, 'r') as f:
		pairs = [json.loads(line) for line in f]
	if pair_index >= len(pairs):
		print(f"Pair index {pair_index} out of range (max: {len(pairs)-1})")
		return
	pair = pairs[pair_index]
	print(f"=== TESTING PAIR {pair_index} ===")
	print(f"CSV A: {pair['csv_a']}")
	print(f"CSV B: {pair['csv_b']}")
	print(f"Resource: {pair['resource']}")
	print(f"GT Related: {pair['is_related']}")
	# Get raw CSV content
	table_a_raw = csv_to_raw_text(pair.get('full_path_a', ''))
	table_b_raw = csv_to_raw_text(pair.get('full_path_b', ''))
	# Build prompt
	prompt = build_table_prompt(table_a_raw, table_b_raw)
	print(f"\n=== PROMPT ===")
	print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
	# Call LLM
	print(f"\n=== CALLING LLM ===")
	try:
		response, _ = LLM_response(prompt, llm_model=llm_model, history=[], kwargs={}, max_tokens=1000)
		print(f"Raw response: {response}")
		yaml_response = parse_llm_yaml(response)
		
		# Save to logs directory
		try:
			os.makedirs("logs", exist_ok=True)
			log_name = f"logs/table_single_{pair['csv_a'].replace('/', '_')}__{pair['csv_b'].replace('/', '_')}.yaml"
			with open(log_name, 'w', encoding='utf-8') as f:
				try:
					import yaml as _yaml
					f.write(_yaml.dump(yaml_response, default_flow_style=False))
				except Exception:
					f.write(str(yaml_response))
			print(f"Saved parsed output to {log_name}")
		except Exception as e:
			print(f"Failed to write logs: {e}")
		
		# Print parsed
		try:
			import yaml as _yaml
			print(f"\n=== PARSED YAML RESPONSE ===")
			print(_yaml.dump(yaml_response, default_flow_style=False))
		except Exception:
			print(yaml_response)
	except Exception as e:
		print(f"LLM call error: {e}")

def test_single_model_pair(pairs_file: str, pair_index: int = 0, llm_model: str = "gpt-4o-mini", model_ids_csv: str = ""):
	"""Test a single model-card pair. Use --model-ids idA,idB to directly load from raw combined data.
	Otherwise expects pairs_file JSONL with 'model_a'/'model_b' or 'card_a'/'card_b' or paths.
	"""
	model_a_id = None
	model_b_id = None
	pairs = []
	if model_ids_csv:
		parts = [p.strip() for p in model_ids_csv.split(',') if p.strip()]
		if len(parts) >= 2:
			model_a_id, model_b_id = parts[0], parts[1]
	else:
		if os.path.exists(pairs_file):
			with open(pairs_file, 'r') as f:
				pairs = [json.loads(line) for line in f]
			if pair_index >= len(pairs):
				print(f"Pair index {pair_index} out of range (max: {len(pairs)-1})")
				return
			pair = pairs[pair_index]
			model_a_id = pair.get('model_a') or pair.get('id_a')
			model_b_id = pair.get('model_b') or pair.get('id_b')
		else:
			print(f"Pairs file not found: {pairs_file}")
			return

	print(f"=== TESTING MODEL PAIR {pair_index} ===")
	print(f"Model A: {model_a_id}")
	print(f"Model B: {model_b_id}")

	card_a_raw = None
	card_b_raw = None
	# Direct content from pairs if present
	if pairs:
		pair = pairs[pair_index]
		if 'card_a' in pair and 'card_b' in pair and isinstance(pair['card_a'], str) and isinstance(pair['card_b'], str):
			card_a_raw = pair['card_a']
			card_b_raw = pair['card_b']
		elif 'card_path_a' in pair or 'card_path_b' in pair:
			card_a_raw = read_text_file(pair.get('card_path_a', ''))
			card_b_raw = read_text_file(pair.get('card_path_b', ''))

	# Otherwise load from combined raw data
	if card_a_raw is None or card_b_raw is None:
		print("Loading model cards from raw combined data...")
		df_raw = load_combined_data(data_type="modelcard", file_path="~/Repo/CitationLake/data/raw", columns=["modelId", "card"])  # minimal columns
		if df_raw is None or len(df_raw) == 0:
			print("Failed to load combined raw modelcard data.")
			return
		# normalize columns
		if 'card' not in df_raw.columns and 'card_readme' in df_raw.columns:
			df_raw = df_raw.rename(columns={'card_readme': 'card'})
		# build case-insensitive lookup
		lookup_ci = {}
		for mid, txt in zip(df_raw.get('modelId', []), df_raw.get('card', [])):
			if isinstance(mid, str):
				lookup_ci[mid] = txt
				lookup_ci[mid.lower()] = txt
		# try direct and lowercased
		card_a_raw = lookup_ci.get(model_a_id) or lookup_ci.get((model_a_id or '').lower())
		card_b_raw = lookup_ci.get(model_b_id) or lookup_ci.get((model_b_id or '').lower())

	# Validate content; skip LLM if missing/too short
	def _is_meaningful(txt: str) -> bool:
		return isinstance(txt, str) and len(txt.strip()) >= 50  # require some substance
	if not _is_meaningful(card_a_raw) or not _is_meaningful(card_b_raw):
		print("Insufficient card text for one or both models; skip LLM call.")
		print(f"Model A card present: {_is_meaningful(card_a_raw)}  | length: {0 if card_a_raw is None else len(str(card_a_raw))}")
		print(f"Model B card present: {_is_meaningful(card_b_raw)}  | length: {0 if card_b_raw is None else len(str(card_b_raw))}")
		return

	# Build prompt
	prompt = build_model_prompt(card_a_raw, card_b_raw)
	print(f"\n=== PROMPT ===")
	print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
	# Call LLM
	print(f"\n=== CALLING LLM ===")
	try:
		response, _ = LLM_response(prompt, llm_model=llm_model, history=[], kwargs={}, max_tokens=1000)
		print(f"Raw response: {response}")
		yaml_response = parse_llm_yaml(response)
		# Persist to logs
		try:
			os.makedirs("logs", exist_ok=True)
			log_name = f"logs/model_single_{(model_a_id or 'A').replace('/', '_')}__{(model_b_id or 'B').replace('/', '_')}.yaml"
			with open(log_name, 'w', encoding='utf-8') as f:
				try:
					import yaml as _yaml
					f.write(_yaml.dump(yaml_response, default_flow_style=False))
				except Exception:
					f.write(str(yaml_response))
			print(f"Saved parsed output to {log_name}")
		except Exception as e:
			print(f"Failed to write logs: {e}")

		# Print parsed
		try:
			import yaml as _yaml
			print(f"\n=== PARSED YAML RESPONSE ===")
			print(_yaml.dump(yaml_response, default_flow_style=False))
		except Exception:
			print(yaml_response)

	except Exception as e:
		print(f"LLM call error: {e}")

def demo_table_prompt(llm_model: str = "gpt-4o-mini"):
	"""Run a minimal demo of the table prompt using tiny synthetic CSVs."""
	table_a_raw = "col1,col2\nalpha,1\nbeta,2\n"
	table_b_raw = "col1,col2\nalpha,3\ngamma,4\n"
	prompt = build_table_prompt(table_a_raw, table_b_raw)
	print("\n=== DEMO TABLE PROMPT ===")
	print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
	print("\n=== CALLING LLM ===")
	try:
		response, _ = LLM_response(prompt, llm_model=llm_model, history=[], kwargs={}, max_tokens=800)
		print(f"Raw response: {response}")
		parsed = parse_llm_yaml(response)
		try:
			import yaml as _yaml
			print("\n=== PARSED YAML RESPONSE ===")
			print(_yaml.dump(parsed, default_flow_style=False))
		except Exception:
			print(parsed)
	except Exception as e:
		print(f"LLM call error: {e}")

def demo_model_prompt(llm_model: str = "gpt-4o-mini"):
	"""Run a minimal demo of the model prompt using tiny synthetic card texts."""
	card_a_raw = (
		"Model: TinyT5-Small\n"
		"Org: ExampleAI\n"
		"Task: Summarization\n"
		"Datasets: CNN/DailyMail\n"
		"Notes: finetuned from TinyT5-Base."
	)
	card_b_raw = (
		"Model: TinyT5-Base\n"
		"Org: ExampleAI\n"
		"Task: Summarization\n"
		"Datasets: CNN/DailyMail\n"
		"Notes: base model referenced by TinyT5-Small."
	)
	prompt = build_model_prompt(card_a_raw, card_b_raw)
	print("\n=== DEMO MODEL PROMPT ===")
	print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
	print("\n=== CALLING LLM ===")
	try:
		response, _ = LLM_response(prompt, llm_model=llm_model, history=[], kwargs={}, max_tokens=800)
		print(f"Raw response: {response}")
		parsed = parse_llm_yaml(response)
		try:
			import yaml as _yaml
			print("\n=== PARSED YAML RESPONSE ===")
			print(_yaml.dump(parsed, default_flow_style=False))
		except Exception:
			print(parsed)
	except Exception as e:
		print(f"LLM call error: {e}")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["tables", "models"], default="tables")
	parser.add_argument("--pairs", default="output/table_pairs_with_content.jsonl")
	parser.add_argument("--index", type=int, default=0, help="Pair index to test")
	parser.add_argument("--llm", default="gpt-4o-mini")
	parser.add_argument("--demo", choices=["none", "tables", "models"], default="none", help="Run built-in minimal demos")
	parser.add_argument("--model-ids", default="", help="Comma-separated modelIdA,modelIdB to directly load from raw data")
	args = parser.parse_args()
	if args.demo == "tables":
		demo_table_prompt(llm_model=args.llm)
		return
	if args.demo == "models":
		demo_model_prompt(llm_model=args.llm)
		return
	if args.mode == "tables":
		test_single_table_pair(args.pairs, args.index, llm_model=args.llm)
	else:
		test_single_model_pair(args.pairs, args.index, llm_model=args.llm, model_ids_csv=args.model_ids)

if __name__ == "__main__":
    main()
