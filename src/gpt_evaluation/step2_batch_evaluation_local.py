#!/usr/bin/env python3
"""
Author: Zhengyuan Dong
Date: 2025-10-17
Description: Local batch evaluation for table/model relatedness without OpenAI Batch API.
Reads a JSONL of pairs (with ground-truth fields), builds prompts using
the reusable builders, calls the LLM synchronously, parses YAML, and
writes results JSONL for downstream analysis.

Usage:
python -m src.gpt_evaluation.batch_evaluation_local --mode tables --pairs output/table_pairs_with_content.jsonl --output output/table_results.jsonl --llm gpt-4o-mini
python -m src.gpt_evaluation.batch_evaluation_local --mode models --pairs output/model_pairs.jsonl --output output/model_results.jsonl --llm gpt-4o-mini
"""

import os
import json
import argparse
from typing import Dict, Any, Iterable

from src.llm.model import LLM_response
from src.gpt_evaluation.test_single_prompt import (
	build_table_prompt,
	build_model_prompt,
	parse_llm_yaml,
	csv_to_raw_text,
)
from src.utils import load_combined_data


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def load_model_cards_from_raw(model_a: str, model_b: str) -> Dict[str, str]:
	"""Load raw card texts from combined raw data by modelId with case-insensitive match."""
	df_raw = load_combined_data(
		data_type="modelcard",
		file_path="~/Repo/CitationLake/data/raw",
		columns=["modelId", "card"],
	)
	if df_raw is None or len(df_raw) == 0:
		return {"card_a": None, "card_b": None}
	if 'card' not in df_raw.columns and 'card_readme' in df_raw.columns:
		df_raw = df_raw.rename(columns={'card_readme': 'card'})
	lookup_ci = {}
	for mid, txt in zip(df_raw.get('modelId', []), df_raw.get('card', [])):
		if isinstance(mid, str):
			lookup_ci[mid] = txt
			lookup_ci[mid.lower()] = txt
	return {
		"card_a": lookup_ci.get(model_a) or lookup_ci.get((model_a or '').lower()),
		"card_b": lookup_ci.get(model_b) or lookup_ci.get((model_b or '').lower()),
	}


def main():
	parser = argparse.ArgumentParser(description="Local batch GPT evaluation")
	parser.add_argument("--mode", choices=["tables", "models"], default="tables")
	parser.add_argument("--pairs", required=True, help="Input pairs JSONL with GT fields")
	parser.add_argument("--output", required=True, help="Output results JSONL")
	parser.add_argument("--llm", default="gpt-4o-mini")
	parser.add_argument("--limit", type=int, default=0, help="Limit number of pairs (0=all)")
	args = parser.parse_args()

	results = []
	for idx, pair in enumerate(iter_jsonl(args.pairs)):
		if args.limit and idx >= args.limit:
			break

		if args.mode == "tables":
			# Prefer provided raw content; otherwise read from paths
			t_a = pair.get('table_a_md') or pair.get('table_a_raw')
			t_b = pair.get('table_b_md') or pair.get('table_b_raw')
			if not t_a:
				t_a = csv_to_raw_text(pair.get('full_path_a', ''))
			if not t_b:
				t_b = csv_to_raw_text(pair.get('full_path_b', ''))
			prompt = build_table_prompt(t_a, t_b)
		else:
			# Models
			card_a = pair.get('card_a')
			card_b = pair.get('card_b')
			if not (isinstance(card_a, str) and isinstance(card_b, str)):
				model_a = pair.get('model_a') or pair.get('id_a')
				model_b = pair.get('model_b') or pair.get('id_b')
				loaded = load_model_cards_from_raw(model_a, model_b)
				card_a = loaded.get('card_a')
				card_b = loaded.get('card_b')
			prompt = build_model_prompt(card_a or "", card_b or "")

		# Call LLM
		resp_text, _ = LLM_response(prompt, llm_model=args.llm, history=[], kwargs={}, max_tokens=900)
		parsed = parse_llm_yaml(resp_text)

		results.append({
			"id": pair.get('id', f"pair-{idx+1}"),
			"mode": args.mode,
			"llm": args.llm,
			"prompt_len": len(prompt),
			"gt": {
				"related": pair.get('gt_related') or pair.get('is_related'),
				"signals": pair.get('gt_signals', {}),
			},
			"identifiers": {
				"csv_a": pair.get('csv_a'),
				"csv_b": pair.get('csv_b'),
				"model_a": pair.get('model_a') or pair.get('id_a'),
				"model_b": pair.get('model_b') or pair.get('id_b'),
			},
			"response": parsed,
		})

	# Write results
	os.makedirs(os.path.dirname(args.output), exist_ok=True)
	with open(args.output, 'w', encoding='utf-8') as f:
		for r in results:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")
	print(f"✅ Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
	main()


