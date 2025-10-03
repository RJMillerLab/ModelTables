"""
LLM-based evaluation of relatedness for table pairs or model-card pairs.

Features:
  - Two prompt templates: tables and model cards (mirrors your human-eval form)
  - Loads pairs from JSON/JSONL/Parquet; optional random sampling
  - Calls src.llm.model.LLM_response to score each pair
  - Saves structured results to JSONL (one object per pair)

Input pairs format (JSON/JSONL): list of objects or lines with keys:
  For tables mode:
    {"id": "pair1", "table_a_md": "|...|\n|...|", "table_b_md": "..."}
  For modelcards mode:
    {"id": "pair1", "card_a": "text...", "card_b": "text..."}

CLI examples:
  python src/gpt_evaluation/evaluate_pairs.py \
    --mode tables \
    --pairs input/table_pairs.jsonl \
    --output output/llm_eval_tables.jsonl \
    --llm gpt-3.5-turbo-0125

  python src/gpt_evaluation/evaluate_pairs.py \
    --mode modelcards \
    --pairs input/card_pairs.parquet \
    --id-col id --a-col card_a --b-col card_b \
    --sample-n 100 \
    --output output/llm_eval_cards.jsonl
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any

import pandas as pd

# Import the LLM caller
from src.llm.model import LLM_response


TABLE_PROMPT = (
    "Human Evaluation: Semantic Table Search over Model Lake, A Benchmark\n"
    "You will be given two tables (A and B) in Markdown.\n"
    "Task: determine whether/how/why they are related.\n\n"
    "Return a strict JSON object with these fields:\n"
    "  related: one of ['YES','NO','UNSURE']\n"
    "  relation_types: array of any of ['JOINABLE','UNIONABLE','KEYWORDS','SEMANTIC','RELATED','OTHER']\n"
    "  closeness: integer 1-5 (Loosely Related=1 ... Very Closely Related=5)\n"
    "  rationale: short text\n"
    "  confidence: integer 1-5\n\n"
    "Table A:\n{table_a}\n\n"
    "Table B:\n{table_b}\n\n"
    "Respond with JSON only, no extra text."
)

CARD_PROMPT = (
    "Human Evaluation: Semantic Model Card Relatedness\n"
    "You will be given two model cards (A and B).\n"
    "Task: determine whether/how/why they are related.\n\n"
    "Return a strict JSON object with these fields:\n"
    "  related: one of ['YES','NO','UNSURE']\n"
    "  relation_types: array of any of ['KEYWORDS','SEMANTIC','RELATED','BENCHMARK','TASK','BASELINE','OTHER']\n"
    "  closeness: integer 1-5 (Loosely Related=1 ... Very Closely Related=5)\n"
    "  rationale: short text\n"
    "  confidence: integer 1-5\n\n"
    "Model Card A:\n{card_a}\n\n"
    "Model Card B:\n{card_b}\n\n"
    "Respond with JSON only, no extra text."
)


def load_pairs(path: str, mode: str, id_col: str, a_col: str, b_col: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jsonl", ".ndjson"}:
        pairs: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs.append(json.loads(line))
        return pairs
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        return data
    if ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        assert id_col in df.columns and a_col in df.columns and b_col in df.columns
        out: List[Dict[str, Any]] = []
        for _, row in df[[id_col, a_col, b_col]].iterrows():
            if mode == "tables":
                out.append({"id": row[id_col], "table_a_md": row[a_col], "table_b_md": row[b_col]})
            else:
                out.append({"id": row[id_col], "card_a": row[a_col], "card_b": row[b_col]})
        return out
    raise ValueError(f"Unsupported pairs file extension: {ext}")


def call_llm(prompt: str, llm: str, max_tokens: int = 800) -> Dict[str, Any]:
    resp, _hist = LLM_response(prompt, llm_model=llm, history=[], kwargs={}, max_tokens=max_tokens)
    # Try parse JSON from the assistant output
    text = resp if isinstance(resp, str) else str(resp)
    try:
        # Find first JSON object if extra text leaked
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return json.loads(text)
    except Exception:
        return {"raw": text}


def evaluate_pairs(mode: str,
                   pairs: List[Dict[str, Any]],
                   llm: str,
                   sample_n: int = 0,
                   seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    items = pairs
    if sample_n and sample_n > 0 and sample_n < len(items):
        items = rng.sample(items, sample_n)

    results: List[Dict[str, Any]] = []
    for item in items:
        pid = item.get("id")
        if mode == "tables":
            prompt = TABLE_PROMPT.format(
                table_a=item.get("table_a_md", ""),
                table_b=item.get("table_b_md", ""),
            )
        else:
            prompt = CARD_PROMPT.format(
                card_a=item.get("card_a", ""),
                card_b=item.get("card_b", ""),
            )
        out = call_llm(prompt, llm=llm)
        rec = {"id": pid, "mode": mode, "prompt": prompt, "response": out}
        results.append(rec)
    return results


def save_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-based evaluation for tables/modelcards pairs")
    ap.add_argument("--mode", choices=["tables", "modelcards"], required=True)
    ap.add_argument("--pairs", required=True, help="Path to pairs file (json/jsonl/parquet)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--llm", default="gpt-3.5-turbo-0125")
    ap.add_argument("--sample-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # For parquet input
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--a-col", default="a")
    ap.add_argument("--b-col", default="b")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs, args.mode, args.id_col, args.a_col, args.b_col)
    results = evaluate_pairs(args.mode, pairs, args.llm, sample_n=args.sample_n, seed=args.seed)
    save_jsonl(results, args.output)
    print(f"✅ Saved {len(results)} LLM evaluations to {args.output}")


if __name__ == "__main__":
    main()


