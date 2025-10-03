"""
Convert LLM evaluation JSONL (from evaluate_pairs.py) into a readable Markdown report.

Input JSONL lines schema (per evaluate_pairs.py):
  {
    "id": "pair-001",
    "mode": "tables" | "modelcards",
    "prompt": "...",
    "response": {
        "related": "YES|NO|UNSURE",
        "relation_types": [...],
        "closeness": 1-5,
        "rationale": "...",
        "confidence": 1-5,
        ... (or raw if parsing failed)
    }
  }

Usage:
  python -m src.gpt_evaluation.jsonl_to_markdown \
    --input output/llm_eval_tables.jsonl \
    --output output/llm_eval_tables.md \
    --show-prompt  # optional, include (truncated) prompt context
"""

import os
import json
import argparse
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def truncate_block(text: str, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def to_markdown(items: List[Dict[str, Any]], show_prompt: bool = False) -> str:
    lines: List[str] = []
    lines.append("# LLM Evaluation Report\n")
    counts = {"YES": 0, "NO": 0, "UNSURE": 0}

    for idx, obj in enumerate(items, 1):
        pid = obj.get("id", f"pair-{idx}")
        mode = obj.get("mode", "tables")
        response = obj.get("response", {}) or {}
        related = str(response.get("related", "")).upper()
        relation_types = response.get("relation_types", [])
        closeness = response.get("closeness", "")
        confidence = response.get("confidence", "")
        rationale = response.get("rationale", response.get("raw", ""))
        if related in counts:
            counts[related] += 1

        lines.append(f"## Pair {idx}: {pid} ({mode})\n")
        if show_prompt:
            prompt = obj.get("prompt", "")
            if prompt:
                lines.append("<details><summary>Prompt (click to expand)</summary>\n\n")
                lines.append("```\n" + truncate_block(prompt) + "\n```\n")
                lines.append("</details>\n")

        lines.append("**Related**: " + (related or "UNKNOWN"))
        lines.append("")
        if relation_types:
            lines.append("- **relation_types**: " + ", ".join(map(str, relation_types)))
        if closeness != "":
            lines.append(f"- **closeness**: {closeness}")
        if confidence != "":
            lines.append(f"- **confidence**: {confidence}")
        if rationale:
            lines.append("- **rationale**: " + truncate_block(str(rationale), 1000))
        lines.append("")

    # Summary
    total = len(items)
    lines.insert(1, f"**Summary**: total={total}, YES={counts['YES']}, NO={counts['NO']}, UNSURE={counts['UNSURE']}\n")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert LLM eval JSONL to Markdown")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    items = load_jsonl(args.input)
    md = to_markdown(items, show_prompt=args.show_prompt)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ Wrote Markdown to {args.output}")


if __name__ == "__main__":
    main()


