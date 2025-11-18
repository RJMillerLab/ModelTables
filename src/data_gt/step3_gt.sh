#!/bin/bash

# Usage: bash src/data_gt/step3_gt.sh [TAG]
# Example: bash src/data_gt/step3_gt.sh 251117

TAG=${1:-""}
LOG=logs/step3_gt${TAG:+_${TAG}}.log

REL_KEYS=(
  direct_label
  direct_label_influential
  direct_label_methodology_or_result
  direct_label_methodology_or_result_influential
  max_pr
  max_pr_influential
  max_pr_methodology_or_result
  max_pr_methodology_or_result_influential
)

echo "===== step3_gt started at $(date) =====" > "$LOG"
if [ -n "$TAG" ]; then
  echo "Using tag: $TAG" >> "$LOG"
fi

for key in "${REL_KEYS[@]}"; do
  echo "---- Processing rel_key = $key ----" >> "$LOG"
  if [ -n "$TAG" ]; then
    python -m src.data_gt.step3_gt \
      --overlap_rate_threshold 0.0 \
      --rel_key "$key" \
      --tag "$TAG" \
      >> "$LOG" 2>&1
  else
    python -m src.data_gt.step3_gt \
      --overlap_rate_threshold 0.0 \
      --rel_key "$key" \
      >> "$LOG" 2>&1
  fi
  echo "" >> "$LOG"
done

echo "===== step3_gt finished at $(date) =====" >> "$LOG"
