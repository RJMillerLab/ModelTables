#!/bin/bash

LOG=logs/step3_gt_0526.log

REL_KEYS=(
  #direct_label
  #direct_label_influential
  #direct_label_methodology_or_result
  #direct_label_methodology_or_result_influential
  max_pr
  max_pr_influential
  max_pr_methodology_or_result
  max_pr_methodology_or_result_influential
)

echo "===== step3_gt started at $(date) =====" > "$LOG"

for key in "${REL_KEYS[@]}"; do
  echo "---- Processing rel_key = $key ----" >> "$LOG"
  python -m src.data_gt.step3_gt \
    --overlap_rate_threshold 0.0 \
    --rel_key "$key" \
    >> "$LOG" 2>&1
  echo "" >> "$LOG"
done

echo "===== step3_gt finished at $(date) =====" >> "$LOG"
