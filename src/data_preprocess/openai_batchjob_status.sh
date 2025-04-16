#!/bin/bash

# === Load API key from .env file ===
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)  ######## Load variables from .env
else
    echo "Error: .env file not found."
    exit 1
fi

# === Validate required env var =========
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set in .env"
    exit 1
fi
BATCH_ID="batch_67fdd91b62e48190954edde236ba74d8"

# === API endpoint and headers ===========
BASE_URL="https://api.openai.com/v1"
STATUS_URL="$BASE_URL/batches/$BATCH_ID"
HEADERS=(-H "Authorization: Bearer $OPENAI_API_KEY")

# === Send request to get batch status ===
response=$(curl -s "${HEADERS[@]}" "$STATUS_URL")

# === Extract status and progress info =========
status=$(echo "$response" | jq -r '.status')
total=$(echo "$response" | jq -r '.request_counts.total // 0')
completed=$(echo "$response" | jq -r '.request_counts.completed // 0')
failed=$(echo "$response" | jq -r '.request_counts.failed // 0')

# === Print status summary ============
echo "Batch ID: $BATCH_ID"
echo "Status: $status"
echo "Total Requests: $total"
echo "Completed: $completed"
echo "Failed: $failed"

# === Calculate and show progress percentage ============
if [ "$total" -gt 0 ]; then
    percent=$(echo "scale=2; ($completed + $failed) / $total * 100" | bc)
    echo "Progress: $percent %"
else
    echo "Progress: Unknown (total requests = 0)"
fi
