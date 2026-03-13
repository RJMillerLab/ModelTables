
set -e

TAG="$1"

if [ -z "$TAG" ]; then
    echo "Usage: $0 <tag>"
    echo "Example: $0 251117"
    exit 1
fi

MASK_FILE="data/analysis/all_valid_title_valid_v2_${TAG}.txt"

if [ ! -f "$MASK_FILE" ]; then
    echo "❌ Mask file not found: $MASK_FILE"
    exit 1
fi

echo "=========================================="
echo "Zipping files from mask into 3 resource zips"
echo "=========================================="
echo "Tag:        $TAG"
echo "Mask file:  $MASK_FILE"
echo ""

TMP_HUGGING="$(mktemp)"
TMP_GITHUB="$(mktemp)"
TMP_HTML="$(mktemp)"
trap 'rm -f "$TMP_HUGGING" "$TMP_GITHUB" "$TMP_HTML"' EXIT

COUNT_TOTAL=0
COUNT_EXIST=0
COUNT_HUGGING=0
COUNT_GITHUB=0
COUNT_HTML=0

while IFS= read -r path; do
    if [ -z "$path" ]; then
        continue
    fi
    COUNT_TOTAL=$((COUNT_TOTAL + 1))
    if [ -f "$path" ]; then
        if [[ "$path" == data/processed/deduped_hugging_csvs_v2_${TAG}/* ]]; then
            echo "$path" >> "$TMP_HUGGING"
            COUNT_HUGGING=$((COUNT_HUGGING + 1))
        elif [[ "$path" == data/processed/deduped_github_csvs_v2_${TAG}/* ]]; then
            echo "$path" >> "$TMP_GITHUB"
            COUNT_GITHUB=$((COUNT_GITHUB + 1))
        elif [[ "$path" == data/processed/tables_output_v2_${TAG}/* ]]; then
            echo "$path" >> "$TMP_HTML"
            COUNT_HTML=$((COUNT_HTML + 1))
        else
            echo "⚠️  Path not in known resource dirs (skipped): $path"
        fi
        COUNT_EXIST=$((COUNT_EXIST + 1))
    else
        echo "⚠️  Missing file (skipped): $path"
    fi
done < "$MASK_FILE"

if [ "$COUNT_EXIST" -eq 0 ]; then
    echo "❌ No existing files from mask, aborting."
    exit 1
fi

echo ""
echo "Files in mask:    $COUNT_TOTAL"
echo "Existing files:   $COUNT_EXIST"
echo "  - hugging:      $COUNT_HUGGING"
echo "  - github:       $COUNT_GITHUB"
echo "  - html:         $COUNT_HTML"
echo ""
echo "Creating zip files in current directory (only files in mask)..."

# 为每个资源各自生成一个 zip（如果有文件）
if [ "$COUNT_HUGGING" -gt 0 ]; then
    HUGGING_ZIP="hugging_tables_mask_v2_${TAG}.zip"
    zip -q -@ "$HUGGING_ZIP" < "$TMP_HUGGING"
    SIZE_HUGGING=$(du -h "$HUGGING_ZIP" | cut -f1)
    echo "✅ Created: $HUGGING_ZIP ($SIZE_HUGGING)"
fi

if [ "$COUNT_GITHUB" -gt 0 ]; then
    GITHUB_ZIP="github_tables_mask_v2_${TAG}.zip"
    zip -q -@ "$GITHUB_ZIP" < "$TMP_GITHUB"
    SIZE_GITHUB=$(du -h "$GITHUB_ZIP" | cut -f1)
    echo "✅ Created: $GITHUB_ZIP ($SIZE_GITHUB)"
fi

if [ "$COUNT_HTML" -gt 0 ]; then
    HTML_ZIP="html_tables_mask_v2_${TAG}.zip"
    zip -q -@ "$HTML_ZIP" < "$TMP_HTML"
    SIZE_HTML=$(du -h "$HTML_ZIP" | cut -f1)
    echo "✅ Created: $HTML_ZIP ($SIZE_HTML)"
fi

echo ""
echo "Done."

