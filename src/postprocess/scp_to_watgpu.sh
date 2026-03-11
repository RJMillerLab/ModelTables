#!/bin/bash

# Script to zip table directories and scp files to remote server (watgpu or chippie)
# Usage: ./scp_to_watgpu.sh [mode] [tag] [--dry-run]
# Example: ./scp_to_watgpu.sh chippie 251117
# Example: ./scp_to_watgpu.sh watgpu 251117
# Example: ./scp_to_watgpu.sh chippie 251117 --dry-run
# 
# Modes:
#   chippie: /u1/z6dong/Repo/...
#   watgpu:  /u501/z6dong/Repo/...
#
# Note: Files are transferred maintaining the same relative path structure
#       e.g., Repo/CitationLake/data/processed/xxx -> Repo/CitationLake/data/processed/xxx

set -e

# Parse arguments
MODE="${1:-chippie}"
if [ "$MODE" = "--dry-run" ]; then
    # If first arg is --dry-run, use default mode
    MODE="chippie"
    TAG="${2:-251117}"
    DRY_RUN="--dry-run"
elif [ "$2" = "--dry-run" ]; then
    TAG="${3:-251117}"
    DRY_RUN="--dry-run"
else
    TAG="${2:-251117}"
    DRY_RUN="${3:-}"
fi

REMOTE_USER="z6dong"

# Set remote host and base path based on mode
case "$MODE" in
    chippie)
        REMOTE_HOST="chippie.cs.uwaterloo.ca"
        REMOTE_BASE="/u1/z6dong/Repo"
        ;;
    watgpu)
        REMOTE_HOST="watgpu.cs.uwaterloo.ca"
        REMOTE_BASE="/u501/z6dong/Repo"
        ;;
    *)
        echo "❌ Error: Invalid mode '$MODE'. Use 'chippie' or 'watgpu'"
        exit 1
        ;;
esac

LOCAL_BASE="/Users/doradong/Repo/CitationLake"

echo "=========================================="
echo "Preparing files for transfer"
echo "=========================================="
echo "Mode: $MODE"
echo "Tag: $TAG"
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}"
echo "Dry run: ${DRY_RUN:-no}"
echo ""

# Table directories to zip
TABLE_DIRS=(
    "data/processed/deduped_hugging_csvs_v2_${TAG}"
    "data/processed/deduped_github_csvs_v2_${TAG}"
    "data/processed/tables_output_v2_${TAG}"
    "data/processed/llm_tables_${TAG}"
)

# Parquet files to transfer directly (no zip)
# Core files needed for starmie (optional but useful for metadata)
PARQUET_FILES=(
    "data/processed/modelcard_step3_dedup_v2_${TAG}.parquet"
    "data/processed/final_integration_with_paths_v2_${TAG}.parquet"
)

# Text files to transfer directly
TEXT_FILES=(
    "data/analysis/all_valid_title_valid_${TAG}.txt"
)

# File lists for starmie (only base filelists, variants are deprecated)
FILE_LISTS=(
    "scilake_final_filelist.txt"
    "scilake_final_filelist_val.txt"
)

# Ground truth files to transfer (zip all GT files with tag)
GT_FILES_PATTERN="data/gt/*${TAG}*"

# Create temporary directory for zip files
TMP_DIR=$(mktemp -d)
echo "Temporary directory: $TMP_DIR"

# Function to zip directory
zip_directory() {
    local dir_path="$1"
    local dir_name=$(basename "$dir_path")
    local zip_path="${TMP_DIR}/${dir_name}.zip"
    
    if [ ! -d "$dir_path" ]; then
        echo "⚠️  Warning: Directory $dir_path does not exist, skipping..."
        return 1
    fi
    
    echo "📦 Zipping $dir_path..."
    if [ "$DRY_RUN" != "--dry-run" ]; then
        cd "$(dirname "$dir_path")"
        zip -rq "$zip_path" "$dir_name"
        cd - > /dev/null
        echo "   ✓ Created: $zip_path ($(du -h "$zip_path" | cut -f1))"
    else
        echo "   [DRY RUN] Would create: $zip_path"
    fi
}

# Function to scp file
scp_file() {
    local local_path="$1"
    local remote_path="$2"
    
    if [ ! -f "$local_path" ]; then
        echo "⚠️  Warning: File $local_path does not exist, skipping..."
        return 1
    fi
    
    # Create remote directory if needed
    # remote_path is already full path like /u1/z6dong/Repo/CitationLake/data/processed/...
    remote_dir=$(dirname "$remote_path")
    if [ "$DRY_RUN" != "--dry-run" ]; then
        ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${remote_dir}" 2>/dev/null || true
    fi
    
    echo "📤 Transferring $local_path..."
    if [ "$DRY_RUN" != "--dry-run" ]; then
        scp "$local_path" "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}"
        echo "   ✓ Transferred to: ${remote_path}"
    else
        echo "   [DRY RUN] Would transfer to: ${remote_path}"
    fi
}

# Zip table directories
echo ""
echo "=== Zipping table directories ==="
for dir in "${TABLE_DIRS[@]}"; do
    zip_directory "$dir"
done

# Transfer parquet files
echo ""
echo "=== Transferring parquet files ==="
for file in "${PARQUET_FILES[@]}"; do
    # Maintain relative path: Repo/CitationLake/data/processed/xxx
    remote_file="${REMOTE_BASE}/CitationLake/${file}"
    scp_file "$file" "$remote_file"
done

# Transfer text files
echo ""
echo "=== Transferring text files ==="
for file in "${TEXT_FILES[@]}"; do
    filename=$(basename "$file")
    
    # Transfer to CitationLake/data/analysis/ (maintain relative path)
    remote_file="${REMOTE_BASE}/CitationLake/${file}"
    scp_file "$file" "$remote_file"
    
    # Also transfer to starmie_internal/val_file/ (if starmie_internal exists)
    remote_starmie_file="${REMOTE_BASE}/starmie_internal/val_file/${filename}"
    if [ "$DRY_RUN" != "--dry-run" ]; then
        ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_BASE}/starmie_internal/val_file" 2>/dev/null || true
    fi
    scp_file "$file" "$remote_starmie_file" || echo "   ⚠️  starmie_internal not found, skipping..."
done

# Transfer zip files
echo ""
echo "=== Transferring zip files ==="
for dir in "${TABLE_DIRS[@]}"; do
    dir_name=$(basename "$dir")
    zip_file="${TMP_DIR}/${dir_name}.zip"
    # Maintain relative path: Repo/CitationLake/data/processed/xxx.zip
    remote_zip="${REMOTE_BASE}/CitationLake/data/processed/${dir_name}.zip"
    
    if [ -f "$zip_file" ]; then
        scp_file "$zip_file" "$remote_zip"
        # Extract on remote server
        if [ "$DRY_RUN" != "--dry-run" ]; then
            echo "📥 Extracting ${dir_name}.zip on remote server..."
            ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_BASE}/CitationLake/data/processed && unzip -q -o ${dir_name}.zip && rm ${dir_name}.zip" 2>/dev/null || true
        fi
    fi
done

# Transfer file lists (only base filelists, variants deprecated)
echo ""
echo "=== Transferring file lists ==="
for file in "${FILE_LISTS[@]}"; do
    if [ -f "$file" ]; then
        # Maintain relative path: Repo/CitationLake/scilake_final_filelist.txt
        remote_file="${REMOTE_BASE}/CitationLake/${file}"
        scp_file "$file" "$remote_file"
        # Also transfer to starmie_internal if it exists (same relative path)
        if [ "$DRY_RUN" != "--dry-run" ]; then
            ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_BASE}/starmie_internal" 2>/dev/null || true
            scp_file "$file" "${REMOTE_BASE}/starmie_internal/${file}" || echo "   ⚠️  starmie_internal not found, skipping..."
        fi
    fi
done

# Transfer ground truth files
echo ""
echo "=== Transferring ground truth files ==="
echo "📦 Zipping GT files..."
cd data/gt
GT_ZIP="${TMP_DIR}/gt_${TAG}.zip"
if [ "$DRY_RUN" != "--dry-run" ]; then
    # Zip all GT files with the tag
    zip -q -r "$GT_ZIP" . -i "*${TAG}*" 2>/dev/null || {
        echo "⚠️  Warning: No GT files found with tag ${TAG}, trying all GT files..."
        zip -q -r "$GT_ZIP" . 2>/dev/null || echo "⚠️  No GT files found"
    }
    if [ -f "$GT_ZIP" ]; then
        # Maintain relative path: Repo/CitationLake/data/gt/gt_xxx.zip
        remote_gt_zip="${REMOTE_BASE}/CitationLake/data/gt/gt_${TAG}.zip"
        echo "📤 Transferring: gt_${TAG}.zip -> ${REMOTE_BASE}/CitationLake/data/gt/"
        scp "$GT_ZIP" ${REMOTE_USER}@${REMOTE_HOST}:${remote_gt_zip}
        echo "📥 Extracting on remote server..."
        ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_BASE}/CitationLake/data/gt && unzip -q -o gt_${TAG}.zip && rm gt_${TAG}.zip" 2>/dev/null || true
        echo "✅ Done: GT files"
    fi
else
    echo "   [DRY RUN] Would zip and transfer GT files"
fi
cd - > /dev/null

# Cleanup
echo ""
echo "=== Cleanup ==="
if [ "$DRY_RUN" != "--dry-run" ]; then
    echo "Removing temporary directory: $TMP_DIR"
    rm -rf "$TMP_DIR"
else
    echo "[DRY RUN] Would remove: $TMP_DIR"
fi

echo ""
echo "=========================================="
echo "✅ Transfer complete!"
echo "=========================================="
echo ""
echo "📋 Summary of transferred files for Starmie:"
echo "   ✓ Table CSV directories (4): deduped_hugging_csvs, deduped_github_csvs, tables_output, llm_tables"
echo "   ✓ Ground truth files: data/gt/*${TAG}*.npz and *.pkl"
echo "   ✓ Analysis files: all_valid_title_valid_${TAG}.txt"
echo "   ✓ Base file lists: scilake_final_filelist.txt, scilake_final_filelist_val.txt"
echo "   ✓ Metadata parquet files (optional)"
echo ""
echo "   Note: Only base filelists are transferred. Variants (_s, _t, _s_t) are deprecated."
echo ""
echo "📝 Next steps on ${MODE}:"
echo "   1. ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "   2. cd ${REMOTE_BASE}/CitationLake"
echo "   3. Run symlink scripts: python -m src.data_symlink.ln_scilake --repo_root ${REMOTE_BASE} --mode all --dir-name scilake_final_${TAG}"
echo "   4. Run Starmie scripts: TAG=${TAG} bash scripts/step2_extractvectors.sh"
echo ""

