#!/bin/bash
# Workflow 2: Correlation Detection & Feature Grouping

set -e

INPUT="${1:-data/preprocessed.parquet}"
OUTPUT="${2:-config/privacy_blocks.json}"
CORR_THRESHOLD="${3:-0.3}"
MI_THRESHOLD="${4:-0.1}"

echo "=========================================="
echo "WORKFLOW 2: CORRELATION & FEATURE GROUPING"
echo "=========================================="
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Correlation threshold: $CORR_THRESHOLD"
echo "MI threshold: $MI_THRESHOLD"
echo ""

python -m src.correlation.block_builder \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --corr-threshold "$CORR_THRESHOLD" \
    --mi-threshold "$MI_THRESHOLD" \
    --use-mi

echo ""
echo "Correlation detection complete!"
