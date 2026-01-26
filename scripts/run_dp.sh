#!/bin/bash
# Workflow 3: Differential Privacy & Budget Allocation

set -e

INPUT="${1:-data/preprocessed.parquet}"
BLOCKS="${2:-config/privacy_blocks.json}"
BOUNDS="${3:-config/bounds.json}"
EPSILON="${4:-1.0}"
OUTPUT="${5:-data/privatized.parquet}"
MECHANISM="${6:-laplace}"

echo "=========================================="
echo "WORKFLOW 3: DIFFERENTIAL PRIVACY"
echo "=========================================="
echo "Input: $INPUT"
echo "Blocks: $BLOCKS"
echo "Bounds: $BOUNDS"
echo "Epsilon: $EPSILON"
echo "Mechanism: $MECHANISM"
echo "Output: $OUTPUT"
echo ""

python -m src.dp.privatize \
    --input "$INPUT" \
    --blocks "$BLOCKS" \
    --bounds "$BOUNDS" \
    --epsilon "$EPSILON" \
    --output "$OUTPUT" \
    --mechanism "$MECHANISM"

echo ""
echo "Differential privacy application complete!"
